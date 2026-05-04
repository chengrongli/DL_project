import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
from urllib.error import HTTPError, URLError

# ==================== 1. 扩散参数设置 ====================
class DiffusionConfig:
    """DDPM的扩散参数配置"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # 线性方差调度（论文中使用的）
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # 计算alpha参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) #cumulative product

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 用于计算前向过程q(x_t|x_0)的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 用于计算后验分布q(x_{t-1}|x_t, x_0)的参数（公式7）
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_process(self, x0, t, noise=None):
        """前向扩散过程：q(x_t|x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

"""Now we need to do time embedding, and also we know that only add $t$ is bad.
What we need to implement? Use several $sin$ function as time embedding. (from high frequency to low frequency)
"""

# ==================== 2. UNet模型架构 ====================
def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """安全地创建 GroupNorm：groups 取 min(max_groups, channels) 并能整除 channels。"""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class TimeEmbedding(nn.Module):
    """时间步的Transformer正弦位置编码（论文3.2节）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings
        # time embeddings

class ConvBlock(nn.Module):
    """卷积残差块（论文附录B）"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = _gn(in_channels)  # 使用组归一化而不是权重归一化
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = _gn(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)

        # 第一个卷积 + 时间嵌入
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 加入时间信息
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[:, :, None, None]

        # 第二个卷积
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual

class AttentionBlock(nn.Module):
    """自注意力块（论文4节，在16x16分辨率使用）"""
    def __init__(self, channels):
        super().__init__()
        self.norm = _gn(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        # c^2+c parameters
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        # 计算query, key, value
        x = self.norm(x)
        q = self.q(x).view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        k = self.k(x).view(B, C, -1)  # [B, C, H*W]
        v = self.v(x).view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        # 注意力机制
        attention = torch.bmm(q, k) * (C ** -0.5)
        attention = F.softmax(attention, dim=-1)
        h = torch.bmm(attention, v)  # [B, H*W, C]

        h = h.permute(0, 2, 1).view(B, C, H, W)
        h = self.proj_out(h)

        return h + residual

class UNet(nn.Module):
    """DDPM的U-Net架构（修复版本）"""
    def __init__(self, in_channels=3, model_channels=128, out_channels=3,
                 channel_mult=(1, 2, 2, 2), num_res_blocks=2,
                 dropout=0.1, time_emb_dim=256, attention_resolutions=(16,)):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.out_channels = out_channels

        # ==================== 下采样部分 ====================
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        # 跟踪当前通道数
        channels = model_channels
        # 保存各层输出通道数，用于上采样的拼接
        self.skip_connection_channels = [channels]

        # 构建下采样层
        for i, mult in enumerate(channel_mult):
            # 每个分辨率的多个残差块
            out_ch = mult * model_channels

            for _ in range(num_res_blocks):
                # 创建ConvBlock（可能后接AttentionBlock）
                layers = []
                layers.append(ConvBlock(channels, out_ch, time_emb_dim, dropout))
                channels = out_ch

                # 检查当前分辨率是否需要注意力
                resolution = 64 // (2 ** i)
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels))

                self.down_blocks.append(nn.ModuleList(layers))
                self.skip_connection_channels.append(channels)

            # 如果不是最后一层，添加下采样
            if i != len(channel_mult) - 1:
                self.downsample_blocks.append(
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                )
                self.skip_connection_channels.append(channels)

        # ==================== 中间部分 ====================
        self.mid_block1 = ConvBlock(channels, channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ConvBlock(channels, channels, time_emb_dim, dropout)

        # ==================== 上采样部分 ====================
        self.up_blocks = nn.ModuleList()

        # 反向遍历，从高分辨率到低分辨率
        for i, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = mult * model_channels

            # 每个分辨率的残差块（比下采样多一个，用于处理跳跃连接）
            for j in range(num_res_blocks + 1):
                # 关键修复：正确计算输入通道数
                # 跳跃连接会带来额外的通道
                skip_channels = self.skip_connection_channels.pop()
                # 输入通道 = 当前通道 + 跳跃连接的通道
                in_channels_up = channels + skip_channels

                # 创建ConvBlock
                layers = []
                layers.append(ConvBlock(in_channels_up, out_ch, time_emb_dim, dropout))
                channels = out_ch

                # 检查是否需要注意力
                resolution = 64 // (2 ** i)
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels))

                # 如果是该分辨率的最后一个块且不是最高分辨率，添加上采样
                if i != 0 and j == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # ==================== 输出层 ====================
        # 最后输出通道数应该是model_channels（128）
        # 但经过上采样后，channels可能是256，需要调整
        final_channels = model_channels
        self.output_norm = _gn(channels)
        self.output_conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

    def forward(self, x, t):
        # 1. 时间嵌入
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # 2. 初始卷积
        h = self.input_conv(x)
        # 保存跳跃连接的特征
        skip_features = [h]

        # 3. 下采样（按分辨率顺序交替处理残差块与下采样）
        down_block_idx = 0
        for i in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                module_list = self.down_blocks[down_block_idx]
                down_block_idx += 1
                for layer in module_list:
                    if isinstance(layer, ConvBlock):
                        h = layer(h, t_emb)
                    else:  # AttentionBlock
                        h = layer(h)
                skip_features.append(h)

            # 下采样（分辨率减半）
            if i != len(self.channel_mult) - 1:
                h = self.downsample_blocks[i](h)
                skip_features.append(h)

        # 4. 中间瓶颈
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # 5. 上采样（关键修复部分）
        for module_list in self.up_blocks:
            for layer in module_list:
                if isinstance(layer, ConvBlock):
                    # 从skip_features取出对应层的特征
                    skip = skip_features.pop()
                    # 拼接：当前特征 + 跳跃连接特征
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, t_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:  # 转置卷积上采样
                    h = layer(h)

        # 6. 输出
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h

# ==================== 3. DDPM训练和采样类 ====================
class DDPM(nn.Module):
    """完整的DDPM模型"""
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model  # UNet模型，预测噪声ϵ

    def compute_losses(
        self,
        x0,
        fg_mask=None,
        background_weight: float = 1.0,
        foreground_weight: float = 1.0,
        alpha_weight: float = 1.0,
        return_components: bool = False,
    ):
        """计算简化损失 L_simple（论文公式14）并可返回分解项（便于 debug）。

        Args:
            x0:  (B, C, H, W) 干净图像。
            fg_mask: 可选前景 mask，(B,1,H,W) 或 (B,C',H,W)，值域 [0,1]。
            background_weight: 背景损失权重（<=1 会降低背景权重）。
            foreground_weight: 前景额外放大系数（>=1 时提升前景梯度）。
            alpha_weight: RGBA 下 alpha 通道损失权重。
            return_components: True 时返回 (loss, components_dict)。
        """
        batch_size = x0.shape[0]

        # 随机选择时间步
        t = torch.randint(0, self.config.timesteps, (batch_size,), device=x0.device)

        # 采样随机噪声
        noise = torch.randn_like(x0)

        # 前向扩散得到 x_t
        xt = self.config.forward_process(x0, t, noise)

        # 预测噪声
        noise_pred = self.model(xt, t)
        diff = (noise_pred - noise) ** 2

        C = x0.shape[1]
        rgb_slice = slice(0, min(3, C))
        alpha_slice = slice(3, 4) if C >= 4 else None

        def _safe_div(num, den):
            return num / den.clamp_min(1e-6)

        # 默认不加权
        weights = torch.ones_like(diff)
        fg1 = None

        if fg_mask is not None:
            fg_mask = fg_mask.to(x0)
            fg1 = fg_mask[:, :1, :, :] if fg_mask.shape[1] >= 1 else fg_mask

            # 扩展到通道维
            if fg_mask.shape[1] != x0.shape[1]:
                if x0.shape[1] % fg_mask.shape[1] == 0:
                    repeat_factor = x0.shape[1] // fg_mask.shape[1]
                    fg_mask = fg_mask.repeat_interleave(repeat_factor, dim=1)
                else:
                    repeat_factor = (x0.shape[1] + fg_mask.shape[1] - 1) // fg_mask.shape[1]
                    fg_mask = fg_mask.repeat(1, repeat_factor, 1, 1)[:, : x0.shape[1], :, :]

            # 背景/前景基础权重
            bg_w = float(max(background_weight, 0.05))
            weights = bg_w + (1.0 - bg_w) * fg_mask

            # 进一步放大前景区域
            fg_w = float(max(foreground_weight, 1.0))
            weights = weights * (1.0 + (fg_w - 1.0) * fg_mask)

        # alpha 通道可单独加权
        if alpha_slice is not None:
            aw = float(max(alpha_weight, 0.0))
            weights[:, alpha_slice, :, :] = weights[:, alpha_slice, :, :] * aw

        loss = _safe_div((diff * weights).sum(), weights.sum())

        if not return_components:
            return loss

        components = {
            "loss_total": loss.detach(),
            "loss_mse_raw": diff.mean().detach(),
        }

        rgb_diff = diff[:, rgb_slice, :, :] if rgb_slice.stop > 0 else diff

        if fg1 is not None:
            fg_rgb = _safe_div(
                (rgb_diff * fg1).sum(),
                fg1.sum() * rgb_diff.shape[1],
            )
            bg1 = 1.0 - fg1
            bg_rgb = _safe_div(
                (rgb_diff * bg1).sum(),
                bg1.sum() * rgb_diff.shape[1],
            )
            components["loss_fg_rgb"] = fg_rgb.detach()
            components["loss_bg_rgb"] = bg_rgb.detach()
            components["fg_ratio"] = fg1.mean().detach()

        if alpha_slice is not None:
            components["loss_alpha"] = diff[:, alpha_slice, :, :].mean().detach()

        return loss, components

    @torch.no_grad()
    def p_sample(self, xt, t):
        """反向过程单步采样：p_θ(x_{t-1}|x_t)"""
        # 预测噪声
        noise_pred = self.model(xt, t)

        # 提取参数
        alpha_t = self.config.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.config.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.config.betas[t].view(-1, 1, 1, 1)

        # 计算均值（论文公式11）
        mean = (1 / torch.sqrt(alpha_t)) * (xt - beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_pred)

        if t[0] > 0:
            # 计算方差（固定方差，论文3.2节）
            variance = self.config.posterior_variance[t].view(-1, 1, 1, 1)

            # 采样
            noise = torch.randn_like(xt)
            xt_prev = mean + torch.sqrt(variance) * noise
        else:
            # t=0时，无随机噪声
            xt_prev = mean

        return xt_prev

    @torch.no_grad()
    def sample(self, sample_shape, progress_bar=True):
        """完整采样过程（论文算法2）"""
        device = next(self.model.parameters()).device
        xt = torch.randn(sample_shape, device=device)

        timesteps = list(range(self.config.timesteps))[::-1]
        if progress_bar:
            timesteps = tqdm(timesteps)

        for t in timesteps:
            t_batch = torch.full((sample_shape[0],), t, device=device, dtype=torch.long)
            xt = self.p_sample(xt, t_batch)

            # 将值裁剪到[-1, 1]范围（论文3.3节数据缩放）
            # xt = torch.clamp(xt, -1.0, 1.0)

        # 将输出从[-1, 1]转换回[0, 1]
        x0 = (xt + 1) / 2
        x0 = torch.clamp(x0, 0.0, 1.0) #把前面clamp注释了兜底
        return x0

# ==================== 4. 训练函数 ====================
def train_ddpm(config, model, dataloader, epochs=100, lr=2e-4,
               save_dir="./ddpm_results", device=None):
    """训练DDPM模型（论文算法1）"""
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    # device = torch.device("cuda")

    ddpm = DDPM(config, model).to(device)
    optimizer = optim.Adam(ddpm.parameters(), lr=lr)

    # 指数移动平均（论文附录B）
    ema_decay = 0.9999
    ema_model = DDPM(config, copy.deepcopy(model).to(device))
    ema_model.load_state_dict(ddpm.state_dict())

    losses = []
    step = 0

    for epoch in range(epochs):
        ddpm.train()
        epoch_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            # 计算损失并更新
            optimizer.zero_grad()
            loss = ddpm.compute_losses(images)
            loss.backward()
            optimizer.step()

            # 更新EMA模型
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), ddpm.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1-ema_decay)

            epoch_loss += loss.item()
            step += 1

            # 更新进度条
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': ddpm.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"ddpm_epoch_{epoch+1}.pth"))

        # 采样并保存示例图像
        if (epoch + 1) % 5 == 0:
            ddpm.eval()
            with torch.no_grad():
                samples = ema_model.sample(sample_shape=(16, 3, 64, 64), progress_bar=False)

                # 保存网格图像
                grid = utils.make_grid(samples, nrow=4, padding=2, normalize=True)
                utils.save_image(grid, os.path.join(save_dir, f"samples_epoch_{epoch+1}.png"))

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    return ddpm, ema_model


def load_cifar10_with_mirrors(root, train, transform, download=True):
    """尝试多个镜像源下载CIFAR-10，避免503错误"""
    mirrors = [
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz",
        "https://huggingface.co/datasets/uoft-cs-ml/CIFAR-10/resolve/main/cifar-10-python.tar.gz?download=1",
    ]
    os.makedirs(root, exist_ok=True)
    archive_path = os.path.join(root, "cifar-10-python.tar.gz")
    original_url = getattr(datasets.CIFAR10, "url", mirrors[0])
    last_error = None

    for mirror in mirrors:
        try:
            datasets.CIFAR10.url = mirror
            dataset = datasets.CIFAR10(
                root=root,
                train=train,
                transform=transform,
                download=download,
            )
            if mirror != original_url:
                print(f"CIFAR-10 downloaded from mirror: {mirror}")
            datasets.CIFAR10.url = original_url
            return dataset
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            last_error = exc
            print(f"Failed to download CIFAR-10 from {mirror}: {exc}")
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass

    datasets.CIFAR10.url = original_url
    raise RuntimeError(f"All CIFAR-10 mirrors failed. Last error: {last_error}")


# ==================== 5. 主程序 ====================
def main():
    # 设置设备（优先使用MPS）
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # device = torch.device("cuda")
    print(f"Using device: {device}")

    # 1. 创建扩散配置
    config = DiffusionConfig(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    # 2. 创建UNet模型
    model = UNet(
        in_channels=3,
        model_channels=128,  # 论文CIFAR10模型使用128个通道
        out_channels=3,
        channel_mult=(1, 2, 2, 2),  # 4个分辨率级别
        num_res_blocks=2,
        dropout=0.1,  # CIFAR10使用0.1的dropout（论文附录B）
        attention_resolutions=(16,8)  # 在16x16分辨率使用注意力
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 准备数据（使用CIFAR10示例）
    transform = transforms.Compose([
      transforms.Resize(64),
      transforms.CenterCrop(64),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_root = "./data"
    try:
        dataset = load_cifar10_with_mirrors(
            root=dataset_root,
            train=True,
            transform=transform,
            download=True,
        )
    except RuntimeError as err:
        print("CIFAR-10 download failed after trying backup mirrors.")
        print(err)
        print("Attempting to load an existing local copy (download=False)...")
        try:
            datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            dataset = datasets.CIFAR10(
                root=dataset_root,
                train=True,
                download=False,
                transform=transform,
            )
            print("Loaded existing CIFAR-10 dataset without downloading.")
        except Exception as local_err:
            raise RuntimeError(
                "无法下载或加载CIFAR-10数据集。请手动下载并将解压后的 'cifar-10-batches-py' 目录放入 ./data，"
                "随后重新运行该脚本。"
            ) from local_err

    dataloader = DataLoader(
        dataset,
        batch_size=128,  # 论文使用128的batch size
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # 4. 训练模型
    print("Starting training...")
    trained_model, ema_model = train_ddpm(
        config=config,
        model=model,
        dataloader=dataloader,
        epochs=500,  # 可以增加更多epoch以获得更好效果
        lr=2e-4,
        save_dir="./ddpm_cifar10",
        device=device
    )

    # 5. 最终采样
    print("Generating final samples...")
    ema_model.eval()
    with torch.no_grad():
        # 生成64个样本
        final_samples = ema_model.sample(sample_shape=(64, 3, 64, 64))

        # 保存最终样本网格
        grid = utils.make_grid(final_samples, nrow=8, padding=2, normalize=False)
        utils.save_image(grid, "./ddpm_cifar10/final_samples.png")

        # 显示样本
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.title("DDPM Generated Samples")
        plt.show()

    print("Training completed!")

# ==================== 6. 辅助函数 ====================
def generate_interpolation(self, x0, x0_prime, t, lambda_val):
    """在加噪空间插值"""
    # 固定同一个噪声
    noise = torch.randn_like(x0)

    # 把两张图加噪到第t步
    xt = self.config.forward_process(x0, t, noise)
    xt_prime = self.config.forward_process(x0_prime, t, noise)

    # 在加噪空间插值
    xt_interp = (1 - lambda_val) * xt + lambda_val * xt_prime

    # 从插值点反向还原
    x0_interp = self.sample_from_xt(xt_interp, t)
    return x0_interp

def progressive_generation(model, config, timesteps_to_show=[0, 250, 500, 750, 999], device=None):
    """渐进生成过程可视化（论文图6）"""
    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    with torch.no_grad():
        xt = torch.randn((1, 3, 64, 64), device=device)
        samples_at_t = []

        for t in reversed(range(config.timesteps)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            xt = model.p_sample(xt, t_batch)

            if t in timesteps_to_show:
                # 使用公式15估计x0
                alpha_cumprod_t = config.alphas_cumprod[t].view(-1, 1, 1, 1)
                x0_estimate = (xt - config.sqrt_one_minus_alphas_cumprod[t] *
                              model.model(xt, t_batch)) / config.sqrt_alphas_cumprod[t]
                samples_at_t.append(x0_estimate)

        return torch.cat(samples_at_t, dim=0)

if __name__ == "__main__":
    main()