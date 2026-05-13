import os
from PIL import Image

# 你的原始文件夹（装满分开的图片）
input_dir = "../data/pairs/random_batch" 
# input_dir = "gen/DL_project_final/data/pairs/random_batch" 
# 拼好后保存的新文件夹（给训练代码用的）
output_dir = "pairs/processed" 

os.makedirs(output_dir, exist_ok=True)

# 假设你的文件前缀是名字，比如 hero_front.png 和 hero_back.png
# 我们先找出所有带 "_front" 的文件
for filename in os.listdir(input_dir):
    if "_front" in filename:
        # 找到对应的背面文件名
        back_filename = filename.replace("_front", "_back")
        
        front_path = os.path.join(input_dir, filename)
        back_path = os.path.join(input_dir, back_filename)
        
        if os.path.exists(back_path):
            # 打开两张图
            img_front = Image.open(front_path).convert("RGBA")
            img_back = Image.open(back_path).convert("RGBA")
            
            # 获取宽高 (假设都是 64x64)
            w, h = img_front.size
            
            # 创建一张两倍宽的透明新图 (宽128, 高64)
            new_img = Image.new("RGBA", (w * 2, h), (0, 0, 0, 0))
            
            # 左边贴正面，右边贴背面
            new_img.paste(img_front, (0, 0))
            new_img.paste(img_back, (w, 0))
            
            # 保存拼好的图
            save_name = filename.replace("_front", "_paired")
            new_img.save(os.path.join(output_dir, save_name))
            print(f"拼接成功: {save_name}")