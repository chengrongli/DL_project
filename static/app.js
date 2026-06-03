// ============================================================================
// Pixel Sprite Generator — Front-end Logic
// ============================================================================

// ---- Color map for color dots ----
const COLOR_HEX = {
    black:  '#1a1a1a', white:  '#f0f0f0', gray:   '#888888', brown:  '#8B4513',
    red:    '#e94560', pink:   '#ff69b4', orange:  '#ff8c00', yellow: '#ffd700',
    green:  '#22c55e', teal:   '#14b8a6', blue:   '#3b82f6', purple: '#a855f7',
    gold:   '#d4a017', silver: '#c0c0c0', copper: '#b87333',
};

// ---- Tab switching ----
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
    });
});

// ---- Utilities ----
function showLoading(text = '生成中…') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading').classList.remove('hidden');
}
function hideLoading() { document.getElementById('loading').classList.add('hidden'); }
function showError(msg) {
    const el = document.getElementById('error');
    el.textContent = msg;
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 5000);
}
function hideError() { document.getElementById('error').classList.add('hidden'); }

function randomInt() { return Math.floor(Math.random() * 2147483647); }

// ---- Render a batch of results into a container ----
function renderResults(containerId, images) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    images.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = 'sprite-card';
        card.innerHTML = `
            <div class="sprite-label-row">
                <span class="sprite-label">正面</span>
                <span class="sprite-label">反面</span>
            </div>
            <div class="sprite-row">
                <div class="sprite-frame">
                    <img class="sprite-img" src="data:image/png;base64,${item.front}" alt="front">
                </div>
                <div class="sprite-frame">
                    <img class="sprite-img" src="data:image/png;base64,${item.back}" alt="back">
                </div>
            </div>
            <div class="sprite-index">#${idx + 1}</div>
        `;
        container.appendChild(card);
    });

    container.classList.remove('hidden');
}

// ============================================================================
// Vocabulary loading — populate dropdowns from /api/vocabs
// ============================================================================
async function loadVocabs() {
    try {
        const resp = await fetch('/api/vocabs');
        const vocabs = await resp.json();

        // Friendly display names
        const LABEL_MAP = {
            // body_type
            male: '♂ 男性', female: '♀ 女性', muscular: '💪 壮硕',
            teen: '🧒 少年', child: '👶 儿童', adult: '🧑 成人',
            // hair_style
            short: '短发', medium: '中发', long: '长发', ponytail: '马尾',
            braid: '编辫', curly: '卷发', spiked: '刺头', bangs: '刘海',
            pigtails: '双马尾', dreadlocks: '脏辫', messy: '凌乱', parted: '分头', bun: '丸子头',
            // torso_type
            bare: '赤裸', clothes: '普通上衣', jacket: '夹克', armour: '盔甲',
            // legs_type
            pants: '长裤', shorts: '短裤', skirt: '裙子', dress: '连衣裙',
            leggings: '紧身裤', armour: '盔甲',
            // feet_type
            boots: '靴子', shoes: '鞋子', sandals: '凉鞋',
            // colors
            black: '黑色', white: '白色', gray: '灰色', brown: '棕色',
            red: '红色', pink: '粉色', orange: '橙色', yellow: '黄色',
            green: '绿色', teal: '青色', blue: '蓝色', purple: '紫色',
            gold: '金色', silver: '银色', copper: '铜色',
        };

        document.querySelectorAll('.attr-select[data-field]').forEach(select => {
            const field = select.dataset.field;
            const options = vocabs[field];
            if (!options) return;

            options.forEach(val => {
                const opt = document.createElement('option');
                opt.value = val;
                opt.textContent = LABEL_MAP[val] || capitalize(val);
                select.appendChild(opt);
            });
        });

        // Color dot updates
        document.querySelectorAll('.color-select').forEach(sel => {
            sel.addEventListener('change', updateColorDot);
        });
    } catch (err) {
        console.error('Failed to load vocabs:', err);
    }
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function updateColorDot(e) {
    const select = e.target;
    const field = select.dataset.field;
    const dot = document.querySelector(`.color-dot[data-for="${field}"]`);
    if (!dot) return;

    const color = select.value;
    if (color && COLOR_HEX[color]) {
        dot.style.background = COLOR_HEX[color];
        dot.style.borderColor = COLOR_HEX[color];
    } else {
        dot.style.background = 'transparent';
        dot.style.borderColor = '#333';
    }
}

loadVocabs();

// ============================================================================
// Tab 1: Conditional Generation
// ============================================================================

// CFG slider
const cfgSlider = document.getElementById('cfg-scale');
const cfgValue = document.getElementById('cfg-value');
cfgSlider.addEventListener('input', () => { cfgValue.textContent = cfgSlider.value; });

// Seed random buttons
document.getElementById('cond-seed-random').addEventListener('click', () => {
    document.getElementById('cond-seed').value = randomInt();
});
document.getElementById('random-seed-random').addEventListener('click', () => {
    document.getElementById('random-seed').value = randomInt();
});

document.getElementById('btn-conditional').addEventListener('click', async () => {
    hideError();
    showLoading('正在生成属性角色…');

    // Collect selected attributes
    const attrs = {};
    document.querySelectorAll('.attr-select[data-field]').forEach(sel => {
        if (sel.value) attrs[sel.dataset.field] = sel.value;
    });

    const body = {
        attrs,
        count: parseInt(document.getElementById('cond-count').value),
        guidance_scale: parseFloat(cfgSlider.value),
    };

    const seedInput = document.getElementById('cond-seed').value;
    if (seedInput) body.seed = parseInt(seedInput);

    try {
        const resp = await fetch('/api/generate_conditional', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Generation failed');

        renderResults('cond-results', data.images);
    } catch (err) {
        showError(err.message);
    } finally {
        hideLoading();
    }
});

// ============================================================================
// Tab 2: Random Generation
// ============================================================================

document.getElementById('btn-random').addEventListener('click', async () => {
    hideError();
    showLoading('正在随机生成…');

    const body = {
        count: parseInt(document.getElementById('random-count').value),
    };

    const seedInput = document.getElementById('random-seed').value;
    if (seedInput) body.seed = parseInt(seedInput);

    try {
        const resp = await fetch('/api/generate_random', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Generation failed');

        renderResults('random-results', data.images);
    } catch (err) {
        showError(err.message);
    } finally {
        hideLoading();
    }
});
