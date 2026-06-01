// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
    });
});

function showLoading() { document.getElementById('loading').classList.remove('hidden'); }
function hideLoading() { document.getElementById('loading').classList.add('hidden'); }
function showError(msg) {
    const el = document.getElementById('error');
    el.textContent = msg;
    el.classList.remove('hidden');
}
function hideError() { document.getElementById('error').classList.add('hidden'); }

// ========== Tab 1: 正面 → 反面 ==========
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('front-upload');
const uploadPreview = document.getElementById('upload-preview');
const btnTask2 = document.getElementById('btn-task2');
let uploadedFile = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreview.src = e.target.result;
        uploadPreview.classList.remove('hidden');
        // Hide upload text content
        document.querySelector('.upload-content').style.display = 'none';
        btnTask2.disabled = false;
    };
    reader.readAsDataURL(file);
}

btnTask2.addEventListener('click', async () => {
    if (!uploadedFile) return;
    hideError();
    showLoading();
    document.getElementById('task2-result').classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('image', uploadedFile);

        const resp = await fetch('/api/generate_back', {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Generation failed');

        document.getElementById('task2-front').src = 'data:image/png;base64,' + data.front;
        document.getElementById('task2-back').src = 'data:image/png;base64,' + data.back;
        document.getElementById('task2-result').classList.remove('hidden');
    } catch (err) {
        showError(err.message);
    } finally {
        hideLoading();
    }
});
