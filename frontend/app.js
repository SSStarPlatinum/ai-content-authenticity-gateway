/**
 * AI 内容鉴真推理网关 - 前端演示页逻辑
 */
(function () {
    'use strict';

    const API_BASE = '/api/v1';
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const resultSection = document.getElementById('resultSection');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const previewImage = document.getElementById('previewImage');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    // ===== 健康检查 =====
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            if (data.status === 'ok' || data.status === 'degraded') {
                statusDot.className = 'status-dot online';
                statusText.textContent = `服务在线 · ${data.models_loaded}/${data.models_total} 模型`;
            } else {
                statusDot.className = 'status-dot offline';
                statusText.textContent = '服务异常';
            }
        } catch (e) {
            statusDot.className = 'status-dot offline';
            statusText.textContent = '无法连接服务';
        }
    }
    checkHealth();
    setInterval(checkHealth, 30000);

    // ===== 上传交互 =====
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // ===== 处理文件 =====
    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('请选择图片文件');
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            alert('图片大小不能超过 20MB');
            return;
        }

        // 预览
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // 显示加载
        loadingOverlay.style.display = 'flex';
        resultSection.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch(`${API_BASE}/detect`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            renderResult(data, file);
        } catch (err) {
            alert('检测失败: ' + err.message);
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    // ===== 渲染结果 =====
    function renderResult(data, file) {
        resultSection.style.display = 'block';

        // 判决标签
        const labelEl = document.getElementById('verdictLabel');
        labelEl.textContent = getLabelText(data.label);
        labelEl.className = 'verdict-label ' + data.label;

        // 概率
        const probPercent = (data.fake_prob * 100).toFixed(1);
        document.getElementById('probValue').textContent = probPercent;
        document.getElementById('probFill').style.width = probPercent + '%';

        // 置信度 & 耗时
        document.getElementById('confidenceValue').textContent =
            (data.confidence * 100).toFixed(1) + '%';
        document.getElementById('latencyValue').textContent =
            data.total_latency_ms.toFixed(1);

        // 缓存命中
        const cacheBadge = document.getElementById('cacheBadge');
        cacheBadge.style.display = data.cache_hit ? 'inline-block' : 'none';

        // 图片元信息
        const meta = data.image_meta || {};
        document.getElementById('imageMeta').innerHTML = `
            <span>尺寸: <strong>${meta.width}×${meta.height}</strong></span>
            <span>格式: <strong>${meta.format}</strong></span>
            <span>大小: <strong>${(file.size / 1024).toFixed(1)} KB</strong></span>
            <span>后处理: <strong>${data.postprocessing_type}</strong></span>
            <span>融合策略: <strong>${data.strategy}</strong></span>
        `;

        // 各模型分数
        renderModelScores(data.per_model || []);

        // 滚动到结果
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function renderModelScores(models) {
        const container = document.getElementById('modelScores');
        container.innerHTML = '';

        if (!models.length) {
            container.innerHTML = '<p style="color:var(--text-dim)">无模型数据</p>';
            return;
        }

        models.forEach((m) => {
            const item = document.createElement('div');
            item.className = 'model-score-item';

            const percent = (m.fake_prob * 100).toFixed(1);
            const level = m.fake_prob < 0.33 ? 'low' : m.fake_prob < 0.66 ? 'mid' : 'high';

            item.innerHTML = `
                <div class="model-name" title="${m.model_id}">${m.model_id}</div>
                <div class="model-bar-track">
                    <div class="model-bar-fill ${level}" style="width:${percent}%"></div>
                </div>
                <div class="model-prob">${percent}%</div>
            `;
            if (m.error) {
                const err = document.createElement('div');
                err.className = 'model-error';
                err.style.gridColumn = '1 / -1';
                err.textContent = '⚠ ' + m.error;
                item.appendChild(err);
            }
            container.appendChild(item);
        });
    }

    function getLabelText(label) {
        const map = { fake: 'AI 生成', real: '真实图像', uncertain: '无法判定' };
        return map[label] || label;
    }
})();
