class SentinelClient {
    constructor() {
        this.video = document.getElementById('video-player');
        this.image = document.getElementById('image-preview');
        this.placeholder = document.getElementById('placeholder');
        this.dropZone = document.getElementById('drop-zone');
        this.fileInput = document.getElementById('file-input');

        this.initListeners();
    }

    initListeners() {
        // Browse Button
        document.getElementById('btn-browse').addEventListener('click', () => {
            this.fileInput.click();
        });

        // File Input Change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) this.handleFile(e.target.files[0]);
        });

        // Drag & Drop
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.style.borderColor = 'var(--primary)';
        });

        this.dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.dropZone.style.borderColor = '';
        });

        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.style.borderColor = '';
            if (e.dataTransfer.files.length) this.handleFile(e.dataTransfer.files[0]);
        });
    }

    handleFile(file) {
        // 1. Update UI Info
        document.getElementById('upload-info').classList.remove('hidden');
        document.getElementById('filename').textContent = file.name;

        // 2. Show Preview
        const url = URL.createObjectURL(file);
        this.placeholder.classList.add('hidden');

        if (file.type.startsWith('video/')) {
            this.video.src = url;
            this.video.classList.remove('hidden');
            this.image.classList.add('hidden');
        } else {
            this.image.src = url;
            this.image.classList.remove('hidden');
            this.video.classList.add('hidden');
        }

        // 3. Upload & Analyze
        this.analyze(file);
    }

    async analyze(file) {
        // Reset Stats
        document.getElementById('verdict-text').textContent = "ANALYZING...";
        document.getElementById('verdict-text').style.color = "var(--text-main)";
        document.getElementById('confidence-text').textContent = "0%";
        document.getElementById('evidence-grid').innerHTML = '<div class="empty-state">Scanning frames...</div>';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/scan', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Analysis failed");

            const data = await response.json();
            this.displayResults(data);

        } catch (err) {
            console.error(err);
            document.getElementById('verdict-text').textContent = "ERROR";
            document.getElementById('verdict-text').style.color = "var(--accent)";
            alert("Analysis failed. See console for details.");
        }
    }

    displayResults(data) {
        const isFake = data.video_is_fake;
        const verdictEl = document.getElementById('verdict-text');

        let typeLabel = "FOOTAGE";
        if (data.input_type === "image") typeLabel = "IMAGE";
        if (data.input_type === "video") typeLabel = "VIDEO";

        if (isFake) {
            verdictEl.textContent = `FAKE ${typeLabel} DETECTED`;
            verdictEl.style.color = "var(--accent)";
        } else {
            verdictEl.textContent = `REAL ${typeLabel}`;
            verdictEl.style.color = "#34d399";
        }

        const conf = (data.overall_confidence * 100).toFixed(1);
        document.getElementById('confidence-text').textContent = `${conf}%`;

        // Display Evidence
        const grid = document.getElementById('evidence-grid');
        grid.innerHTML = '';

        if (data.evidence && data.evidence.length > 0) {
            data.evidence.forEach(item => {
                const card = document.createElement('div');
                card.className = 'evidence-card';

                const ts = item.timestamp || 0;

                card.innerHTML = `
                    <img src="${item.path}" loading="lazy">
                    <div class="evidence-meta">
                        <span class="tag-anomaly">ANOMALY</span>
                        <i class="fa-solid fa-clock"></i> ${ts.toFixed(2)}s
                    </div>
                `;

                card.onclick = () => {
                    if (!this.video.classList.contains('hidden')) {
                        this.video.currentTime = ts;
                        this.video.play();
                    }
                };

                grid.appendChild(card);
            });
        } else {
            grid.innerHTML = '<div class="empty-state">No anomalies found.</div>';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SentinelClient();
});
