// Sentinel Client Logic v3.0 // NERO SPATIAL
// Encapsulates the Gatekeeper and UI interactions

class SentinelTerminal {
    constructor() {
        this.container = document.getElementById('terminal-content');
        this.box = document.getElementById('terminal-box');
    }

    log(message, type = 'INFO') {
        if (!this.container) return;

        const line = document.createElement('div');
        const color = type === 'ALERT' ? '#ff4444' : (type === 'PROCESS' ? '#3388ff' : '#aaaaaa');
        const timestamp = new Date().toLocaleTimeString().split(' ')[0];

        line.innerHTML = `<span style="color:#555">[${timestamp}]</span> <span style="color:${color}">${message}</span>`;
        line.style.marginBottom = '2px';

        this.container.appendChild(line);

        // Auto-scroll
        if (this.box) this.box.scrollTop = this.box.scrollHeight;
    }
}

class Gatekeeper {
    constructor() {
        this.terminal = new SentinelTerminal();
        this.video = document.getElementById('video-player');
        this.image = document.getElementById('image-preview');
        this.fileInput = document.getElementById('file-input');

        // Global Error Handler for UI Debugging
        window.onerror = (msg, url, lineNo, columnNo, error) => {
            this.terminal.log(`SYSTEM ERROR: ${msg} (Line ${lineNo})`, 'ALERT');
            return false;
        };

        // Listeners
        this.initListeners();
    }

    initListeners() {
        // File Input is handled via HTML shim, but we can double check
        // The HTML script tag handles the UI transitions. 
        // This class handles the LOGIC/API.

        // Expose handle function to window for the HTML script to call
        window.handleFileSelect = (e) => {
            if (e.target.files.length) {
                this.handleFile(e.target.files[0]);
            }
        };

        document.getElementById('btn-reset').addEventListener('click', () => {
            location.reload();
        });
    }

    handleFile(file) {
        const isVideo = file.type.startsWith('video/');
        const isImage = file.type.startsWith('image/');

        if (!isVideo && !isImage) {
            this.terminal.log('INVALID FILE TYPE.', 'ALERT');
            return;
        }

        this.terminal.log(`SIGNAL RECEIVED: ${file.name.toUpperCase()}`, 'PROCESS');
        const url = URL.createObjectURL(file);

        // Upload Preview Logic
        const dropZone = document.getElementById('drop-zone');
        dropZone.style.backgroundImage = `url('${url}')`; // Use blob for image or video poster if simple

        // If it's a video, we might not get a frame easily without creating a hidden video element to canvas snapshot
        // For simplicity, we just rely on type.

        // Update Label
        document.querySelector('.upload-icon').className = 'fa-solid fa-check upload-icon';
        dropZone.querySelector('.text-technical').textContent = file.name;
        dropZone.querySelector('.text-technical').style.background = 'rgba(0,0,0,0.7)';
        dropZone.querySelector('.text-technical').style.padding = '2px 8px';

        // Switch Active Step to Metrics
        document.getElementById('step-upload').classList.remove('active');
        document.getElementById('step-metrics').classList.add('active'); // FIX: Makes results visible

        // UI Updates
        document.querySelector('.placeholder-signal').classList.add('hidden');

        if (isVideo) {
            this.video.src = url;
            this.video.classList.remove('hidden');
            this.image.classList.add('hidden');
        } else {
            this.image.src = url;
            this.image.classList.remove('hidden');
            this.video.classList.add('hidden');
        }

        // Start Analysis
        this.uploadAndAnalyze(file);
    }

    async uploadAndAnalyze(file) {
        this.terminal.log('INITIATING FORENSIC SCAN...', 'PROCESS');

        // Reset Metrics
        document.getElementById('stat-verdict').textContent = "scanning...";
        document.getElementById('stat-confidence').textContent = "0%";
        document.getElementById('score-bar').style.width = "0%";

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/scan', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server Error: ${response.statusText}`);
            }

            const result = await response.json();
            this.handleResults(result);

        } catch (error) {
            this.terminal.log(`SCAN ERROR: ${error.message.toUpperCase()}`, 'ALERT');
            console.error(error);
            document.getElementById('stat-verdict').textContent = "ERROR";
            document.getElementById('stat-verdict').style.color = "red";
        }
    }

    handleResults(data) {
        this.terminal.log('ANALYSIS COMPLETE. DECODING RESULTS.', 'PROCESS');

        const verdict = data.video_is_fake ? "DEEPFAKE DETECTED" : "REAL FOOTAGE";
        let confidenceVal = data.overall_confidence * 100;
        if (isNaN(confidenceVal)) confidenceVal = 0.0;

        let fakeScore = data.fake_score;
        if (isNaN(fakeScore) || fakeScore === undefined) fakeScore = 0.0;

        // Update Stats (Nero Style)
        const verdictEl = document.getElementById('stat-verdict');
        verdictEl.textContent = verdict;
        verdictEl.style.color = data.video_is_fake ? '#ff3333' : '#0066ff'; // Red vs Nero Blue

        document.getElementById('stat-confidence').textContent = `${confidenceVal.toFixed(1)}%`;

        // Animate Bar
        setTimeout(() => {
            document.getElementById('score-bar').style.width = `${fakeScore}%`;
            document.getElementById('score-bar').style.backgroundColor = data.video_is_fake ? '#ff3333' : '#0066ff';
        }, 100);

        if (data.video_is_fake) {
            this.terminal.log(`[ALERT] HIGH PROBABILITY OF MANIPULATION DETECTED.`, 'ALERT');
            if (data.evidence && data.evidence.length > 0) {
                this.terminal.log(`LOGGED ${data.evidence.length} ANOMALIES.`, 'WARN');
                this.displayEvidence(data.evidence);
            }
        } else {
            this.terminal.log(`[SUCCESS] INTEGRITY VERIFIED.`, 'PROCESS');
        }
    }

    displayEvidence(evidenceItems) {
        // Robustness: Handle potential cache mismatch where 'evidence-grid' doesn't exist yet
        const grid = document.getElementById('evidence-grid') || document.getElementById('evidence-list');
        const count = document.getElementById('evidence-count');

        if (!grid) {
            console.error("Evidence container not found (Cache mismatch?)");
            return;
        }

        grid.innerHTML = ''; // Clear placeholder
        if (count) count.textContent = `${evidenceItems.length} FRAMES LOGGED`;

        evidenceItems.forEach(item => {
            const card = document.createElement('div');
            card.className = 'evidence-card';

            const img = document.createElement('img');
            img.src = item.path;
            img.className = 'evidence-thumb';

            // Safety: Ensure timestamp is a number
            const ts = typeof item.timestamp === 'number' ? item.timestamp : 0.0;

            const meta = document.createElement('div');
            meta.className = 'evidence-meta';
            // Added explicit TAG as requested
            meta.innerHTML = `
                <div style="display:flex; justify-content:space-between; width:100%; align-items:center;">
                    <span style="color:#ff3333; font-size:0.7em; letter-spacing:1px; font-weight:bold;">ANOMALY</span>
                    <span>TS: ${ts.toFixed(2)}s <i class="fa-solid fa-play" style="margin-left:5px;"></i></span>
                </div>`;

            card.onclick = () => {
                const vid = this.video;
                if (vid && !vid.classList.contains('hidden')) {
                    vid.currentTime = ts;
                    vid.play();
                    // Scroll to player
                    vid.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            };

            card.appendChild(img);
            card.appendChild(meta);
            grid.appendChild(card);
        });
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.gatekeeper = new Gatekeeper();
});
