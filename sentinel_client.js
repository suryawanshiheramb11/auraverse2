// Sentinel Client Logic v2.5 // RESTORED BLUE THEME
// Encapsulates the Gatekeeper and UI interactions

class SentinelTerminal {
    constructor() {
        this.console = document.getElementById('terminal-content');
    }

    log(message, type = 'INFO') {
        if (!this.console) return;
        const line = document.createElement('div');
        const timestamp = new Date().toLocaleTimeString();
        let colorClass = 'text-emerald-500';

        if (type === 'ALERT') colorClass = 'text-red-500 font-bold';
        if (type === 'WARN') colorClass = 'text-yellow-500';
        if (type === 'PROCESS') colorClass = 'text-cyan-400';

        line.className = `${colorClass} font-mono text-xs break-words`;
        line.innerHTML = `<span class="opacity-50">[${timestamp}]</span> ${message}`;

        this.console.appendChild(line);
        this.console.scrollTop = this.console.scrollHeight;
    }
}

class Gatekeeper {
    constructor() {
        this.terminal = new SentinelTerminal();
        this.video = document.getElementById('video-player');
        this.image = document.getElementById('image-preview');
        this.canvas = document.getElementById('process-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d', { willReadFrequently: true }) : null;
        this.fileInput = document.getElementById('file-input');
        this.dropZone = document.getElementById('drop-zone');
        this.controls = document.getElementById('video-controls');
        this.placeholder = document.getElementById('placeholder-text');

        this.initListeners();
        this.initPlayerControls();
    }

    initListeners() {
        // Drag and Drop
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('active');
        });

        this.dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('active');
        });

        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('active');
            if (e.dataTransfer.files.length) {
                this.handleFile(e.dataTransfer.files[0]);
            }
        });

        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.handleFile(e.target.files[0]);
            }
        });
    }

    initPlayerControls() {
        const btnPlay = document.getElementById('btn-play');
        const btnRew = document.getElementById('btn-rewind');
        const btnFwd = document.getElementById('btn-forward');
        const seekSlider = document.getElementById('seek-slider');
        const seekFill = document.getElementById('seek-fill');
        const seekThumb = document.getElementById('seek-thumb');

        if (!btnPlay) return;

        // Play/Pause
        btnPlay.addEventListener('click', () => {
            if (this.video.paused) {
                this.video.play();
                btnPlay.innerHTML = '<i class="fa-solid fa-pause"></i>';
            } else {
                this.video.pause();
                btnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
            }
        });

        // Skips
        btnRew.addEventListener('click', () => { this.video.currentTime -= 10; });
        btnFwd.addEventListener('click', () => { this.video.currentTime += 10; });

        // Seek Bar (Update visual)
        this.video.addEventListener('timeupdate', () => {
            if (!this.video.duration) return;
            const pct = (this.video.currentTime / this.video.duration) * 100;
            seekSlider.value = pct;
            seekFill.style.width = `${pct}%`;
            seekThumb.style.left = `${pct}%`;

            document.getElementById('time-current').textContent = this.formatTime(this.video.currentTime);
        });

        // Seek Input
        seekSlider.addEventListener('input', (e) => {
            const time = (e.target.value / 100) * this.video.duration;
            this.video.currentTime = time;
        });

        this.video.addEventListener('loadedmetadata', () => {
            document.getElementById('time-total').textContent = this.formatTime(this.video.duration);
            console.log(`Video loaded: ${this.video.videoWidth}x${this.video.videoHeight}`);
        });

        this.video.addEventListener('ended', () => {
            btnPlay.innerHTML = '<i class="fa-solid fa-rotate-left"></i>';
        });
    }

    formatTime(seconds) {
        if (!seconds) return "00:00";
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    handleFile(file) {
        const isVideo = file.type.startsWith('video/');
        const isImage = file.type.startsWith('image/');

        if (!isVideo && !isImage) {
            this.terminal.log('Invalid file type. Video or Image required.', 'WARN');
            return;
        }

        this.terminal.log(`File Loading: ${file.name}`, 'PROCESS');
        const url = URL.createObjectURL(file);

        // UI Reset
        this.video.classList.add('hidden');
        this.image.classList.add('hidden');
        this.controls.classList.add('pointer-events-none', 'opacity-50');
        this.controls.classList.remove('pointer-events-auto', 'opacity-100');

        if (this.placeholder) this.placeholder.classList.add('hidden');

        // Reset Evidence
        document.getElementById('evidence-list').innerHTML = '<div class="col-span-2 text-center text-slate-600 text-xs py-10 italic">Analysis in progress...</div>';
        document.getElementById('stat-verdict').textContent = "ANALYZING...";
        document.getElementById('stat-verdict').className = "text-lg font-bold text-yellow-500 animate-pulse";
        document.getElementById('stat-confidence').textContent = "--%";
        document.getElementById('score-bar').style.width = "0%";
        document.getElementById('fake-score-text').textContent = "0.00 / 100";


        if (isVideo) {
            this.video.src = url;
            this.video.classList.remove('hidden');
            this.controls.classList.remove('pointer-events-none', 'opacity-50');
            this.controls.classList.add('pointer-events-auto', 'opacity-100');

            // Upload
            this.uploadAndAnalyze(file);
        } else {
            this.image.src = url;
            this.image.classList.remove('hidden');
            this.uploadAndAnalyze(file);
        }
    }

    async uploadAndAnalyze(file) {
        this.terminal.log('Transmitting data to Sentinel Backend...', 'PROCESS');

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
            this.terminal.log(`ANALYSIS FAILED: ${error.message}`, 'ALERT');
            document.getElementById('stat-verdict').textContent = "ERROR";
            document.getElementById('stat-verdict').className = "text-lg font-bold text-red-500";
        }
    }

    handleResults(data) {
        this.terminal.log('Analysis Complete. Processing Results...', 'PROCESS');

        const verdict = data.video_is_fake ? "FAKE DETECTED" : "AUTHENTIC";

        let confidenceVal = data.overall_confidence * 100;
        if (isNaN(confidenceVal)) confidenceVal = 0.0;
        const confidence = confidenceVal.toFixed(1);

        // Update Stats
        const verdictEl = document.getElementById('stat-verdict');
        verdictEl.textContent = verdict;
        verdictEl.className = data.video_is_fake
            ? "text-lg font-bold text-red-500 neon-text"
            : "text-lg font-bold text-emerald-500 neon-text";

        document.getElementById('stat-confidence').textContent = `${confidence}%`;

        // Fake Score Bar
        let fakeScore = data.fake_score;
        if (isNaN(fakeScore) || fakeScore === undefined) fakeScore = 0.0;

        document.getElementById('score-bar').style.width = `${fakeScore}%`;
        document.getElementById('score-bar').className = `h-full transition-all duration-1000 ${fakeScore > 50 ? 'bg-red-500' : 'bg-yellow-500'}`;
        document.getElementById('fake-score-text').textContent = `${fakeScore.toFixed(2)} / 100`;

        if (data.video_is_fake) {
            this.terminal.log(`[ALERT] AI SYNTHESIS DETECTED.`, 'ALERT');

            // Evidence
            if (data.evidence && data.evidence.length > 0) {
                this.displayEvidence(data.evidence);
            } else {
                this.terminal.log("[INFO] No visual evidence extracted (Advanced Analysis used).", 'WARN');
                document.getElementById('evidence-list').innerHTML = '<div class="col-span-2 text-center text-slate-500 text-xs py-10">Confirmed by Advanced Analysis</div>';
            }
        } else {
            this.terminal.log(`[SUCCESS] Content appears authentic.`, 'PROCESS');
            document.getElementById('evidence-list').innerHTML = '<div class="col-span-2 text-center text-emerald-800 text-xs py-10">No Evidence Found</div>';
            document.getElementById('score-bar').style.width = "0%";
            document.getElementById('score-bar').className = "h-full bg-emerald-500";
        }
    }

    displayEvidence(evidenceItems) {
        const list = document.getElementById('evidence-list');
        list.className = "flex-grow overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-4 content-start pr-2";
        list.innerHTML = ''; // Clear

        for (let i = 0; i < evidenceItems.length; i++) {
            const item = evidenceItems[i];
            const container = document.createElement('div');
            container.className = "relative group cursor-pointer";

            const img = document.createElement('img');
            img.src = item.path;
            img.className = "w-full h-48 object-cover rounded border-2 border-slate-700 group-hover:border-red-500 transition-all shadow-lg";

            // Timestamp Badge
            const badge = document.createElement('div');
            badge.className = "absolute bottom-2 right-2 bg-black/80 text-red-500 text-xs font-mono px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity";
            badge.textContent = `Seek: ${item.timestamp.toFixed(2)}s`;

            // Click to Seek
            container.onclick = () => {
                this.video.currentTime = item.timestamp;
                this.video.play();
                this.terminal.log(`Seeking to timestamp: ${item.timestamp.toFixed(2)}s`, 'PROCESS');
            };

            container.appendChild(img);
            container.appendChild(badge);
            list.appendChild(container);
        }

        this.terminal.log(`[EVIDENCE] ${evidenceItems.length} frames flagged. Click to seek.`, 'WARN');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.gatekeeper = new Gatekeeper();
});
