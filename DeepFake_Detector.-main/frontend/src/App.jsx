import { useState, useRef } from 'react'
import axios from 'axios'
import { API_URL } from './config.js'
import TimeLine from './components/Timeline.jsx'
import { Upload, AlertTriangle, CheckCircle, FileVideo, Play, Pause } from 'lucide-react'

function App() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('idle') // idle, uploading, analyzing, complete
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const videoRef = useRef(null)

  const handleDrop = (e) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      if (droppedFile.type.startsWith('video/') || droppedFile.type.startsWith('image/')) {
        setFile(droppedFile)
        startAnalysis(droppedFile)
      }
    }
  }

  const startAnalysis = async (selectedFile) => {
    setStatus('uploading')
    setProgress(10)

    const formData = new FormData()
    formData.append('file', selectedFile)

    const isImage = selectedFile.type.startsWith('image/')
    const endpoint = isImage ? '/api/analyze/image' : '/api/analyze'

    try {
      setProgress(30)
      const response = await axios.post(`${API_URL}${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(Math.min(60, percentCompleted));
        }
      })

      setProgress(80)
      setTimeout(() => {
        setResult(response.data)
        setProgress(100)
        setStatus('complete')
      }, 500)

    } catch (error) {
      console.error("Analysis Failed", error)
      setStatus('error')
      alert("Analysis Failed: " + (error.response?.data?.detail || error.message))
      setStatus('idle')
    }
  }

  // ... (Keep handleTimeUpdate, handleLoadedMetadata as is) ...
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white font-sans selection:bg-red-500/30">
      {/* Header */}
      <header className="border-b border-slate-800 p-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-red-600 p-2 rounded-lg">
            <AlertTriangle size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">DeepForged <span className="text-red-500">Forensics</span></h1>
            <p className="text-xs text-slate-400">Temporal Localization Engine v2.0</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-500 uppercase tracking-widest font-semibold">System Status</div>
          <div className="text-sm text-green-400 flex items-center justify-end gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            Online
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-8">
        {/* Upload Area */}
        {status === 'idle' && (
          <div
            className="border-2 border-dashed border-slate-700 rounded-2xl p-24 text-center hover:border-red-500/50 hover:bg-slate-900/50 transition-all cursor-pointer group"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <input
              type="file"
              id="fileInput"
              className="hidden"
              accept="video/*,image/*"
              onChange={(e) => {
                if (e.target.files[0]) {
                  setFile(e.target.files[0])
                  startAnalysis(e.target.files[0])
                }
              }}
            />
            <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
              <Upload size={32} className="text-slate-400 group-hover:text-red-400" />
            </div>
            <h2 className="text-2xl font-bold mb-2">Drop Evidence Here</h2>
            <p className="text-slate-400">Video (MP4, AVI) or Image (JPG, PNG)</p>
          </div>
        )}

        {/* Progress Area */}
        {(status === 'uploading' || status === 'analyzing') && (
          <div className="max-w-md mx-auto mt-20 text-center">
            <div className="mb-4 flex justify-between text-sm">
              <span className="text-slate-300">Analyzing Artifacts...</span>
              <span className="text-red-400 font-mono">{progress}%</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-red-600 transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Main Dashboard (Result) */}
        {status === 'complete' && result && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left: Player OR Image Preview */}
            <div className="lg:col-span-2">
              <div className="bg-black rounded-xl overflow-hidden shadow-2xl border border-slate-800 relative">
                {file && file.type.startsWith('video/') ? (
                  <>
                    <video
                      ref={videoRef}
                      src={URL.createObjectURL(file)}
                      className="w-full aspect-video"
                      controls={false}
                      onTimeUpdate={handleTimeUpdate}
                      onLoadedMetadata={handleLoadedMetadata}
                    />
                    <div className="absolute bottom-4 left-4 right-4 flex justify-center">
                      <button
                        className="bg-white/10 backdrop-blur hover:bg-white/20 p-3 rounded-full text-white"
                        onClick={() => videoRef.current.paused ? videoRef.current.play() : videoRef.current.pause()}
                      >
                        <Play fill="currentColor" size={20} />
                      </button>
                    </div>
                  </>
                ) : (
                  <img src={URL.createObjectURL(file)} className="w-full object-contain max-h-[500px]" alt="Evidence" />
                )}
              </div>

              {/* Timeline only for videos */}
              {file && file.type.startsWith('video/') && (
                <TimeLine
                  duration={duration || 10}
                  currentTime={currentTime}
                  segments={result.manipulated_segments}
                  onSeek={(time) => {
                    if (videoRef.current) {
                      videoRef.current.currentTime = time;
                      videoRef.current.play();
                    }
                  }}
                />
              )}
            </div>

            {/* Right: Analysis Report */}
            <div className="space-y-6">
              {/* Overall Verdict */}
              <div className={`p-6 rounded-xl border ${result.video_is_fake ? 'bg-red-950/30 border-red-900/50' : 'bg-green-950/30 border-green-900/50'}`}>
                <div className="flex items-center gap-3 mb-2">
                  {result.video_is_fake ? <AlertTriangle className="text-red-500" /> : <CheckCircle className="text-green-500" />}
                  <h3 className={`text-lg font-bold ${result.video_is_fake ? 'text-red-400' : 'text-green-400'}`}>
                    {result.video_is_fake ? 'MANIPULATION DETECTED' : 'AUTHENTIC MEDIA'}
                  </h3>
                </div>
                <div className="text-4xl font-mono font-bold mb-1">
                  {(result.overall_confidence * 100).toFixed(1)}%
                </div>
                <div className="text-xs uppercase tracking-widest text-slate-500">Confidence Score</div>
              </div>

              {/* Segment List (Only for Videos) */}
              {file && file.type.startsWith('video/') && (
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                  <h3 className="font-bold mb-4 flex items-center gap-2">
                    <FileVideo size={18} className="text-slate-400" />
                    <span>Forensic Log</span>
                  </h3>
                  <div className="space-y-3 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                    {result.manipulated_segments.map((seg, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-3 bg-slate-900 rounded border border-slate-800 hover:border-red-500/30 cursor-pointer group transition-colors"
                        onClick={() => {
                          const parts = seg.start_time.split(':').map(Number);
                          const time = parts[0] * 3600 + parts[1] * 60 + parts[2];
                          if (videoRef.current) {
                            videoRef.current.currentTime = time;
                            videoRef.current.play();
                          }
                        }}
                      >
                        <div>
                          <div className="font-mono text-red-400 text-sm group-hover:text-red-300">
                            {seg.start_time} - {seg.end_time}
                          </div>
                          <div className="text-xs text-slate-500">Segment #{i + 1}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-bold text-slate-300">{(seg.confidence * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Export Button */}
              <button
                className="w-full py-4 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded-xl border border-slate-700 transition-colors flex items-center justify-center gap-2 mt-6"
                onClick={() => {
                  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
                  const downloadAnchorNode = document.createElement('a');
                  downloadAnchorNode.setAttribute("href", dataStr);
                  downloadAnchorNode.setAttribute("download", "forensic_report.json");
                  document.body.appendChild(downloadAnchorNode);
                  downloadAnchorNode.click();
                  downloadAnchorNode.remove();
                }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" x2="12" y1="15" y2="3" /></svg>
                Export Forensic Report (JSON)
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
