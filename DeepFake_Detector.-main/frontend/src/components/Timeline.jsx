import React, { useMemo } from 'react';

const Timeline = ({ duration, segments, currentTime, onSeek }) => {
    // duration in seconds
    // segments: [{start, end, confidence}]
    // currentTime: current video time in seconds

    const formatTime = (time) => {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="w-full bg-slate-900 p-4 rounded-lg mt-4">
            <div className="flex justify-between text-slate-400 text-xs mb-1">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
            </div>

            <div
                className="relative h-12 bg-slate-800 rounded cursor-pointer overflow-hidden border border-slate-700"
                onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const percent = x / rect.width;
                    onSeek(percent * duration);
                }}
            >
                {/* Progress Bar */}
                <div
                    className="absolute top-0 bottom-0 left-0 bg-blue-500/30 border-r-2 border-blue-500 z-10 pointer-events-none transition-all duration-100 ease-linear"
                    style={{ width: `${(currentTime / duration) * 100}%` }}
                />

                {/* Fake Segments (Red Zones) */}
                {segments.map((seg, idx) => {
                    // Convert "00:00:02" to seconds (mock parser for now)
                    const parseTime = (t) => {
                        const parts = t.split(':').map(Number);
                        if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
                        return 0;
                    };

                    const start = parseTime(seg.start_time);
                    const end = parseTime(seg.end_time);
                    const left = (start / duration) * 100;
                    const width = ((end - start) / duration) * 100;

                    return (
                        <div
                            key={idx}
                            className="absolute top-0 bottom-0 bg-red-500/60 border-l border-r border-red-400 z-0"
                            style={{ left: `${left}%`, width: `${width}%` }}
                            title={`Fake Detected: ${seg.start_time} - ${seg.end_time} (${seg.confidence * 100}%)`}
                        />
                    );
                })}

                {/* Tick Marks (Optional) */}
                <div className="absolute bottom-0 w-full flex justify-between px-1">
                    {[...Array(10)].map((_, i) => (
                        <div key={i} className="h-2 w-px bg-slate-600/50" />
                    ))}
                </div>
            </div>

            <div className="mt-2 flex gap-4 text-xs">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500/60 rounded-sm"></div>
                    <span className="text-slate-300">Manipulated Segment</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-500/30 rounded-sm"></div>
                    <span className="text-slate-300">Current Playback</span>
                </div>
            </div>
        </div>
    );
};

export default Timeline;
