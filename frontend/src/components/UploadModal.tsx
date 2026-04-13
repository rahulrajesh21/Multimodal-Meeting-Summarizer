'use client';
import { useState, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { createTeamsMeeting, uploadTeamsRecording, uploadTeamsTranscript } from '@/lib/api';

export default function UploadModal({ onClose }: { onClose: () => void }) {
    const router = useRouter();
    const videoRef    = useRef<HTMLInputElement>(null);
    const transcriptRef = useRef<HTMLInputElement>(null);
    const [videoFile, setVideoFile]           = useState<File | null>(null);
    const [transcriptFile, setTranscriptFile] = useState<File | null>(null);
    const [dragging, setDragging]             = useState(false);
    const [title, setTitle]                   = useState('');
    const [organizer, setOrganizer]           = useState('');
    const [loading, setLoading]               = useState(false);
    const [stage, setStage]                   = useState('');
    const [error, setError]                   = useState('');

    const onVideoDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        if (f) setVideoFile(f);
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!videoFile) { setError('Please select a video or audio file.'); return; }
        setLoading(true);
        setError('');
        try {
            // 1. Create Teams meeting entry
            setStage('Creating meeting…');
            const subject = title || videoFile.name.replace(/\.[^.]+$/, '');
            const meeting = await createTeamsMeeting({
                subject,
                organizer: organizer || undefined,
                startDateTime: new Date().toISOString(),
            });

            // 2. Upload recording to Teams server
            setStage('Uploading recording…');
            await uploadTeamsRecording(meeting.id, videoFile);

            // 3. Upload transcript if provided
            if (transcriptFile) {
                setStage('Uploading transcript…');
                await uploadTeamsTranscript(meeting.id, transcriptFile);
            }

            onClose();
            router.push(`/meetings/teams/${meeting.id}`);
        } catch (err: any) {
            setError(err.message || 'Upload failed');
        } finally {
            setLoading(false);
            setStage('');
        }
    };

    return (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(6px)' }}>
            <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-xl)', width: 'min(640px, 95vw)', padding: '32px', maxHeight: '90vh', overflowY: 'auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                    <div>
                        <div style={{ fontSize: '20px', fontWeight: 700 }}>Upload to Teams Media Server</div>
                        <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Adds recording &amp; transcript to the Teams-compatible media store
                        </div>
                    </div>
                    <button onClick={onClose} className="btn btn-secondary btn-sm">✕ Close</button>
                </div>

                <form onSubmit={handleSubmit}>
                    {/* Video drop zone */}
                    <div
                        className={`dropzone${dragging ? ' active' : ''}`}
                        style={{ marginBottom: '20px' }}
                        onDragOver={e => { e.preventDefault(); setDragging(true); }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={onVideoDrop}
                        onClick={() => videoRef.current?.click()}
                    >
                        <input ref={videoRef} type="file" accept="video/*,audio/*" style={{ display: 'none' }}
                            onChange={e => e.target.files?.[0] && setVideoFile(e.target.files[0])} />
                        <div className="dropzone-icon">{videoFile ? '✅' : '🎬'}</div>
                        <div className="dropzone-title">{videoFile ? videoFile.name : 'Drop recording here'}</div>
                        <div className="dropzone-sub">
                            {videoFile ? `${(videoFile.size / 1024 / 1024).toFixed(1)} MB` : 'MP4, MOV, WebM, WAV, MP3'}
                        </div>
                    </div>

                    {/* Transcript file picker */}
                    <div className="form-group" style={{ marginBottom: '16px' }}>
                        <label className="form-label">Transcript File (VTT) — optional</label>
                        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                            <input ref={transcriptRef} type="file" accept=".vtt,.txt,.srt" style={{ display: 'none' }}
                                onChange={e => e.target.files?.[0] && setTranscriptFile(e.target.files[0])} />
                            <button
                                type="button"
                                className="btn btn-secondary"
                                style={{ flex: 1 }}
                                onClick={() => transcriptRef.current?.click()}
                            >
                                {transcriptFile ? `📄 ${transcriptFile.name}` : '📄 Choose .vtt transcript…'}
                            </button>
                            {transcriptFile && (
                                <button type="button" className="btn btn-secondary btn-sm" onClick={() => setTranscriptFile(null)}>✕</button>
                            )}
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                            WebVTT format with <code>&lt;v SpeakerName&gt;</code> tags for diarization (same as Teams Graph API)
                        </div>
                    </div>

                    <div className="form-group" style={{ marginBottom: '12px' }}>
                        <label className="form-label">Meeting Subject</label>
                        <input className="input" value={title} onChange={e => setTitle(e.target.value)} placeholder="e.g. Weekly Standup — Feb 20" />
                    </div>

                    <div className="form-group" style={{ marginBottom: '20px' }}>
                        <label className="form-label">Organizer Name</label>
                        <input className="input" value={organizer} onChange={e => setOrganizer(e.target.value)} placeholder="e.g. Rahul Kumar" />
                    </div>

                    {error && <div style={{ color: 'var(--danger)', fontSize: '13px', marginBottom: '16px' }}>⚠ {error}</div>}

                    <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: '100%', justifyContent: 'center' }}>
                        {loading
                            ? <><span className="spinner" /> {stage || 'Uploading…'}</>
                            : '☁️ Upload to Teams Server'}
                    </button>
                </form>
            </div>
        </div>
    );
}
