'use client';
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useParams } from 'next/navigation';
import {
    fetchTeamsMeeting,
    fetchVttSegments,
    teamsVideoUrl,
    teamsTranscriptUrl,
    TeamsMeeting,
    TeamsTranscript,
    TeamsRecording,
    VttSegment,
    TEAMS_API,
} from '@/lib/api';
import Link from 'next/link';

/* ── helpers ──────────────────────────────────────────────────────────────── */
function initials(name: string) {
    return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
}
const AVATAR_COLORS = [
    'linear-gradient(135deg,#7c6ff7,#06b6d4)',
    'linear-gradient(135deg,#f59e0b,#ef4444)',
    'linear-gradient(135deg,#22c55e,#06b6d4)',
    'linear-gradient(135deg,#ec4899,#8b5cf6)',
    'linear-gradient(135deg,#f97316,#facc15)',
    'linear-gradient(135deg,#0ea5e9,#6366f1)',
];
function avatarColor(name: string) {
    let h = 0; for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return AVATAR_COLORS[h % AVATAR_COLORS.length];
}
function fmtDate(iso: string) {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-IN', {
        weekday: 'short', year: 'numeric', month: 'short',
        day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
}
function vttToSec(ts: string): number {
    const parts = ts.split(':').reverse();
    return (parseFloat(parts[0]) || 0)
        + (parseFloat(parts[1]) || 0) * 60
        + (parseFloat(parts[2]) || 0) * 3600;
}

/* ── VTT Transcript line ─────────────────────────────────────────────────── */
function VttLine({
    seg,
    isActive,
    onClick,
    colorMap,
}: {
    seg: VttSegment;
    isActive: boolean;
    onClick: () => void;
    colorMap: Record<string, string>;
}) {
    return (
        <div onClick={onClick} style={{
            display: 'flex', gap: '12px', padding: '10px 14px', borderRadius: '8px',
            cursor: 'pointer',
            background: isActive ? 'var(--bg-hover)' : 'transparent',
            borderLeft: `3px solid ${isActive ? (colorMap[seg.speaker] || '#555') : 'transparent'}`,
            transition: 'all 0.15s',
        }}>
            <div style={{
                width: 32, height: 32, borderRadius: '50%',
                background: avatarColor(seg.speaker),
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '11px', fontWeight: 700, flexShrink: 0,
            }}>{initials(seg.speaker)}</div>
            <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'baseline', marginBottom: '4px' }}>
                    <span style={{ fontWeight: 600, fontSize: '13px' }}>{seg.speaker}</span>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                        {seg.start}
                    </span>
                </div>
                <div style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                    {seg.text}
                </div>
            </div>
        </div>
    );
}

/* ── Speaker stats ───────────────────────────────────────────────────────── */
function SpeakerBar({ name, pct, secCount, color }: { name: string; pct: number; secCount: number; color: string }) {
    return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 80px auto', gap: '12px', alignItems: 'center' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <div style={{
                        width: 24, height: 24, borderRadius: '50%', background: avatarColor(name),
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '9px', fontWeight: 700, flexShrink: 0,
                    }}>{initials(name)}</div>
                    <span style={{ fontSize: '13px', fontWeight: 500 }}>{name}</span>
                </div>
                <div style={{ height: 4, background: 'var(--bg-hover)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2, transition: 'width 0.5s ease' }} />
                </div>
            </div>
            <div style={{ fontSize: '13px', fontWeight: 600, textAlign: 'right' }}>
                {Math.round(secCount / 60)}m {Math.round(secCount % 60)}s
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{pct}%</div>
        </div>
    );
}

/* ── Diarization info card ─────────────────────────────────────────────────── */
function DiarizationCard({ transcript }: { transcript: TeamsTranscript }) {
    const colors = ['#7c6ff7', '#06b6d4', '#22c55e', '#f59e0b', '#ec4899', '#ef4444'];
    return (
        <div style={{
            background: 'rgba(6,182,212,0.06)', border: '1px solid rgba(6,182,212,0.2)',
            borderRadius: '10px', padding: '14px 16px', marginBottom: '16px',
        }}>
            <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--accent2)', marginBottom: '10px' }}>
                🎤 Diarization — {transcript.speakerCount} speakers · {transcript.segmentCount} segments
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {(transcript.diarization || []).map((d, i) => (
                    <div key={d.speakerLabel} style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        background: 'var(--bg-surface)', borderRadius: '100px',
                        padding: '4px 12px', border: `1px solid ${colors[i % colors.length]}40`,
                    }}>
                        <span style={{
                            width: 8, height: 8, borderRadius: '50%',
                            background: colors[i % colors.length], display: 'inline-block',
                        }} />
                        <span style={{ fontSize: '12px', fontWeight: 600 }}>{d.speakerLabel}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{d.segments?.length ?? 0} segs</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ══ Main page ══════════════════════════════════════════════════════════════ */
export default function TeamsMeetingDetailPage() {
    const { id } = useParams<{ id: string }>();
    const [meeting, setMeeting]           = useState<TeamsMeeting | null>(null);
    const [segments, setSegments]         = useState<VttSegment[]>([]);
    const [loadingMeta, setLoadingMeta]   = useState(true);
    const [loadingVtt, setLoadingVtt]     = useState(false);
    const [activeIdx, setActiveIdx]       = useState<number | null>(null);
    const [speakerFilter, setSpeakerFilter] = useState('all');
    const [rightTab, setRightTab]         = useState<'transcript' | 'info'>('transcript');
    const [selectedTranscript, setSelectedTranscript] = useState<TeamsTranscript | null>(null);
    const [selectedRecording, setSelectedRecording]   = useState<TeamsRecording  | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    const load = useCallback(async () => {
        try {
            const m = await fetchTeamsMeeting(id);
            setMeeting(m);
            // Auto-select first transcript and recording
            if (m.transcripts?.length && !selectedTranscript) {
                const tx = m.transcripts[0];
                setSelectedTranscript(tx);
                setLoadingVtt(true);
                try {
                    const segs = await fetchVttSegments(tx.meetingId, tx.id);
                    setSegments(segs);
                } catch { setSegments([]); }
                finally { setLoadingVtt(false); }
            }
            if (m.recordings?.length && !selectedRecording) {
                setSelectedRecording(m.recordings[0]);
            }
        } catch { } finally { setLoadingMeta(false); }
    }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => { load(); }, [load]);

    const switchTranscript = async (tx: TeamsTranscript) => {
        setSelectedTranscript(tx);
        setSegments([]);
        setLoadingVtt(true);
        try {
            const segs = await fetchVttSegments(tx.meetingId, tx.id);
            setSegments(segs);
        } catch { setSegments([]); }
        finally { setLoadingVtt(false); }
    };

    // Build speaker → color map
    const speakerColors = useMemo(() => {
        const COLORS = ['#7c6ff7', '#06b6d4', '#22c55e', '#f59e0b', '#ec4899', '#ef4444', '#8b5cf6', '#0ea5e9'];
        const map: Record<string, string> = {};
        let ci = 0;
        for (const seg of segments) {
            if (!map[seg.speaker]) map[seg.speaker] = COLORS[ci++ % COLORS.length];
        }
        return map;
    }, [segments]);

    const speakers = useMemo(() => ['all', ...Object.keys(speakerColors)], [speakerColors]);

    const filteredSegs = useMemo(() =>
        segments.filter(s => speakerFilter === 'all' || s.speaker === speakerFilter),
        [segments, speakerFilter]
    );

    // Speaker talk-time stats
    const speakerStats = useMemo(() => {
        const dur: Record<string, number> = {};
        for (const s of segments) {
            const secs = vttToSec(s.end) - vttToSec(s.start);
            dur[s.speaker] = (dur[s.speaker] || 0) + secs;
        }
        const total = Math.max(1, Object.values(dur).reduce((a, b) => a + b, 0));
        return Object.entries(dur)
            .sort((a, b) => b[1] - a[1])
            .map(([name, secs]) => ({ name, secs, pct: Math.round((secs / total) * 100) }));
    }, [segments]);

    const seekTo = (segIdx: number) => {
        setActiveIdx(segIdx);
        if (videoRef.current) {
            const seg = filteredSegs[segIdx];
            if (seg) videoRef.current.currentTime = vttToSec(seg.start);
        }
    };

    if (loadingMeta) return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh' }}>
            <div style={{ textAlign: 'center' }}>
                <div className="spinner" style={{ width: 40, height: 40, margin: '0 auto 16px' }} />
                <div style={{ color: 'var(--text-secondary)' }}>Loading meeting…</div>
            </div>
        </div>
    );

    if (!meeting) return (
        <div className="page">
            <div className="empty-state">
                <div className="empty-icon">❌</div>
                <div className="empty-title">Meeting not found on Teams server</div>
            </div>
        </div>
    );

    const videoSrc = selectedRecording
        ? teamsVideoUrl(meeting.id, selectedRecording.id)
        : null;

    return (
        <>
            {/* ── Topbar ── */}
            <div className="topbar" style={{ gap: '0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                    <Link href="/meetings"><button className="btn btn-secondary btn-sm">←</button></Link>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: '16px' }}>{meeting.subject}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                            {fmtDate(meeting.startDateTime)} &nbsp;·&nbsp; {meeting.organizer.displayName}
                            &nbsp;·&nbsp;
                            <span style={{ color: 'var(--accent2)' }}>
                                Teams Server
                            </span>
                        </div>
                    </div>
                    <span className="badge badge-purple" style={{ marginLeft: '4px' }}>
                        {meeting.transcripts?.length ?? 0} transcript{meeting.transcripts?.length !== 1 ? 's' : ''}
                    </span>
                    <span className="badge badge-green" style={{ marginLeft: '4px' }}>
                        {meeting.recordings?.length ?? 0} recording{meeting.recordings?.length !== 1 ? 's' : ''}
                    </span>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    {selectedTranscript && (
                        <a
                            href={teamsTranscriptUrl(meeting.id, selectedTranscript.id)}
                            target="_blank"
                            rel="noreferrer"
                            className="btn btn-secondary btn-sm"
                        >
                            ⬇ VTT
                        </a>
                    )}
                    {selectedRecording && (
                        <a
                            href={teamsVideoUrl(meeting.id, selectedRecording.id)}
                            target="_blank"
                            rel="noreferrer"
                            className="btn btn-secondary btn-sm"
                        >
                            ⬇ Video
                        </a>
                    )}
                </div>
            </div>

            {/* ── Body: 60/40 ── */}
            <div style={{
                display: 'grid', gridTemplateColumns: '1fr 440px',
                height: 'calc(100vh - 61px)', overflow: 'hidden',
            }}>

                {/* ── LEFT: video + speaker stats ─────────────────────────────── */}
                <div style={{ borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

                    {/* Video player */}
                    <div style={{ background: '#000', flexShrink: 0 }}>
                        {videoSrc ? (
                            <video
                                ref={videoRef}
                                controls
                                key={videoSrc}
                                style={{ width: '100%', maxHeight: '360px', display: 'block' }}
                                src={videoSrc}
                            />
                        ) : (
                            <div style={{
                                height: 220, display: 'flex', alignItems: 'center',
                                justifyContent: 'center', flexDirection: 'column', gap: '10px',
                                color: 'var(--text-muted)',
                            }}>
                                <span style={{ fontSize: '36px' }}>🎬</span>
                                <div style={{ fontSize: '14px' }}>No recording attached</div>
                                <div style={{ fontSize: '12px' }}>Upload a recording in the Teams Media Server UI</div>
                            </div>
                        )}
                    </div>

                    {/* Recording selector (if multiple) */}
                    {(meeting.recordings?.length ?? 0) > 1 && (
                        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                            <select
                                className="input"
                                value={selectedRecording?.id ?? ''}
                                onChange={e => {
                                    const r = meeting.recordings.find(r => r.id === e.target.value);
                                    if (r) setSelectedRecording(r);
                                }}
                                style={{ fontSize: '13px', padding: '6px 10px' }}
                            >
                                {meeting.recordings.map(r => (
                                    <option key={r.id} value={r.id}>{r.filename}</option>
                                ))}
                            </select>
                        </div>
                    )}

                    {/* Left tabs */}
                    <div style={{ padding: '0 20px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                        <div className="tabs" style={{ margin: '0', borderBottom: 'none' }}>
                            <div
                                className={`tab${rightTab === 'info' ? ' active' : ''}`}
                                onClick={() => setRightTab('info')}
                            >
                                📊 Speaker Stats
                            </div>
                        </div>
                    </div>

                    <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
                        {speakerStats.length > 0 ? (
                            <>
                                <div style={{
                                    display: 'grid', gridTemplateColumns: '1fr 80px auto',
                                    gap: '0 12px', marginBottom: '12px',
                                }}>
                                    <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)' }}>Speaker</div>
                                    <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)' }}>Talk Time</div>
                                    <div />
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                                    {speakerStats.map(({ name, secs, pct }, i) => (
                                        <SpeakerBar key={name} name={name} pct={pct} secCount={secs}
                                            color={speakerColors[name] || '#7c6ff7'} />
                                    ))}
                                </div>
                            </>
                        ) : (
                            <div className="empty-state" style={{ padding: '40px 0' }}>
                                <div className="empty-icon">📊</div>
                                <div className="empty-title">No transcript loaded</div>
                                <div style={{ color: 'var(--text-muted)', fontSize: '13px', marginTop: '6px' }}>
                                    Upload a VTT transcript to see speaker stats
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* ── RIGHT: transcript panel ───────────────────────────────── */}
                <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                        <span style={{ fontWeight: 700, fontSize: '16px' }}>Transcript</span>
                        {selectedTranscript && (
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '8px' }}>
                                {selectedTranscript.filename}
                            </span>
                        )}
                    </div>

                    {/* Transcript selector + speaker filter */}
                    <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', flexShrink: 0, display: 'flex', gap: '10px' }}>
                        {(meeting.transcripts?.length ?? 0) > 1 ? (
                            <select
                                className="input"
                                value={selectedTranscript?.id ?? ''}
                                onChange={e => {
                                    const tx = meeting.transcripts.find(t => t.id === e.target.value);
                                    if (tx) switchTranscript(tx);
                                }}
                                style={{ fontSize: '13px', padding: '6px 10px', flex: 1 }}
                            >
                                {meeting.transcripts.map(t => (
                                    <option key={t.id} value={t.id}>{t.filename}</option>
                                ))}
                            </select>
                        ) : (
                            <div style={{ flex: 1 }} />
                        )}
                        <select
                            className="input"
                            value={speakerFilter}
                            onChange={e => setSpeakerFilter(e.target.value)}
                            style={{ fontSize: '13px', padding: '6px 10px', flex: 1 }}
                        >
                            {speakers.map(s => (
                                <option key={s} value={s}>{s === 'all' ? 'All speakers' : s}</option>
                            ))}
                        </select>
                    </div>

                    {/* Diarization badge */}
                    {selectedTranscript && (selectedTranscript.speakerCount ?? 0) > 0 && (
                        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                            <DiarizationCard transcript={selectedTranscript} />
                        </div>
                    )}

                    {/* Transcript lines */}
                    <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
                        {loadingVtt ? (
                            <div style={{ display: 'flex', justifyContent: 'center', paddingTop: '40px' }}>
                                <span className="spinner" />
                            </div>
                        ) : filteredSegs.length === 0 ? (
                            <div className="empty-state">
                                <div className="empty-icon">📝</div>
                                <div className="empty-title">
                                    {(meeting.transcripts?.length ?? 0) === 0
                                        ? 'No transcript attached'
                                        : 'No segments match filter'}
                                </div>
                                {(meeting.transcripts?.length ?? 0) === 0 && (
                                    <div style={{ color: 'var(--text-muted)', fontSize: '13px', marginTop: '6px' }}>
                                        Upload a VTT file in the Teams Media Server UI
                                    </div>
                                )}
                            </div>
                        ) : (
                            filteredSegs.map((seg, i) => (
                                <VttLine
                                    key={i}
                                    seg={seg}
                                    isActive={activeIdx === i}
                                    onClick={() => seekTo(i)}
                                    colorMap={speakerColors}
                                />
                            ))
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}
