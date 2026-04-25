'use client';
import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { fetchMeeting, teamsVideoUrl, patchSpeakers, chatWithMeeting, fetchModels, Job, ChatMessage, AgentStep } from '@/lib/api';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChevronRight, Users, Clock, Calendar, Download, Scissors, Send, RotateCcw, Sparkles, MessageCircle, FileText, BarChart3, ChevronDown, Paperclip, CheckSquare, Music } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/* ── Design Tokens ── */
const BRAND = '#534AB7';
const PAGE_BG = '#F8F7F4';
const CARD_BG = '#FFFFFF';
const BORDER = 'rgba(0,0,0,0.08)';
const TEXT_PRIMARY = '#1A1A18';
const TEXT_SEC = '#6B6A66';
const TEXT_MUTED = '#9B9891';

/* ── Fixed speaker palette ── */
const SPEAKER_COLORS = ['#534AB7', '#F59E0B', '#1D9E75', '#E11D48', '#0EA5E9', '#8B5CF6'];
function speakerColor(name: string, speakerList: string[]) {
    const idx = speakerList.indexOf(name);
    return SPEAKER_COLORS[idx >= 0 ? idx % SPEAKER_COLORS.length : name.split('').reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 0) % SPEAKER_COLORS.length];
}

/* ── Topic pill colors ── */
const TOPIC_STYLE: Record<string, { bg: string; color: string }> = {
    decision: { bg: '#EEF2FF', color: '#534AB7' },
    discussion: { bg: '#F0FDFA', color: '#0D9488' },
    problem: { bg: '#FEF2F2', color: '#DC2626' },
    update: { bg: '#FFFBEB', color: '#D97706' },
};
function topicStyle(type: string) {
    return TOPIC_STYLE[type] || { bg: '#F4F4F5', color: '#71717A' };
}

/* ── Helpers ── */
function initials(name: string) { return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase(); }
function fmtDate(iso: string) { return new Date(iso).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }); }
function fmtTime(s: number) { const m = Math.floor(s / 60); return m > 0 ? `${m}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`; }

/* ── Avatar ── */
function Avatar({ name, size = 36, color }: { name: string; size?: number; color: string }) {
    return (
        <div style={{ width: size, height: size, borderRadius: '50%', background: color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: size * 0.33, fontWeight: 700, color: '#fff', flexShrink: 0 }}>
            {initials(name)}
        </div>
    );
}

/* ── Speaker Assign Panel ── */
function SpeakerAssignPanel({ job, onSaved }: { job: Job; onSaved: () => void }) {
    const [map, setMap] = useState<Record<string, string>>(job.speaker_map || {});
    const [saving, setSaving] = useState(false);
    const [open, setOpen] = useState(false);
    useEffect(() => { setMap(job.speaker_map || {}); }, [job.speaker_map]);
    const hasUnknown = Object.values(map).some(v => !v || v.startsWith('SPEAKER_') || v === 'Unknown');
    const participantNames = job.participants.map(p => p.name);
    const save = async () => { setSaving(true); try { await patchSpeakers(job.job_id, map); onSaved(); setOpen(false); } catch { } finally { setSaving(false); } };
    const autoFill = () => { const keys = Object.keys(map); const newMap = { ...map }; keys.forEach((k, i) => { if (i < participantNames.length) newMap[k] = participantNames[i]; }); setMap(newMap); };
    if (Object.keys(map).length === 0) return null;
    return (
        <div style={{ background: hasUnknown ? '#FEF2F2' : '#F0FDF4', border: `0.5px solid ${hasUnknown ? '#FCA5A5' : '#BBF7D0'}`, borderRadius: 12, marginBottom: 20, overflow: 'hidden' }}>
            <div onClick={() => setOpen(o => !o)} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px', cursor: 'pointer' }}>
                <span>{hasUnknown ? '⚠️' : '✅'}</span>
                <span style={{ fontWeight: 600, fontSize: 13, color: TEXT_PRIMARY }}>{hasUnknown ? 'Speaker mapping needs review' : 'Speaker mapping confirmed'}</span>
                <span style={{ fontSize: 12, color: TEXT_MUTED }}>({Object.keys(map).length} speakers)</span>
                <ChevronDown style={{ marginLeft: 'auto', width: 16, height: 16, color: TEXT_MUTED, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </div>
            {open && (
                <div style={{ padding: '0 16px 16px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(220px,1fr))', gap: 10, marginBottom: 12 }}>
                        {Object.entries(map).map(([raw, display]) => (
                            <div key={raw} style={{ display: 'flex', alignItems: 'center', gap: 10, background: CARD_BG, borderRadius: 8, padding: '10px 12px', border: `0.5px solid ${BORDER}` }}>
                                <Avatar name={display || raw} size={28} color={BRAND} />
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: 11, color: TEXT_MUTED, marginBottom: 4 }}>{raw}</div>
                                    <select value={display} onChange={e => setMap(m => ({ ...m, [raw]: e.target.value }))}
                                        style={{ width: '100%', fontSize: 12, padding: '4px 8px', borderRadius: 6, border: `0.5px solid ${BORDER}`, background: PAGE_BG, color: TEXT_PRIMARY, outline: 'none' }}>
                                        <option value="">— unassigned —</option>
                                        {participantNames.map(n => <option key={n} value={n}>{n}</option>)}
                                        {display && !participantNames.includes(display) && <option value={display}>{display}</option>}
                                    </select>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <button onClick={autoFill} style={{ padding: '7px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, border: `0.5px solid ${BORDER}`, background: CARD_BG, color: TEXT_SEC, cursor: 'pointer' }}>🪄 Auto-fill</button>
                        <button onClick={save} disabled={saving} style={{ padding: '7px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, border: 'none', background: BRAND, color: '#fff', cursor: 'pointer' }}>{saving ? 'Saving…' : '✅ Save & Reprocess'}</button>
                    </div>
                </div>
            )}
        </div>
    );
}

/* ── Chat Panel ── */
const SUGGESTED = ['What were the key decisions?', 'Any unresolved issues?', 'Summarize cost discussions', 'How does this connect to past meetings?'];
function ChatPanel({ jobId, onSeek }: { jobId: string; onSeek?: (t: number) => void }) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [liveSteps, setLiveSteps] = useState<AgentStep[]>([]);
    const [models, setModels] = useState<{ id: string }[]>([]);
    const [selectedModel, setSelectedModel] = useState('');
    const scrollRef = useRef<HTMLDivElement>(null);
    useEffect(() => { fetchModels().then(m => { setModels(m); setSelectedModel(m.find(x => x.id.toLowerCase().includes('gemma'))?.id || m[0]?.id || ''); }).catch(() => { }); }, []);
    useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' }); }, [messages, loading]);
    const send = async (text?: string) => {
        const msg = (text || input).trim(); if (!msg || loading) return;
        const updated = [...messages, { role: 'user' as const, content: msg }];
        setMessages(updated); setInput(''); setLoading(true); setLiveSteps([]);
        try {
            const { reply, steps, model } = await chatWithMeeting(jobId, msg, updated, s => setLiveSteps(s), selectedModel || undefined);
            setLiveSteps([]);
            setMessages(prev => [...prev, { role: 'assistant', content: reply, steps, model }]);
        } catch (e: any) {
            setLiveSteps([]); setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }]);
        } finally { setLoading(false); }
    };
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: 560, background: CARD_BG, border: `0.5px solid ${BORDER}`, borderRadius: 12, overflow: 'hidden' }}>
            <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
                {messages.length === 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 20, padding: 20 }}>
                        <div style={{ width: 52, height: 52, borderRadius: '50%', background: BRAND, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>🧠</div>
                        <div style={{ textAlign: 'center', fontSize: 13, color: TEXT_MUTED }}>Ask anything about this meeting</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, width: '100%', maxWidth: 360 }}>
                            {SUGGESTED.map((p, i) => (
                                <button key={i} onClick={() => send(p)} style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, borderRadius: 10, padding: '10px 12px', fontSize: 12, color: TEXT_SEC, cursor: 'pointer', textAlign: 'left', lineHeight: 1.4 }}>{p}</button>
                            ))}
                        </div>
                    </div>
                )}
                {messages.map((msg, i) => (
                    <div key={i} style={{ display: 'flex', gap: 10, flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', alignItems: 'flex-start' }}>
                        {msg.role === 'assistant' && <div style={{ width: 28, height: 28, borderRadius: '50%', background: BRAND, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, flexShrink: 0 }}>🧠</div>}
                        <div style={{ maxWidth: '85%', background: msg.role === 'user' ? BRAND : PAGE_BG, color: msg.role === 'user' ? '#fff' : TEXT_PRIMARY, border: msg.role === 'user' ? 'none' : `0.5px solid ${BORDER}`, borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '4px 16px 16px 16px', padding: '10px 14px', fontSize: 13, lineHeight: 1.6 }}>
                            {msg.role === 'user' ? msg.content : (
                                <div className="markdown-body">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                                        a: ({ node, ...props }) => {
                                            if (props.href?.startsWith('#cite:')) {
                                                const citation = decodeURIComponent(props.href.replace('#cite:', ''));
                                                return <span className="citation-pill" title={citation} onClick={() => { const m = citation.match(/(\d+):(\d+)/); if (m && onSeek) onSeek(parseInt(m[1]) * 60 + parseInt(m[2])); }}>{props.children}</span>;
                                            }
                                            return <a {...props} target="_blank" rel="noopener noreferrer" />;
                                        }
                                    }}>
                                        {msg.content.replace(/\(\s*(?:AMI\s+)?(?:Meeting|Speaker_)[^)]+\)/gi, (match) => `[${match.split(',')[0].replace('(', '').trim()} +1](#cite:${encodeURIComponent(match)})`)}
                                    </ReactMarkdown>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                {loading && (
                    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                        <div style={{ width: 28, height: 28, borderRadius: '50%', background: BRAND, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13 }}>🧠</div>
                        <div style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, borderRadius: '4px 16px 16px 16px', padding: '14px 18px', display: 'flex', gap: 5, alignItems: 'center' }}>
                            {liveSteps.length > 0 ? <span style={{ fontSize: 12, color: TEXT_SEC }}>{liveSteps[liveSteps.length - 1].name || 'Thinking'}…</span> : <>
                                <span className="typing-dot" style={{ animationDelay: '0ms' }} />
                                <span className="typing-dot" style={{ animationDelay: '150ms' }} />
                                <span className="typing-dot" style={{ animationDelay: '300ms' }} />
                            </>}
                        </div>
                    </div>
                )}
            </div>
            <div style={{ padding: '12px 16px', borderTop: `0.5px solid ${BORDER}`, background: PAGE_BG, flexShrink: 0 }}>
                {messages.length > 0 && <div style={{ textAlign: 'center', marginBottom: 8 }}><button onClick={() => setMessages([])} style={{ background: 'transparent', border: 'none', color: TEXT_MUTED, fontSize: 11, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 }}><RotateCcw style={{ width: 11, height: 11 }} /> Clear</button></div>}
                <div style={{ display: 'flex', gap: 8, background: CARD_BG, borderRadius: 10, border: `0.5px solid ${BORDER}`, padding: '4px 4px 4px 14px', alignItems: 'center' }}>
                    <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                        placeholder="Ask anything about this meeting…" disabled={loading}
                        style={{ flex: 1, border: 'none', background: 'transparent', color: TEXT_PRIMARY, fontSize: 13, outline: 'none', padding: '8px 0', fontFamily: 'inherit' }} />
                    {models.length > 0 && <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, color: TEXT_SEC, fontSize: 11, borderRadius: 6, padding: '4px 8px', outline: 'none', maxWidth: 100 }}>{models.map(m => <option key={m.id} value={m.id}>{m.id.split('/').pop()}</option>)}</select>}
                    <button onClick={() => send()} disabled={loading || !input.trim()} style={{ width: 32, height: 32, borderRadius: 8, border: 'none', background: input.trim() ? BRAND : PAGE_BG, color: input.trim() ? '#fff' : TEXT_MUTED, cursor: input.trim() ? 'pointer' : 'default', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <Send style={{ width: 14, height: 14 }} />
                    </button>
                </div>
            </div>
        </div>
    );
}

/* ══ MAIN PAGE ══ */
export default function MeetingDetailPage() {
    const { id } = useParams<{ id: string }>();
    const [job, setJob] = useState<Job | null>(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState<'summary' | 'transcript' | 'chat'>('summary');
    const [expandedSpeaker, setExpandedSpeaker] = useState<string | null>(null);
    const [speakerFilter, setSpeakerFilter] = useState('all');
    const [topicFilter, setTopicFilter] = useState('all');
    const [activeEvent, setActiveEvent] = useState<number | null>(null);
    const [hoveredEvent, setHoveredEvent] = useState<number | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const [extractingVideo, setExtractingVideo] = useState(false);

    const load = useCallback(async () => {
        try { const j = await fetchMeeting(id); setJob(j); if (j.status === 'queued' || j.status === 'processing') pollRef.current = setTimeout(load, 3000); }
        catch { } finally { setLoading(false); }
    }, [id]);

    useEffect(() => { load(); return () => { if (pollRef.current) clearTimeout(pollRef.current); }; }, [load]);

    const events = useMemo(() => job?.graph_events || [], [job]);
    const summaries = job?.summaries || {};

    const rawTranscriptLines = useMemo(() => {
        if (events.length > 0) return [];
        return (job?.transcript || '').split('\n').filter((l: string) => l.trim()).map((line: string) => {
            const idx = line.indexOf(':');
            return { speaker: idx > 0 ? line.slice(0, idx).trim() : 'Unknown', text: idx > 0 ? line.slice(idx + 1).trim() : line };
        });
    }, [events, job]);

    const participants = useMemo(() => {
        if (job?.participants && job.participants.length > 0) return job.participants;
        return Object.keys(summaries).map(name => ({ name, role: 'Attendee' }));
    }, [job, summaries]);

    const speakerList = useMemo(() => {
        const names = events.length > 0
            ? [...new Set<string>(events.map((e: any) => e.speaker || 'Unknown'))]
            : [...new Set<string>(rawTranscriptLines.map((l: any) => l.speaker))];
        return names;
    }, [events, rawTranscriptLines]);

    const speakers = useMemo(() => ['all', ...speakerList], [speakerList]);

    const topics = useMemo(() => {
        const s = new Set<string>(); for (const ev of events) if (ev.event_type) s.add(ev.event_type); return Array.from(s).sort();
    }, [events]);

    const speakerStats = useMemo(() => {
        const counts: Record<string, number> = {};
        const source = events.length > 0 ? events : rawTranscriptLines.map((l: any) => ({ speaker: l.speaker }));
        for (const ev of source) { const s = (ev as any).speaker || 'Unknown'; counts[s] = (counts[s] || 0) + 1; }
        const total = Math.max(1, Object.values(counts).reduce((a, b) => a + b, 0));
        return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name, count]) => ({ name, count, pct: Math.round((count / total) * 100), secs: count * 8 }));
    }, [events, rawTranscriptLines]);

    // Deduplicated + filtered events
    const filteredEvents = useMemo(() => {
        const seen = new Set<string>();
        return events.filter((ev: any) => {
            if (speakerFilter !== 'all' && ev.speaker !== speakerFilter) return false;
            if (topicFilter !== 'all' && ev.event_type !== topicFilter) return false;
            const key = `${ev.speaker}|${ev.summary}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }, [events, speakerFilter, topicFilter]);

    const seekToEvent = (i: number) => { setActiveEvent(i); const ev = filteredEvents[i]; if (videoRef.current && ev) videoRef.current.currentTime = ev.start_time || i * 8; };

    const exportSummary = () => {
        const lines = [`Meeting: ${job?.title}`, `Date: ${job ? fmtDate(job.created_at) : ''}`, `Participants: ${participants.map(p => `${p.name} (${p.role})`).join(', ')}`, '', '--- SUMMARIES ---', ...Object.entries(summaries).map(([n, s]) => `\n${n}:\n${s}`), '', '--- TRANSCRIPT ---', ...events.map((ev: any) => `[${ev.speaker}]: ${ev.summary}`)];
        const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = `${job?.title || 'meeting'}.txt`; a.click(); URL.revokeObjectURL(url);
    };

    const extractVideo = async () => {
        setExtractingVideo(true);
        try {
            const res = await fetch(`${API_BASE}/api/meetings/${id}/extract-video`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ speaker: speakerFilter, topic: topicFilter }) });
            const data = await res.json().catch(async () => ({ detail: await res.text() }));
            if (!res.ok) throw new Error(data.detail || 'Extract failed');
            const a = document.createElement('a'); a.href = `${API_BASE}/api/meetings/${id}/download-video/${data.filename}`; a.download = data.filename; a.click();
        } catch (e: any) { alert(`Failed: ${e.message}`); } finally { setExtractingVideo(false); }
    };

    if (loading) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', background: PAGE_BG }}><div style={{ textAlign: 'center' }}><div className="spinner" style={{ width: 36, height: 36, margin: '0 auto 12px' }} /><div style={{ color: TEXT_SEC, fontSize: 14 }}>Loading…</div></div></div>;
    if (!job) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', background: PAGE_BG, color: TEXT_MUTED }}>Meeting not found</div>;

    const videoSrc = job.teams_meeting_id && job.teams_recording_id ? teamsVideoUrl(job.teams_meeting_id, job.teams_recording_id) : job.video_filename ? `${API_BASE}/api/meetings/${job.job_id}/video` : null;
    const decisions = filteredEvents.filter((ev: any) => ev.event_type === 'decision');
    const tabs = [{ key: 'summary' as const, label: 'Summary', icon: FileText }, { key: 'transcript' as const, label: 'Transcript', icon: BarChart3 }, ...(job.status === 'done' ? [{ key: 'chat' as const, label: 'Ask AI', icon: MessageCircle }] : [])];

    /* ── Shared card style ── */
    const card = { background: CARD_BG, border: `0.5px solid ${BORDER}`, borderRadius: 12, padding: '18px 20px' };
    const secHeader = { fontSize: 13, fontWeight: 500 as const, textTransform: 'uppercase' as const, letterSpacing: '0.06em', color: TEXT_SEC, marginBottom: 16 };

    return (
        <div style={{ minHeight: '100vh', background: PAGE_BG, fontFamily: 'system-ui, -apple-system, sans-serif' }}>
            <div style={{ maxWidth: 1280, margin: '0 auto', padding: '28px 32px', display: 'grid', gridTemplateColumns: '1fr 280px', gap: 28 }}>

                {/* ── CENTER COLUMN ── */}
                <div style={{ minWidth: 0 }}>

                    {/* Breadcrumb */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: TEXT_MUTED, marginBottom: 18 }}>
                        <Link href="/meetings" style={{ color: BRAND, fontWeight: 500 }}>Meetings</Link>
                        <ChevronRight style={{ width: 14, height: 14 }} />
                        <span style={{ color: TEXT_PRIMARY, fontWeight: 500 }}>{job.title}</span>
                    </div>

                    {/* Header */}
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 16, gap: 16 }}>
                        <div>
                            <h1 style={{ fontSize: 26, fontWeight: 700, color: TEXT_PRIMARY, lineHeight: 1.25, marginBottom: 10 }}>{job.title}</h1>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 14, fontSize: 13, color: TEXT_MUTED, flexWrap: 'wrap' }}>
                                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><Users style={{ width: 14, height: 14 }} />{participants.length} Person</span>
                                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><Calendar style={{ width: 14, height: 14 }} />{fmtDate(job.created_at)}</span>
                                {events.length > 0 && <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><Clock style={{ width: 14, height: 14 }} />{events.length} events</span>}
                                {job.status === 'done' && <span style={{ fontSize: 11, fontWeight: 500, padding: '3px 10px', borderRadius: 6, background: '#DCFCE7', color: '#15803D' }}>DONE</span>}
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
                            <button onClick={exportSummary} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '7px 14px', borderRadius: 8, fontSize: 12, fontWeight: 600, border: `0.5px solid ${BORDER}`, background: CARD_BG, color: TEXT_SEC, cursor: 'pointer' }}>
                                <Download style={{ width: 13, height: 13 }} /> Export
                            </button>
                            {job.status === 'done' && (
                                <button onClick={extractVideo} disabled={extractingVideo} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '7px 14px', borderRadius: 8, fontSize: 12, fontWeight: 600, border: 'none', background: BRAND, color: '#fff', cursor: 'pointer' }}>
                                    <Scissors style={{ width: 13, height: 13 }} />{extractingVideo ? 'Generating…' : 'Extract Highlights'}
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Speaker Assignment */}
                    {job.speaker_map && Object.keys(job.speaker_map).length > 0 && job.status === 'done' && <SpeakerAssignPanel job={job} onSaved={load} />}

                    {/* Video Player */}
                    <div style={{ borderRadius: 12, overflow: 'hidden', marginBottom: 20, border: `0.5px solid ${BORDER}` }}>
                        {job.status === 'processing' || job.status === 'queued' ? (
                            <div style={{ height: 260, background: '#1C1C1E', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 14 }}>
                                <span className="spinner" />
                                <div style={{ fontSize: 14, color: 'rgba(255,255,255,0.6)' }}>{job.stage || 'Processing…'}</div>
                                <div style={{ width: '50%', height: 4, background: 'rgba(255,255,255,0.1)', borderRadius: 2 }}>
                                    <div style={{ height: '100%', width: `${job.progress || 0}%`, background: BRAND, borderRadius: 2, transition: 'width 0.5s' }} />
                                </div>
                            </div>
                        ) : videoSrc ? (
                            <video ref={videoRef} controls style={{ width: '100%', maxHeight: 400, display: 'block', background: '#000' }} src={videoSrc} />
                        ) : (
                            /* Styled audio-only empty state — never a plain black box */
                            <div style={{ height: 240, background: '#1C1C1E', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 14 }}>
                                <div style={{ width: 52, height: 52, borderRadius: '50%', border: '1.5px solid rgba(255,255,255,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Music style={{ width: 22, height: 22, color: 'rgba(255,255,255,0.5)' }} />
                                </div>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontSize: 14, color: 'rgba(255,255,255,0.7)', fontWeight: 500 }}>Audio only</div>
                                    <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.35)', marginTop: 4 }}>No video available</div>
                                </div>
                                {/* Simple waveform bars animation */}
                                <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                                    {[40, 60, 80, 50, 70, 90, 55, 75, 45, 65].map((h, i) => (
                                        <div key={i} style={{ width: 3, height: h * 0.4, background: `rgba(83,74,183,${0.4 + (i % 3) * 0.2})`, borderRadius: 2, animation: `pulse ${0.8 + (i % 4) * 0.2}s ease-in-out ${i * 0.1}s infinite alternate` }} />
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Tab Bar */}
                    <div style={{ display: 'flex', borderBottom: `0.5px solid ${BORDER}`, marginBottom: 24 }}>
                        {tabs.map(t => {
                            const active = activeTab === t.key;
                            const Icon = t.icon;
                            return (
                                <button key={t.key} onClick={() => setActiveTab(t.key)} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '11px 18px', fontSize: 14, fontWeight: active ? 600 : 400, color: active ? BRAND : TEXT_MUTED, background: 'transparent', border: 'none', borderBottom: active ? `2px solid ${BRAND}` : '2px solid transparent', marginBottom: -1, cursor: 'pointer', transition: 'all 0.15s', fontFamily: 'inherit' }}>
                                    <Icon style={{ width: 15, height: 15 }} />{t.label}
                                </button>
                            );
                        })}
                    </div>

                    {/* ── Summary Tab ── */}
                    {activeTab === 'summary' && (
                        <div>
                            {/* Meeting Purpose */}
                            <div style={{ ...card, marginBottom: 20 }}>
                                <h3 style={{ fontSize: 16, fontWeight: 700, color: TEXT_PRIMARY, marginBottom: 10 }}>Meeting Purpose</h3>
                                <p style={{ fontSize: 14, color: TEXT_SEC, lineHeight: 1.7 }}>
                                    This meeting involved {participants.length} participant{participants.length !== 1 ? 's' : ''}
                                    {topics.length > 0 ? ` discussing ${topics.join(', ')}.` : '.'}
                                    {events.length > 0 && ` ${events.length} events were captured across the session.`}
                                </p>
                            </div>
                            {/* Per-person summaries */}
                            {Object.keys(summaries).length === 0 ? (
                                <div style={{ textAlign: 'center', padding: '56px 24px', color: TEXT_MUTED }}>
                                    <Sparkles style={{ width: 28, height: 28, color: BRAND, margin: '0 auto 12px', display: 'block' }} />
                                    <div style={{ fontSize: 15, fontWeight: 600, color: TEXT_PRIMARY, marginBottom: 4 }}>Summaries generating…</div>
                                    <div style={{ fontSize: 13 }}>Available once processing completes</div>
                                </div>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                                    <div style={secHeader}>Per-Person Summaries</div>
                                    {Object.entries(summaries).map(([name, summary]) => {
                                        const profile = participants.find(p => p.name.toLowerCase().includes(name.toLowerCase()));
                                        const color = speakerColor(name, speakerList);
                                        const expanded = expandedSpeaker === name;
                                        return (
                                            <div key={name} style={{ ...card, padding: 0, overflow: 'hidden' }}>
                                                <div onClick={() => setExpandedSpeaker(expanded ? null : name)} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '16px 18px', cursor: 'pointer', background: expanded ? PAGE_BG : 'transparent', transition: 'background 0.15s' }}>
                                                    <Avatar name={name} size={38} color={color} />
                                                    <div style={{ flex: 1 }}>
                                                        <div style={{ fontWeight: 600, fontSize: 15, color: TEXT_PRIMARY }}>{name}</div>
                                                        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 100, background: `${color}15`, color }}>{profile?.role || 'Attendee'}</span>
                                                    </div>
                                                    <ChevronDown style={{ width: 17, height: 17, color: TEXT_MUTED, transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'none' }} />
                                                </div>
                                                {expanded && <div style={{ padding: '14px 18px', borderTop: `0.5px solid ${BORDER}`, fontSize: 14, color: TEXT_SEC, lineHeight: 1.7 }}>{String(summary).split('\n').filter(Boolean).map((l, i) => <p key={i} style={{ marginBottom: 10 }}>{l}</p>)}</div>}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    )}

                    {/* ── Transcript Tab ── */}
                    {activeTab === 'transcript' && (
                        <div>
                            {/* Filters */}
                            <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
                                {[{ val: speakerFilter, opts: speakers, onChange: setSpeakerFilter, fmtOpt: (s: string) => s === 'all' ? 'All speakers' : s },
                                { val: topicFilter, opts: ['all', ...topics], onChange: setTopicFilter, fmtOpt: (t: string) => t === 'all' ? 'All topics' : t }].map((sel, si) => (
                                    <div key={si} style={{ flex: 1, position: 'relative' }}>
                                        <select value={sel.val} onChange={e => sel.onChange(e.target.value)}
                                            style={{ width: '100%', padding: '8px 36px 8px 12px', fontSize: 13, borderRadius: 8, border: `0.5px solid ${BORDER}`, background: CARD_BG, color: TEXT_PRIMARY, outline: 'none', appearance: 'none', cursor: 'pointer' }}>
                                            {sel.opts.map(o => <option key={o} value={o}>{sel.fmtOpt(o)}</option>)}
                                        </select>
                                        <ChevronDown style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', width: 14, height: 14, color: TEXT_MUTED, pointerEvents: 'none' }} />
                                    </div>
                                ))}
                            </div>

                            {/* Lines */}
                            {filteredEvents.length > 0 ? (
                                <div style={{ display: 'flex', flexDirection: 'column' }}>
                                    {filteredEvents.map((ev: any, i: number) => {
                                        const isActive = activeEvent === i;
                                        const isHovered = hoveredEvent === i;
                                        const color = speakerColor(ev.speaker || 'Unknown', speakerList);
                                        const ts = topicStyle(ev.event_type);
                                        return (
                                            <div key={i} onClick={() => seekToEvent(i)}
                                                onMouseEnter={() => setHoveredEvent(i)} onMouseLeave={() => setHoveredEvent(null)}
                                                style={{ display: 'flex', gap: 12, padding: '12px 16px', cursor: 'pointer', background: i % 2 === 1 ? '#FAFAF9' : 'transparent', borderLeft: (isActive || isHovered) ? `2px solid ${color}` : '2px solid transparent', transition: 'all 0.15s' }}>
                                                <Avatar name={ev.speaker || '?'} size={32} color={color} />
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                                                        <span style={{ fontWeight: 600, fontSize: 13, color: TEXT_PRIMARY }}>{ev.speaker || 'Unknown'}</span>
                                                        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 100, background: ts.bg, color: ts.color }}>{ev.event_type}</span>
                                                        <span style={{ marginLeft: 'auto', fontSize: 12, color: TEXT_MUTED, fontFamily: 'monospace' }}>{ev.confidence != null ? `${((ev.confidence || 0) * 100).toFixed(0)}%` : ''}</span>
                                                    </div>
                                                    <div style={{ fontSize: 14, color: TEXT_SEC, lineHeight: 1.6 }}>{ev.summary}</div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : rawTranscriptLines.length > 0 ? (
                                <div style={{ display: 'flex', flexDirection: 'column' }}>
                                    {rawTranscriptLines.map((line: any, i: number) => {
                                        const color = speakerColor(line.speaker, speakerList);
                                        return (
                                            <div key={i} style={{ display: 'flex', gap: 12, padding: '10px 16px', background: i % 2 === 1 ? '#FAFAF9' : 'transparent', borderLeft: '2px solid transparent' }}>
                                                <Avatar name={line.speaker} size={32} color={color} />
                                                <div style={{ flex: 1 }}>
                                                    <div style={{ fontWeight: 600, fontSize: 13, color: TEXT_PRIMARY, marginBottom: 2 }}>{line.speaker}</div>
                                                    <div style={{ fontSize: 14, color: TEXT_SEC, lineHeight: 1.6 }}>{line.text}</div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div style={{ textAlign: 'center', padding: '56px 24px', color: TEXT_MUTED }}>
                                    <FileText style={{ width: 28, height: 28, margin: '0 auto 12px', display: 'block' }} />
                                    <div style={{ fontSize: 15, fontWeight: 600, color: TEXT_PRIMARY, marginBottom: 4 }}>Transcript processing…</div>
                                    <div style={{ fontSize: 13 }}>Check back in a moment</div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* ── Ask AI Tab ── */}
                    {activeTab === 'chat' && <ChatPanel jobId={id} onSeek={t => { if (videoRef.current) videoRef.current.currentTime = t; }} />}
                </div>

                {/* ── RIGHT SIDEBAR ── */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

                    {/* Meeting Attendance */}
                    <div style={card}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                            <div style={secHeader}>Meeting Attendance</div>
                            <button style={{ fontSize: 12, color: BRAND, background: 'none', border: 'none', cursor: 'pointer', fontWeight: 500 }}>Send recording</button>
                        </div>
                        {participants.length === 0 ? <div style={{ fontSize: 13, color: TEXT_MUTED }}>No attendees</div> : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                {participants.map((p, i) => {
                                    const color = speakerColor(p.name, speakerList);
                                    return (
                                        <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }}>
                                            <Avatar name={p.name} size={36} color={color} />
                                            <div>
                                                <div style={{ fontSize: 14, fontWeight: 600, color: TEXT_PRIMARY }}>{p.name}{i === 0 ? ' (Author)' : ''}</div>
                                                <div style={{ fontSize: 12, color: TEXT_MUTED }}>{p.role || 'Attendee'}</div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                    {/* Talk Time */}
                    {speakerStats.length > 0 && (
                        <div style={card}>
                            <div style={secHeader}>Talk Time</div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                {speakerStats.map(({ name, pct, secs }) => {
                                    const color = speakerColor(name, speakerList);
                                    return (
                                        <div key={name}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                                <span style={{ fontSize: 13, fontWeight: 500, color: TEXT_PRIMARY }}>{name}</span>
                                                <span style={{ fontSize: 12, color: TEXT_MUTED, fontFamily: 'monospace' }}>{fmtTime(secs)} · {pct}%</span>
                                            </div>
                                            <div style={{ height: 4, background: PAGE_BG, borderRadius: 2 }}>
                                                <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2, transition: 'width 0.5s' }} />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Attachments — always rendered */}
                    <div style={card}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                            <div style={secHeader}>Attachments</div>
                            <button style={{ fontSize: 12, color: BRAND, background: 'none', border: 'none', cursor: 'pointer' }}>View all ›</button>
                        </div>
                        <div style={{ fontSize: 13, color: TEXT_MUTED, display: 'flex', alignItems: 'center', gap: 6 }}>
                            <Paperclip style={{ width: 14, height: 14 }} /> No attachments
                        </div>
                    </div>

                    {/* Topics */}
                    {topics.length > 0 && (
                        <div style={card}>
                            <div style={secHeader}>Topics</div>
                            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                {topics.map(t => {
                                    const ts = topicStyle(t);
                                    return <span key={t} onClick={() => { setTopicFilter(topicFilter === t ? 'all' : t); setActiveTab('transcript'); }} style={{ fontSize: 12, fontWeight: 600, padding: '4px 12px', borderRadius: 100, background: ts.bg, color: ts.color, cursor: 'pointer' }}>{t}</span>;
                                })}
                            </div>
                        </div>
                    )}

                    {/* Actionable Tasks (from decisions) */}
                    {decisions.length > 0 && (
                        <div style={card}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                                <div style={secHeader}>Actionable tasks</div>
                                <button style={{ fontSize: 16, color: TEXT_MUTED, background: 'none', border: 'none', cursor: 'pointer' }}>⋯</button>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: BRAND, marginBottom: 14, fontWeight: 500 }}>
                                <Sparkles style={{ width: 12, height: 12 }} /> Generated by AI
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                {decisions.slice(0, 4).map((ev: any, i: number) => (
                                    <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                                        <CheckSquare style={{ width: 16, height: 16, color: TEXT_MUTED, flexShrink: 0, marginTop: 2 }} />
                                        <div>
                                            <div style={{ fontSize: 13, color: TEXT_PRIMARY, lineHeight: 1.5 }}>{(ev.summary || '').slice(0, 80)}</div>
                                            <div style={{ fontSize: 11, color: TEXT_MUTED, marginTop: 2, fontFamily: 'monospace' }}>@{ev.speaker}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Quick Actions */}
                    <div style={card}>
                        <div style={secHeader}>Quick Actions</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <Link href={`/meetings/${id}/graph`} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '9px 12px', borderRadius: 8, border: `0.5px solid ${BORDER}`, fontSize: 13, fontWeight: 500, color: TEXT_PRIMARY, transition: 'background 0.15s' }}
                                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = PAGE_BG; }} onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}>
                                🕸️ Knowledge Graph
                            </Link>
                            <button onClick={() => setActiveTab('chat')} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '9px 12px', borderRadius: 8, border: `0.5px solid ${BORDER}`, fontSize: 13, fontWeight: 500, color: TEXT_PRIMARY, background: 'transparent', cursor: 'pointer', fontFamily: 'inherit', textAlign: 'left', transition: 'background 0.15s', width: '100%' }}
                                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = PAGE_BG; }} onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}>
                                💬 Ask AI
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Waveform pulse animation */}
            <style>{`@keyframes pulse { 0% { opacity:0.5; transform:scaleY(0.7); } 100% { opacity:1; transform:scaleY(1); } }`}</style>
        </div>
    );
}
