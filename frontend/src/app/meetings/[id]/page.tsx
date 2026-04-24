'use client';
import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { fetchMeeting, teamsVideoUrl, patchSpeakers, reprocessMeeting, chatWithMeeting, fetchModels, Job, ChatMessage, AgentStep, TEAMS_API } from '@/lib/api';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/* ── helpers ── */
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
const TOPIC_COLORS = ['#7c6ff7', '#06b6d4', '#22c55e', '#f59e0b', '#ec4899', '#ef4444', '#8b5cf6', '#0ea5e9'];
function fmtTime(s: number) {
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`;
}
function fmtDate(iso: string) {
    return new Date(iso).toLocaleDateString('en-IN', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/* ── Speaker Assignment Panel ── */
function SpeakerAssignPanel({ job, onSaved }: { job: Job; onSaved: () => void }) {
    const [map, setMap] = useState<Record<string, string>>(job.speaker_map || {});
    const [saving, setSaving] = useState(false);
    const [open, setOpen] = useState(false);

    // Sync if job.speaker_map changes
    useEffect(() => { setMap(job.speaker_map || {}); }, [job.speaker_map]);

    const hasUnknown = Object.values(map).some(v => !v || v.startsWith('SPEAKER_') || v === 'Unknown');
    const participantNames = job.participants.map(p => p.name);

    const save = async () => {
        setSaving(true);
        try {
            await patchSpeakers(job.job_id, map);
            onSaved();
            setOpen(false);
        } catch (e) { console.error(e); }
        finally { setSaving(false); }
    };

    // Auto-fill: assign speakers in order of participants
    const autoFill = () => {
        const keys = Object.keys(map);
        const newMap = { ...map };
        keys.forEach((k, i) => {
            if (i < participantNames.length) newMap[k] = participantNames[i];
        });
        setMap(newMap);
    };

    if (Object.keys(map).length === 0) return null;

    return (
        <div style={{
            background: hasUnknown ? 'rgba(239,68,68,0.07)' : 'rgba(34,197,94,0.07)',
            borderBottom: `1px solid ${hasUnknown ? 'rgba(239,68,68,0.3)' : 'var(--border)'}`,
            padding: '0 24px',
        }}>
            <div
                onClick={() => setOpen(o => !o)}
                style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 0', cursor: 'pointer' }}
            >
                <span style={{ fontSize: '15px' }}>{hasUnknown ? '⚠️' : '✅'}</span>
                <span style={{ fontWeight: 600, fontSize: '13px' }}>
                    {hasUnknown ? 'Speaker mapping needs review' : 'Speaker mapping confirmed'}
                </span>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                    ({Object.keys(map).length} speakers detected)
                </span>
                <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '18px', transition: 'transform 0.2s', transform: open ? 'rotate(90deg)' : 'none' }}>›</span>
            </div>

            {open && (
                <div style={{ paddingBottom: '16px' }}>
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                        The AI detected {Object.keys(map).length} unique speakers. Match them to your participants:
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '10px', marginBottom: '14px' }}>
                        {Object.entries(map).map(([rawLabel, displayName], i) => (
                            <div key={rawLabel} style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'var(--bg-surface)', borderRadius: '8px', padding: '10px 12px', border: '1px solid var(--border)' }}>
                                <div style={{ width: 32, height: 32, borderRadius: '50%', background: avatarColor(displayName || rawLabel), display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 700, flexShrink: 0 }}>
                                    {initials(displayName || rawLabel)}
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>{rawLabel}</div>
                                    <select
                                        className="input"
                                        value={displayName}
                                        onChange={e => setMap(m => ({ ...m, [rawLabel]: e.target.value }))}
                                        style={{ padding: '5px 8px', fontSize: '13px', width: '100%' }}
                                    >
                                        <option value="">— unassigned —</option>
                                        {participantNames.map(n => <option key={n} value={n}>{n}</option>)}
                                        {/* Also allow typing custom name */}
                                        {displayName && !participantNames.includes(displayName) && (
                                            <option value={displayName}>{displayName}</option>
                                        )}
                                    </select>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ display: 'flex', gap: '10px' }}>
                        <button className="btn btn-secondary btn-sm" onClick={autoFill}>
                            🪄 Auto-fill in order
                        </button>
                        <button className="btn btn-primary btn-sm" onClick={save} disabled={saving}>
                            {saving ? <><span className="spinner" /> Saving…</> : '✅ Save & Reprocess'}
                        </button>
                        <span style={{ fontSize: '12px', color: 'var(--text-muted)', alignSelf: 'center', marginLeft: '4px' }}>
                            This will re-run summaries with the new speaker names
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
}

/* ── Status badge ── */
function StatusPill({ status }: { status: string }) {
    const cls = { queued: 'badge-yellow', processing: 'badge-purple', done: 'badge-green', error: 'badge-red' }[status] || 'badge-muted';
    const icon = { queued: '⏳', processing: '⚙️', done: '✅', error: '❌' }[status] || '';
    return <span className={`badge ${cls}`}>{icon} {status}</span>;
}

/* ── Transcript line ── */
function TranscriptLine({ ev, isActive, onClick }: { ev: any; isActive: boolean; onClick: () => void }) {
    const typeColor: Record<string, string> = { decision: '#7c6ff7', problem: '#ef4444', discussion: '#06b6d4' };
    return (
        <div onClick={onClick} style={{
            display: 'flex', gap: '12px', padding: '10px 14px', borderRadius: '8px', cursor: 'pointer',
            background: isActive ? 'var(--bg-hover)' : 'transparent',
            borderLeft: `3px solid ${isActive ? typeColor[ev.event_type] || '#555' : 'transparent'}`,
            transition: 'all 0.15s',
        }}>
            <div style={{
                width: 32, height: 32, borderRadius: '50%', background: avatarColor(ev.speaker || '?'),
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '11px', fontWeight: 700, flexShrink: 0,
            }}>{initials(ev.speaker || '?')}</div>
            <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'baseline', marginBottom: '4px' }}>
                    <span style={{ fontWeight: 600, fontSize: '13px' }}>{ev.speaker || 'Unknown'}</span>
                    {ev.event_type !== 'discussion' && (
                        <span className={ev.event_type === 'decision' ? 'badge badge-purple' : 'badge badge-red'} style={{ fontSize: '10px', padding: '1px 6px' }}>
                            {ev.event_type}
                        </span>
                    )}
                    {ev.is_screen_sharing && (
                        <span title={ev.ocr_text ? `Screen text: ${ev.ocr_text}` : 'Screen sharing detected'}
                            style={{ fontSize: '12px', cursor: 'help', opacity: 0.8 }}>🖥️</span>
                    )}
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                        {((ev.confidence || 0) * 100).toFixed(0)}% relevance
                    </span>
                </div>
                <div style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                    {ev.summary}
                </div>
                {ev.is_screen_sharing && ev.ocr_text && (
                    <div style={{ fontSize: '11px', color: 'var(--accent)', marginTop: '4px', fontStyle: 'italic', opacity: 0.7 }}>
                        🖥️ {ev.ocr_text.length > 100 ? ev.ocr_text.slice(0, 100) + '…' : ev.ocr_text}
                    </div>
                )}
            </div>
        </div>
    );
}

/* ── Speaker summary card ── */
function SpeakerSummaryCard({ name, role, summary, expanded, onToggle }: {
    name: string; role: string; summary: string; expanded: boolean; onToggle: () => void;
}) {
    return (
        <div className="summary-card">
            <div className="summary-header" onClick={onToggle} style={{ background: expanded ? 'var(--bg-hover)' : 'transparent' }}>
                <div style={{ width: 38, height: 38, borderRadius: '50%', background: avatarColor(name), display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 14, flexShrink: 0 }}>
                    {initials(name)}
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 600, fontSize: '15px' }}>{name}</div>
                    <span className="badge badge-purple" style={{ fontSize: '11px', marginTop: '2px' }}>{role}</span>
                </div>
                <span style={{ color: 'var(--text-muted)', fontSize: '22px', lineHeight: 1, transition: 'transform 0.2s', display: 'inline-block', transform: expanded ? 'rotate(90deg)' : 'none' }}>›</span>
            </div>
            {expanded && (
                <div className="summary-body">
                    {summary.split('\n').filter(Boolean).map((line, i) => (
                        <p key={i} style={{ marginBottom: '10px' }}>{line}</p>
                    ))}
                </div>
            )}
        </div>
    );
}

/* ── Agent Steps Panel (collapsible thinking + tool calls) ── */
function AgentStepsPanel({ steps, model }: { steps: AgentStep[]; model?: string }) {
    const [open, setOpen] = useState(false);
    const thinkingSteps = steps.filter(s => s.type === 'thinking');
    const toolCalls = steps.filter(s => s.type === 'tool_call');
    const toolResults = steps.filter(s => s.type === 'tool_result');

    return (
        <div style={{
            fontSize: '12px',
            borderRadius: '10px',
            border: '1px solid var(--border)',
            background: 'rgba(124,111,247,0.04)',
            overflow: 'hidden',
        }}>
            <div
                onClick={() => setOpen(o => !o)}
                style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '8px 12px', cursor: 'pointer',
                    color: 'var(--text-muted)', userSelect: 'none',
                }}
            >
                <span style={{ fontSize: '14px', transition: 'transform 0.2s', display: 'inline-block', transform: open ? 'rotate(90deg)' : 'none' }}>›</span>
                <span style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                    {thinkingSteps.length > 0 && <span title="Model reasoning">💭 Thinking</span>}
                    {toolCalls.map((tc, i) => (
                        <span key={i} style={{
                            background: 'rgba(124,111,247,0.12)',
                            borderRadius: '6px', padding: '2px 8px',
                            color: 'var(--accent)', fontWeight: 600,
                        }}>
                            🔧 {tc.name}({tc.args ? Object.values(tc.args).join(', ') : ''})
                        </span>
                    ))}
                    {model && <span style={{ marginLeft: 'auto', opacity: 0.5 }}>⚡ {model.split('/').pop()}</span>}
                </span>
            </div>

            {open && (
                <div style={{ padding: '0 12px 10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {steps.map((step, i) => {
                        if (step.type === 'thinking') return (
                            <div key={i} style={{
                                background: 'rgba(250,204,21,0.07)',
                                border: '1px solid rgba(250,204,21,0.2)',
                                borderRadius: '8px', padding: '8px 10px',
                                color: 'var(--text-secondary)', lineHeight: 1.5,
                                whiteSpace: 'pre-wrap', maxHeight: '200px', overflowY: 'auto',
                            }}>
                                <strong style={{ color: 'var(--text-muted)' }}>💭 Thinking</strong>
                                <div style={{ marginTop: '4px' }}>{step.content}</div>
                            </div>
                        );
                        if (step.type === 'tool_call') return (
                            <div key={i} style={{
                                background: 'rgba(124,111,247,0.07)',
                                border: '1px solid rgba(124,111,247,0.2)',
                                borderRadius: '8px', padding: '8px 10px',
                            }}>
                                <strong style={{ color: 'var(--accent)' }}>🔧 {step.name}</strong>
                                {step.args && (
                                    <pre style={{
                                        margin: '4px 0 0', fontSize: '11px',
                                        color: 'var(--text-secondary)',
                                        whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                    }}>{JSON.stringify(step.args, null, 2)}</pre>
                                )}
                                {step.error && <div style={{ color: '#ef4444', marginTop: '4px' }}>❌ {step.error}</div>}
                            </div>
                        );
                        if (step.type === 'tool_result') return (
                            <div key={i} style={{
                                background: 'rgba(34,197,94,0.05)',
                                border: '1px solid rgba(34,197,94,0.2)',
                                borderRadius: '8px', padding: '8px 10px',
                                maxHeight: '150px', overflowY: 'auto',
                            }}>
                                <strong style={{ color: '#22c55e' }}>📋 Result from {step.name}</strong>
                                <pre style={{
                                    margin: '4px 0 0', fontSize: '11px',
                                    color: 'var(--text-secondary)',
                                    whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                }}>{step.result}</pre>
                            </div>
                        );
                        return null;
                    })}
                </div>
            )}
        </div>
    );
}

/* ── Chat Panel ── */
const SUGGESTED_PROMPTS = [
    '🎯 What were the key decisions?',
    '⚠️ Any unresolved issues?',
    '📊 Summarize cost discussions',
    '🔗 How does this connect to past meetings?',
];

function ChatPanel({ jobId, onSeek }: { jobId: string, onSeek?: (time: number) => void }) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [liveSteps, setLiveSteps] = useState<AgentStep[]>([]);
    const [models, setModels] = useState<{ id: string }[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const init = async () => {
            try {
                const fetchedModels = await fetchModels();
                setModels(fetchedModels);
                // Try to find Gemma as a default, fallback to first available
                const defaultModel = fetchedModels.find(m => m.id.toLowerCase().includes('gemma'))?.id || (fetchedModels.length > 0 ? fetchedModels[0].id : '');
                setSelectedModel(defaultModel);
            } catch (e) { console.error('Failed to fetch models', e); }
        };
        init();
    }, []);

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }, [messages, loading, liveSteps]);

    const send = async (text?: string) => {
        const msg = (text || input).trim();
        if (!msg || loading) return;

        const userMsg: ChatMessage = { role: 'user', content: msg };
        const updated = [...messages, userMsg];
        setMessages(updated);
        setInput('');
        setLoading(true);
        setLiveSteps([]);

        try {
            const { reply, steps, model } = await chatWithMeeting(
                jobId, msg, updated,
                (steps) => setLiveSteps(steps),
                selectedModel || undefined
            );
            setLiveSteps([]);
            setMessages(prev => [...prev, { role: 'assistant', content: reply, steps, model }]);
        } catch (e: any) {
            setLiveSteps([]);
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message || 'Failed to get response'}` }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Messages */}
            <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: '20px 16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {messages.length === 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, padding: '20px', gap: '20px' }}>
                        <div style={{
                            width: 56, height: 56, borderRadius: '50%',
                            background: 'linear-gradient(135deg, #7c6ff7, #06b6d4)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '28px', boxShadow: '0 0 30px rgba(124,111,247,0.3)',
                        }}>🧠</div>
                        <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '16px', fontWeight: 700, marginBottom: '6px', color: 'var(--text-primary)' }}>
                                MeetingIQ Assistant
                            </div>
                            <div style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                Ask about this meeting or query cross-meeting history.<br />
                                I can cite speakers, timestamps, and past decisions.
                            </div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', width: '100%', maxWidth: '360px' }}>
                            {SUGGESTED_PROMPTS.map((p, i) => (
                                <button key={i} onClick={() => send(p.replace(/^[^\s]+\s/, ''))}
                                    style={{
                                        background: 'var(--bg-surface)', border: '1px solid var(--border)',
                                        borderRadius: '10px', padding: '10px 12px', fontSize: '12px',
                                        color: 'var(--text-secondary)', cursor: 'pointer', textAlign: 'left',
                                        transition: 'all 0.15s', lineHeight: 1.4,
                                    }}
                                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent-dim)'; e.currentTarget.style.background = 'var(--bg-hover)'; }}
                                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.background = 'var(--bg-surface)'; }}
                                >{p}</button>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div key={i} style={{
                        display: 'flex', gap: '10px', alignItems: 'flex-start',
                        flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                    }}>
                        {/* Avatar */}
                        {msg.role === 'assistant' && (
                            <div style={{
                                width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                                background: 'linear-gradient(135deg, #7c6ff7, #06b6d4)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '14px', marginTop: '2px',
                            }}>🧠</div>
                        )}
                        {/* Bubble with steps */}
                        <div style={{ maxWidth: '88%', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {/* Agent Steps (thinking + tool calls) */}
                            {msg.role === 'assistant' && msg.steps && msg.steps.length > 0 && (
                                <AgentStepsPanel steps={msg.steps} model={msg.model} />
                            )}
                            <div style={{
                                background: msg.role === 'user'
                                    ? 'linear-gradient(135deg, #7c6ff7, #6366f1)'
                                    : 'var(--bg-surface)',
                                color: msg.role === 'user' ? '#fff' : 'var(--text-primary)',
                                border: msg.role === 'user' ? 'none' : '1px solid var(--border)',
                                borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '4px 16px 16px 16px',
                                padding: msg.role === 'user' ? '10px 14px' : '12px 16px',
                                fontSize: '13px',
                                lineHeight: 1.55,
                                overflowX: 'auto',
                            }}>
                                {msg.role === 'user' ? (
                                    <span>{msg.content}</span>
                                ) : (
                                    <div className="markdown-body">
                                        <ReactMarkdown
                                            remarkPlugins={[remarkGfm]}
                                            components={{
                                                a: ({ node, ...props }) => {
                                                    if (props.href?.startsWith('#cite:')) {
                                                        const fullCitation = decodeURIComponent(props.href.replace('#cite:', ''));
                                                        return (
                                                            <span
                                                                className="citation-pill"
                                                                title={fullCitation}
                                                                onClick={() => {
                                                                    const timeMatch = fullCitation.match(/(\d+):(\d+)(?::(\d+))?/);
                                                                    if (timeMatch && onSeek && !fullCitation.includes('Meeting')) {
                                                                        let secs = 0;
                                                                        if (timeMatch[3]) secs = parseInt(timeMatch[1]) * 3600 + parseInt(timeMatch[2]) * 60 + parseInt(timeMatch[3]);
                                                                        else secs = parseInt(timeMatch[1]) * 60 + parseInt(timeMatch[2]);
                                                                        onSeek(Math.max(0, secs - 2));
                                                                    } else {
                                                                        alert(`Source Context:\n${fullCitation}`);
                                                                    }
                                                                }}
                                                            >
                                                                {props.children}
                                                            </span>
                                                        );
                                                    }
                                                    return <a {...props} target="_blank" rel="noopener noreferrer" />;
                                                }
                                            }}
                                        >
                                            {msg.content.replace(/\(\s*(?:AMI\s+)?(?:Meeting|Speaker_)[^)]+\)/gi, (match) => {
                                                const pText = match.split(',')[0].replace('(', '').trim();
                                                return `[${pText} +1](#cite:${encodeURIComponent(match)})`;
                                            })}
                                        </ReactMarkdown>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}

                {/* Live streaming indicator */}
                {loading && (
                    <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                        <div style={{
                            width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                            background: 'linear-gradient(135deg, #7c6ff7, #06b6d4)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '14px',
                        }}>🧠</div>
                        <div style={{ maxWidth: '88%', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {/* Live steps as they stream in */}
                            {liveSteps.length > 0 ? (
                                <div style={{
                                    fontSize: '12px', borderRadius: '10px',
                                    border: '1px solid var(--border)',
                                    background: 'rgba(124,111,247,0.04)',
                                    padding: '10px 12px',
                                    display: 'flex', flexDirection: 'column', gap: '8px',
                                }}>
                                    {liveSteps.map((step, i) => {
                                        if (step.type === 'thinking') return (
                                            <div key={i} style={{
                                                display: 'flex', alignItems: 'center', gap: '8px',
                                                color: 'var(--text-secondary)',
                                            }}>
                                                <span style={{ animation: 'pulse 1.5s ease-in-out infinite' }}>💭</span>
                                                <span>Thinking...</span>
                                            </div>
                                        );
                                        if (step.type === 'tool_call') return (
                                            <div key={i} style={{
                                                display: 'flex', alignItems: 'center', gap: '8px',
                                                background: 'rgba(124,111,247,0.08)',
                                                borderRadius: '8px', padding: '6px 10px',
                                            }}>
                                                <span style={{ animation: 'pulse 1.5s ease-in-out infinite' }}>🔧</span>
                                                <span style={{ color: 'var(--accent)', fontWeight: 600 }}>
                                                    Calling {step.name}({step.args ? Object.values(step.args).join(', ') : ''})
                                                </span>
                                            </div>
                                        );
                                        if (step.type === 'tool_result') return (
                                            <div key={i} style={{
                                                display: 'flex', alignItems: 'center', gap: '8px',
                                                color: '#22c55e',
                                            }}>
                                                <span>✅</span>
                                                <span>Got results from {step.name}</span>
                                            </div>
                                        );
                                        return null;
                                    })}
                                    {/* Still waiting for more */}
                                    <div style={{
                                        display: 'flex', gap: '5px', alignItems: 'center',
                                        color: 'var(--text-muted)', marginTop: '4px',
                                    }}>
                                        <span className="typing-dot" style={{ animationDelay: '0ms' }} />
                                        <span className="typing-dot" style={{ animationDelay: '150ms' }} />
                                        <span className="typing-dot" style={{ animationDelay: '300ms' }} />
                                    </div>
                                </div>
                            ) : (
                                /* Initial loading state before any steps arrive */
                                <div style={{
                                    background: 'var(--bg-surface)', border: '1px solid var(--border)',
                                    borderRadius: '4px 16px 16px 16px', padding: '14px 18px',
                                    display: 'flex', gap: '5px', alignItems: 'center',
                                }}>
                                    <span className="typing-dot" style={{ animationDelay: '0ms' }} />
                                    <span className="typing-dot" style={{ animationDelay: '150ms' }} />
                                    <span className="typing-dot" style={{ animationDelay: '300ms' }} />
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Input */}
            <div style={{
                padding: '12px 16px',
                borderTop: '1px solid var(--border)',
                background: 'var(--bg-surface)',
                flexShrink: 0,
            }}>
                {messages.length > 0 && (
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '8px' }}>
                        <button
                            onClick={() => setMessages([])}
                            style={{
                                background: 'transparent', border: 'none', color: 'var(--text-muted)',
                                fontSize: '11px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px'
                            }}
                        >
                            <span style={{ fontSize: '14px' }}>↺</span> Clear Chat History
                        </button>
                    </div>
                )}
                <div style={{
                    display: 'flex', gap: '8px',
                    background: 'var(--bg-hover)', borderRadius: '12px',
                    border: '1px solid var(--border)', padding: '4px 4px 4px 14px',
                    transition: 'border-color 0.15s',
                    alignItems: 'center',
                }}>
                    <input
                        className="chat-input"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                        placeholder="Ask about this meeting…"
                        disabled={loading}
                        style={{
                            flex: 1, border: 'none', background: 'transparent',
                            color: 'var(--text-primary)', fontSize: '13px',
                            outline: 'none', padding: '8px 0',
                            fontFamily: 'inherit',
                        }}
                    />
                    {models.length > 0 && (
                        <select
                            value={selectedModel}
                            onChange={e => setSelectedModel(e.target.value)}
                            style={{
                                background: 'var(--bg-surface)', border: '1px solid var(--border)',
                                color: 'var(--text-secondary)', fontSize: '11px', borderRadius: '6px',
                                padding: '4px 8px', outline: 'none', cursor: 'pointer',
                                maxWidth: '120px', textOverflow: 'ellipsis',
                            }}
                            title="Select Model"
                        >
                            {models.map(m => (
                                <option key={m.id} value={m.id}>{m.id.split('/').pop()}</option>
                            ))}
                        </select>
                    )}
                    <button
                        onClick={() => send()}
                        disabled={loading || !input.trim()}
                        style={{
                            width: 34, height: 34, borderRadius: '10px', border: 'none',
                            background: input.trim() ? 'var(--accent)' : 'var(--bg-surface)',
                            color: input.trim() ? '#fff' : 'var(--text-muted)',
                            cursor: input.trim() ? 'pointer' : 'default',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '16px', transition: 'all 0.15s', flexShrink: 0,
                        }}
                    >↑</button>
                </div>
                <div style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px', opacity: 0.6 }}>
                    Powered by Cross-Meeting Graph · Responses may cite past meetings
                </div>
            </div>
        </div>
    );
}

/* ══ Main Page ══ */
export default function MeetingDetailPage() {
    const { id } = useParams<{ id: string }>();
    const [job, setJob] = useState<Job | null>(null);
    const [loading, setLoading] = useState(true);
    const [rightTab, setRightTab] = useState<'transcript' | 'summary' | 'insights' | 'chat'>('transcript');
    const [expandedSpeaker, setExpandedSpeaker] = useState<string | null>(null);
    const [speakerFilter, setSpeakerFilter] = useState('all');
    const [topicFilter, setTopicFilter] = useState('all');
    const [activeEvent, setActiveEvent] = useState<number | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const [rightWidth, setRightWidth] = useState(440);
    const dragging = useRef(false);
    const [extractingVideo, setExtractingVideo] = useState(false);

    const load = useCallback(async () => {
        try {
            const j = await fetchMeeting(id);
            setJob(j);
            if (j.status === 'queued' || j.status === 'processing') {
                pollRef.current = setTimeout(load, 3000);
            }
        } catch { } finally { setLoading(false); }
    }, [id]);

    useEffect(() => {
        load();
        return () => { if (pollRef.current) clearTimeout(pollRef.current); };
    }, [load]);

    // ── Resize drag handler ──
    const startDrag = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        dragging.current = true;
        const startX = e.clientX;
        const startW = rightWidth;
        const onMove = (ev: MouseEvent) => {
            if (!dragging.current) return;
            const delta = startX - ev.clientX;
            setRightWidth(Math.min(800, Math.max(320, startW + delta)));
        };
        const onUp = () => {
            dragging.current = false;
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    }, [rightWidth]);

    const events = useMemo(() => job?.graph_events || [], [job]);
    const summaries = job?.summaries || {};

    // Parse raw transcript lines for fallback data
    const rawTranscriptLines = useMemo(() => {
        if (events.length > 0) return [];
        const raw = job?.transcript || '';
        return raw.split('\n').filter((l: string) => l.trim()).map((line: string) => {
            const idx = line.indexOf(':');
            return {
                speaker: idx > 0 ? line.slice(0, idx).trim() : 'Unknown',
                text: idx > 0 ? line.slice(idx + 1).trim() : line,
            };
        });
    }, [events, job]);

    // Derive participants: use job.participants, or fall back to summaries keys
    const participants = useMemo(() => {
        if (job?.participants && job.participants.length > 0) return job.participants;
        return Object.keys(summaries).map(name => ({ name, role: 'Attendee' }));
    }, [job, summaries]);

    const speakers = useMemo(() => {
        if (events.length > 0) {
            const s = new Set<string>(events.map((e: any) => e.speaker || 'Unknown'));
            return ['all', ...Array.from(s)];
        }
        const s = new Set<string>(rawTranscriptLines.map(l => l.speaker));
        return s.size > 0 ? ['all', ...Array.from(s)] : ['all'];
    }, [events, rawTranscriptLines]);

    const topics = useMemo(() => {
        const typeSet = new Set<string>();
        for (const ev of events) {
            if (ev.event_type) typeSet.add(ev.event_type);
        }
        return Array.from(typeSet).sort();
    }, [events]);

    const speakerStats = useMemo(() => {
        const counts: Record<string, number> = {};
        if (events.length > 0) {
            for (const ev of events) {
                const s = ev.speaker || 'Unknown';
                counts[s] = (counts[s] || 0) + 1;
            }
        } else {
            for (const l of rawTranscriptLines) {
                counts[l.speaker] = (counts[l.speaker] || 0) + 1;
            }
        }
        const total = Math.max(1, Object.values(counts).reduce((a, b) => a + b, 0));
        return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name, count]) => ({
            name, count, pct: Math.round((count / total) * 100), secs: count * 8,
        }));
    }, [events, rawTranscriptLines]);

    const filteredEvents = useMemo(() => {
        return events.filter((ev: any) => {
            if (speakerFilter !== 'all' && ev.speaker !== speakerFilter) return false;
            if (topicFilter !== 'all' && ev.event_type !== topicFilter) return false;
            return true;
        });
    }, [events, speakerFilter, topicFilter]);

    const seekToEvent = (i: number) => {
        setActiveEvent(i);
        const ev = filteredEvents[i];
        if (videoRef.current && ev) {
            videoRef.current.currentTime = ev.start_time || i * 8;
        }
    };

    const exportSummary = () => {
        const lines = [
            `Meeting: ${job?.title}`, `Date: ${job ? fmtDate(job.created_at) : ''}`,
            `Participants: ${participants.map(p => `${p.name} (${p.role})`).join(', ')}`,
            '', '--- SUMMARIES ---',
            ...Object.entries(summaries).map(([name, s]) => `\n${name}:\n${s}`),
            '', '--- TRANSCRIPT ---',
            ...events.map((ev: any) => `[${ev.speaker}]: ${ev.summary}`),
        ];
        const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = `${job?.title || 'meeting'}.txt`; a.click();
        URL.revokeObjectURL(url);
    };

    const extractVideo = async () => {
        setExtractingVideo(true);
        try {
            const res = await fetch(`${API_BASE}/api/meetings/${id}/extract-video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speaker: speakerFilter, topic: topicFilter })
            });
            let data;
            try {
                data = await res.json();
            } catch (err) {
                data = { detail: await res.text() };
            }
            if (!res.ok) throw new Error(data.detail || 'Extract failed');

            const url = `${API_BASE}/api/meetings/${id}/download-video/${data.filename}`;
            const a = document.createElement('a');
            a.href = url;
            a.download = data.filename;
            a.click();
        } catch (e: any) {
            alert(`Failed to extract video: ${e.message}`);
        } finally {
            setExtractingVideo(false);
        }
    };


    if (loading) return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh' }}>
            <div style={{ textAlign: 'center' }}>
                <div className="spinner" style={{ width: 40, height: 40, margin: '0 auto 16px' }} />
                <div style={{ color: 'var(--text-secondary)' }}>Loading…</div>
            </div>
        </div>
    );
    if (!job) return <div className="page"><div className="empty-state"><div className="empty-icon">❌</div><div className="empty-title">Meeting not found</div></div></div>;

    return (
        <>
            {/* Topbar */}
            <div className="topbar" style={{ gap: '0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                    <Link href="/meetings"><button className="btn btn-secondary btn-sm">←</button></Link>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: '16px' }}>{job.title}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                            {fmtDate(job.created_at)} &nbsp;·&nbsp; {participants.length} participants
                            {participants.slice(0, 5).map(p => ` · ${p.name.split(' ')[0]}`)}
                            {participants.length > 5 && ` · +${participants.length - 5} more`}
                        </div>
                    </div>
                    <StatusPill status={job.status} />
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <Link href={`/meetings/${id}/graph`}><button className="btn btn-secondary btn-sm">🕸️ Graph</button></Link>
                    <button className="btn btn-primary btn-sm" onClick={exportSummary}>⬇ Export Summary</button>
                    {job.status === 'done' && (
                        <button className="btn btn-primary btn-sm" onClick={extractVideo} disabled={extractingVideo} title="Download a shortened video combining only the highlighted regions currently filtered.">
                            {extractingVideo ? <><span className="spinner" /> Generating...</> : '✂️ Extract Highlights'}
                        </button>
                    )}
                </div>
            </div>

            {/* Speaker Assignment Panel — shows under topbar when mapping is available */}
            {job.speaker_map && Object.keys(job.speaker_map).length > 0 && job.status === 'done' && (
                <SpeakerAssignPanel job={job} onSaved={load} />
            )}

            {/* Body: resizable split */}
            <div style={{ display: 'grid', gridTemplateColumns: `1fr 4px ${rightWidth}px`, height: 'calc(100vh - 61px)', overflow: 'hidden' }}>

                {/* ── LEFT: video + insights ── */}
                <div style={{ borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    {/* Video */}
                    <div style={{ background: '#000', flexShrink: 0 }}>
                        {job.status === 'processing' || job.status === 'queued' ? (
                            <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '12px', color: 'var(--text-secondary)' }}>
                                <span className="spinner" />
                                <div style={{ fontSize: '14px' }}>{job.stage}</div>
                                <div className="progress-bar" style={{ width: '60%' }}>
                                    <div className="progress-fill" style={{ width: `${job.progress}%` }} />
                                </div>
                            </div>
                        ) : (() => {
                            // Determine video source: Teams recording or local upload
                            const videoSrc = job.teams_meeting_id && job.teams_recording_id
                                ? teamsVideoUrl(job.teams_meeting_id, job.teams_recording_id)
                                : job.video_filename
                                    ? `${API_BASE}/api/meetings/${job.job_id}/video`
                                    : null;
                            return videoSrc ? (
                                <video ref={videoRef} controls style={{ width: '100%', maxHeight: '380px', display: 'block' }} src={videoSrc} />
                            ) : (
                                <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '14px' }}>
                                    No video available
                                </div>
                            );
                        })()}
                    </div>

                    {/* Event timeline scrubber */}
                    {events.length > 0 && (
                        <div style={{ padding: '10px 20px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                            <div className="timeline-bar">
                                {events.slice(0, 200).map((ev: any, i: number) => {
                                    const total = Math.min(events.length, 200);
                                    const typeColor: Record<string, string> = { decision: '#7c6ff7', problem: '#ef4444', discussion: '#06b6d4' };
                                    return (
                                        <div key={i} onClick={() => seekToEvent(i)} className="timeline-marker"
                                            title={`${ev.speaker}: ${(ev.summary || '').slice(0, 60)}`}
                                            style={{ left: `${(i / total) * 100}%`, width: `${Math.max(0.5, 100 / total - 0.1)}%`, background: typeColor[ev.event_type] || '#555' }} />
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Left tabs */}
                    <div style={{ padding: '0 20px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                        <div className="tabs" style={{ margin: '0', borderBottom: 'none' }}>
                            <div className={`tab${rightTab === 'insights' ? ' active' : ''}`} onClick={() => setRightTab('insights')}>📊 Insights</div>
                        </div>
                    </div>

                    <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
                        {topics.length > 0 && (
                            <div style={{ marginBottom: '20px' }}>
                                <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)', marginBottom: '10px' }}>Topics</div>
                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                    {topics.map((t, i) => (
                                        <button key={t} onClick={() => setTopicFilter(topicFilter === t ? 'all' : t)}
                                            style={{
                                                background: topicFilter === t ? TOPIC_COLORS[i % TOPIC_COLORS.length] : 'rgba(255,255,255,0.05)',
                                                color: topicFilter === t ? '#fff' : TOPIC_COLORS[i % TOPIC_COLORS.length],
                                                border: `1px solid ${TOPIC_COLORS[i % TOPIC_COLORS.length]}`,
                                                borderRadius: '100px', padding: '4px 13px', fontSize: '12px',
                                                fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s',
                                            }}>
                                            {t}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {speakerStats.length > 0 && (
                            <div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '0 12px', marginBottom: '8px' }}>
                                    <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)' }}>Speakers</div>
                                    <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)' }}>Talk Time</div>
                                    <div />
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    {speakerStats.map(({ name, pct, secs }, i) => {
                                        const profile = participants.find(p => p.name.toLowerCase().includes(name.toLowerCase()) || name.toLowerCase().includes(p.name.split(' ')[0].toLowerCase()));
                                        return (
                                            <div key={name} style={{ display: 'grid', gridTemplateColumns: '1fr 80px auto', gap: '12px', alignItems: 'center' }}>
                                                <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                                                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                                        <div style={{ width: 24, height: 24, borderRadius: '50%', background: avatarColor(name), display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '9px', fontWeight: 700, flexShrink: 0 }}>
                                                            {initials(name)}
                                                        </div>
                                                        <span style={{ fontSize: '13px', fontWeight: 500 }}>{name}</span>
                                                        {profile?.role && <span className="badge badge-muted" style={{ fontSize: '10px', padding: '1px 6px' }}>{profile.role}</span>}
                                                    </div>
                                                    <div style={{ height: 4, background: 'var(--bg-hover)', borderRadius: 2, overflow: 'hidden' }}>
                                                        <div style={{ height: '100%', width: `${pct}%`, background: TOPIC_COLORS[i % TOPIC_COLORS.length], borderRadius: 2, transition: 'width 0.5s ease' }} />
                                                    </div>
                                                </div>
                                                <div style={{ fontSize: '13px', fontWeight: 600, textAlign: 'right' }}>{fmtTime(secs)}</div>
                                                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{pct}%</div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* ── Drag handle ── */}
                <div
                    onMouseDown={startDrag}
                    className="resize-handle"
                />

                {/* ── RIGHT: Output panel ── */}
                <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                        <span style={{ fontWeight: 700, fontSize: '16px' }}>Output</span>
                    </div>

                    <div style={{ padding: '0 20px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                        <div className="tabs" style={{ margin: '0', borderBottom: 'none' }}>
                            <div className={`tab${rightTab === 'transcript' ? ' active' : ''}`} style={{ fontSize: '13px' }} onClick={() => setRightTab('transcript')}>📝 Transcript</div>
                            <div className={`tab${rightTab === 'summary' ? ' active' : ''}`} style={{ fontSize: '13px' }} onClick={() => setRightTab('summary')}>🧠 Summaries</div>
                            <div className={`tab${rightTab === 'insights' ? ' active' : ''}`} style={{ fontSize: '13px' }} onClick={() => setRightTab('insights')}>📌 Events</div>
                            {job.status === 'done' && <div className={`tab${rightTab === 'chat' ? ' active' : ''}`} style={{ fontSize: '13px' }} onClick={() => setRightTab('chat')}>💬 Chat</div>}
                        </div>
                    </div>

                    {(rightTab === 'transcript' || rightTab === 'insights') && (
                        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', display: 'flex', gap: '10px', flexShrink: 0 }}>
                            <select className="input" value={speakerFilter} onChange={e => setSpeakerFilter(e.target.value)}
                                style={{ padding: '7px 12px', fontSize: '13px', flex: 1 }}>
                                {speakers.map(s => <option key={s} value={s}>{s === 'all' ? 'All speakers' : s}</option>)}
                            </select>
                            <select className="input" value={topicFilter} onChange={e => setTopicFilter(e.target.value)}
                                style={{ padding: '7px 12px', fontSize: '13px', flex: 1 }}>
                                <option value="all">All topics</option>
                                {topics.map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                        </div>
                    )}

                    <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
                        {rightTab === 'transcript' && (() => {
                            if (filteredEvents.length > 0) {
                                return filteredEvents.map((ev: any, i: number) => (
                                    <TranscriptLine key={i} ev={ev} isActive={activeEvent === i} onClick={() => seekToEvent(i)} />
                                ));
                            }
                            // Fallback: show raw transcript text when graph_events are empty
                            const rawTranscript = job.transcript || '';
                            if (rawTranscript.trim()) {
                                const lines = rawTranscript.split('\n').filter((l: string) => l.trim());
                                return (
                                    <div style={{ padding: '4px 16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                        {lines.map((line: string, i: number) => {
                                            const colonIdx = line.indexOf(':');
                                            const speaker = colonIdx > 0 ? line.slice(0, colonIdx).trim() : 'Unknown';
                                            const text = colonIdx > 0 ? line.slice(colonIdx + 1).trim() : line;
                                            return (
                                                <div key={i} style={{ display: 'flex', gap: '12px', padding: '8px 14px', borderRadius: '8px' }}>
                                                    <div style={{
                                                        width: 32, height: 32, borderRadius: '50%', background: avatarColor(speaker),
                                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                        fontSize: '11px', fontWeight: 700, flexShrink: 0,
                                                    }}>{initials(speaker)}</div>
                                                    <div style={{ flex: 1 }}>
                                                        <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '2px' }}>{speaker}</div>
                                                        <div style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.55 }}>{text}</div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                );
                            }
                            return <div className="empty-state"><div className="empty-icon">📝</div><div className="empty-title">No transcript yet</div></div>;
                        })()}

                        {rightTab === 'summary' && (
                            Object.keys(summaries).length === 0
                                ? <div className="empty-state" style={{ padding: '48px 24px' }}>
                                    <div className="empty-icon">🧠</div>
                                    <div className="empty-title">Summaries generating…</div>
                                    <div style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '4px' }}>Available after processing completes</div>
                                </div>
                                : <div style={{ padding: '4px 16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                    {Object.entries(summaries).map(([name, summary]) => {
                                        const profile = participants.find(p => p.name.toLowerCase().includes(name.toLowerCase()));
                                        return <SpeakerSummaryCard key={name} name={name} role={profile?.role || 'Attendee'} summary={summary} expanded={expandedSpeaker === name} onToggle={() => setExpandedSpeaker(expandedSpeaker === name ? null : name)} />;
                                    })}
                                </div>
                        )}

                        {rightTab === 'insights' && (() => {
                            // Use graph_events if available, otherwise fall back to raw transcript
                            const eventData = filteredEvents.length > 0
                                ? filteredEvents
                                : rawTranscriptLines
                                    .filter(l => (speakerFilter === 'all' || l.speaker === speakerFilter) && l.text.length > 20)
                                    .map(l => ({ speaker: l.speaker, summary: l.text, event_type: 'discussion', confidence: 0 }));

                            if (eventData.length === 0) {
                                return <div className="empty-state"><div className="empty-icon">📌</div><div className="empty-title">No events yet</div></div>;
                            }
                            const typeColor: Record<string, string> = { decision: 'badge-purple', problem: 'badge-red', discussion: 'badge-cyan' };
                            return (
                                <div style={{ padding: '4px 16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    {eventData.map((ev: any, i: number) => (
                                        <div key={i} onClick={() => seekToEvent(i)} style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '8px', padding: '12px 14px', cursor: 'pointer', transition: 'border-color 0.15s' }}
                                            onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--border-bright)')}
                                            onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}>
                                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '6px' }}>
                                                <div style={{ width: 24, height: 24, borderRadius: '50%', background: avatarColor(ev.speaker || '?'), display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '9px', fontWeight: 700, flexShrink: 0 }}>
                                                    {initials(ev.speaker || '?')}
                                                </div>
                                                <span style={{ fontWeight: 600, fontSize: '13px' }}>{ev.speaker}</span>
                                                <span className={`badge ${typeColor[ev.event_type] || 'badge-muted'}`} style={{ fontSize: '10px', padding: '1px 7px' }}>{ev.event_type}</span>
                                                {ev.confidence > 0 && <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>{((ev.confidence || 0) * 100).toFixed(0)}%</span>}
                                            </div>
                                            <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{(ev.summary || '').slice(0, 140)}</div>
                                        </div>
                                    ))}
                                </div>
                            );
                        })()}

                        <div style={{ display: rightTab === 'chat' ? 'block' : 'none', height: '100%' }}>
                            {job.status === 'done' && <ChatPanel jobId={id} onSeek={t => { if (videoRef.current) videoRef.current.currentTime = t; }} />}
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
