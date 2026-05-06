'use client';
import { useEffect, useRef, useState } from 'react';
import { chatWithGlobal, fetchModels, ChatMessage, AgentStep } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, RotateCcw, Plus, Search, Library, MoreVertical, ChevronDown, MessageSquare, PanelLeft, PanelLeftClose } from 'lucide-react';
import { stepsToTags } from '@/components/ToolResultBubble';
import { LiveToolStep } from '@/components/LiveToolStep';
import Header from '@/components/Header';

/* ── Design Tokens ── */
const BRAND = '#7C3AED';
const PAGE_BG = '#FFFFFF';
const CARD_BG = '#FFFFFF';
const BORDER = '#E5E7EB';
const TEXT_PRIMARY = '#111827';
const TEXT_SEC = '#4B5563';
const TEXT_MUTED = '#9CA3AF';

const SUGGESTED = [
    'What were the key decisions in recent meetings?',
    'Any unresolved issues across projects?',
    'Summarize our discussion about the budget.',
    'What action items are assigned to me?'
];

type ChatSession = {
    id: string;
    title: string;
    messages: ChatMessage[];
    updatedAt: number;
};

export default function GlobalAIPage() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [liveSteps, setLiveSteps] = useState<AgentStep[]>([]);
    const [models, setModels] = useState<{ id: string }[]>([]);
    const [selectedModel, setSelectedModel] = useState('');
    
    // History State
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        fetchModels().then(m => {
            setModels(m);
            setSelectedModel(m.find(x => x.id.toLowerCase().includes('gemma'))?.id || m[0]?.id || '');
        }).catch(() => { });
        
        // Load sessions
        const stored = localStorage.getItem('vela_ai_sessions');
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                setSessions(parsed);
            } catch (e) {}
        }
    }, []);

    // Save sessions whenever they change
    useEffect(() => {
        localStorage.setItem('vela_ai_sessions', JSON.stringify(sessions));
    }, [sessions]);

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }, [messages, loading, liveSteps]);

    const handleNewChat = () => {
        setCurrentSessionId(null);
        setMessages([]);
        setInput('');
        setLiveSteps([]);
    };

    const handleSelectSession = (id: string) => {
        if (loading) return;
        const session = sessions.find(s => s.id === id);
        if (session) {
            setCurrentSessionId(id);
            setMessages(session.messages);
            setInput('');
            setLiveSteps([]);
        }
    };

    const send = async (text?: string) => {
        const msg = (text || input).trim();
        if (!msg || loading) return;
        
        const updated = [...messages, { role: 'user' as const, content: msg }];
        setMessages(updated);
        setInput('');
        setLoading(true);
        setLiveSteps([]);

        // Determine session
        let sessionId = currentSessionId;
        if (!sessionId) {
            sessionId = Date.now().toString();
            setCurrentSessionId(sessionId);
            
            const title = msg.length > 40 ? msg.substring(0, 40) + '...' : msg;
            setSessions(prev => [
                { id: sessionId!, title, messages: updated, updatedAt: Date.now() },
                ...prev
            ]);
        } else {
            setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, messages: updated, updatedAt: Date.now() } : s));
        }

        try {
            const { reply, steps, model } = await chatWithGlobal(msg, updated, s => setLiveSteps(s), selectedModel || undefined);
            setLiveSteps([]);
            
            const finalMessages = [...updated, { role: 'assistant' as const, content: reply, steps, model }];
            setMessages(finalMessages);
            
            setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, messages: finalMessages, updatedAt: Date.now() } : s));
            
        } catch (e: any) {
            setLiveSteps([]);
            const finalMessages = [...updated, { role: 'assistant' as const, content: `Error: ${e.message}` }];
            setMessages(finalMessages);
            setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, messages: finalMessages, updatedAt: Date.now() } : s));
        } finally {
            setLoading(false);
        }
    };

    const groupSessions = () => {
        const groups: Record<string, ChatSession[]> = {
            'Today': [],
            'Yesterday': [],
            'Previous 7 Days': [],
            'Older': []
        };
        
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
        const yesterday = today - 86400000;
        const weekAgo = today - 7 * 86400000;
        
        [...sessions].sort((a, b) => b.updatedAt - a.updatedAt).forEach(s => {
            if (s.updatedAt >= today) groups['Today'].push(s);
            else if (s.updatedAt >= yesterday) groups['Yesterday'].push(s);
            else if (s.updatedAt >= weekAgo) groups['Previous 7 Days'].push(s);
            else groups['Older'].push(s);
        });
        
        return groups;
    };

    const sessionGroups = groupSessions();

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: PAGE_BG, fontFamily: 'system-ui, -apple-system, sans-serif' }}>
            <Header onUpload={() => {}} />

            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                
                {/* Inner Sidebar for History */}
                <div style={{ 
                    width: sidebarOpen ? 260 : 0, 
                    borderRight: sidebarOpen ? `1px solid ${BORDER}` : 'none', 
                    background: '#FAFAFA', 
                    display: 'flex', 
                    flexDirection: 'column',
                    overflowY: 'auto',
                    overflowX: 'hidden',
                    transition: 'width 0.2s ease',
                    flexShrink: 0
                }}>
                    <div style={{ minWidth: 260, display: 'flex', flexDirection: 'column', height: '100%' }}>
                        <div style={{ padding: '20px 16px 12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                                <span style={{ fontSize: 18, fontWeight: 600, color: TEXT_PRIMARY }}>Vela AI</span>
                                <Plus style={{ width: 20, height: 20, color: TEXT_MUTED, cursor: 'pointer' }} onClick={handleNewChat} />
                            </div>
                            
                            <button 
                                onClick={handleNewChat}
                                style={{ 
                                    width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                                    padding: '10px', background: '#0F172A', color: '#fff', borderRadius: 8, border: 'none',
                                    fontSize: 14, fontWeight: 500, cursor: 'pointer', marginBottom: 24
                                }}
                            >
                                <Plus style={{ width: 16, height: 16 }} /> New Chat
                            </button>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 24 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: TEXT_SEC, fontSize: 14, cursor: 'pointer' }}>
                                    <Search style={{ width: 18, height: 18 }} /> Search Chat
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: TEXT_SEC, fontSize: 14, cursor: 'pointer' }}>
                                    <Library style={{ width: 18, height: 18 }} /> Library
                                </div>
                            </div>
                        </div>

                        <div style={{ height: 1, background: BORDER, margin: '0 16px' }} />

                        <div style={{ padding: '16px', flex: 1 }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                                <span style={{ fontSize: 13, fontWeight: 600, color: TEXT_PRIMARY }}>Topic list</span>
                                <MoreVertical style={{ width: 16, height: 16, color: TEXT_MUTED, cursor: 'pointer' }} />
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                                {['Today', 'Yesterday', 'Previous 7 Days', 'Older'].map(group => {
                                    const groupItems = sessionGroups[group];
                                    if (groupItems.length === 0) return null;
                                    
                                    return (
                                        <div key={group}>
                                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: TEXT_MUTED, fontSize: 12, marginBottom: 8 }}>
                                                {group}
                                                <ChevronDown style={{ width: 14, height: 14 }} />
                                            </div>
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                                {groupItems.map(s => (
                                                    <div 
                                                        key={s.id} 
                                                        onClick={() => handleSelectSession(s.id)}
                                                        style={{ 
                                                            padding: '8px 10px', 
                                                            borderRadius: 6, 
                                                            fontSize: 13, 
                                                            color: currentSessionId === s.id ? TEXT_PRIMARY : TEXT_SEC, 
                                                            background: currentSessionId === s.id ? '#F1F5F9' : 'transparent',
                                                            cursor: 'pointer',
                                                            whiteSpace: 'nowrap',
                                                            overflow: 'hidden',
                                                            textOverflow: 'ellipsis',
                                                            fontWeight: currentSessionId === s.id ? 500 : 400
                                                        }}
                                                    >
                                                        {s.title}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Chat Area */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '32px 48px', overflow: 'hidden', background: '#F9FAFB', position: 'relative' }}>
                    <button 
                        onClick={() => setSidebarOpen(!sidebarOpen)}
                        style={{ position: 'absolute', top: 16, left: 16, background: 'transparent', border: 'none', cursor: 'pointer', color: TEXT_MUTED, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 8, borderRadius: 8, transition: 'background 0.2s' }}
                        onMouseOver={e => e.currentTarget.style.background = '#E2E8F0'}
                        onMouseOut={e => e.currentTarget.style.background = 'transparent'}
                        title={sidebarOpen ? "Close Sidebar" : "Open Sidebar"}
                    >
                        {sidebarOpen ? <PanelLeftClose style={{ width: 20, height: 20 }} /> : <PanelLeft style={{ width: 20, height: 20 }} />}
                    </button>

                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: CARD_BG, border: `0.5px solid ${BORDER}`, borderRadius: 12, overflow: 'hidden', maxWidth: 1000, width: '100%', margin: '0 auto', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
                            <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
                                {messages.length === 0 && (
                                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 20, padding: 20 }}>
                                        <img src="/vela_logo.svg" alt="Vela" style={{ width: 52, height: 52, borderRadius: 10 }} />
                                        <div style={{ textAlign: 'center', fontSize: 13, color: TEXT_MUTED }}>Ask anything across your workspace</div>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, width: '100%', maxWidth: 360 }}>
                                            {SUGGESTED.map((p, i) => (
                                                <button key={i} onClick={() => send(p)} style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, borderRadius: 10, padding: '10px 12px', fontSize: 12, color: TEXT_SEC, cursor: 'pointer', textAlign: 'left', lineHeight: 1.4 }}>{p}</button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {messages.map((msg, i) => (
                                    <div key={i} style={{ display: 'flex', gap: 10, flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', alignItems: 'flex-start' }}>
                                        {msg.role === 'assistant' && (
                                            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 72 72" style={{ flexShrink: 0 }}>
                                                <rect x="0" y="0" width="72" height="72" rx="14" fill="#1a1a18" />
                                                <path d="M22 56 L34 16 L50 16 L38 56 Z" fill="#ffffff" opacity="0.15" />
                                                <path d="M30 56 L42 16 L52 16 L40 56 Z" fill="#4F46E5" />
                                                <circle cx="20" cy="20" r="5" fill="#ffffff" />
                                            </svg>
                                        )}
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '85%' }}>
                                            {msg.role === 'assistant' && ((msg.steps && stepsToTags(msg.steps).length > 0) || (msg.steps && msg.steps.some(s => s.type === 'thinking' && s.content))) ? (
                                                <>
                                                    {(() => {
                                                        const tStep = msg.steps?.slice().reverse().find(s => s.type === 'thinking' && s.content);
                                                        if (!tStep) return null;
                                                        return (
                                                            <details style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: '8px 12px', fontSize: 13 }}>
                                                                <summary style={{ cursor: 'pointer', color: TEXT_SEC, fontWeight: 500, userSelect: 'none' }}>View AI thought process</summary>
                                                                <div style={{ marginTop: 8, fontFamily: '"JetBrains Mono", monospace', fontSize: 11, color: TEXT_SEC, whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
                                                                    {tStep.content}
                                                                </div>
                                                            </details>
                                                        );
                                                    })()}
                                                    {msg.steps && stepsToTags(msg.steps).map(tag => (
                                                        <LiveToolStep key={tag.name} toolName={tag.name} argsRaw={tag.argsRaw} done />
                                                    ))}
                                                    <div style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, borderRadius: '4px 16px 16px 16px', padding: '10px 14px', fontSize: 13, lineHeight: 1.6, color: TEXT_PRIMARY }}>
                                                        <div className="markdown-body">
                                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                                {msg.content}
                                                            </ReactMarkdown>
                                                        </div>
                                                    </div>
                                                </>
                                            ) : (
                                                <div style={{ background: msg.role === 'user' ? BRAND : PAGE_BG, color: msg.role === 'user' ? '#fff' : TEXT_PRIMARY, border: msg.role === 'user' ? 'none' : `0.5px solid ${BORDER}`, borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '4px 16px 16px 16px', padding: '10px 14px', fontSize: 13, lineHeight: 1.6 }}>
                                                    {msg.role === 'user' ? msg.content : (
                                                        <div className="markdown-body">
                                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                                {msg.content}
                                                            </ReactMarkdown>
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                                {loading && (
                                    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 72 72" style={{ flexShrink: 0 }}>
                                            <rect x="0" y="0" width="72" height="72" rx="14" fill="#1a1a18" />
                                            <path d="M22 56 L34 16 L50 16 L38 56 Z" fill="#ffffff" opacity="0.15" />
                                            <path d="M30 56 L42 16 L52 16 L40 56 Z" fill="#4F46E5" />
                                            <circle cx="20" cy="20" r="5" fill="#ffffff" />
                                        </svg>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                            {(() => {
                                                const resolvedNames = new Set<string>();
                                                liveSteps.forEach(s => { if (s.type === 'tool_result' && s.name) resolvedNames.add(s.name); });
                                                const callEntries: { id: string; name: string; idx: number; argsRaw?: string }[] = [];
                                                const seen = new Set<string>();
                                                liveSteps.forEach((s, i) => {
                                                    if (s.type === 'tool_call' && (s.name || s.id)) {
                                                        const stableId = s.id || s.name || `unknown_${i}`;
                                                        if (!seen.has(stableId)) {
                                                            seen.add(stableId);
                                                            callEntries.push({ id: stableId, name: s.name || 'Resolving...', idx: i, argsRaw: s.args_raw });
                                                        } else {
                                                            const existing = callEntries.find(c => c.id === stableId);
                                                            if (existing) {
                                                                if (s.args_raw) existing.argsRaw = s.args_raw;
                                                                if (s.name) existing.name = s.name;
                                                            }
                                                        }
                                                    }
                                                });
                                                const tStep = liveSteps.slice().reverse().find(s => s.type === 'thinking' && s.content);

                                                return (
                                                    <>
                                                        {callEntries.map(({ id, name, argsRaw }) => (
                                                            <LiveToolStep key={id} toolName={name} argsRaw={argsRaw} done={resolvedNames.has(name)} />
                                                        ))}

                                                        {tStep && (
                                                            <details style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: '8px 12px', fontSize: 13, marginBottom: 8 }}>
                                                                <summary style={{ cursor: 'pointer', color: TEXT_SEC, fontWeight: 500, userSelect: 'none' }}>View active thought process</summary>
                                                                <div style={{ marginTop: 8, fontFamily: '"JetBrains Mono", monospace', fontSize: 11, color: TEXT_SEC, whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
                                                                    {tStep.content}
                                                                </div>
                                                            </details>
                                                        )}

                                                        <div style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, borderRadius: '4px 16px 16px 16px', padding: '10px 16px', display: 'inline-flex', gap: 5, alignItems: 'center', width: 'fit-content' }}>
                                                            <span style={{ fontSize: 13, color: TEXT_SEC, fontWeight: 500, marginRight: 4 }}>Thinking</span>
                                                            <span className="typing-dot" style={{ animationDelay: '0ms' }} />
                                                            <span className="typing-dot" style={{ animationDelay: '150ms' }} />
                                                            <span className="typing-dot" style={{ animationDelay: '300ms' }} />
                                                        </div>
                                                    </>
                                                );
                                            })()}
                                        </div>
                                    </div>
                                )}
                            </div>
                            <div style={{ padding: '12px 16px', borderTop: `0.5px solid ${BORDER}`, background: PAGE_BG, flexShrink: 0 }}>
                                {messages.length > 0 && <div style={{ textAlign: 'center', marginBottom: 8 }}><button onClick={handleNewChat} style={{ background: 'transparent', border: 'none', color: TEXT_MUTED, fontSize: 11, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 }}><RotateCcw style={{ width: 11, height: 11 }} /> Start New</button></div>}
                                <div style={{ display: 'flex', gap: 8, background: CARD_BG, borderRadius: 10, border: `0.5px solid ${BORDER}`, padding: '4px 4px 4px 14px', alignItems: 'center' }}>
                                    <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                                        placeholder="Ask anything about your workspace…" disabled={loading}
                                        style={{ flex: 1, border: 'none', background: 'transparent', color: TEXT_PRIMARY, fontSize: 13, outline: 'none', padding: '8px 0', fontFamily: 'inherit' }} />
                                    {models.length > 0 && <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ background: PAGE_BG, border: `0.5px solid ${BORDER}`, color: TEXT_SEC, fontSize: 11, borderRadius: 6, padding: '4px 8px', outline: 'none', maxWidth: 100 }}>{models.map(m => <option key={m.id} value={m.id}>{m.id.split('/').pop()}</option>)}</select>}
                                    <button onClick={() => send()} disabled={loading || !input.trim()} style={{ width: 32, height: 32, borderRadius: 8, border: 'none', background: input.trim() ? BRAND : PAGE_BG, color: input.trim() ? '#fff' : TEXT_MUTED, cursor: input.trim() ? 'pointer' : 'default', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                                        <Send style={{ width: 14, height: 14 }} />
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <style>{`
                    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0; } 100% { opacity: 1; } }
                    .typing-dot { width: 4px; height: 4px; border-radius: 50%; background: #9CA3AF; animation: typing 1.4s infinite ease-in-out; }
                    @keyframes typing { 0%, 100% { transform: translateY(0); opacity: 0.5; } 50% { transform: translateY(-4px); opacity: 1; } }
                    .markdown-body { font-size: inherit; color: inherit; }
                    .markdown-body p { margin-bottom: 0.8em; }
                    .markdown-body p:last-child { margin-bottom: 0; }
                    .markdown-body code { background: rgba(0,0,0,0.05); padding: 0.2em 0.4em; border-radius: 4px; font-size: 0.9em; font-family: "JetBrains Mono", monospace; }
                    .markdown-body pre { background: #1A1A18; color: #fff; padding: 1em; border-radius: 8px; overflow-x: auto; margin-bottom: 1em; }
                    .markdown-body pre code { background: transparent; color: inherit; padding: 0; }
                `}</style>
            </div>
    );
}
