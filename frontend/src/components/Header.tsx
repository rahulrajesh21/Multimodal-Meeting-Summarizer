'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Search, Upload, Bell, ArrowLeft, ArrowRight, Clock, HelpCircle, Eye, AlertTriangle, X } from 'lucide-react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Header() {
    const [uploadHovered, setUploadHovered] = useState(false);
    const router = useRouter();

    // LLM health state
    const [degraded, setDegraded] = useState(false);
    const [warnings, setWarnings] = useState<string[]>([]);
    const [dismissed, setDismissed] = useState(false);

    useEffect(() => {
        const check = async () => {
            try {
                const res = await fetch(`${API}/api/health/llm`);
                if (res.ok) {
                    const data = await res.json();
                    setDegraded(data.degraded);
                    setWarnings(data.warnings || []);
                    if (data.degraded) setDismissed(false);
                }
            } catch {
                setDegraded(true);
                setWarnings(['Cannot reach backend — check that the API server is running.']);
                setDismissed(false);
            }
        };
        check();
        const id = setInterval(check, 30_000);
        return () => clearInterval(id);
    }, []);

    const showBanner = degraded && !dismissed;

    return (
        <div style={{ position: 'sticky', top: 0, zIndex: 60, width: '100%' }}>
            {/* ── Degraded-mode warning banner ── */}
            {showBanner && (
                <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '0 20px', height: '36px',
                    background: '#FFFBEB', borderBottom: '1px solid #FCD34D',
                    fontSize: '12px', color: '#92400E',
                    fontFamily: '"DM Sans", system-ui, sans-serif',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
                        <AlertTriangle style={{ width: 14, height: 14, color: '#D97706', flexShrink: 0 }} />
                        <span>
                            <strong style={{ marginRight: 4 }}>Degraded mode:</strong>
                            {warnings[0]}
                            {warnings.length > 1 && (
                                <span style={{ marginLeft: 6, color: '#B45309' }}>
                                    +{warnings.length - 1} more issue{warnings.length > 2 ? 's' : ''}
                                </span>
                            )}
                        </span>
                        <span style={{
                            marginLeft: 8, padding: '1px 8px', borderRadius: 4,
                            background: '#FEF3C7', border: '1px solid #FCD34D',
                            fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em',
                            textTransform: 'uppercase' as const, color: '#D97706',
                        }}>
                            Start LM Studio to restore
                        </span>
                    </div>
                    <button
                        onClick={() => setDismissed(true)}
                        style={{
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            width: 20, height: 20, borderRadius: 4,
                            background: 'transparent', border: 'none', cursor: 'pointer',
                            color: '#B45309', flexShrink: 0,
                        }}
                        title="Dismiss"
                    >
                        <X style={{ width: 13, height: 13 }} />
                    </button>
                </div>
            )}

            {/* ── Main header bar ── */}
            <div style={{
                width: '100%',
                display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center',
                height: '44px',
                background: '#FFFFFF',
                borderBottom: '1px solid #E8E6E1',
            }}>
                {/* Left section */}
                <div />

                {/* Center: Navigation, Search, Upload */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <ArrowLeft style={{ width: 16, height: 16, color: '#9B9891', cursor: 'pointer' }} />
                        <ArrowRight style={{ width: 16, height: 16, color: '#D4D2CC', cursor: 'pointer' }} />
                        <Clock style={{ width: 16, height: 16, color: '#9B9891', cursor: 'pointer' }} />
                    </div>

                    <div style={{ width: '560px' }}>
                        <label style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                            <Search style={{ position: 'absolute', left: '10px', width: 14, height: 14, color: '#9B9891' }} />
                            <input
                                type="text"
                                placeholder="Describe what you are looking for"
                                style={{
                                    width: '100%', paddingLeft: '32px', paddingRight: '10px',
                                    height: '28px', fontSize: '13px', fontWeight: 400,
                                    borderRadius: '6px', border: '1px solid #E8E6E1',
                                    background: '#F7F6F3', color: '#1A1A18', outline: 'none',
                                    fontFamily: '"DM Sans", system-ui, sans-serif',
                                    transition: 'border-color 0.15s, box-shadow 0.15s, background 0.15s',
                                }}
                                onFocus={e => {
                                    e.currentTarget.style.borderColor = '#4F46E5';
                                    e.currentTarget.style.boxShadow = '0 0 0 3px rgba(79,70,229,0.08)';
                                    e.currentTarget.style.background = '#FFFFFF';
                                }}
                                onBlur={e => {
                                    e.currentTarget.style.borderColor = '#E8E6E1';
                                    e.currentTarget.style.boxShadow = 'none';
                                    e.currentTarget.style.background = '#F7F6F3';
                                }}
                            />
                        </label>
                    </div>

                    <button
                        onClick={() => window.dispatchEvent(new Event('open-upload'))}
                        onMouseEnter={() => setUploadHovered(true)}
                        onMouseLeave={() => setUploadHovered(false)}
                        style={{
                            display: 'flex', alignItems: 'center',
                            height: '28px',
                            maxWidth: uploadHovered ? '90px' : '28px',
                            paddingRight: uploadHovered ? '12px' : '0',
                            borderRadius: '6px', color: '#FFFFFF',
                            background: uploadHovered ? '#4338CA' : '#4F46E5',
                            border: 'none', cursor: 'pointer',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                            transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                            overflow: 'hidden', whiteSpace: 'nowrap',
                        }}>
                        <div style={{ width: '28px', height: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                            <Upload style={{ width: 14, height: 14 }} />
                        </div>
                        <span style={{ fontSize: '12px', fontWeight: 600, opacity: uploadHovered ? 1 : 0, transition: 'opacity 0.2s' }}>
                            Upload
                        </span>
                    </button>
                </div>

                {/* Right: User controls */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '20px', gap: '14px' }}>
                    <button
                        onClick={() => router.push('/landing-preview')}
                        style={{
                            display: 'flex', alignItems: 'center', gap: '6px',
                            height: '28px', paddingLeft: '10px', paddingRight: '12px',
                            borderRadius: '6px', background: 'transparent',
                            border: '1px solid #E8E6E1', cursor: 'pointer',
                            color: '#6B6A66', fontSize: '11px', fontWeight: 600,
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                            transition: 'all 0.15s',
                        }}
                        onMouseEnter={e => {
                            (e.currentTarget as HTMLElement).style.background = '#F7F6F3';
                            (e.currentTarget as HTMLElement).style.borderColor = '#D4D2CC';
                            (e.currentTarget as HTMLElement).style.color = '#1A1A18';
                        }}
                        onMouseLeave={e => {
                            (e.currentTarget as HTMLElement).style.background = 'transparent';
                            (e.currentTarget as HTMLElement).style.borderColor = '#E8E6E1';
                            (e.currentTarget as HTMLElement).style.color = '#6B6A66';
                        }}
                    >
                        <Eye style={{ width: 14, height: 14 }} />
                        Preview
                    </button>

                    {/* Bell — dot turns amber when degraded */}
                    <button style={{
                        position: 'relative', width: 28, height: 28,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        borderRadius: '6px', background: 'transparent',
                        border: 'none', cursor: 'pointer',
                        color: '#9B9891', transition: 'color 0.15s, background 0.15s',
                    }}
                        onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = '#F7F6F3'; (e.currentTarget as HTMLElement).style.color = '#1A1A18'; }}
                        onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent'; (e.currentTarget as HTMLElement).style.color = '#9B9891'; }}>
                        <Bell style={{ width: 16, height: 16 }} />
                        <div style={{
                            position: 'absolute', top: 4, right: 6,
                            width: 5, height: 5, borderRadius: '50%',
                            background: degraded ? '#D97706' : '#4F46E5',
                            transition: 'background 0.3s',
                        }} />
                    </button>

                    <HelpCircle style={{ width: 18, height: 18, color: '#9B9891', cursor: 'pointer' }} />
                </div>
            </div>
        </div>
    );
}
