'use client';
import { useState, useEffect } from 'react';
import { RefreshCw, MoreVertical, ExternalLink } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ink = '#1A1A18';
const inkSec = '#6B6A66';
const inkMuted = '#9B9891';
const borderColor = '#E8E6E1';
const warmBg = '#F7F6F3';
const indigo = '#4F46E5';
const positive = '#16A34A';

interface Category {
    label: string;
    description: string;
    icon: string;
    tools: string[];
}

interface McpConfig {
    enabled_categories: string[];
    categories: Record<string, Category>;
    active_tools: string[];
    mcp_connected: boolean;
}

/* ── iOS-style toggle ── */
function Toggle({ on, onChange }: { on: boolean; onChange: () => void }) {
    return (
        <div
            onClick={e => { e.stopPropagation(); onChange(); }}
            style={{
                width: 44, height: 26, borderRadius: 100, position: 'relative', cursor: 'pointer', flexShrink: 0,
                background: on ? indigo : '#D4D2CC',
                transition: 'background 0.2s',
                boxShadow: on ? `0 0 0 3px rgba(79,70,229,0.15)` : 'none',
            }}
        >
            <span style={{
                position: 'absolute', top: 3,
                left: on ? 21 : 3,
                width: 20, height: 20, borderRadius: '50%',
                background: '#FFFFFF',
                transition: 'left 0.2s',
                boxShadow: '0 1px 3px rgba(0,0,0,0.18)',
            }} />
        </div>
    );
}

/* ── Integration card matching reference design ── */
function IntegrationCard({
    catId, cat, isOn, onToggle,
}: {
    catId: string; cat: Category; isOn: boolean; onToggle: () => void;
}) {
    // Map backend categories to local static icons
    const getIconSrc = (id: string) => {
        switch (id) {
            case 'docs':
                return '/icons/Google_Docs_logo_(2014-2020).svg';
            case 'sheets':
                return '/icons/Google_Sheets_logo_(2014-2020).svg';
            case 'drive':
                return '/icons/Google_Drive_icon_(2020).svg';
            case 'calendar':
                return '/icons/Google_Calendar_icon_(2020).svg';
            case 'slack':
                return '/icons/Slack_icon_2019.svg';
            case 'email':
                // We don't have a specific gmail icon, fallback to docs to match Google formatting style
                return '/icons/Google_Docs_logo_(2014-2020).svg';
            default:
                return '/icons/Google_Docs_logo_(2014-2020).svg';
        }
    };

    const iconSrc = getIconSrc(catId);

    return (
        <div style={{
            background: '#FFFFFF',
            border: `1px solid ${isOn ? 'rgba(79,70,229,0.3)' : borderColor}`,
            borderRadius: 16,
            padding: '20px 20px 16px',
            display: 'flex', flexDirection: 'column', gap: 0,
            transition: 'border-color 0.15s, box-shadow 0.15s',
            boxShadow: isOn ? '0 2px 12px rgba(79,70,229,0.08)' : '0 1px 3px rgba(0,0,0,0.04)',
        }}>
            {/* Top row: icon + kebab */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                <div style={{
                    width: 48, height: 48, borderRadius: 12, flexShrink: 0,
                    background: warmBg, border: `1px solid ${borderColor}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                    <img src={iconSrc} alt={`${cat.label} Icon`} style={{ width: 24, height: 24, objectFit: 'contain' }} />
                </div>
                <button
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 4, color: inkMuted, borderRadius: 6 }}
                    onClick={e => e.stopPropagation()}
                >
                    <MoreVertical style={{ width: 16, height: 16 }} />
                </button>
            </div>

            {/* Name */}
            <div style={{ fontSize: 16, fontWeight: 700, color: ink, marginBottom: 6 }}>{cat.label}</div>

            {/* Description */}
            <div style={{ fontSize: 13, color: inkSec, lineHeight: 1.5, marginBottom: 12, minHeight: 40 }}>
                {cat.description}
            </div>

            {/* Tool count */}
            <div style={{ fontSize: 13, color: inkMuted, fontWeight: 500, marginBottom: 16 }}>
                {cat.tools.length} {cat.tools.length === 1 ? 'Tool' : 'Tools'}
            </div>

            {/* Divider */}
            <div style={{ height: 1, background: borderColor, margin: '0 -20px', marginBottom: 14 }} />

            {/* Bottom row: View Integration + Toggle */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <button
                    style={{
                        display: 'inline-flex', alignItems: 'center', gap: 5,
                        padding: '6px 12px', borderRadius: 8, border: `1px solid ${borderColor}`,
                        background: '#FFFFFF', color: inkSec, fontSize: 12, fontWeight: 500,
                        cursor: 'pointer', fontFamily: '"DM Sans", system-ui, sans-serif',
                        transition: 'border-color 0.15s, color 0.15s',
                    }}
                    onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = indigo; (e.currentTarget as HTMLButtonElement).style.color = indigo; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = borderColor; (e.currentTarget as HTMLButtonElement).style.color = inkSec; }}
                    onClick={e => e.stopPropagation()}
                >
                    View integration
                </button>
                <Toggle on={isOn} onChange={onToggle} />
            </div>
        </div>
    );
}

export default function IntegrationsPage() {
    const [mcpConfig, setMcpConfig] = useState<McpConfig | null>(null);
    const [enabledCats, setEnabledCats] = useState<string[]>([]);
    const [mcpSaving, setMcpSaving] = useState(false);
    const [mcpStatus, setMcpStatus] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
    const [mcpDirty, setMcpDirty] = useState(false);

    useEffect(() => {
        fetch(`${API_BASE}/api/mcp/config`)
            .then(r => r.json())
            .then((data: McpConfig) => {
                setMcpConfig(data);
                setEnabledCats(data.enabled_categories);
            })
            .catch(() => setMcpStatus({ type: 'error', text: 'Could not connect to API.' }));
    }, []);

    const toggleCategory = (catId: string) => {
        const next = enabledCats.includes(catId)
            ? enabledCats.filter(c => c !== catId)
            : [...enabledCats, catId];
        setEnabledCats(next);
        setMcpDirty(true);
        setMcpStatus(null);
    };

    const saveMcpConfig = async () => {
        setMcpSaving(true);
        setMcpStatus(null);
        try {
            const res = await fetch(`${API_BASE}/api/mcp/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled_categories: enabledCats }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Save failed');
            setMcpDirty(false);
            setMcpStatus({ type: 'info', text: `Saved. ${data.active_tools.length} tools active. Restart the API server for changes to take effect.` });
            setMcpConfig(prev => prev ? { ...prev, enabled_categories: enabledCats, active_tools: data.active_tools } : prev);
        } catch (e: any) {
            setMcpStatus({ type: 'error', text: e.message });
        } finally {
            setMcpSaving(false);
        }
    };

    const totalActiveTools = mcpConfig
        ? enabledCats.reduce((sum, cat) => sum + (mcpConfig.categories[cat]?.tools.length ?? 0), 0)
        : 0;

    return (
        <div style={{
            padding: '36px 40px', maxWidth: 1100, margin: '0 auto',
            fontFamily: '"DM Sans", system-ui, sans-serif',
        }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 28 }}>
                <div>
                    <h1 style={{
                        fontSize: 26, fontWeight: 700, color: ink, marginBottom: 4,
                        fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
                    }}>
                        Integrations &amp; Workflows
                    </h1>
                    <p style={{ fontSize: 14, color: inkMuted }}>
                        Integrate your applications using our comprehensive directory
                    </p>
                </div>

                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                    {/* Connection status badge */}
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: 8,
                        padding: '8px 14px', borderRadius: 8,
                        background: mcpConfig?.mcp_connected ? '#F0FDF4' : warmBg,
                        border: `1px solid ${mcpConfig?.mcp_connected ? 'rgba(22,163,74,0.25)' : borderColor}`,
                        fontSize: 13, fontWeight: 600,
                        color: mcpConfig?.mcp_connected ? positive : inkMuted,
                    }}>
                        <div style={{
                            width: 7, height: 7, borderRadius: '50%',
                            background: mcpConfig?.mcp_connected ? positive : '#B0AEA8',
                            boxShadow: mcpConfig?.mcp_connected ? '0 0 5px rgba(22,163,74,0.5)' : 'none',
                        }} />
                        {mcpConfig?.mcp_connected ? 'Connected' : 'Disconnected'}
                    </div>

                    <button
                        onClick={saveMcpConfig}
                        disabled={!mcpDirty || mcpSaving}
                        style={{
                            display: 'flex', alignItems: 'center', gap: 6,
                            padding: '8px 18px', borderRadius: 8,
                            fontWeight: 600, fontSize: 13, border: 'none',
                            cursor: mcpDirty ? 'pointer' : 'not-allowed',
                            background: mcpDirty ? indigo : '#E8E6E1',
                            color: mcpDirty ? '#FFFFFF' : inkMuted,
                            transition: 'all 0.15s',
                        }}
                    >
                        {mcpSaving
                            ? <><RefreshCw style={{ width: 13, height: 13, animation: 'spin 1s linear infinite' }} /> Saving...</>
                            : 'Save Changes'}
                    </button>
                </div>
            </div>

            {/* Summary bar */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                marginBottom: 24, fontSize: 13, color: inkMuted,
                borderBottom: `1px solid ${borderColor}`, paddingBottom: 16,
            }}>
                <span style={{ fontWeight: 600, color: inkSec }}>Integrations</span>
                <span>·</span>
                <span>{enabledCats.length} active</span>
                <span>·</span>
                <span>{totalActiveTools} tools enabled</span>
            </div>

            {/* Cards grid */}
            {mcpConfig ? (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                    gap: 20,
                    marginBottom: 28,
                }}>
                    {Object.entries(mcpConfig.categories).map(([catId, cat]) => (
                        <IntegrationCard
                            key={catId}
                            catId={catId}
                            cat={cat}
                            isOn={enabledCats.includes(catId)}
                            onToggle={() => toggleCategory(catId)}
                        />
                    ))}
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginBottom: 28 }}>
                    {[0, 1, 2, 3, 4, 5].map(i => (
                        <div key={i} className="skeleton" style={{ height: 220, borderRadius: 16 }} />
                    ))}
                </div>
            )}

            {/* Status message */}
            {mcpStatus && (
                <div style={{
                    marginTop: 16, padding: '11px 16px', borderRadius: 8, fontSize: 13,
                    background: mcpStatus.type === 'error' ? '#FEF2F2' : '#F0FDF4',
                    color: mcpStatus.type === 'error' ? '#DC2626' : positive,
                    border: `1px solid ${mcpStatus.type === 'error' ? 'rgba(220,38,38,0.2)' : 'rgba(22,163,74,0.2)'}`,
                }}>
                    {mcpStatus.text}
                </div>
            )}
        </div>
    );
}
