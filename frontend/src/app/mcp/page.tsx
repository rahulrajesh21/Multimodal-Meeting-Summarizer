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
    // Map backend categories to Simple Icons SVG paths and brand colors
    const getIconSvg = (id: string) => {
        switch (id) {
            case 'docs':
                return { path: 'M14.773 0H2.727C1.227 0 0 1.227 0 2.727v18.546C0 22.773 1.227 24 2.727 24h18.546C22.773 24 24 22.773 24 21.273V9.227L14.773 0zm-1.636 18.545H5.454v-1.636h7.683v1.636zm4.364-3.818H5.454v-1.636h11.999v1.636zm0-3.818H5.454v-1.636h11.999v1.636zm-.546-3.272l-4.91-4.91v4.91h4.91z', color: '#4285F4' }; // Google Docs
            case 'sheets':
                return { path: 'M14.773 0H2.727C1.227 0 0 1.227 0 2.727v18.546C0 22.773 1.227 24 2.727 24h18.546C22.773 24 24 22.773 24 21.273V9.227L14.773 0zM12 18.545H5.455v-3.272H12v3.272zm0-4.363H5.455v-3.273H12v3.273zm6.545 4.363H13.636v-3.272h4.909v3.272zm0-4.363H13.636v-3.273h4.909v3.273zm-2.181-6.546l-4.91-4.91v4.91h4.91z', color: '#0F9D58' }; // Google Sheets
            case 'drive':
                return { path: 'M12.01 1.485c-1.696 0-3.14.782-4.008 2.052l-5.69 9.84a4.85 4.85 0 0 0 0 4.852l1.64 2.825H8.71l6.452-11.144c.433-.748 1.142-1.332 1.996-1.636L14.07 3.32a4.792 4.792 0 0 0-2.06-1.835zM8.344 19.53L2.65 9.77a4.856 4.856 0 0 0-1.22 1.693L0 13.94l5.65 9.761a4.85 4.85 0 0 0 4.225 2.45h3.31L8.344 19.53zm8.307-8.15H3.593v5.698h13.045a4.78 4.78 0 0 0 2.408-1.558l2.91-5.025a4.846 4.846 0 0 0 0-4.852l-1.464 2.525a5.558 5.558 0 0 1-3.84 3.211z', color: '#1FA463' }; // Google Drive
            case 'email':
                return { path: 'M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z', color: '#EA4335' }; // Gmail
            case 'calendar':
                return { path: 'M24 2.181a.545.545 0 0 1-.545.546H19.09v-1.636h-4.363v1.636h-5.455v-1.636H4.91v1.636H.545A.545.545 0 0 1 0 2.18V.545A.545.545 0 0 1 .545 0h4.364v1.636h4.363V0h5.455v1.636h4.363V0h4.364A.545.545 0 0 1 24 .545v1.636zM24 23.455a.545.545 0 0 1-.545.545H.545A.545.545 0 0 1 0 23.454V8.181c0-.3.245-.545.545-.545h22.91c.3 0 .545.245.545.545v15.274zm-22.909-6.546h4.364v4.364H1.09v-4.364zm5.454 0h4.364v4.364h-4.364v-4.364zm10.909 0h5.455v4.364h-5.455v-4.364zm-5.455 0h4.364v4.364h-4.364v-4.364zm-10.909-5.455h4.364v4.364H1.09V11.454zm5.454 0h4.364v4.364h-4.364v-4.364zm10.909 0h5.455v4.364h-5.455v-4.364zm-5.455 0h4.364v4.364h-4.364v-4.364z', color: '#4285F4' }; // Google Calendar
            default:
                return { path: 'M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 18.25c-3.45 0-6.25-2.8-6.25-6.25S8.55 5.75 12 5.75s6.25 2.8 6.25 6.25-2.8 6.25-6.25 6.25zm.8-6.25h3.45v1.6H10.4V8.4h1.6v3.6z', color: inkSec }; // fallback to simple clock or generic
        }
    };

    const iconInfo = getIconSvg(catId);

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
                    width: 48, height: 48, borderRadius: 12,
                    background: warmBg, border: `1px solid ${borderColor}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                    <svg viewBox="0 0 24 24" width={24} height={24} fill={iconInfo.color}>
                        <path d={iconInfo.path} />
                    </svg>
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
