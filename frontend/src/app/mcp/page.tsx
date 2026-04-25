'use client';
import { useState, useEffect } from 'react';
import { Cable, Power, Check, X, ChevronRight, RefreshCw } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ink = '#1A1A18';
const inkSec = '#6B6A66';
const inkMuted = '#9B9891';
const inkFaint = '#B0AEA8';
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
        <div style={{ padding: '36px 40px', maxWidth: '960px', margin: '0 auto' }}>
            <h1 style={{
                fontSize: '24px', fontWeight: 600, marginBottom: '6px',
                fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
                color: ink,
            }}>Integrations</h1>
            <p style={{ color: inkMuted, marginBottom: '32px', fontSize: '14px' }}>
                Manage MCP connections and control which tools the AI agent can access.
            </p>

            {/* Connection Status */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '28px',
                padding: '12px 16px', borderRadius: '8px',
                background: mcpConfig?.mcp_connected ? '#F0FDF4' : warmBg,
                border: `1px solid ${mcpConfig?.mcp_connected ? 'rgba(22,163,74,0.2)' : borderColor}`,
            }}>
                <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: mcpConfig?.mcp_connected ? positive : inkFaint,
                    boxShadow: mcpConfig?.mcp_connected ? `0 0 6px rgba(22,163,74,0.5)` : 'none',
                }} />
                <span style={{
                    fontSize: '13px', fontWeight: 600,
                    color: mcpConfig?.mcp_connected ? positive : inkMuted,
                }}>
                    {mcpConfig?.mcp_connected ? 'Connected' : 'Disconnected'}
                </span>
                <span style={{ fontSize: '12px', color: inkMuted, marginLeft: 'auto' }}>
                    {enabledCats.length} categories -- {totalActiveTools} tools active
                </span>
            </div>

            {/* Tool Categories */}
            {mcpConfig ? (
                <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
                    gap: '14px', marginBottom: '24px',
                }}>
                    {Object.entries(mcpConfig.categories).map(([catId, cat]) => {
                        const isOn = enabledCats.includes(catId);
                        return (
                            <button
                                key={catId}
                                onClick={() => toggleCategory(catId)}
                                style={{
                                    display: 'flex', flexDirection: 'column', gap: '10px',
                                    padding: '18px 20px', borderRadius: '12px', textAlign: 'left',
                                    cursor: 'pointer', transition: 'all 0.15s ease',
                                    border: isOn ? `2px solid ${indigo}` : `1px solid ${borderColor}`,
                                    background: isOn ? '#EEF2FF' : '#FFFFFF',
                                    fontFamily: '"DM Sans", system-ui, sans-serif',
                                }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '22px' }}>{cat.icon}</span>
                                    {/* Toggle */}
                                    <span style={{
                                        width: 38, height: 22, borderRadius: '100px', position: 'relative',
                                        background: isOn ? indigo : '#D4D2CC', transition: 'background 0.15s',
                                        flexShrink: 0,
                                    }}>
                                        <span style={{
                                            position: 'absolute', top: 3,
                                            left: isOn ? 19 : 3,
                                            width: 16, height: 16, borderRadius: '50%',
                                            background: '#FFFFFF', transition: 'left 0.15s',
                                            boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
                                        }} />
                                    </span>
                                </div>
                                <div>
                                    <div style={{ fontWeight: 600, fontSize: '14px', color: ink, marginBottom: '3px' }}>
                                        {cat.label}
                                    </div>
                                    <div style={{ fontSize: '12px', color: inkMuted, lineHeight: 1.4 }}>
                                        {cat.description}
                                    </div>
                                </div>
                                <div style={{
                                    fontSize: '11px', fontWeight: 600,
                                    color: isOn ? indigo : inkMuted,
                                }}>
                                    {cat.tools.length} tools
                                </div>
                            </button>
                        );
                    })}
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '24px' }}>
                    {[0, 1, 2].map(i => (
                        <div key={i} className="skeleton" style={{ height: 140, borderRadius: '12px' }} />
                    ))}
                </div>
            )}

            {/* Save */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                <button
                    onClick={saveMcpConfig}
                    disabled={!mcpDirty || mcpSaving}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        padding: '9px 20px', borderRadius: '6px',
                        fontWeight: 600, fontSize: '13px', border: 'none',
                        cursor: mcpDirty ? 'pointer' : 'not-allowed',
                        background: mcpDirty ? indigo : '#E8E6E1',
                        color: mcpDirty ? '#FFFFFF' : inkMuted,
                        fontFamily: '"DM Sans", system-ui, sans-serif',
                        transition: 'all 0.15s',
                    }}>
                    {mcpSaving ? (
                        <><RefreshCw style={{ width: 13, height: 13, animation: 'spin 1s linear infinite' }} /> Saving...</>
                    ) : 'Save Changes'}
                </button>
            </div>

            {mcpStatus && (
                <div style={{
                    marginTop: '16px', padding: '11px 16px', borderRadius: '8px', fontSize: '13px',
                    background: mcpStatus.type === 'error' ? '#FEF2F2' : '#F0FDF4',
                    color: mcpStatus.type === 'error' ? '#DC2626' : positive,
                    border: `1px solid ${mcpStatus.type === 'error' ? 'rgba(220,38,38,0.2)' : 'rgba(22,163,74,0.2)'}`,
                }}>
                    {mcpStatus.text}
                </div>
            )}

            <style>{`@keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }`}</style>
        </div>
    );
}
