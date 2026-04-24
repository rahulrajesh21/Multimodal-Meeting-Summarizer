'use client';
import { useState, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
            setMcpStatus({ type: 'info', text: `✅ Saved! ${data.active_tools.length} tools active. Restart the API server for changes to take effect.` });
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
        <div style={{ padding: '40px', maxWidth: '860px', margin: '0 auto' }}>
            <h1 style={{ fontSize: '28px', fontWeight: '800', marginBottom: '6px' }}>Integrations</h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '40px', fontSize: '15px' }}>
                Manage Google Workspace MCP connections and limit context usage.
            </p>

            {/* ── MCP Integrations ──────────────────────────────────────── */}
            <section style={{ marginBottom: '36px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                    <h2 style={{ fontSize: '18px', fontWeight: '700', margin: 0 }}>MCP Tools</h2>
                    <span style={{
                        fontSize: '12px', fontWeight: '600', padding: '3px 10px', borderRadius: '99px',
                        background: mcpConfig?.mcp_connected ? 'rgba(56,161,105,0.15)' : 'rgba(160,160,160,0.15)',
                        color: mcpConfig?.mcp_connected ? '#38a169' : 'var(--text-secondary)',
                    }}>
                        {mcpConfig?.mcp_connected ? '● Connected' : '○ Disconnected'}
                    </span>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '20px', lineHeight: '1.6' }}>
                    Toggle which Google Workspace tool categories the AI agent can access. Enabling more categories uses
                    more context tokens — keep only what you need for your model size.
                </p>

                {mcpConfig ? (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '12px' }}>
                        {Object.entries(mcpConfig.categories).map(([catId, cat]) => {
                            const isOn = enabledCats.includes(catId);
                            return (
                                <button
                                    key={catId}
                                    onClick={() => toggleCategory(catId)}
                                    style={{
                                        display: 'flex', flexDirection: 'column', gap: '8px',
                                        padding: '18px 20px', borderRadius: '10px', textAlign: 'left',
                                        cursor: 'pointer', transition: 'all 0.18s ease',
                                        border: isOn ? '2px solid var(--accent-color, #6366f1)' : '2px solid var(--border-color)',
                                        background: isOn ? 'rgba(99,102,241,0.08)' : 'var(--surface-color)',
                                    }}
                                >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span style={{ fontSize: '22px' }}>{cat.icon}</span>
                                        {/* Toggle pill */}
                                        <span style={{
                                            width: '38px', height: '22px', borderRadius: '99px', position: 'relative',
                                            background: isOn ? 'var(--accent-color, #6366f1)' : 'var(--border-color)',
                                            transition: 'background 0.18s',
                                            flexShrink: 0,
                                        }}>
                                            <span style={{
                                                position: 'absolute', top: '3px',
                                                left: isOn ? '19px' : '3px',
                                                width: '16px', height: '16px', borderRadius: '50%',
                                                background: '#fff', transition: 'left 0.18s',
                                            }} />
                                        </span>
                                    </div>
                                    <div>
                                        <div style={{ fontWeight: '700', fontSize: '14px', marginBottom: '3px' }}>{cat.label}</div>
                                        <div style={{ color: 'var(--text-secondary)', fontSize: '12px', lineHeight: '1.4' }}>{cat.description}</div>
                                    </div>
                                    <div style={{ fontSize: '11px', color: isOn ? 'var(--accent-color, #6366f1)' : 'var(--text-secondary)', fontWeight: '600' }}>
                                        {cat.tools.length} tools
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                ) : (
                    <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Loading configuration…</div>
                )}

                {/* Info bar + Save */}
                <div style={{ marginTop: '20px', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                        {enabledCats.length} categor{enabledCats.length === 1 ? 'y' : 'ies'} enabled
                        {' · '}<strong>{totalActiveTools}</strong> tools active
                    </div>
                    <button
                        onClick={saveMcpConfig}
                        disabled={!mcpDirty || mcpSaving}
                        style={{
                            padding: '8px 20px', borderRadius: '7px', fontWeight: '700', fontSize: '13px',
                            border: 'none', cursor: mcpDirty ? 'pointer' : 'not-allowed',
                            background: mcpDirty ? 'var(--accent-color, #6366f1)' : 'var(--border-color)',
                            color: mcpDirty ? '#fff' : 'var(--text-secondary)',
                            transition: 'all 0.18s',
                        }}
                    >
                        {mcpSaving ? 'Saving…' : 'Save Changes'}
                    </button>
                </div>

                {mcpStatus && (
                    <div style={{
                        marginTop: '14px', padding: '11px 16px', borderRadius: '7px', fontSize: '13px',
                        background: mcpStatus.type === 'error' ? 'rgba(229,62,62,0.1)' : 'rgba(56,161,105,0.1)',
                        color: mcpStatus.type === 'error' ? '#e53e3e' : '#38a169',
                        border: `1px solid ${mcpStatus.type === 'error' ? 'rgba(229,62,62,0.2)' : 'rgba(56,161,105,0.2)'}`,
                    }}>
                        {mcpStatus.text}
                    </div>
                )}
            </section>
        </div>
    );
}
