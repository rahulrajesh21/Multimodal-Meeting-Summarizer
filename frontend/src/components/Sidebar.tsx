'use client';
import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Video, Network, Users, Cable, Settings,
    Zap, Upload, ChevronDown, PanelLeftClose, PanelLeft
} from 'lucide-react';

const navSections = [
    {
        label: 'INTELLIGENCE',
        items: [
            { href: '/', icon: LayoutDashboard, label: 'Dashboard' },
            { href: '/ai', icon: Zap, label: 'Ask AI' },
            { href: '/meetings', icon: Video, label: 'Meetings' },
            { href: '/graph', icon: Network, label: 'Knowledge Graph' },
        ],
    },
    {
        label: 'WORKSPACE',
        items: [
            { href: '/roles', icon: Users, label: 'Roles' },
            { href: '/mcp', icon: Cable, label: 'Integrations' },
        ],
    },
    {
        label: 'SYSTEM',
        items: [
            { href: '/settings', icon: Settings, label: 'Settings' },
        ],
    },
];

export default function Sidebar() {
    const pathname = usePathname();
    const [collapsed, setCollapsed] = useState(false);

    return (
        <aside style={{
            width: collapsed ? '72px' : '220px', flexShrink: 0, display: 'flex', flexDirection: 'column',
            position: 'sticky', top: 0, left: 0, height: '100vh', zIndex: 50,
            background: '#FFFFFF', borderRight: '1px solid #E8E6E1',
            transition: 'width 0.2s ease',
            overflow: 'hidden',
        }}>

            {/* Logo */}
            <div style={{ padding: '20px', borderBottom: '1px solid #E8E6E1', display: 'flex', alignItems: 'center', justifyContent: collapsed ? 'center' : 'space-between' }}>
                {!collapsed && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <img src="/vela_logo.svg" alt="Vela Logo" style={{ width: 32, height: 32, flexShrink: 0 }} />
                        <div style={{ whiteSpace: 'nowrap' }}>
                            <div style={{ fontSize: '16px', fontWeight: 700, color: '#1A1A18', letterSpacing: '-0.3px', lineHeight: 1 }}>
                                Vela
                            </div>
                            <div style={{ fontSize: '9px', fontWeight: 600, color: '#4F46E5', letterSpacing: '0.15em', textTransform: 'uppercase' as const, marginTop: '2px' }}>
                                Agent Platform
                            </div>
                        </div>
                    </div>
                )}
                <button 
                    onClick={() => setCollapsed(!collapsed)} 
                    style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#B0AEA8', display: 'flex', padding: 4, borderRadius: 4 }}
                    title="Toggle Sidebar"
                >
                    {collapsed ? <PanelLeft style={{ width: 20, height: 20 }} /> : <PanelLeftClose style={{ width: 20, height: 20 }} />}
                </button>
            </div>

            {/* Upload CTA */}
            <div style={{ padding: collapsed ? '16px 8px 8px' : '16px 16px 8px' }}>
                <Link href="#" onClick={(e) => e.preventDefault()}>
                    <button style={{
                        width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        gap: '8px', padding: collapsed ? '10px 0' : '9px 16px', borderRadius: '6px',
                        background: '#4F46E5', color: '#FFFFFF', border: 'none',
                        fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                        fontFamily: '"DM Sans", system-ui, sans-serif',
                        transition: 'all 0.15s ease',
                    }}>
                        <Upload style={{ width: 14, height: 14 }} />
                        {!collapsed && <span style={{ whiteSpace: 'nowrap' }}>Upload Meeting</span>}
                    </button>
                </Link>
            </div>

            {/* Navigation */}
            <nav style={{ flex: 1, padding: collapsed ? '4px 8px' : '4px 12px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {navSections.map((section) => (
                    <div key={section.label}>
                        <div style={{
                            fontSize: '10px', fontWeight: 600, textTransform: 'uppercase' as const,
                            letterSpacing: '0.1em', color: '#B0AEA8',
                            padding: collapsed ? '16px 0 4px' : '20px 8px 6px',
                            textAlign: collapsed ? 'center' : 'left',
                            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'
                        }}>
                            {!collapsed && section.label}
                        </div>
                        {section.items.map(({ href, icon: Icon, label }) => {
                            const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
                            return (
                                <Link key={href} href={href} style={{ display: 'block', textDecoration: 'none', position: 'relative' }}>
                                    {/* Active left bar */}
                                    {active && !collapsed && (
                                        <div style={{
                                            position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)',
                                            width: '2px', height: '18px', borderRadius: '0 2px 2px 0',
                                            background: '#4F46E5',
                                        }} />
                                    )}
                                    <div style={{
                                        display: 'flex', alignItems: 'center', gap: '10px', justifyContent: collapsed ? 'center' : 'flex-start',
                                        padding: collapsed ? '10px 0' : '9px 12px', borderRadius: '8px',
                                        fontSize: '13px', fontWeight: active ? 500 : 400,
                                        color: active ? '#4F46E5' : '#6B6A66',
                                        background: active ? '#F0EEE9' : 'transparent',
                                        cursor: 'pointer', transition: 'all 0.12s ease',
                                    }}
                                        title={collapsed ? label : undefined}
                                        onMouseEnter={(e) => {
                                            if (!active) {
                                                (e.currentTarget as HTMLElement).style.background = '#F7F6F3';
                                                (e.currentTarget as HTMLElement).style.color = '#1A1A18';
                                            }
                                        }}
                                        onMouseLeave={(e) => {
                                            if (!active) {
                                                (e.currentTarget as HTMLElement).style.background = 'transparent';
                                                (e.currentTarget as HTMLElement).style.color = '#6B6A66';
                                            }
                                        }}>
                                        <Icon style={{ width: 17, height: 17, flexShrink: 0 }} />
                                        {!collapsed && <span style={{ whiteSpace: 'nowrap' }}>{label}</span>}
                                    </div>
                                </Link>
                            );
                        })}
                    </div>
                ))}
            </nav>

            {/* Agent Status + User Avatar */}
            <div style={{ padding: collapsed ? '16px 8px' : '16px', borderTop: '1px solid #E8E6E1', flexShrink: 0 }}>
                {/* Agent Status */}
                {!collapsed && (
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        padding: '10px 12px', borderRadius: '8px',
                        background: '#F7F6F3', marginBottom: '12px',
                        whiteSpace: 'nowrap'
                    }}>
                        <div className="agent-dot-idle" />
                        <span style={{ fontSize: '12px', fontWeight: 500, color: '#9B9891' }}>
                            Agent idle
                        </span>
                    </div>
                )}

                {/* User */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer' }}>
                    <div style={{
                        width: 32, height: 32, borderRadius: '50%', flexShrink: 0,
                        background: '#4F46E5', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '12px', fontWeight: 700, color: '#fff',
                    }}>
                        R
                    </div>
                    {!collapsed && (
                        <>
                            <div style={{ flex: 1, minWidth: 0, whiteSpace: 'nowrap' }}>
                                <div style={{ fontSize: '13px', fontWeight: 500, color: '#1A1A18', lineHeight: 1.2 }}>Rahul</div>
                                <div style={{ fontSize: '11px', color: '#9B9891' }}>Workspace</div>
                            </div>
                            <ChevronDown style={{ width: 14, height: 14, color: '#B0AEA8', flexShrink: 0 }} />
                        </>
                    )}
                </div>
            </div>
        </aside>
    );
}
