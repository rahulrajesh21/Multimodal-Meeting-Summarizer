'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Video, Network, Users, Cable, Settings,
    Zap, Upload, ChevronDown
} from 'lucide-react';

const navSections = [
    {
        label: 'INTELLIGENCE',
        items: [
            { href: '/', icon: LayoutDashboard, label: 'Dashboard' },
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

    return (
        <aside style={{
            width: '220px', flexShrink: 0, display: 'flex', flexDirection: 'column',
            position: 'fixed', top: 0, left: 0, height: '100vh', zIndex: 50,
            background: '#FFFFFF', borderRight: '1px solid #E8E6E1',
        }}>

            {/* Logo */}
            <div style={{ padding: '20px 20px 16px', borderBottom: '1px solid #E8E6E1' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{
                        width: 32, height: 32, borderRadius: '8px',
                        background: '#4F46E5', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                        <Zap style={{ width: 16, height: 16, color: '#fff', fill: '#fff' }} />
                    </div>
                    <div>
                        <div style={{ fontSize: '15px', fontWeight: 700, color: '#1A1A18', letterSpacing: '-0.3px', lineHeight: 1 }}>
                            MeetingIQ
                        </div>
                        <div style={{ fontSize: '9px', fontWeight: 600, color: '#4F46E5', letterSpacing: '0.15em', textTransform: 'uppercase' as const, marginTop: '2px' }}>
                            Agent Platform
                        </div>
                    </div>
                </div>
            </div>

            {/* Upload CTA */}
            <div style={{ padding: '16px 16px 8px' }}>
                <Link href="#" onClick={(e) => e.preventDefault()}>
                    <button style={{
                        width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        gap: '8px', padding: '9px 16px', borderRadius: '6px',
                        background: '#4F46E5', color: '#FFFFFF', border: 'none',
                        fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                        fontFamily: '"DM Sans", system-ui, sans-serif',
                        transition: 'all 0.15s ease',
                    }}>
                        <Upload style={{ width: 14, height: 14 }} />
                        Upload Meeting
                    </button>
                </Link>
            </div>

            {/* Navigation */}
            <nav style={{ flex: 1, padding: '4px 12px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {navSections.map((section) => (
                    <div key={section.label}>
                        <div style={{
                            fontSize: '10px', fontWeight: 600, textTransform: 'uppercase' as const,
                            letterSpacing: '0.1em', color: '#B0AEA8',
                            padding: '20px 8px 6px',
                        }}>
                            {section.label}
                        </div>
                        {section.items.map(({ href, icon: Icon, label }) => {
                            const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
                            return (
                                <Link key={href} href={href} style={{ display: 'block', textDecoration: 'none', position: 'relative' }}>
                                    {/* Active left bar */}
                                    {active && (
                                        <div style={{
                                            position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)',
                                            width: '2px', height: '18px', borderRadius: '0 2px 2px 0',
                                            background: '#4F46E5',
                                        }} />
                                    )}
                                    <div style={{
                                        display: 'flex', alignItems: 'center', gap: '10px',
                                        padding: '9px 12px', borderRadius: '8px',
                                        fontSize: '13px', fontWeight: active ? 500 : 400,
                                        color: active ? '#4F46E5' : '#6B6A66',
                                        background: active ? '#F0EEE9' : 'transparent',
                                        cursor: 'pointer', transition: 'all 0.12s ease',
                                    }}
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
                                        {label}
                                    </div>
                                </Link>
                            );
                        })}
                    </div>
                ))}
            </nav>

            {/* Agent Status + User Avatar */}
            <div style={{ padding: '16px', borderTop: '1px solid #E8E6E1', flexShrink: 0 }}>
                {/* Agent Status */}
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '10px 12px', borderRadius: '8px',
                    background: '#F7F6F3', marginBottom: '12px',
                }}>
                    <div className="agent-dot-idle" />
                    <span style={{ fontSize: '12px', fontWeight: 500, color: '#9B9891' }}>
                        Agent idle
                    </span>
                </div>

                {/* User */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                    <div style={{
                        width: 32, height: 32, borderRadius: '50%',
                        background: '#4F46E5', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '12px', fontWeight: 700, color: '#fff',
                    }}>
                        R
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '13px', fontWeight: 500, color: '#1A1A18', lineHeight: 1.2 }}>Rahul</div>
                        <div style={{ fontSize: '11px', color: '#9B9891' }}>Workspace</div>
                    </div>
                    <ChevronDown style={{ width: 14, height: 14, color: '#B0AEA8' }} />
                </div>
            </div>
        </aside>
    );
}
