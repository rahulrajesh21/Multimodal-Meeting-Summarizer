'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Video, Network, Settings,
    Zap, Moon, Plus, MoreHorizontal
} from 'lucide-react';

const mainNavItems = [
    { href: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { href: '/ai', icon: Zap, label: 'Ask AI' },
    { href: '/meetings', icon: Video, label: 'Meetings' },
    { href: '/graph', icon: Network, label: 'Graph' },
];

export default function Sidebar() {
    const pathname = usePathname();

    const NavItem = ({ href, icon: Icon, label }: { href: string, icon: any, label: string }) => {
        const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
        return (
            <Link href={href} style={{ textDecoration: 'none', display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
                <div style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px',
                    padding: '10px 4px', borderRadius: '10px',
                    width: '64px',
                    color: active ? '#4F46E5' : '#6B6A66',
                    background: active ? '#F0EEE9' : 'transparent',
                    transition: 'all 0.12s ease',
                    cursor: 'pointer'
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
                    <Icon style={{ width: 22, height: 22 }} strokeWidth={active ? 2.5 : 2} />
                    <span style={{ fontSize: '11px', fontWeight: active ? 600 : 500, textAlign: 'center', lineHeight: 1.1 }}>{label}</span>
                </div>
            </Link>
        );
    };

    return (
        <aside style={{
            width: '80px', flexShrink: 0, display: 'flex', flexDirection: 'column', alignItems: 'center',
            position: 'sticky', top: 0, left: 0, height: '100%', zIndex: 50,
            background: '#FFFFFF', borderRight: '1px solid #E8E6E1',
            padding: '16px 0',
        }}>
            {/* Top Workspace Icon */}
            <div style={{
                width: '48px', height: '48px', borderRadius: '12px',
                background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: '20px', color: '#1A1A18', fontWeight: 700, fontSize: '20px',
                cursor: 'pointer', flexShrink: 0
            }}>
                <img src="/vela_logo.svg" alt="Vela Logo" style={{ width: 34, height: 34 }} />
            </div>

            {/* Main Navigation */}
            <nav style={{ display: 'flex', flexDirection: 'column', gap: '6px', width: '100%', alignItems: 'center', flex: 1, overflowY: 'auto' }}>
                {mainNavItems.map((item) => <NavItem key={item.href} {...item} />)}
                
                <NavItem href="/mcp" icon={MoreHorizontal} label="More" />

                {/* Divider */}
                <div style={{ width: '32px', height: '1px', background: '#E8E6E1', margin: '10px 0' }} />

                <NavItem href="/settings" icon={Settings} label="Admin" />
            </nav>

            {/* Bottom Section */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', alignItems: 'center', marginTop: 'auto', paddingTop: '16px' }}>
                {/* Plus Button (Upload CTA) */}
                <Link href="#" onClick={(e) => e.preventDefault()}>
                    <button style={{
                        width: '42px', height: '42px', borderRadius: '50%',
                        background: '#F0EEE9', border: 'none', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        color: '#6B6A66', transition: 'background 0.2s ease'
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = '#E8E6E1')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = '#F0EEE9')}
                    >
                        <Plus style={{ width: 22, height: 22 }} />
                    </button>
                </Link>

                {/* Moon Button */}
                <button style={{
                    width: '42px', height: '42px', borderRadius: '50%',
                    background: '#F0EEE9', border: 'none', cursor: 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#6B6A66', transition: 'background 0.2s ease'
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = '#E8E6E1')}
                onMouseLeave={(e) => (e.currentTarget.style.background = '#F0EEE9')}
                >
                    <Moon style={{ width: 20, height: 20 }} />
                </button>

                {/* User Avatar */}
                <div style={{ position: 'relative', cursor: 'pointer', marginTop: '4px' }}>
                    <div style={{
                        width: '44px', height: '44px', borderRadius: '12px',
                        background: '#4F46E5', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '18px', fontWeight: 700, color: '#fff',
                    }}>
                        R
                    </div>
                    {/* Online Dot */}
                    <div style={{
                        position: 'absolute', bottom: '-3px', right: '-3px',
                        width: '14px', height: '14px', borderRadius: '50%',
                        background: '#10B981', border: '2px solid #FFFFFF'
                    }} />
                </div>
            </div>
        </aside>
    );
}
