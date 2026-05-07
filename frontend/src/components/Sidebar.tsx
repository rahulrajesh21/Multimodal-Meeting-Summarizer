'use client';
import React from 'react';
import { createPortal } from 'react-dom';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard, Video, Network, Settings,
    Zap, Moon, Plus, MoreHorizontal, Blocks
} from 'lucide-react';

const mainNavItems = [
    { href: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { href: '/ai', icon: Zap, label: 'Ask AI' },
    { href: '/meetings', icon: Video, label: 'Meetings' },
    { href: '/graph', icon: Network, label: 'Graph' },
];

export default function Sidebar() {
    const pathname = usePathname();
    const [mounted, setMounted] = React.useState(false);

    React.useEffect(() => {
        setMounted(true);
    }, []);

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

    const [moreOpen, setMoreOpen] = React.useState(false);
    const moreRef = React.useRef<HTMLDivElement>(null);
    const popoverRef = React.useRef<HTMLDivElement>(null);
    const timeoutRef = React.useRef<NodeJS.Timeout | null>(null);

    React.useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            const isClickInsideButton = moreRef.current?.contains(event.target as Node);
            const isClickInsidePopover = popoverRef.current?.contains(event.target as Node);
            
            if (!isClickInsideButton && !isClickInsidePopover) {
                setMoreOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleMoreEnter = () => {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        setMoreOpen(true);
    };

    const handleMoreLeave = () => {
        timeoutRef.current = setTimeout(() => {
            setMoreOpen(false);
        }, 150);
    };

    const MoreButton = () => {
        const active = moreOpen;
        const [popoverCoords, setPopoverCoords] = React.useState({ top: 0, left: 0 });

        React.useEffect(() => {
            if (moreOpen && moreRef.current) {
                const rect = moreRef.current.getBoundingClientRect();
                setPopoverCoords({ top: rect.top - 16, left: rect.right + 12 });
            }
        }, [moreOpen]);

        const popover = (
            <div ref={popoverRef} 
                 onMouseEnter={handleMoreEnter} 
                 onMouseLeave={handleMoreLeave}
                 style={{
                position: 'fixed', top: popoverCoords.top, left: popoverCoords.left,
                width: '320px', background: '#FFFFFF',
                borderRadius: '12px', border: '1px solid #E8E6E1',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
                zIndex: 1000, padding: '16px',
                display: 'flex', flexDirection: 'column',
                fontFamily: '"DM Sans", system-ui, sans-serif'
            }}>
                <h3 style={{ margin: '0 0 12px 0', fontSize: '15px', fontWeight: 700, color: '#1A1A18' }}>More</h3>
                
                <Link href="/mcp" onClick={() => setMoreOpen(false)} style={{ textDecoration: 'none' }}>
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '14px',
                        padding: '12px 10px', margin: '0 -10px', borderRadius: '8px',
                        transition: 'background 0.15s ease',
                        cursor: 'pointer'
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = '#F7F6F3')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}>
                        <div style={{
                            width: '48px', height: '48px', borderRadius: '12px',
                            background: '#F0EEE9', display: 'flex', alignItems: 'center', justifyContent: 'center',
                            color: '#4F46E5', flexShrink: 0
                        }}>
                            <Blocks style={{ width: 24, height: 24 }} />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <span style={{ fontSize: '15px', fontWeight: 600, color: '#1A1A18' }}>MCP Integration</span>
                            <span style={{ fontSize: '13px', color: '#6B6A66', marginTop: '2px' }}>Manage Model Context Protocol external tools</span>
                        </div>
                    </div>
                </Link>
            </div>
        );

        return (
            <div ref={moreRef} 
                 onMouseEnter={handleMoreEnter} 
                 onMouseLeave={handleMoreLeave}
                 style={{ position: 'relative', width: '100%' }}>
                <div onClick={() => setMoreOpen(!moreOpen)} style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px',
                    padding: '10px 4px', borderRadius: '10px',
                    width: '64px', margin: '0 auto',
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
                    <MoreHorizontal style={{ width: 22, height: 22 }} strokeWidth={active ? 2.5 : 2} />
                    <span style={{ fontSize: '11px', fontWeight: active ? 600 : 500, textAlign: 'center', lineHeight: 1.1 }}>More</span>
                </div>

                {/* Popover rendered via Portal */}
                {moreOpen && mounted && document.body && createPortal(popover, document.body)}
            </div>
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
                
                <MoreButton />

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
