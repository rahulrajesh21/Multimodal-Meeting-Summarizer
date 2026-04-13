'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
    { href: '/', icon: '⚡', label: 'Dashboard' },
    { href: '/meetings', icon: '🎬', label: 'Meetings' },
    { href: '/graph', icon: '🕸️', label: 'Cross-Meeting Graph' },
    { href: '/roles', icon: '👥', label: 'Role Management' },
];

export default function Sidebar() {
    const pathname = usePathname();
    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <div className="logo-mark">
                    <div className="logo-icon">🧠</div>
                    <div>
                        <div className="logo-text">MeetingIQ</div>
                        <div className="logo-sub">Intelligence Platform</div>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                <div className="nav-section-label">Navigation</div>
                {navItems.map(({ href, icon, label }) => {
                    const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
                    return (
                        <Link key={href} href={href}>
                            <div className={`nav-item${active ? ' active' : ''}`}>
                                <span className="nav-icon">{icon}</span>
                                {label}
                            </div>
                        </Link>
                    );
                })}
            </nav>
        </aside>
    );
}
