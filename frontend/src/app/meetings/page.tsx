'use client';
import { useEffect, useState } from 'react';
import { fetchTeamsMeetings, TeamsMeeting } from '@/lib/api';
import Link from 'next/link';
import UploadModal from '@/components/UploadModal';
import { motion } from 'framer-motion';
import { Search, Calendar, SlidersHorizontal, Plus, Users, Sparkles, MoreHorizontal, Play } from 'lucide-react';

/* ── Tokens ── */
const ink = '#1A1A18';
const inkSec = '#6B6A66';
const inkMuted = '#9B9891';
const inkFaint = '#B0AEA8';
const borderColor = '#E8E6E1';
const warmBg = '#F7F6F3';
const mutedBg = '#F0EEE9';
const indigo = '#4F46E5';

function SkeletonRow({ delay }: { delay: number }) {
    return (
        <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay }}
            style={{
                display: 'flex', alignItems: 'center', padding: '16px 20px', gap: '24px',
                borderBottom: `1px solid ${borderColor}`,
            }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flex: 2 }}>
                <div className="skeleton" style={{ width: 96, height: 56, borderRadius: '8px' }} />
                <div style={{ flex: 1 }}>
                    <div className="skeleton skeleton-title" style={{ width: '70%' }} />
                    <div className="skeleton skeleton-text" style={{ width: '50%' }} />
                </div>
            </div>
            <div className="skeleton" style={{ width: 80, height: 14, borderRadius: '4px' }} />
            <div className="skeleton" style={{ width: 100, height: 14, borderRadius: '4px' }} />
            <div className="skeleton" style={{ width: 40, height: 14, borderRadius: '4px' }} />
            <div className="skeleton" style={{ width: 80, height: 28, borderRadius: '6px' }} />
        </motion.div>
    );
}

function MeetingRow({ m, index }: { m: TeamsMeeting; index: number }) {
    const [hovered, setHovered] = useState(false);
    const organizerName = m.organizer?.displayName || 'Participant';
    const participantCount = m.participants?.length || 4;

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.05 * index }}
            style={{
                display: 'flex', alignItems: 'center', padding: '14px 20px',
                gap: '24px', borderBottom: `1px solid ${borderColor}`,
                background: hovered ? warmBg : '#FFFFFF',
                cursor: 'pointer', transition: 'background 0.12s ease',
            }}
            onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>

            {/* Thumbnail + Title */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flex: 2, minWidth: '280px' }}>
                <div style={{
                    width: 96, height: 56, borderRadius: '8px', flexShrink: 0,
                    background: `linear-gradient(135deg, ${mutedBg} 0%, #E8E6E1 100%)`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    position: 'relative', overflow: 'hidden',
                }}>
                    {hovered && (
                        <div style={{
                            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
                            background: 'rgba(0,0,0,0.06)',
                        }}>
                            <Play style={{ width: 16, height: 16, color: indigo, fill: indigo }} />
                        </div>
                    )}
                    <div style={{
                        position: 'absolute', bottom: 3, right: 4,
                        fontSize: '9px', fontFamily: '"JetBrains Mono", monospace', fontWeight: 600,
                        padding: '1px 5px', borderRadius: '4px',
                        background: 'rgba(255,255,255,0.85)', color: inkMuted,
                    }}>
                        1:00:00
                    </div>
                </div>
                <div>
                    <div style={{
                        fontSize: '14px', fontWeight: 500, color: hovered ? indigo : ink,
                        transition: 'color 0.12s', marginBottom: '3px',
                    }}>{m.subject}</div>
                    <div style={{
                        fontSize: '11px', color: inkMuted,
                        fontFamily: '"JetBrains Mono", monospace',
                    }}>
                        {new Date(m.startDateTime).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                        {' '}
                        {new Date(m.startDateTime).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
                        {' - '}
                        {new Date(m.endDateTime || Date.now()).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
                    </div>
                </div>
            </div>

            {/* Organizer */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 0.7, minWidth: '100px' }}>
                <div style={{
                    width: 22, height: 22, borderRadius: '50%', background: indigo,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '9px', fontWeight: 700, color: '#fff',
                }}>
                    {organizerName.charAt(0).toUpperCase()}
                </div>
                <span style={{ fontSize: '13px', color: ink }}>{organizerName}</span>
            </div>

            {/* Platform */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 0.7 }}>
                <div style={{
                    width: 18, height: 18, borderRadius: '4px', background: '#FFFFFF',
                    border: `1px solid ${borderColor}`, display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                        <rect x="2" y="6" width="14" height="12" rx="2" fill="#4285F4" />
                        <path d="M16 10l6-4v12l-6-4V10z" fill="#34A853" />
                    </svg>
                </div>
                <span style={{ fontSize: '13px', color: inkSec }}>Google Meet</span>
            </div>

            {/* Participants */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px', width: '60px', color: inkMuted }}>
                <Users style={{ width: 14, height: 14 }} />
                <span style={{ fontSize: '13px' }}>{participantCount}</span>
            </div>

            {/* AI Insight */}
            <Link href={`/meetings/teams/${m.id}`} onClick={e => e.stopPropagation()}>
                <button style={{
                    display: 'flex', alignItems: 'center', gap: '5px',
                    padding: '6px 12px', borderRadius: '6px',
                    border: `1px solid ${borderColor}`, background: '#FFFFFF',
                    color: indigo, fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                    fontFamily: '"DM Sans", system-ui, sans-serif',
                    transition: 'all 0.12s',
                }}
                    onMouseEnter={e => {
                        (e.currentTarget as HTMLElement).style.background = '#EEF2FF';
                        (e.currentTarget as HTMLElement).style.borderColor = 'rgba(79,70,229,0.3)';
                    }}
                    onMouseLeave={e => {
                        (e.currentTarget as HTMLElement).style.background = '#FFFFFF';
                        (e.currentTarget as HTMLElement).style.borderColor = borderColor;
                    }}>
                    <Sparkles style={{ width: 12, height: 12 }} />
                    AI insight
                </button>
            </Link>

            {/* More */}
            <button style={{
                background: 'none', border: 'none', cursor: 'pointer', color: inkFaint,
                padding: '4px', display: 'flex',
            }}>
                <MoreHorizontal style={{ width: 16, height: 16 }} />
            </button>
        </motion.div>
    );
}

export default function MeetingsPage() {
    const [meetings, setMeetings] = useState<TeamsMeeting[]>([]);
    const [showUpload, setShowUpload] = useState(false);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('all');

    const load = async () => {
        try { setMeetings(await fetchTeamsMeetings()); }
        catch { }
        finally { setLoading(false); }
    };

    useEffect(() => {
        load();
        const id = setInterval(load, 5000);
        return () => clearInterval(id);
    }, []);

    const tabs = [
        { id: 'all', label: 'All meetings' },
        { id: 'mine', label: 'My meetings' },
        { id: 'shared', label: 'Shared with me' },
    ];

    return (
        <div style={{ minHeight: '100vh', background: warmBg }}>
            {showUpload && <UploadModal onClose={() => { setShowUpload(false); load(); }} />}

            <div style={{
                padding: '28px 36px', maxWidth: '1400px', margin: '24px auto 0',
                background: '#FFFFFF', borderRadius: '12px', border: `1px solid ${borderColor}`,
            }}>
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                    <h1 style={{
                        fontSize: '24px', fontWeight: 600,
                        fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
                        color: ink, margin: 0,
                    }}>Meetings</h1>
                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                        <button onClick={() => setShowUpload(true)} style={{
                            display: 'flex', alignItems: 'center', gap: '6px',
                            padding: '8px 16px', borderRadius: '6px',
                            background: indigo, color: '#fff', border: 'none',
                            fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                        }}>
                            <Plus style={{ width: 14, height: 14 }} />
                            Add new
                        </button>
                    </div>
                </div>

                {/* Tabs */}
                <div style={{ display: 'flex', gap: '0', borderBottom: `1px solid ${borderColor}`, marginBottom: '20px' }}>
                    {tabs.map(tab => (
                        <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                            padding: '10px 20px', fontSize: '13px', fontWeight: activeTab === tab.id ? 600 : 400,
                            color: activeTab === tab.id ? ink : inkMuted,
                            background: 'none', border: 'none', cursor: 'pointer',
                            borderBottom: activeTab === tab.id ? `2px solid ${ink}` : '2px solid transparent',
                            marginBottom: '-1px', transition: 'all 0.12s',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                        }}>
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Filters */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        padding: '8px 14px', borderRadius: '8px',
                        border: `1px solid ${borderColor}`, background: warmBg, width: '280px',
                    }}>
                        <Search style={{ width: 14, height: 14, color: inkFaint }} />
                        <input type="text" placeholder="Search for meeting" style={{
                            background: 'transparent', border: 'none', color: ink, outline: 'none',
                            width: '100%', fontSize: '13px',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                        }} />
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button style={{
                            display: 'flex', alignItems: 'center', gap: '6px',
                            padding: '7px 12px', borderRadius: '6px',
                            border: `1px solid ${borderColor}`, background: '#FFFFFF',
                            fontSize: '12px', fontWeight: 500, color: inkSec, cursor: 'pointer',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                        }}>
                            <Calendar style={{ width: 13, height: 13 }} />
                            Date and time
                        </button>
                        <button style={{
                            display: 'flex', alignItems: 'center', gap: '6px',
                            padding: '7px 12px', borderRadius: '6px',
                            border: `1px solid ${borderColor}`, background: '#FFFFFF',
                            fontSize: '12px', fontWeight: 500, color: inkSec, cursor: 'pointer',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                        }}>
                            <SlidersHorizontal style={{ width: 13, height: 13 }} />
                            Filter
                        </button>
                    </div>
                </div>

                {/* Table */}
                <div style={{ border: `1px solid ${borderColor}`, borderRadius: '8px', overflow: 'hidden' }}>
                    {loading ? (
                        <div>
                            {[0, 1, 2, 3, 4].map(i => <SkeletonRow key={i} delay={i * 0.08} />)}
                        </div>
                    ) : meetings.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '60px 32px', color: inkMuted }}>
                            <Calendar style={{ width: 40, height: 40, color: inkFaint, marginBottom: '12px' }} />
                            <div style={{ fontSize: '16px', fontWeight: 600, color: inkSec, marginBottom: '6px' }}>No meetings found</div>
                            <div style={{ fontSize: '13px', color: inkMuted, marginBottom: '20px' }}>Upload a new meeting to get started</div>
                            <button onClick={() => setShowUpload(true)} style={{
                                display: 'inline-flex', alignItems: 'center', gap: '6px',
                                padding: '9px 18px', borderRadius: '6px',
                                background: indigo, color: '#fff', border: 'none',
                                fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                                fontFamily: '"DM Sans", system-ui, sans-serif',
                            }}>
                                <Plus style={{ width: 14, height: 14 }} />
                                Upload Meeting
                            </button>
                        </div>
                    ) : (
                        <div>
                            {meetings.map((m, idx) => <MeetingRow key={m.id} m={m} index={idx} />)}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
