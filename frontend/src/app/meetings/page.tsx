'use client';
import { useEffect, useState } from 'react';
import { fetchTeamsMeetings, TeamsMeeting } from '@/lib/api';
import Link from 'next/link';
import UploadModal from '@/components/UploadModal';

function MeetingRow({ m }: { m: TeamsMeeting }) {
    const txCount  = m.transcripts?.length  ?? 0;
    const recCount = m.recordings?.length   ?? 0;
    return (
        <Link href={`/meetings/teams/${m.id}`} style={{ textDecoration: 'none' }}>
            <div className="summary-card" style={{ cursor: 'pointer', padding: '16px 20px' }}>
                <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
                    <div style={{
                        width: 42, height: 42, borderRadius: '10px',
                        background: 'linear-gradient(135deg,#6264a7,#7b83eb)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '18px', flexShrink: 0,
                    }}>📅</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: '15px', marginBottom: '4px' }}>{m.subject}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                            {m.organizer?.displayName}
                            &nbsp;·&nbsp;
                            {new Date(m.startDateTime).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                        {txCount > 0
                            ? <span className="badge badge-purple">📄 {txCount} transcript{txCount > 1 ? 's' : ''}</span>
                            : <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>no transcript</span>}
                        {recCount > 0
                            ? <span className="badge badge-green">🎬 {recCount} recording{recCount > 1 ? 's' : ''}</span>
                            : <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>no recording</span>}
                    </div>
                </div>
            </div>
        </Link>
    );
}

export default function MeetingsPage() {
    const [meetings, setMeetings]   = useState<TeamsMeeting[]>([]);
    const [showUpload, setShowUpload] = useState(false);
    const [loading, setLoading]     = useState(true);

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

    return (
        <>
            {showUpload && <UploadModal onClose={() => { setShowUpload(false); load(); }} />}
            <div className="topbar">
                <span className="topbar-title">📅 Teams Meetings</span>
                <button className="btn btn-primary btn-sm" onClick={() => setShowUpload(true)}>+ Upload</button>
            </div>
            <div className="page">
                {loading ? (
                    <div style={{ display: 'flex', justifyContent: 'center', padding: '60px' }}>
                        <span className="spinner" style={{ width: 36, height: 36 }} />
                    </div>
                ) : meetings.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-icon">📅</div>
                        <div className="empty-title">No meetings yet</div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '6px' }}>
                            Upload a recording to get started
                        </div>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {meetings.map(m => <MeetingRow key={m.id} m={m} />)}
                    </div>
                )}
            </div>
        </>
    );
}

