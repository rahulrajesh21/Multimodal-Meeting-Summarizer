'use client';
import { Job } from '@/lib/api';
import Link from 'next/link';

function statusColor(status: string) {
    return { queued: 'yellow', processing: 'purple', done: 'green', error: 'red' }[status] || 'muted';
}
function statusIcon(status: string) {
    return { queued: '⏳', processing: '⚙️', done: '✅', error: '❌' }[status] || '•';
}
function fmtDate(iso: string) {
    return new Date(iso).toLocaleDateString('en-IN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function QueueCard({ job }: { job: Job }) {
    const isActive = job.status === 'processing' || job.status === 'queued';
    return (
        <div className={`queue-item${job.status === 'processing' ? ' processing' : ''}`}>
            <div className="queue-item-header">
                <div>
                    <div style={{ fontWeight: 600, fontSize: '15px' }}>{job.title}</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>{fmtDate(job.created_at)}</div>
                </div>
                <span className={`status-pill status-${job.status}`}>
                    {statusIcon(job.status)} {job.status}
                </span>
            </div>

            {isActive && (
                <>
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                        <span className={job.status === 'processing' ? 'pulse' : ''}>
                            {job.stage || 'Waiting…'}
                        </span>
                    </div>
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${job.progress}%` }} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                        <span>{job.participants.length} participant{job.participants.length !== 1 ? 's' : ''}</span>
                        <span>{job.progress}%</span>
                    </div>
                </>
            )}

            {job.status === 'done' && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        <span className="badge badge-purple">{job.events} events</span>
                        <span className="badge badge-cyan">{job.topics} topics</span>
                    </div>
                    <Link href={`/meetings/${job.job_id}`}>
                        <button className="btn btn-secondary btn-sm">View →</button>
                    </Link>
                </div>
            )}

            {job.status === 'error' && (
                <div style={{ fontSize: '13px', color: 'var(--danger)' }}>⚠ {job.error}</div>
            )}
        </div>
    );
}

export function MeetingCard({ job }: { job: Job }) {
    return (
        <Link href={`/meetings/${job.job_id}`}>
            <div className="meeting-card">
                <div className="meeting-thumb">
                    🎬
                    <div style={{ position: 'absolute', top: '10px', right: '10px' }}>
                        <span className="badge badge-green">✅ done</span>
                    </div>
                </div>
                <div className="meeting-info">
                    <div className="meeting-title">{job.title}</div>
                    <div className="meeting-meta">{fmtDate(job.created_at)}</div>
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '10px' }}>
                        {job.participants.slice(0, 3).map(p => (
                            <span key={p.name} className="participant-chip">
                                <span>{p.name.split(' ')[0]}</span>
                                <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{p.role}</span>
                            </span>
                        ))}
                        {job.participants.length > 3 && (
                            <span className="participant-chip">+{job.participants.length - 3}</span>
                        )}
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <span className="badge badge-purple">{job.events} events</span>
                        <span className="badge badge-cyan">{job.topics} topics</span>
                    </div>
                </div>
            </div>
        </Link>
    );
}
