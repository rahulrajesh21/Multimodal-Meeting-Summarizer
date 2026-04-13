'use client';
import { useEffect, useState, useCallback } from 'react';
import { fetchMeetings, Job } from '@/lib/api';
import { QueueCard, MeetingCard } from '@/components/MeetingCards';
import UploadModal from '@/components/UploadModal';

export default function DashboardPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [showUpload, setShowUpload] = useState(false);

  const refresh = useCallback(async () => {
    try { setJobs(await fetchMeetings()); } catch { }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  const queue = jobs.filter(j => j.status === 'queued' || j.status === 'processing');
  const done = jobs.filter(j => j.status === 'done');
  const errored = jobs.filter(j => j.status === 'error');

  const totalEvents = done.reduce((s, j) => s + (j.events || 0), 0);
  const totalTopics = done.reduce((s, j) => s + (j.topics || 0), 0);

  return (
    <>
      {showUpload && <UploadModal onClose={() => { setShowUpload(false); refresh(); }} />}

      <div className="topbar">
        <span className="topbar-title">⚡ Dashboard</span>
        <div className="topbar-right">
          <button className="btn btn-primary" onClick={() => setShowUpload(true)}>
            + Upload Meeting
          </button>
        </div>
      </div>

      <div className="page">
        {/* Stats row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '16px', marginBottom: '32px' }}>
          <div className="stat-tile">
            <div className="stat-value">{jobs.length}</div>
            <div className="stat-label">Total Meetings</div>
          </div>
          <div className="stat-tile">
            <div className="stat-value" style={{ color: 'var(--warning)' }}>{queue.length}</div>
            <div className="stat-label">In Queue</div>
          </div>
          <div className="stat-tile">
            <div className="stat-value" style={{ color: 'var(--success)' }}>{done.length}</div>
            <div className="stat-label">Processed</div>
          </div>
          <div className="stat-tile">
            <div className="stat-value" style={{ color: 'var(--accent2)' }}>{totalEvents}</div>
            <div className="stat-label">Total Events</div>
          </div>
        </div>

        {/* Processing Queue */}
        {queue.length > 0 && (
          <section style={{ marginBottom: '36px' }}>
            <div className="section-header">
              <div>
                <div className="section-title">⚙️ Processing Queue</div>
                <div className="section-subtitle">{queue.length} meeting{queue.length !== 1 ? 's' : ''} being processed</div>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {queue.map(j => <QueueCard key={j.job_id} job={j} />)}
            </div>
          </section>
        )}

        {/* Upload CTA if empty */}
        {jobs.length === 0 && (
          <div
            className="dropzone"
            onClick={() => setShowUpload(true)}
            style={{ cursor: 'pointer' }}
          >
            <div className="dropzone-icon">🎬</div>
            <div className="dropzone-title">No meetings yet</div>
            <div className="dropzone-sub">Click here or drag a video file to get started</div>
          </div>
        )}

        {/* Completed meetings */}
        {done.length > 0 && (
          <section style={{ marginBottom: '36px' }}>
            <div className="section-header">
              <div>
                <div className="section-title">🎬 Completed Meetings</div>
                <div className="section-subtitle">{done.length} meeting{done.length !== 1 ? 's' : ''} · {totalEvents} events · {totalTopics} topics</div>
              </div>
            </div>
            <div className="grid-auto">
              {done.map(j => <MeetingCard key={j.job_id} job={j} />)}
            </div>
          </section>
        )}

        {/* Errored */}
        {errored.length > 0 && (
          <section>
            <div className="section-title" style={{ marginBottom: '12px', color: 'var(--danger)' }}>❌ Errors</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {errored.map(j => <QueueCard key={j.job_id} job={j} />)}
            </div>
          </section>
        )}
      </div>
    </>
  );
}
