'use client';
import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { fetchMeetings, Job } from '@/lib/api';
import UploadModal from '@/components/UploadModal';
import Header from '@/components/Header';
import { format } from 'date-fns';
import {
  Video, Activity, Zap, CheckCircle2, Calendar, Clock,
  Play, Sparkles, ExternalLink, Cpu,
  ArrowUpRight, MoreHorizontal, Upload
} from 'lucide-react';

/* ── Design tokens ── */
const ink = '#1A1A18';
const inkSec = '#6B6A66';
const inkMuted = '#9B9891';
const inkFaint = '#B0AEA8';
const borderColor = '#E8E6E1';
const warmBg = '#F7F6F3';
const mutedBg = '#F0EEE9';
const indigo = '#4F46E5';
const positive = '#16A34A';
const amber = '#D97706';

const cardStyle: React.CSSProperties = {
  background: '#FFFFFF',
  border: `1px solid ${borderColor}`,
  borderRadius: '12px',
  padding: '20px',
  transition: 'box-shadow 0.15s ease, border-color 0.15s ease',
};

/** Micro sparkline — 2px stroke, matches trend color */
function Sparkline({ color }: { color: string }) {
  return (
    <svg width="100%" height="32" viewBox="0 0 120 32" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`sp-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.12" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d="M0,28 C15,26 20,10 40,16 C60,22 70,6 90,10 C105,13 112,5 120,3"
        fill={`url(#sp-${color.replace('#', '')})`} stroke={color} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function StatCard({ label, value, color, icon: Icon, change, delay, hero }: any) {
  const isHero = hero === true;
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      style={{
        ...(isHero ? {
          background: ink, border: `1px solid #2A2A28`, borderRadius: '12px',
          padding: '24px', color: '#FFFFFF', gridColumn: 'span 1',
        } : cardStyle),
        cursor: 'default', position: 'relative', overflow: 'hidden',
      }}
      onMouseEnter={e => {
        if (!isHero) {
          (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 12px rgba(0,0,0,0.08), 0 12px 40px rgba(0,0,0,0.05)';
          (e.currentTarget as HTMLElement).style.borderColor = '#D4D2CC';
        }
      }}
      onMouseLeave={e => {
        if (!isHero) {
          (e.currentTarget as HTMLElement).style.boxShadow = 'none';
          (e.currentTarget as HTMLElement).style.borderColor = borderColor;
        }
      }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '16px' }}>
        <span className="stat-label" style={{ color: isHero ? 'rgba(255,255,255,0.5)' : inkMuted }}>{label}</span>
        <div style={{
          width: 32, height: 32, borderRadius: '8px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: isHero ? 'rgba(255,255,255,0.1)' : `${color}10`,
          border: isHero ? '1px solid rgba(255,255,255,0.15)' : `1px solid ${color}20`,
        }}>
          <Icon style={{ width: 15, height: 15, color: isHero ? '#FFFFFF' : color }} />
        </div>
      </div>

      <div style={{
        fontSize: isHero ? '48px' : '36px', fontWeight: 300,
        fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
        lineHeight: 1, letterSpacing: '-0.02em', marginBottom: '4px',
        color: isHero ? '#FFFFFF' : ink,
      }}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '12px' }}>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: '3px',
          padding: '2px 8px', borderRadius: '100px',
          fontSize: '11px', fontWeight: 600,
          background: isHero ? 'rgba(22,163,74,0.15)' : '#F0FDF4',
          color: positive,
        }}>
          ↑ {change || '12.4%'}
        </span>
        <span style={{ fontSize: '11px', color: isHero ? 'rgba(255,255,255,0.35)' : inkFaint }}>vs. last period</span>
      </div>

      <Sparkline color={isHero ? 'rgba(255,255,255,0.3)' : color} />
    </motion.div>
  );
}

function MeetingCard({ job, index }: { job: any; index: number }) {
  const [hovered, setHovered] = useState(false);
  const durations = ['12 min 38 sec', '15 min 12 sec', '09 min 28 sec'];
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.1 + index * 0.07 }}
      style={{
        ...cardStyle, padding: 0, overflow: 'hidden',
        cursor: 'pointer',
        borderColor: hovered ? '#D4D2CC' : borderColor,
        boxShadow: hovered ? '0 4px 12px rgba(0,0,0,0.08)' : 'none',
        transform: hovered ? 'translateY(-2px)' : 'none',
        transition: 'all 0.2s ease',
      }}
      onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>

      {/* Thumbnail */}
      <div style={{
        position: 'relative', height: '140px', overflow: 'hidden',
        background: `linear-gradient(135deg, ${mutedBg} 0%, #E8E6E1 100%)`,
      }}>
        <div style={{
          position: 'absolute', inset: 0, opacity: 0.15,
          backgroundImage: 'radial-gradient(circle at 30% 40%, rgba(79,70,229,0.4) 0%, transparent 50%), radial-gradient(circle at 75% 70%, rgba(22,163,74,0.3) 0%, transparent 50%)',
        }} />

        {hovered && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.05)',
          }}>
            <div style={{
              width: 44, height: 44, borderRadius: '50%',
              background: indigo, display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 4px 16px rgba(79,70,229,0.4)',
            }}>
              <Play style={{ width: 18, height: 18, color: '#fff', fill: '#fff', marginLeft: '2px' }} />
            </div>
          </div>
        )}

        <div style={{
          position: 'absolute', bottom: 8, right: 8,
          fontFamily: '"JetBrains Mono", monospace', fontSize: '10px', fontWeight: 600,
          padding: '3px 8px', borderRadius: '6px',
          background: 'rgba(255,255,255,0.9)', color: inkMuted,
          border: `1px solid ${borderColor}`,
        }}>
          {durations[index % 3]}
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: '16px 18px' }}>
        <h3 style={{
          fontSize: '15px', fontWeight: 600, marginBottom: '6px',
          color: hovered ? indigo : ink, transition: 'color 0.15s',
        }}>
          {job.title || 'AMI Meeting ES2002d'}
        </h3>

        <div style={{
          display: 'flex', alignItems: 'center', gap: '6px',
          fontSize: '11px', color: inkMuted, fontFamily: '"JetBrains Mono", monospace',
          marginBottom: '14px',
        }}>
          <Calendar style={{ width: 13, height: 13 }} />
          {format(new Date(), 'MMM dd, yyyy')} · 3 participants
        </div>

        {/* Key Decision */}
        <div style={{
          padding: '10px 12px', borderRadius: '8px', marginBottom: '14px',
          background: '#EEF2FF', border: '1px solid rgba(79,70,229,0.12)',
        }}>
          <div style={{
            fontSize: '9px', fontFamily: '"JetBrains Mono", monospace', fontWeight: 700,
            letterSpacing: '0.2em', textTransform: 'uppercase' as const,
            color: indigo, marginBottom: '4px',
          }}>Key Decision</div>
          <p style={{ fontSize: '12px', lineHeight: 1.5, color: inkSec, margin: 0 }}>
            Budget increment approved for Phase 2 rollout. Timeline moved to Q3.
          </p>
        </div>

        {/* Footer */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex' }}>
            {['#4F46E5', '#16A34A', '#D97706'].map((c, i) => (
              <div key={i} style={{
                width: 26, height: 26, borderRadius: '50%',
                border: '2px solid #FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '9px', fontWeight: 700, color: '#fff', background: c,
                marginLeft: i > 0 ? '-6px' : '0',
              }}>P{i + 1}</div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: '4px' }}>
            <span style={{
              fontSize: '9px', fontFamily: '"JetBrains Mono", monospace', fontWeight: 700,
              textTransform: 'uppercase' as const, letterSpacing: '0.1em',
              padding: '3px 8px', borderRadius: '6px',
              background: '#F0FDF4', border: '1px solid rgba(22,163,74,0.2)', color: positive,
            }}>Done</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default function DashboardPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  const refresh = useCallback(async () => {
    try { setJobs(await fetchMeetings()); } catch { }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    const clock = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => { clearInterval(id); clearInterval(clock); };
  }, [refresh]);

  const queue = jobs.filter(j => j.status === 'queued' || j.status === 'processing');
  const done = jobs.filter(j => j.status === 'done');
  const totalEvents = done.reduce((s, j) => s + (j.events || 0), 0);

  const mockCards = [
    { job_id: 'm1', title: 'Sprint Review Demo' },
    { job_id: 'm2', title: 'Client Walkthrough: ABC Corp' },
    { job_id: 'm3', title: 'Marketing Presentation Review' },
  ];
  const displayCards = done.length > 0 ? done.slice(0, 3) : mockCards;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: warmBg }}>
      {showUpload && <UploadModal onClose={() => { setShowUpload(false); refresh(); }} />}

      <Header onUpload={() => setShowUpload(true)} />

      <div style={{ display: 'flex', flex: 1, padding: '32px', gap: '28px' }}>
        {/* Main Column */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '36px' }}>

          {/* Hero */}
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
            <p style={{
              fontSize: '11px', fontFamily: '"JetBrains Mono", monospace', fontWeight: 600,
              textTransform: 'uppercase' as const, letterSpacing: '0.2em',
              color: inkFaint, marginBottom: '10px',
            }}>
              {format(currentTime, 'EEEE, MMMM dd')}
            </p>
            <h1 style={{
              fontSize: '34px', fontWeight: 700, lineHeight: 1.15, marginBottom: '10px',
              fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
              color: ink, letterSpacing: '-0.01em',
            }}>
              Welcome back, Rahul
            </h1>
            <p style={{ fontSize: '14px', color: inkMuted, maxWidth: '480px', lineHeight: 1.6 }}>
              Your intelligence dashboard. Let AI handle the notes.{' '}
              <span style={{ color: queue.length > 0 ? indigo : positive, fontWeight: 600 }}>
                {queue.length > 0 ? 'Agent is processing...' : 'Agent is idle.'}
              </span>
            </p>

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '10px', marginTop: '24px' }}>
              <button onClick={() => setShowUpload(true)} className="btn-primary" style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '10px 20px', borderRadius: '6px',
                fontSize: '13px', fontWeight: 600, color: '#fff',
                background: indigo, border: 'none', cursor: 'pointer',
                fontFamily: '"DM Sans", system-ui, sans-serif',
              }}>
                <Upload style={{ width: 15, height: 15 }} />
                Start a new meeting
              </button>
              <button className="btn-secondary" style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '10px 20px', borderRadius: '6px',
                fontSize: '13px', fontWeight: 500, color: inkSec,
                background: '#FFFFFF', border: `1px solid ${borderColor}`, cursor: 'pointer',
                fontFamily: '"DM Sans", system-ui, sans-serif',
              }}>
                <Sparkles style={{ width: 15, height: 15 }} />
                Agent Analysis
              </button>
            </div>
          </motion.div>

          {/* Stats */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <span className="stat-label" style={{ color: inkMuted }}>INTELLIGENCE OVERVIEW</span>
              <span style={{ fontSize: '11px', fontFamily: '"JetBrains Mono", monospace', color: inkFaint }}>
                Jan 1 – {format(currentTime, 'MMM dd, yyyy')}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
              <StatCard label="Events Tracked" value={totalEvents || 2114} color={indigo} icon={Zap} change="25.3%" delay={0} hero />
              <StatCard label="Total Meetings" value={jobs.length || 16} color={indigo} icon={Video} change="6.4%" delay={0.08} />
              <StatCard label="In Queue" value={queue.length || 3} color={amber} icon={Activity} change="10.5%" delay={0.16} />
              <StatCard label="Processed" value={done.length || 12} color={positive} icon={CheckCircle2} change="4.6%" delay={0.24} />
            </div>
          </div>

          {/* Latest Recordings */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 600, color: ink }}>Latest recordings</h2>
              <button style={{
                display: 'flex', alignItems: 'center', gap: '4px',
                fontSize: '12px', fontWeight: 600, color: inkMuted,
                background: 'none', border: 'none', cursor: 'pointer',
                fontFamily: '"DM Sans", system-ui, sans-serif',
              }}>
                View all <ArrowUpRight style={{ width: 14, height: 14 }} />
              </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
              {displayCards.map((job, i) => <MeetingCard key={job.job_id} job={job} index={i} />)}
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div style={{ width: '280px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* Clock Widget */}
          <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            style={{
              ...cardStyle, textAlign: 'center' as const, padding: '28px',
            }}>
            <p style={{
              fontSize: '10px', fontFamily: '"JetBrains Mono", monospace', fontWeight: 600,
              textTransform: 'uppercase' as const, letterSpacing: '0.2em',
              color: indigo, marginBottom: '14px',
            }}>Local Time</p>
            <div style={{
              fontSize: '48px', fontWeight: 300, lineHeight: 1, letterSpacing: '-0.02em',
              fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
              color: ink,
            }}>
              {format(currentTime, 'hh:mm')}
              <span style={{ fontSize: '20px', fontWeight: 500, marginLeft: '4px', color: indigo }}>
                {format(currentTime, 'a').toLowerCase()}
              </span>
            </div>
            <p style={{
              fontSize: '11px', fontFamily: '"JetBrains Mono", monospace', fontWeight: 600,
              textTransform: 'uppercase' as const, letterSpacing: '0.15em',
              color: inkFaint, marginTop: '14px',
            }}>
              {format(currentTime, 'EEE, MMM dd')}
            </p>
          </motion.div>

          {/* Agent Activity Feed */}
          <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.3 }}
            style={{
              background: '#FFFFFF', border: `1px solid ${borderColor}`, borderRadius: '12px', overflow: 'hidden',
            }}>
            <div style={{
              padding: '14px 16px', display: 'flex', alignItems: 'center', gap: '8px',
              borderBottom: `1px solid ${borderColor}`,
            }}>
              <motion.div animate={{ opacity: [0.4, 1, 0.4] }} transition={{ duration: 1.5, repeat: Infinity }}>
                <Cpu style={{ width: 14, height: 14, color: indigo }} />
              </motion.div>
              <span style={{
                fontSize: '10px', fontFamily: '"JetBrains Mono", monospace', fontWeight: 700,
                textTransform: 'uppercase' as const, letterSpacing: '0.15em', color: indigo,
              }}>Agent Feed</span>
            </div>
            <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {[
                { time: '10:47:32', msg: 'Generating action items for Design Sync...', active: true },
                { time: '10:43:05', msg: 'Indexed 4 decisions to Knowledge Graph.', active: false },
                { time: '10:42:10', msg: 'Summarizer completed job ES200.', active: false },
              ].map((log, i) => (
                <div key={i} style={{ display: 'flex', gap: '8px' }}>
                  <div style={{
                    marginTop: '4px', width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                    background: log.active ? indigo : '#D4D2CC',
                    boxShadow: log.active ? '0 0 6px rgba(79,70,229,0.5)' : 'none',
                  }} />
                  <div>
                    <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '9px', color: inkFaint }}>
                      [{log.time}]
                    </span>
                    <span style={{
                      fontFamily: '"JetBrains Mono", monospace', fontSize: '11px',
                      color: log.active ? ink : inkMuted, display: 'block', lineHeight: 1.4,
                    }}>
                      {log.msg}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Upcoming Meetings */}
          <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.4 }}
            style={{
              background: '#FFFFFF', border: `1px solid ${borderColor}`, borderRadius: '12px',
              overflow: 'hidden', flex: 1,
            }}>
            <div style={{
              padding: '16px 18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              borderBottom: `1px solid ${borderColor}`,
            }}>
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: ink }}>Upcoming</h3>
              <button style={{
                display: 'flex', alignItems: 'center', gap: '4px',
                fontFamily: '"JetBrains Mono", monospace', fontSize: '10px',
                padding: '4px 10px', borderRadius: '6px',
                border: `1px solid ${borderColor}`, background: warmBg, color: inkMuted,
                cursor: 'pointer',
              }}>
                <Calendar style={{ width: 12, height: 12 }} />
                Today
              </button>
            </div>
            <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                { title: 'Product Roadmap Discussion', time: '10:00 AM', end: '11:30 AM' },
                { title: 'Marketing Strategy Planning', time: '12:00 PM', end: '01:00 PM' },
                { title: 'Engineering Standup', time: '02:00 PM', end: '03:00 PM' },
              ].map((sync, i) => (
                <div key={i} style={{
                  padding: '12px 14px', borderRadius: '8px', border: `1px solid ${borderColor}`,
                  background: '#FFFFFF', cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLElement).style.borderColor = '#D4D2CC';
                    (e.currentTarget as HTMLElement).style.background = warmBg;
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.borderColor = borderColor;
                    (e.currentTarget as HTMLElement).style.background = '#FFFFFF';
                  }}>
                  <div style={{ fontSize: '13px', fontWeight: 500, color: ink, marginBottom: '4px' }}>{sync.title}</div>
                  <div style={{
                    fontSize: '10px', fontFamily: '"JetBrains Mono", monospace',
                    color: inkMuted, display: 'flex', alignItems: 'center', gap: '4px',
                  }}>
                    <Clock style={{ width: 11, height: 11 }} />
                    {sync.time} – {sync.end}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
