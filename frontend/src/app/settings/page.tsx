'use client';
import { useState } from 'react';
import { AlertTriangle, Trash2, Shield } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ink = '#1A1A18';
const inkSec = '#6B6A66';
const inkMuted = '#9B9891';
const borderColor = '#E8E6E1';
const warmBg = '#F7F6F3';
const negative = '#DC2626';

export default function SettingsPage() {
  const [isDeleting, setIsDeleting] = useState(false);
  const [systemStatus, setSystemStatus] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleDeleteMemory = async () => {
    if (!confirm('Are you absolutely sure? This will delete all meetings, graphs, embeddings, and uploaded videos.')) return;
    setIsDeleting(true);
    setSystemStatus(null);
    try {
      const res = await fetch(`${API_BASE}/api/system/reset`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text() || 'Failed to delete memory');
      setSystemStatus({ type: 'success', text: 'All system memory and data has been successfully deleted.' });
    } catch (error: any) {
      setSystemStatus({ type: 'error', text: error.message });
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div style={{ padding: '36px 40px', maxWidth: '860px', margin: '0 auto' }}>
      <h1 style={{
        fontSize: '24px', fontWeight: 600, marginBottom: '6px',
        fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
        color: ink,
      }}>Settings</h1>
      <p style={{ color: inkMuted, marginBottom: '40px', fontSize: '14px' }}>
        Manage system preferences and data retention.
      </p>

      {/* General Preferences */}
      <section style={{ marginBottom: '36px' }}>
        <h2 style={{ fontSize: '16px', fontWeight: 600, color: ink, marginBottom: '16px' }}>General</h2>
        <div style={{
          background: '#FFFFFF', border: `1px solid ${borderColor}`, borderRadius: '12px',
          padding: '20px',
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div>
              <label style={{ fontSize: '13px', fontWeight: 500, color: inkSec, display: 'block', marginBottom: '6px' }}>
                Default Language
              </label>
              <select style={{
                width: '100%', padding: '9px 12px', borderRadius: '8px',
                border: `1px solid ${borderColor}`, background: '#FFFFFF', color: ink,
                fontSize: '13px', fontFamily: '"DM Sans", system-ui, sans-serif', outline: 'none',
              }}>
                <option>English</option>
                <option>Spanish</option>
                <option>French</option>
              </select>
            </div>
            <div>
              <label style={{ fontSize: '13px', fontWeight: 500, color: inkSec, display: 'block', marginBottom: '6px' }}>
                Timezone
              </label>
              <select style={{
                width: '100%', padding: '9px 12px', borderRadius: '8px',
                border: `1px solid ${borderColor}`, background: '#FFFFFF', color: ink,
                fontSize: '13px', fontFamily: '"DM Sans", system-ui, sans-serif', outline: 'none',
              }}>
                <option>Asia/Kolkata (IST)</option>
                <option>America/New_York (EST)</option>
                <option>Europe/London (GMT)</option>
              </select>
            </div>
          </div>
        </div>
      </section>

      {/* Danger Zone */}
      <section>
        <h2 style={{
          fontSize: '16px', fontWeight: 600, color: negative,
          display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '16px',
        }}>
          <AlertTriangle style={{ width: 16, height: 16 }} />
          Danger Zone
        </h2>
        <div style={{
          background: '#FEF2F2', border: '1px solid #FCA5A5', borderRadius: '12px', padding: '24px',
        }}>
          <p style={{ fontSize: '14px', color: inkSec, marginBottom: '20px', lineHeight: 1.6 }}>
            Deleting system memory will permanently erase all meetings, generated insights, temporal graph data,
            embeddings, and uploaded videos. Participant roles will be preserved.
          </p>
          <button
            onClick={handleDeleteMemory}
            disabled={isDeleting}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '10px 20px', borderRadius: '6px',
              background: isDeleting ? borderColor : negative,
              color: isDeleting ? inkMuted : '#FFFFFF',
              border: 'none', fontSize: '13px', fontWeight: 600,
              cursor: isDeleting ? 'not-allowed' : 'pointer',
              fontFamily: '"DM Sans", system-ui, sans-serif',
              transition: 'all 0.15s',
            }}>
            <Trash2 style={{ width: 14, height: 14 }} />
            {isDeleting ? 'Deleting...' : 'Delete All Memory'}
          </button>
          {systemStatus && (
            <div style={{
              marginTop: '16px', padding: '11px 16px', borderRadius: '8px', fontSize: '13px',
              background: systemStatus.type === 'success' ? '#F0FDF4' : '#FEF2F2',
              color: systemStatus.type === 'success' ? '#16A34A' : negative,
              border: `1px solid ${systemStatus.type === 'success' ? 'rgba(22,163,74,0.2)' : 'rgba(220,38,38,0.2)'}`,
            }}>
              {systemStatus.text}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
