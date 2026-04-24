'use client';
import { useState, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// System dependencies only

export default function SettingsPage() {
  // ── System reset state ──
  const [isDeleting, setIsDeleting] = useState(false);
  const [systemStatus, setSystemStatus] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Removed MCP state

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

  // No additional constants needed

  return (
    <div style={{ padding: '40px', maxWidth: '860px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '28px', fontWeight: '800', marginBottom: '6px' }}>Settings</h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '40px', fontSize: '15px' }}>
        Manage integrations, MCP tools, and data retention.
      </p>

      {/* Moved MCP configuration to /mcp page */}

      {/* ── Danger Zone ───────────────────────────────────────────── */}
      <section>
        <div style={{
          background: 'var(--surface-color)', border: '1px solid rgba(229,62,62,0.35)',
          borderRadius: '10px', padding: '24px',
        }}>
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#e53e3e', marginBottom: '10px' }}>Danger Zone</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '20px', lineHeight: '1.6' }}>
            Deleting system memory will permanently erase all meetings, generated insights, temporal graph data,
            embeddings, and uploaded videos. Participant roles will be preserved.
          </p>
          <button
            onClick={handleDeleteMemory}
            disabled={isDeleting}
            style={{
              background: isDeleting ? 'var(--border-color)' : '#e53e3e',
              color: isDeleting ? 'var(--text-secondary)' : '#fff',
              border: 'none', padding: '10px 20px', borderRadius: '7px',
              fontSize: '14px', fontWeight: '700', cursor: isDeleting ? 'not-allowed' : 'pointer',
              transition: 'background 0.2s', display: 'flex', alignItems: 'center', gap: '8px',
            }}
          >
            {isDeleting ? (
              <><span className="spinner" style={{ width: '15px', height: '15px', border: '2px solid transparent', borderTopColor: 'currentColor', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />Deleting…</>
            ) : 'Delete All Memory'}
          </button>
          {systemStatus && (
            <div style={{
              marginTop: '16px', padding: '11px 16px', borderRadius: '7px', fontSize: '13px',
              background: systemStatus.type === 'success' ? 'rgba(56,161,105,0.1)' : 'rgba(229,62,62,0.1)',
              color: systemStatus.type === 'success' ? '#38a169' : '#e53e3e',
              border: `1px solid ${systemStatus.type === 'success' ? 'rgba(56,161,105,0.2)' : 'rgba(229,62,62,0.2)'}`,
            }}>
              {systemStatus.text}
            </div>
          )}
        </div>
      </section>

      <style>{`@keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }`}</style>
    </div>
  );
}
