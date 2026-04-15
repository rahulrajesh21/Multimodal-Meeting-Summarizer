'use client';
import { useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function SettingsPage() {
    const [isDeleting, setIsDeleting] = useState(false);
    const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

    const handleDeleteMemory = async () => {
        if (!confirm('Are you absolutely sure you want to delete all memory? This action cannot be undone and will delete all meeting jobs, graphs, embeddings, and uploaded videos.')) {
            return;
        }

        setIsDeleting(true);
        setStatusMessage(null);

        try {
            const res = await fetch(`${API_BASE}/api/system/reset`, { method: 'POST' });
            if (!res.ok) throw new Error(await res.text() || 'Failed to delete memory');

            setStatusMessage({ type: 'success', text: 'All system memory and data has been successfully deleted.' });
        } catch (error: any) {
            setStatusMessage({ type: 'error', text: error.message });
        } finally {
            setIsDeleting(false);
        }
    };

    return (
        <div style={{ padding: '40px', maxWidth: '800px', margin: '0 auto' }}>
            <h1 style={{ fontSize: '28px', fontWeight: 'bold', marginBottom: '10px' }}>Settings</h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '30px' }}>Manage system configurations and data retention.</p>
            
            <div style={{
                background: 'var(--surface-color)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                padding: '24px',
                marginTop: '20px'
            }}>
                <h2 style={{ fontSize: '18px', fontWeight: '600', color: 'var(--danger-color, #e53e3e)', marginBottom: '12px' }}>Danger Zone</h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', lineHeight: '1.5' }}>
                    Deleting system memory will permanently erase all meetings, generated insights, temporal graph data, embeddings, and uploaded videos. Participant roles will be preserved.
                </p>
                
                <button 
                    onClick={handleDeleteMemory}
                    disabled={isDeleting}
                    style={{
                        background: isDeleting ? 'var(--border-color)' : '#e53e3e',
                        color: isDeleting ? 'var(--text-secondary)' : '#fff',
                        border: 'none',
                        padding: '10px 20px',
                        borderRadius: '6px',
                        fontSize: '14px',
                        fontWeight: '600',
                        cursor: isDeleting ? 'not-allowed' : 'pointer',
                        transition: 'background 0.2s',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}
                >
                    {isDeleting ? (
                        <>
                            <span className="spinner" style={{ width: '16px', height: '16px', border: '2px solid transparent', borderTopColor: 'currentColor', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                            Deleting...
                        </>
                    ) : (
                        'Delete All Memory'
                    )}
                </button>

                {statusMessage && (
                    <div style={{
                        marginTop: '20px',
                        padding: '12px 16px',
                        borderRadius: '6px',
                        fontSize: '14px',
                        background: statusMessage.type === 'success' ? 'rgba(56, 161, 105, 0.1)' : 'rgba(229, 62, 62, 0.1)',
                        color: statusMessage.type === 'success' ? '#38a169' : '#e53e3e',
                        border: `1px solid ${statusMessage.type === 'success' ? 'rgba(56, 161, 105, 0.2)' : 'rgba(229, 62, 62, 0.2)'}`,
                    }}>
                        {statusMessage.text}
                    </div>
                )}
            </div>

            <style>{`
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
}
