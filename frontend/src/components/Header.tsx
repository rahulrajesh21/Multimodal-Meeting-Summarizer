'use client';
import { Search, Upload, Bell, Shield } from 'lucide-react';

export default function Header({ onUpload }: { onUpload: () => void }) {
    return (
        <div style={{
            position: 'sticky', top: 0, zIndex: 40, width: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0 32px', height: '60px',
            background: 'rgba(247, 246, 243, 0.85)', backdropFilter: 'blur(16px)',
            borderBottom: '1px solid #E8E6E1',
        }}>
            {/* Search */}
            <div style={{ flex: 1, maxWidth: '400px' }}>
                <label style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                    <Search style={{ position: 'absolute', left: '12px', width: 16, height: 16, color: '#B0AEA8' }} />
                    <input
                        type="text"
                        placeholder="Search anything..."
                        style={{
                            width: '100%', paddingLeft: '38px', paddingRight: '50px',
                            height: '38px', fontSize: '13px', fontWeight: 400,
                            borderRadius: '8px', border: '1px solid #E8E6E1',
                            background: '#FFFFFF', color: '#1A1A18', outline: 'none',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                            transition: 'border-color 0.15s, box-shadow 0.15s',
                        }}
                        onFocus={e => {
                            e.currentTarget.style.borderColor = '#4F46E5';
                            e.currentTarget.style.boxShadow = '0 0 0 3px rgba(79,70,229,0.08)';
                        }}
                        onBlur={e => {
                            e.currentTarget.style.borderColor = '#E8E6E1';
                            e.currentTarget.style.boxShadow = 'none';
                        }}
                    />
                    <div style={{ position: 'absolute', right: '10px' }}>
                        <kbd style={{
                            fontFamily: '"JetBrains Mono", monospace', fontSize: '10px',
                            padding: '2px 6px', borderRadius: '4px',
                            background: '#F7F6F3', border: '1px solid #E8E6E1', color: '#9B9891',
                        }}>
                            Cmd+K
                        </kbd>
                    </div>
                </label>
            </div>

            {/* Right */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                {/* System Status */}
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '5px 12px', borderRadius: '100px',
                    background: '#F0FDF4', border: '1px solid rgba(22,163,74,0.2)',
                }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#16A34A', boxShadow: '0 0 6px rgba(22,163,74,0.6)' }} />
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#16A34A' }}>System Nominal</span>
                </div>

                {/* Notifications */}
                <button style={{
                    position: 'relative', width: 36, height: 36,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    borderRadius: '8px', background: '#FFFFFF',
                    border: '1px solid #E8E6E1', cursor: 'pointer',
                    color: '#9B9891', transition: 'all 0.15s',
                }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = '#D4D2CC'; (e.currentTarget as HTMLElement).style.color = '#1A1A18'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = '#E8E6E1'; (e.currentTarget as HTMLElement).style.color = '#9B9891'; }}>
                    <Bell style={{ width: 17, height: 17 }} />
                    <div style={{
                        position: 'absolute', top: 7, right: 7,
                        width: 6, height: 6, borderRadius: '50%',
                        background: '#4F46E5', border: '2px solid #FFFFFF',
                    }} />
                </button>

                {/* Upload */}
                <button
                    onClick={onUpload}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        height: '38px', padding: '0 18px', borderRadius: '6px',
                        fontSize: '13px', fontWeight: 600, color: '#FFFFFF',
                        background: '#4F46E5', border: 'none', cursor: 'pointer',
                        fontFamily: '"DM Sans", system-ui, sans-serif',
                        transition: 'all 0.15s ease',
                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                    }}
                    onMouseEnter={e => {
                        (e.currentTarget as HTMLElement).style.background = '#4338CA';
                        (e.currentTarget as HTMLElement).style.boxShadow = '0 2px 8px rgba(79,70,229,0.3)';
                    }}
                    onMouseLeave={e => {
                        (e.currentTarget as HTMLElement).style.background = '#4F46E5';
                        (e.currentTarget as HTMLElement).style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)';
                    }}>
                    <Upload style={{ width: 15, height: 15 }} />
                    Upload Meeting
                </button>
            </div>
        </div>
    );
}
