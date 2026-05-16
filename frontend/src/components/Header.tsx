'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Search, Upload, Bell, ArrowLeft, ArrowRight, Clock, HelpCircle, Eye } from 'lucide-react';

export default function Header() {
    const [uploadHovered, setUploadHovered] = useState(false);
    const router = useRouter();

    return (
        <div style={{
            position: 'sticky', top: 0, zIndex: 60, width: '100%',
            display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center',
            height: '44px',
            background: '#FFFFFF',
        }}>
            {/* Left section: Empty to balance the grid */}
            <div />

            {/* Center section: Navigation, Search, Upload */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                {/* Navigation */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <ArrowLeft style={{ width: 16, height: 16, color: '#9B9891', cursor: 'pointer' }} />
                    <ArrowRight style={{ width: 16, height: 16, color: '#D4D2CC', cursor: 'pointer' }} />
                    <Clock style={{ width: 16, height: 16, color: '#9B9891', cursor: 'pointer' }} />
                </div>

                {/* Search Bar */}
                <div style={{ width: '560px' }}>
                    <label style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                        <Search style={{ position: 'absolute', left: '10px', width: 14, height: 14, color: '#9B9891' }} />
                        <input
                            type="text"
                            placeholder="Describe what you are looking for"
                            style={{
                                width: '100%', paddingLeft: '32px', paddingRight: '10px',
                                height: '28px', fontSize: '13px', fontWeight: 400,
                                borderRadius: '6px', border: '1px solid #E8E6E1',
                                background: '#F7F6F3', color: '#1A1A18', outline: 'none',
                                fontFamily: '"DM Sans", system-ui, sans-serif',
                                transition: 'border-color 0.15s, box-shadow 0.15s, background 0.15s',
                            }}
                            onFocus={e => {
                                e.currentTarget.style.borderColor = '#4F46E5';
                                e.currentTarget.style.boxShadow = '0 0 0 3px rgba(79,70,229,0.08)';
                                e.currentTarget.style.background = '#FFFFFF';
                            }}
                            onBlur={e => {
                                e.currentTarget.style.borderColor = '#E8E6E1';
                                e.currentTarget.style.boxShadow = 'none';
                                e.currentTarget.style.background = '#F7F6F3';
                            }}
                        />
                    </label>
                </div>

                {/* Upload button */}
                <button
                    onClick={() => window.dispatchEvent(new Event('open-upload'))}
                    onMouseEnter={() => setUploadHovered(true)}
                    onMouseLeave={() => setUploadHovered(false)}
                    style={{
                        display: 'flex', alignItems: 'center', 
                        height: '28px', 
                        maxWidth: uploadHovered ? '90px' : '28px',
                        paddingRight: uploadHovered ? '12px' : '0',
                        borderRadius: '6px',
                        color: '#FFFFFF',
                        background: uploadHovered ? '#4338CA' : '#4F46E5', 
                        border: 'none', cursor: 'pointer',
                        fontFamily: '"DM Sans", system-ui, sans-serif',
                        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                    }}>
                    <div style={{ width: '28px', height: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <Upload style={{ width: 14, height: 14 }} />
                    </div>
                    <span style={{ 
                        fontSize: '12px', fontWeight: 600,
                        opacity: uploadHovered ? 1 : 0,
                        transition: 'opacity 0.2s',
                    }}>
                        Upload
                    </span>
                </button>
            </div>

            {/* Right section: User controls */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '20px', gap: '14px' }}>
                {/* Preview Landing Page */}
                <button 
                    onClick={() => router.push('/landing-preview')}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        height: '28px', paddingLeft: '10px', paddingRight: '12px',
                        borderRadius: '6px', background: 'transparent',
                        border: '1px solid #E8E6E1', cursor: 'pointer',
                        color: '#6B6A66', fontSize: '11px', fontWeight: 600,
                        fontFamily: '"DM Sans", system-ui, sans-serif',
                        transition: 'all 0.15s',
                    }}
                    onMouseEnter={e => { 
                        (e.currentTarget as HTMLElement).style.background = '#F7F6F3'; 
                        (e.currentTarget as HTMLElement).style.borderColor = '#D4D2CC';
                        (e.currentTarget as HTMLElement).style.color = '#1A1A18'; 
                    }}
                    onMouseLeave={e => { 
                        (e.currentTarget as HTMLElement).style.background = 'transparent'; 
                        (e.currentTarget as HTMLElement).style.borderColor = '#E8E6E1';
                        (e.currentTarget as HTMLElement).style.color = '#6B6A66'; 
                    }}
                >
                    <Eye style={{ width: 14, height: 14 }} />
                    Preview
                </button>

                {/* Notifications */}
                <button style={{
                    position: 'relative', width: 28, height: 28,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    borderRadius: '6px', background: 'transparent',
                    border: 'none', cursor: 'pointer',
                    color: '#9B9891', transition: 'color 0.15s, background 0.15s',
                }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = '#F7F6F3'; (e.currentTarget as HTMLElement).style.color = '#1A1A18'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent'; (e.currentTarget as HTMLElement).style.color = '#9B9891'; }}>
                    <Bell style={{ width: 16, height: 16 }} />
                    <div style={{
                        position: 'absolute', top: 4, right: 6,
                        width: 5, height: 5, borderRadius: '50%',
                        background: '#4F46E5',
                    }} />
                </button>

                {/* Help Icon */}
                <HelpCircle style={{ width: 18, height: 18, color: '#9B9891', cursor: 'pointer' }} />

            </div>
        </div>
    );
}
