'use client';
import { useEffect, useState } from 'react';

/* ─────────────────────────────────────────
   SVG ICONS (18px)
───────────────────────────────────────── */
const ICONS: Record<string, string> = {
    search: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
    database: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
    docs: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`,
    sheets: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>`,
    calendar: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>`,
    mail: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>`,
    slack: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 10c-.83 0-1.5-.67-1.5-1.5v-5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5z"/><path d="M20.5 10H19V8.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/><path d="M9.5 14c.83 0 1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5S8 21.33 8 20.5v-5c0-.83.67-1.5 1.5-1.5z"/><path d="M3.5 14H5v1.5c0 .83-.67 1.5-1.5 1.5S2 16.33 2 15.5 2.67 14 3.5 14z"/><path d="M14 14.5c0-.83.67-1.5 1.5-1.5h5c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5h-5c-.83 0-1.5-.67-1.5-1.5z"/><path d="M15.5 19H14v1.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5-.67-1.5-1.5-1.5z"/><path d="M10 9.5C10 8.67 9.33 8 8.5 8h-5C2.67 8 2 8.67 2 9.5S2.67 11 3.5 11h5c.83 0 1.5-.67 1.5-1.5z"/><path d="M8.5 5H10V3.5C10 2.67 9.33 2 8.5 2S7 2.67 7 3.5 7.67 5 8.5 5z"/></svg>`,
};

function iconFor(name: string): string {
    if (/search|transcript|rag|context/i.test(name)) return ICONS.search;
    if (/database|sql|db/i.test(name)) return ICONS.database;
    if (/docs|drive/i.test(name)) return ICONS.docs;
    if (/sheet/i.test(name)) return ICONS.sheets;
    if (/calendar/i.test(name)) return ICONS.calendar;
    if (/mail|gmail|email/i.test(name)) return ICONS.mail;
    if (/slack/i.test(name)) return ICONS.slack;
    return ICONS.search;
}

/* ──────────────────────────────────
   Humanize tool name
────────────────────────────────── */
function humanizeTool(name: string): string {
    const map: Record<string, string> = {
        search_context: 'Meeting Context',
        search_meeting: 'Meeting Context',
        web_search: 'Web Search',
        google_docs: 'Google Docs',
        google_drive: 'Google Drive',
        google_sheets: 'Google Sheets',
        database: 'Database',
        slack: 'Slack',
        calendar: 'Calendar',
        gmail: 'Gmail',
    };
    const lower = name.toLowerCase();
    for (const [key, label] of Object.entries(map)) {
        if (lower.includes(key.replace('_', ''))) return label;
    }
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/* ──────────────────────────────────
   Single tool step row
   Line 1: [Vela logo]  Using {Tool Name}
   Line 2: [scan bar]
────────────────────────────────── */
export function LiveToolStep({
    toolName,
    argsRaw,
    done = false,
}: {
    toolName: string;
    argsRaw?: string;
    done?: boolean;
}) {
    const [visible, setVisible] = useState(false);
    const [minTimeMet, setMinTimeMet] = useState(false);

    useEffect(() => {
        // Mount transition
        const t = requestAnimationFrame(() => setVisible(true));
        // Force minimum animation duration (1.5 seconds) so fast queries don't skip the UI
        const minTimer = setTimeout(() => setMinTimeMet(true), 1500);
        return () => {
            cancelAnimationFrame(t);
            clearTimeout(minTimer);
        };
    }, []);

    // Only settle if backend says done AND the minimum visual time has passed
    const isSettled = done && minTimeMet;

    const label = humanizeTool(toolName);

    return (
        <>
            <style>{`
        @keyframes scanProgress {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(400%);  }
        }
        @keyframes ltsIn {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: translateY(0);   }
        }
      `}</style>

            <div style={{
                display: 'inline-flex',
                flexDirection: 'column',
                gap: 8,
                opacity: visible ? 1 : 0,
                transform: visible ? 'translateY(0)' : 'translateY(4px)',
                transition: 'opacity 0.2s ease, transform 0.2s ease',
            }}>
                {/* Line 1: logo + label */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{
                        width: 24, height: 24, borderRadius: 6,
                        background: isSettled ? '#F0EEE9' : '#1A1A18',
                        color: isSettled ? '#9B9891' : '#FFFFFF',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        flexShrink: 0
                    }} dangerouslySetInnerHTML={{ __html: iconFor(toolName) }} />
                    <span style={{
                        color: isSettled ? '#9B9891' : '#1A1A18',
                        fontSize: 14,
                        fontWeight: 500,
                        fontFamily: 'system-ui, -apple-system, sans-serif',
                    }}>
                        {isSettled ? `Used ${label}` : `Using ${label}`}
                    </span>
                </div>

                {/* Line 2: scan bar — only while active */}
                {!isSettled && (
                    <div style={{
                        height: 2,
                        borderRadius: 2,
                        background: '#F0EEE9',
                        overflow: 'hidden',
                        width: '100%',
                        position: 'relative',
                    }}>
                        <div style={{
                            position: 'absolute',
                            left: 0, top: 0, bottom: 0,
                            width: '30%',
                            background: 'linear-gradient(90deg, transparent, #4F46E5 40%, #7C3AED 60%, transparent)',
                            animation: 'scanProgress 1.4s linear infinite',
                        }} />
                    </div>
                )}

                {/* Arguments display block */}
                {argsRaw && (
                    <details style={{
                        background: isSettled ? 'transparent' : '#FAFAF9',
                        border: isSettled ? 'none' : '0.5px solid #E5E7EB',
                        borderRadius: '6px',
                        padding: isSettled ? '0 0 0 34px' : '8px 12px',
                        marginTop: isSettled ? '-4px' : '0',
                        fontSize: '11px',
                        fontFamily: '"JetBrains Mono", monospace',
                        color: isSettled ? '#9B9891' : '#71717A',
                        transition: 'all 0.3s ease',
                    }}>
                        <summary style={{ cursor: 'pointer', userSelect: 'none' }}>View arguments</summary>
                        <div style={{ marginTop: 8, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                            {argsRaw}
                        </div>
                    </details>
                )}
            </div>
        </>
    );
}
