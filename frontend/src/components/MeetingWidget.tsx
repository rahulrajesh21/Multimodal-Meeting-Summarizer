import React, { useState } from 'react';

type ActionItem = { owner: string; task: string };
type Meeting = { id: string; title: string; keyDecisions: string[]; majorUpdates: string[]; actionItems: ActionItem[] };
type Overview = { themes: string[]; nextSteps: string[] };

export type MeetingData = {
  overview?: Overview;
  meetings: Meeting[];
};

const SPEAKER_COLORS: Record<string, string> = {
  'Speaker A': '#9333EA', // purple
  'Speaker B': '#0D9488', // teal
  'Speaker C': '#E11D48', // coral
  'Speaker D': '#D97706', // amber
};

const getSpeakerColor = (speaker: string) => {
  for (const key in SPEAKER_COLORS) {
    if (speaker.includes(key)) return SPEAKER_COLORS[key];
  }
  return '#4B5563'; // default gray
};

export const MeetingWidget: React.FC<{ data: MeetingData }> = ({ data }) => {
  const [activeTab, setActiveTab] = useState<string>(data.overview ? 'overview' : (data.meetings[0]?.id || ''));

  const tabs = [];
  if (data.overview) tabs.push({ id: 'overview', label: 'Overview' });
  if (data.meetings) {
    data.meetings.forEach(m => tabs.push({ id: m.id, label: m.title || m.id }));
  }

  // Helper to fix LLM outputs that merge items using <br> or bullets
  const cleanAndSplit = (items: string[]) => {
    if (!items) return [];
    return items.flatMap(item => 
      item.split(/<br\s*\/?>|\n/i)
          .map(s => s.replace(/^[\s•\-\*]+/, '').trim())
          .filter(Boolean)
    );
  };

  return (
    <div className="meeting-widget" style={{ border: '1px solid #E5E7EB', borderRadius: '12px', background: '#FFFFFF', overflow: 'hidden', fontFamily: 'system-ui, sans-serif', width: '100%', minWidth: '400px' }}>
      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #E5E7EB', background: '#F9FAFB', overflowX: 'auto' }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 16px',
              border: 'none',
              background: activeTab === tab.id ? '#FFFFFF' : 'transparent',
              borderBottom: activeTab === tab.id ? '2px solid #7C3AED' : '2px solid transparent',
              cursor: 'pointer',
              fontWeight: activeTab === tab.id ? 600 : 500,
              color: activeTab === tab.id ? '#111827' : '#4B5563',
              whiteSpace: 'nowrap',
              outline: 'none'
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: '20px', maxHeight: '400px', overflowY: 'auto' }}>
        {activeTab === 'overview' && data.overview && (
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#111827', marginBottom: '12px', marginTop: 0 }}>Themes & Trends</h3>
            <ul style={{ listStyleType: 'none', padding: 0, margin: '0 0 20px 0' }}>
              {cleanAndSplit(data.overview.themes).map((theme, i) => (
                <li key={i} style={{ padding: '8px 12px', background: '#F3F4F6', borderRadius: '6px', marginBottom: '8px', fontSize: '14px', color: '#374151' }}>
                  {theme}
                </li>
              ))}
            </ul>
            <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#111827', marginBottom: '12px', marginTop: 0 }}>Consolidated Next Steps</h3>
            <ul style={{ listStyleType: 'none', padding: 0, margin: 0 }}>
              {cleanAndSplit(data.overview.nextSteps).map((step, i) => (
                <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '8px', fontSize: '14px', color: '#374151' }}>
                  <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#7C3AED', flexShrink: 0, marginTop: '6px' }} />
                  <span style={{ lineHeight: 1.4 }}>{step}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {data.meetings?.map(meeting => {
          if (activeTab !== meeting.id) return null;

          const decisions = cleanAndSplit(meeting.keyDecisions || []);
          const updates = cleanAndSplit(meeting.majorUpdates || []);
          
          // Fix action items that LLM merged into a single task with <br>
          const actions = (meeting.actionItems || []).flatMap(item => {
            const tasks = item.task.split(/<br\s*\/?>|\n/i);
            return tasks.map(t => {
              // Sometimes the LLM puts the owner in the task like "• Owner - Task"
              let owner = item.owner;
              let task = t.replace(/^[\s•\-\*]+/, '').trim();
              
              const match = task.match(/^([A-Za-z\s]+)\s*(?:\([^)]+\))?\s*[-–—]\s*(.+)$/);
              if (match && tasks.length > 1) {
                 owner = match[1].trim();
                 task = match[2].trim();
              }
              return { owner, task };
            }).filter(a => a.task);
          });

          return (
            <div key={meeting.id}>
              {decisions.length > 0 && (
                <div style={{ marginBottom: '20px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#111827', marginBottom: '8px', marginTop: 0 }}>Key Decisions</h3>
                  <ul style={{ paddingLeft: '20px', margin: 0 }}>
                    {decisions.map((decision, i) => (
                      <li key={i} style={{ color: '#374151', fontSize: '14px', marginBottom: '4px', lineHeight: 1.4 }}>
                        {decision}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {updates.length > 0 && (
                <div style={{ marginBottom: '20px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#111827', marginBottom: '8px', marginTop: 0 }}>Major Updates</h3>
                  <ul style={{ paddingLeft: '20px', margin: 0, listStyleType: 'disc' }}>
                    {updates.map((update, i) => (
                      <li key={i} style={{ color: '#374151', fontSize: '14px', marginBottom: '4px', lineHeight: 1.4 }}>
                        {update}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {actions.length > 0 && (
                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#111827', marginBottom: '8px', marginTop: 0 }}>Action Items</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {actions.map((item, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', background: '#F9FAFB', padding: '10px 12px', borderRadius: '8px', border: '1px solid #F3F4F6' }}>
                        <span style={{
                          background: getSpeakerColor(item.owner),
                          color: '#fff',
                          fontSize: '11px',
                          fontWeight: 600,
                          padding: '2px 8px',
                          borderRadius: '12px',
                          whiteSpace: 'nowrap',
                          marginTop: '2px'
                        }}>
                          {item.owner}
                        </span>
                        <span style={{ fontSize: '14px', color: '#374151', lineHeight: '1.4' }}>
                          {item.task}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
