// ── Main Processing API (summaries / insights) ─────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Teams Media Server (transcript + video source) ─────────────────────────
export const TEAMS_API = process.env.NEXT_PUBLIC_TEAMS_API_URL || 'http://localhost:8001';

// ─────────────────────────────────────────────────────────────────────────────
//  Teams Media Server types  (mirrors Microsoft Graph API shape)
// ─────────────────────────────────────────────────────────────────────────────

export interface TeamsMeeting {
  id: string;
  subject: string;
  organizer: { displayName: string };
  startDateTime: string;
  endDateTime: string;
  createdDateTime: string;
  joinUrl: string;
  transcripts: TeamsTranscript[];
  recordings: TeamsRecording[];
  participants?: { name: string; role?: string }[];
}

export interface TeamsTranscript {
  id: string;
  meetingId: string;
  createdDateTime: string;
  transcriptContentUrl: string;
  filename: string;
  speakerCount: number;
  segmentCount: number;
  diarization: TeamsSpeakerSegment[];
}

export interface TeamsSpeakerSegment {
  speakerLabel: string;
  displayName: string;
  segments: { start: string; end: string; text: string; speaker: string }[];
}

export interface TeamsRecording {
  id: string;
  meetingId: string;
  createdDateTime: string;
  recordingContentUrl: string;
  filename: string;
  fileSizeBytes: number | null;
  duration: string | null;
}

export interface VttSegment {
  speaker: string;
  start: string;
  end: string;
  text: string;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Teams Media Server API calls
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchTeamsMeetings(): Promise<TeamsMeeting[]> {
  const res = await fetch(`${TEAMS_API}/v1.0/me/onlineMeetings`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch Teams meetings');
  const data = await res.json();
  return data.value ?? [];
}

export async function fetchTeamsMeeting(mid: string): Promise<TeamsMeeting> {
  const res = await fetch(`${TEAMS_API}/v1.0/me/onlineMeetings/${mid}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Teams meeting not found');
  return res.json();
}

export async function createTeamsMeeting(opts: {
  subject: string;
  organizer?: string;
  startDateTime?: string;
  endDateTime?: string;
}): Promise<TeamsMeeting> {
  const fd = new FormData();
  fd.append('subject', opts.subject);
  if (opts.organizer) fd.append('organizer', opts.organizer);
  if (opts.startDateTime) fd.append('start_datetime', opts.startDateTime);
  if (opts.endDateTime) fd.append('end_datetime', opts.endDateTime);
  const res = await fetch(`${TEAMS_API}/v1.0/me/onlineMeetings`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadTeamsTranscript(mid: string, file: File): Promise<TeamsTranscript> {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${TEAMS_API}/v1.0/me/onlineMeetings/${mid}/transcripts`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadTeamsRecording(mid: string, file: File): Promise<TeamsRecording> {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${TEAMS_API}/v1.0/me/onlineMeetings/${mid}/recordings`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchTeamsTranscriptList(mid: string): Promise<TeamsTranscript[]> {
  const res = await fetch(`${TEAMS_API}/v1.0/me/onlineMeetings/${mid}/transcripts`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch transcripts');
  const data = await res.json();
  return data.value ?? [];
}

export async function fetchTeamsRecordingList(mid: string): Promise<TeamsRecording[]> {
  const res = await fetch(`${TEAMS_API}/v1.0/me/onlineMeetings/${mid}/recordings`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch recordings');
  const data = await res.json();
  return data.value ?? [];
}

/** Absolute URL to stream a Teams recording (video) */
export function teamsVideoUrl(mid: string, rid: string): string {
  return `${TEAMS_API}/v1.0/me/onlineMeetings/${mid}/recordings/${rid}/content`;
}

/** Absolute URL to download a Teams VTT transcript */
export function teamsTranscriptUrl(mid: string, tid: string): string {
  return `${TEAMS_API}/v1.0/me/onlineMeetings/${mid}/transcripts/${tid}/content`;
}

/** Fetch and parse a VTT transcript into structured segments */
export async function fetchVttSegments(mid: string, tid: string): Promise<VttSegment[]> {
  const res = await fetch(teamsTranscriptUrl(mid, tid), { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch VTT transcript');
  const vtt = await res.text();
  return parseVtt(vtt);
}

/** Parse a VTT string into segments */
export function parseVtt(vtt: string): VttSegment[] {
  const segments: VttSegment[] = [];
  const blocks = vtt.split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    const tsIdx = lines.findIndex(l => l.includes('-->'));
    if (tsIdx < 0) continue;
    const parts = lines[tsIdx].split('-->');
    const start = parts[0].trim();
    const end = parts[1]?.trim().split(' ')[0] ?? '';
    const full = lines.slice(tsIdx + 1).join(' ');
    const m = full.match(/^<v ([^>]+)>(.*)/);
    const speaker = m ? m[1].trim() : 'Unknown';
    const text = (m ? m[2] : full).replace(/<[^>]+>/g, '').trim();
    if (text) segments.push({ speaker, start, end, text });
  }
  return segments;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main-API job types
// ─────────────────────────────────────────────────────────────────────────────

export interface Job {
  job_id: string;
  title: string;
  status: 'queued' | 'processing' | 'done' | 'error';
  progress: number;
  stage: string;
  meeting_id: string | null;
  participants: { name: string; role: string; department?: string; is_external?: boolean }[];
  summaries: Record<string, string>;
  transcript: string;
  scored_count: number;
  events: number;
  topics: number;
  created_at: string;
  completed_at: string | null;
  error: string | null;
  video_filename: string;
  graph_events?: any[];
  speaker_map?: Record<string, string>;
  /** Teams Media Server meeting ID — links job to /v1.0/me/onlineMeetings/{id} */
  teams_meeting_id?: string;
  /** Teams recording ID for streaming video from Teams server */
  teams_recording_id?: string;
}

export interface Participant {
  display_name: string;
  role: string;
  department: string;
  is_external: boolean;
  authority_score: number;
  weights: Record<string, number>;
  weight_source: string;
  added_at: string;
}

export interface GraphData {
  nodes: { id: string; label: string; type: string; mentions?: number; recurrence?: number; unresolved?: number; date?: string }[];
  links: { source: string; target: string; value: number }[];
}

export interface Thread {
  thread_id: string;
  label: string;
  meeting_count: number;
  first_seen: string;
  last_seen: string;
  keywords: string[];
  appearances: { date: string; meeting_title: string; topic: string; keywords: string[] }[];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main-API read-only helpers  (diarization/transcription stays in Streamlit)
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchMeetings(): Promise<Job[]> {
  const res = await fetch(`${API_BASE}/api/meetings`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch meetings');
  return res.json();
}

export async function fetchMeeting(jobId: string): Promise<Job> {
  const res = await fetch(`${API_BASE}/api/meetings/${jobId}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Meeting not found');
  return res.json();
}

export async function fetchGraph(): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/api/graph`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Graph fetch failed');
  return res.json();
}

export async function fetchThreads(threshold = 0.5): Promise<{ threads: Thread[] }> {
  const res = await fetch(`${API_BASE}/api/threads?threshold=${threshold}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Thread fetch failed');
  return res.json();
}

export async function fetchRoles(): Promise<{ participants: Participant[] }> {
  const res = await fetch(`${API_BASE}/api/roles`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Role fetch failed');
  return res.json();
}

export async function addRole(data: { name: string; role: string; department?: string; is_external?: boolean }): Promise<Participant> {
  const res = await fetch(`${API_BASE}/api/roles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteRole(name: string): Promise<void> {
  await fetch(`${API_BASE}/api/roles/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

export async function patchSpeakers(jobId: string, speakerMap: Record<string, string>): Promise<void> {
  const res = await fetch(`${API_BASE}/api/meetings/${jobId}/speakers`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speaker_map: speakerMap }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function reprocessMeeting(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/meetings/${jobId}/reprocess`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
}

export async function processTeamsMeeting(opts: {
  teams_meeting_id: string;
  transcript_id?: string;
  recording_id?: string;
}): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/meetings/process-teams`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Fetch available LM Studio models. */
export async function fetchModels(): Promise<{ id: string }[]> {
  const res = await fetch(`${API_BASE}/api/models`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export interface AgentStep {
  id?: string;
  type: 'thinking' | 'tool_call' | 'tool_result';
  content?: string;
  name?: string;
  args?: Record<string, any>;
  args_raw?: string;
  result?: string;
  error?: string;
  turn?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  steps?: AgentStep[];
  model?: string;
}

export async function chatWithMeeting(
  jobId: string,
  message: string,
  history: ChatMessage[],
  onStepsUpdate?: (steps: AgentStep[]) => void,
  model?: string,
): Promise<{ reply: string; steps?: AgentStep[]; model?: string }> {
  const llmBackend = typeof window !== 'undefined' ? localStorage.getItem('chatLlmBackend') || 'lmstudio' : 'lmstudio';
  const res = await fetch(`${API_BASE}/api/meetings/${jobId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, model, llm_backend: llmBackend }),
  });
  if (!res.ok) throw new Error(await res.text());

  // Read SSE stream
  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';
  let finalReply = '';
  let finalSteps: AgentStep[] = [];
  let finalModel = 'unknown';
  const activeToolCalls: Record<string, AgentStep> = {};

  // Small helper to yield to the React render loop
  const tick = (ms = 80) => new Promise<void>(r => setTimeout(r, ms));

  // Collect parsed events from one read() chunk, then process them
  // with deliberate async yields so React can render intermediate states
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // keep incomplete line

    // Parse all events from this chunk first
    const events: any[] = [];
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') continue;
      try {
        events.push(JSON.parse(data));
      } catch {
        // ignore partial JSON
      }
    }

    // Process events with async yields between step and reply events
    for (const event of events) {
      try {
        if (event.type === 'text') {
          // Fast-stream proxy: accumulate raw text independently if passed through
          const textContent = event.content || '';
        } else if (event.type === 'tool_call' && event.tool_calls) {
          const streamTurn = event.turn || 0;
          // Incrementally assemble the raw tool call JSON arguments
          for (const tc of event.tool_calls) {
            const idx = `${streamTurn}_${tc.index || 0}`;
            if (!activeToolCalls[idx]) {
              activeToolCalls[idx] = { id: tc.id || `tc_stream_${idx}`, type: 'tool_call', name: '', args_raw: '', turn: streamTurn };
              finalSteps.push(activeToolCalls[idx]);
            }
            if (tc.function?.name) activeToolCalls[idx].name += tc.function.name;
            if (tc.function?.arguments) activeToolCalls[idx].args_raw += (tc.function.arguments || '');
          }
          // Force layout repaint
          onStepsUpdate?.([...finalSteps]);
          await tick(20);
        } else if (event.type === 'step' && event.step) {
          if (event.step.type === 'thinking' && event.step.status === 'done') {
            const existingIdx = finalSteps.findIndex(s => s.type === 'thinking' && s.turn === event.step.turn);
            if (existingIdx >= 0) {
              finalSteps[existingIdx] = event.step;
            } else {
              finalSteps.push(event.step);
            }
          } else if (event.step.type === 'tool_call') {
            // Reconcile agent_chat's 'started' step loopback with UI state
            const existing = finalSteps.find(s => s.type === 'tool_call' && s.name === event.step.name && !s.args);
            if (existing) {
              if (event.step.args) existing.args = event.step.args;
              if (event.step.error) existing.error = event.step.error;
            } else {
              finalSteps.push(event.step);
            }
          } else {
            finalSteps.push(event.step);
          }
          onStepsUpdate?.([...finalSteps]);
          // Yield to the browser render loop so React can paint the LiveToolStep
          await tick(80);
        } else if (event.type === 'reply') {
          // CRITICAL: Wait before processing the reply so the last tool
          // animation is visible for a meaningful duration before unmounting
          await tick(400);
          finalReply = event.content || '';
          finalSteps = finalSteps.length > 0 ? finalSteps : (event.steps || []);
          finalModel = event.model || finalModel;
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Chat failed');
        }
      } catch (e: any) {
        if (e.message && !e.message.includes('JSON')) throw e;
      }
    }
  }

  return { reply: finalReply, steps: finalSteps, model: finalModel };
}

export async function chatWithGlobal(
  message: string,
  history: ChatMessage[],
  onStepsUpdate?: (steps: AgentStep[]) => void,
  model?: string,
): Promise<{ reply: string; steps?: AgentStep[]; model?: string }> {
  const llmBackend = typeof window !== 'undefined' ? localStorage.getItem('chatLlmBackend') || 'lmstudio' : 'lmstudio';
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, model, llm_backend: llmBackend }),
  });
  if (!res.ok) throw new Error(await res.text());

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';
  let finalReply = '';
  let finalSteps: AgentStep[] = [];
  let finalModel = 'unknown';
  const activeToolCalls: Record<string, AgentStep> = {};

  const tick = (ms = 80) => new Promise<void>(r => setTimeout(r, ms));

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    const events: any[] = [];
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') continue;
      try {
        events.push(JSON.parse(data));
      } catch {}
    }

    for (const event of events) {
      try {
        if (event.type === 'text') {
          // ignore stream text pieces, just relying on final reply
        } else if (event.type === 'tool_call' && event.tool_calls) {
          const streamTurn = event.turn || 0;
          for (const tc of event.tool_calls) {
            const idx = `${streamTurn}_${tc.index || 0}`;
            if (!activeToolCalls[idx]) {
              activeToolCalls[idx] = { id: tc.id || `tc_stream_${idx}`, type: 'tool_call', name: '', args_raw: '', turn: streamTurn };
              finalSteps.push(activeToolCalls[idx]);
            }
            if (tc.function?.name) activeToolCalls[idx].name += tc.function.name;
            if (tc.function?.arguments) activeToolCalls[idx].args_raw += (tc.function.arguments || '');
          }
          onStepsUpdate?.([...finalSteps]);
          await tick(20);
        } else if (event.type === 'step' && event.step) {
          if (event.step.type === 'thinking' && event.step.status === 'done') {
            const existingIdx = finalSteps.findIndex(s => s.type === 'thinking' && s.turn === event.step.turn);
            if (existingIdx >= 0) {
              finalSteps[existingIdx] = event.step;
            } else {
              finalSteps.push(event.step);
            }
          } else if (event.step.type === 'tool_call') {
            const existing = finalSteps.find(s => s.type === 'tool_call' && s.name === event.step.name && !s.args);
            if (existing) {
              if (event.step.args) existing.args = event.step.args;
              if (event.step.error) existing.error = event.step.error;
            } else {
              finalSteps.push(event.step);
            }
          } else {
            finalSteps.push(event.step);
          }
          onStepsUpdate?.([...finalSteps]);
          await tick(80);
        } else if (event.type === 'reply') {
          await tick(400);
          finalReply = event.content || '';
          finalSteps = finalSteps.length > 0 ? finalSteps : (event.steps || []);
          finalModel = event.model || finalModel;
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Chat failed');
        }
      } catch (e: any) {
        if (e.message && !e.message.includes('JSON')) throw e;
      }
    }
  }

  return { reply: finalReply, steps: finalSteps, model: finalModel };
}
