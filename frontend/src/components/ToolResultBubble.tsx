'use client';
import { AgentStep } from '@/lib/api';

/**
 * Convert a list of AgentSteps from a completed turn into a deduplicated
 * list of { name, argsRaw } tags suitable for rendering settled LiveToolSteps.
 */
export function stepsToTags(steps: AgentStep[]): { name: string; argsRaw?: string }[] {
    const seen = new Set<string>();
    const tags: { name: string; argsRaw?: string }[] = [];
    for (const s of steps) {
        if (s.type === 'tool_call' && s.name) {
            const key = s.id || s.name;
            if (!seen.has(key)) {
                seen.add(key);
                // args_raw is populated during streaming; args (dict) is what the backend
                // sends in the final reply event — fall back to pretty-printing the dict.
                const argsRaw = s.args_raw
                    || (s.args ? JSON.stringify(s.args, null, 2) : undefined);
                tags.push({ name: s.name, argsRaw });
            }
        }
    }
    return tags;
}

/**
 * ToolResultBubble — renders a single settled tool call result.
 * Currently a thin wrapper; the primary UI is handled by LiveToolStep.
 */
export default function ToolResultBubble({ name, argsRaw }: { name: string; argsRaw?: string }) {
    return null; // rendering delegated to LiveToolStep
}
