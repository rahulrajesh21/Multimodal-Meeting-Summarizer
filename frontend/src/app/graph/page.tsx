'use client';
import { useEffect, useRef, useState } from 'react';
import { fetchGraph, fetchThreads, GraphData, Thread } from '@/lib/api';
import * as d3 from 'd3';

function GraphCanvas({ data }: { data: GraphData }) {
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        if (!svgRef.current || !data.nodes.length) return;
        const el = svgRef.current;
        const W = el.clientWidth || 900;
        const H = el.clientHeight || 580;

        d3.select(el).selectAll('*').remove();

        const svg = d3.select(el)
            .attr('width', W)
            .attr('height', H);

        // Zoom
        const g = svg.append('g');
        svg.call(d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.3, 3]).on('zoom', e => g.attr('transform', e.transform)) as any);

        const nodeColor = (type: string) => {
            const map: Record<string, string> = {
                meeting: '#60a5fa', topic: '#7c6ff7', decision: '#22c55e',
                problem: '#ef4444', discussion: '#06b6d4',
            };
            return map[type] || '#888';
        };

        const nodeRadius = (d: any) => d.type === 'meeting' ? 18 : Math.min(14, 6 + (d.mentions || 1) * 1.2);

        const links = data.links.map(l => ({ ...l }));
        const nodes = data.nodes.map(n => ({ ...n }));

        const sim = d3.forceSimulation(nodes as any)
            .force('link', d3.forceLink(links).id((d: any) => d.id).distance(120).strength(0.5))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(W / 2, H / 2))
            .force('collision', d3.forceCollide().radius(28));

        // Links
        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', '#2a2a3d')
            .attr('stroke-width', d => Math.min(3, 1 + (d as any).value * 0.3))
            .attr('stroke-opacity', 0.6);

        // Node groups
        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .join('g')
            .attr('cursor', 'pointer')
            .call(d3.drag<SVGGElement, any>()
                .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
                .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
                .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
            );

        node.append('circle')
            .attr('r', d => nodeRadius(d))
            .attr('fill', d => nodeColor(d.type))
            .attr('fill-opacity', 0.9)
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5)
            .attr('stroke-opacity', 0.1);

        // Glow for meetings
        node.filter(d => d.type === 'meeting')
            .append('circle')
            .attr('r', d => nodeRadius(d) + 8)
            .attr('fill', 'none')
            .attr('stroke', '#60a5fa')
            .attr('stroke-opacity', 0.15)
            .attr('stroke-width', 6);

        node.append('text')
            .text(d => d.label.slice(0, 18))
            .attr('dy', d => nodeRadius(d) + 14)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#9090b0')
            .style('pointer-events', 'none');

        // Tooltip
        const tooltip = d3.select('body').append('div')
            .style('position', 'fixed')
            .style('background', '#1a1a25')
            .style('border', '1px solid #3d3d5c')
            .style('border-radius', '8px')
            .style('padding', '10px 14px')
            .style('font-size', '12px')
            .style('color', '#f0f0fa')
            .style('pointer-events', 'none')
            .style('opacity', '0')
            .style('z-index', '9999')
            .style('max-width', '220px')
            .style('box-shadow', '0 4px 20px rgba(0,0,0,0.5)');

        node
            .on('mouseover', (e, d: any) => {
                tooltip.style('opacity', '1')
                    .html(`<strong>${d.label}</strong><br/><span style="color:#7c6ff7">${d.type}</span>` +
                        (d.mentions ? `<br/>Mentions: ${d.mentions}` : '') +
                        (d.date ? `<br/>Date: ${d.date.slice(0, 10)}` : '') +
                        (d.recurrence != null ? `<br/>Recurrence: ${(d.recurrence * 100).toFixed(0)}%` : ''));
            })
            .on('mousemove', e => {
                tooltip.style('left', (e.clientX + 12) + 'px').style('top', (e.clientY - 10) + 'px');
            })
            .on('mouseout', () => tooltip.style('opacity', '0'));

        sim.on('tick', () => {
            link
                .attr('x1', (d: any) => d.source.x)
                .attr('y1', (d: any) => d.source.y)
                .attr('x2', (d: any) => d.target.x)
                .attr('y2', (d: any) => d.target.y);
            node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
        });

        return () => { sim.stop(); tooltip.remove(); };
    }, [data]);

    return <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />;
}

export default function GraphPage() {
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
    const [threads, setThreads] = useState<Thread[]>([]);
    const [threshold, setThreshold] = useState(0.5);
    const [tab, setTab] = useState<'graph' | 'threads'>('graph');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const load = async () => {
            try {
                const [g, t] = await Promise.all([fetchGraph(), fetchThreads(threshold)]);
                setGraphData(g);
                setThreads(t.threads);
            } catch { } finally { setLoading(false); }
        };
        load();
    }, [threshold]);

    return (
        <>
            <div className="topbar">
                <span className="topbar-title">🕸️ Cross-Meeting Graph</span>
                <div className="topbar-right">
                    <span style={{ fontSize: '13px', color: 'var(--text-secondary)', marginRight: '8px' }}>
                        {graphData.nodes.length} nodes · {graphData.links.length} edges
                    </span>
                </div>
            </div>
            <div className="page">
                <div className="tabs">
                    <div className={`tab${tab === 'graph' ? ' active' : ''}`} onClick={() => setTab('graph')}>🕸️ Entity Graph</div>
                    <div className={`tab${tab === 'threads' ? ' active' : ''}`} onClick={() => setTab('threads')}>🧵 Recurring Threads</div>
                </div>

                {tab === 'graph' && (
                    <>
                        <div style={{ display: 'flex', gap: '16px', marginBottom: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
                            <div style={{ display: 'flex', gap: '12px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                                <span>🔵 Meeting</span>
                                <span style={{ color: '#7c6ff7' }}>🟣 Topic</span>
                                <span style={{ color: '#22c55e' }}>🟢 Decision</span>
                                <span style={{ color: '#ef4444' }}>🔴 Problem</span>
                            </div>
                            <div style={{ marginLeft: 'auto', fontSize: '12px', color: 'var(--text-muted)' }}>Drag to move · Scroll to zoom</div>
                        </div>
                        <div className="graph-canvas">
                            {loading
                                ? <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                                    <span className="spinner" />
                                </div>
                                : graphData.nodes.length === 0
                                    ? <div className="empty-state">
                                        <div className="empty-icon">🕸️</div>
                                        <div className="empty-title">No graph data yet</div>
                                        <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Process at least one meeting to see the entity graph</div>
                                    </div>
                                    : <GraphCanvas data={graphData} />
                            }
                        </div>
                    </>
                )}

                {tab === 'threads' && (
                    <>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '20px' }}>
                            <div style={{ flex: 1 }}>
                                <label className="form-label">Similarity Threshold: {threshold.toFixed(2)}</label>
                                <input
                                    type="range" min="0.2" max="0.9" step="0.05"
                                    value={threshold}
                                    onChange={e => setThreshold(+e.target.value)}
                                    style={{ width: '100%', marginTop: '6px', accentColor: 'var(--accent)' }}
                                />
                            </div>
                            <div>
                                <span className="badge badge-purple">{threads.length} threads</span>
                            </div>
                        </div>

                        {threads.length === 0
                            ? <div className="empty-state">
                                <div className="empty-icon">🧵</div>
                                <div className="empty-title">No recurring threads found</div>
                                <div style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '4px' }}>
                                    Try lowering the threshold, or process more meetings
                                </div>
                            </div>
                            : <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                {threads.map((t, i) => (
                                    <div key={t.thread_id} className="card card-sm">
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px' }}>
                                            <div>
                                                <div style={{ fontWeight: 600, fontSize: '15px' }}>{t.label}</div>
                                                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                                                    {t.first_seen.slice(0, 10)} → {t.last_seen.slice(0, 10)}
                                                </div>
                                            </div>
                                            <span className="badge badge-purple">{t.meeting_count} meetings</span>
                                        </div>
                                        {t.keywords.length > 0 && (
                                            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                                                {t.keywords.map(k => <span key={k} className="badge badge-muted">{k}</span>)}
                                            </div>
                                        )}
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                            {t.appearances.map((a, j) => {
                                                const icons = ['🟡', '🔵', '🔵', '🔵', '🔴'];
                                                return (
                                                    <div key={j} style={{ display: 'flex', gap: '10px', fontSize: '13px' }}>
                                                        <span style={{ flexShrink: 0 }}>{icons[Math.min(j, icons.length - 1)]}</span>
                                                        <div>
                                                            <span style={{ fontWeight: 500 }}>{a.meeting_title}</span>
                                                            <span style={{ color: 'var(--text-muted)', marginLeft: '8px' }}>
                                                                {a.date.slice(0, 10)}
                                                            </span>
                                                            <div style={{ color: 'var(--text-secondary)', fontSize: '12px', marginTop: '2px' }}>
                                                                {a.topic}
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        }
                    </>
                )}
            </div>
        </>
    );
}
