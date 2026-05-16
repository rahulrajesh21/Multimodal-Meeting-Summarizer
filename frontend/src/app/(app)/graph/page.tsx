'use client';
import { useEffect, useRef, useState, useMemo } from 'react';
import { fetchGraph, fetchThreads, GraphData, Thread } from '@/lib/api';
import * as d3 from 'd3';
import { Maximize2, Plus, Minus, RotateCcw, Filter, ChevronDown, Search, ArrowUpDown } from 'lucide-react';

const COLORS: Record<string, string> = {
    meeting: '#534AB7',
    topic: '#AC88E8',
    decision: '#1D9E75',
    problem: '#D85A30',
    default: '#888888'
};

const BG_COLOR = '#F8F8F6';
const BORDER_COLOR = '#E8E6E1';
const TEXT_DARK = '#1A1A18';
const TEXT_MUTED = '#6B6A66';

function trimLabel(label: string) {
    if (!label) return '';
    if (label.includes('[Archived')) return '[Archived cluster]';
    const words = label.split(' ');
    // max 4 words as requested
    if (words.length <= 4) return label;
    return words.slice(0, 4).join(' ') + '...';
}

function GraphCanvas({
    nodes, links, degrees, onSetControls
}: {
    nodes: any[], links: any[], degrees: Record<string, number>, onSetControls: (fns: any) => void
}) {
    const containerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!svgRef.current || !containerRef.current) return;
        const el = svgRef.current;
        const tooltip = tooltipRef.current!;
        const W = containerRef.current.clientWidth || 800;
        const H = containerRef.current.clientHeight || 600;

        d3.select(el).selectAll('*').remove();

        const svg = d3.select(el)
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${W} ${H}`);

        const g = svg.append('g');

        const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.05, 10])
            .on('zoom', (e) => {
                g.attr('transform', e.transform);
                // Reveal labels if zoomed in sufficiently
                g.selectAll('.node-label').style('opacity', e.transform.k > 0.7 ? 1 : 0);
            });

        svg.call(zoomBehavior as any);

        onSetControls({
            zoomIn: () => svg.transition().duration(300).call(zoomBehavior.scaleBy as any, 1.3),
            zoomOut: () => svg.transition().duration(300).call(zoomBehavior.scaleBy as any, 0.7),
            zoomReset: () => svg.transition().duration(400).call(zoomBehavior.transform as any, d3.zoomIdentity)
        });

        const nodeRadius = (d: any) => {
            const deg = degrees[d.id] || 0;
            return Math.max(10, Math.min(36, 10 + deg * 1.5));
        };

        const sim = d3.forceSimulation(nodes as any)
            .force('link', d3.forceLink(links).id((d: any) => d.id).distance(120).strength(0.15))
            .force('charge', d3.forceManyBody().strength(-150))
            .force('center', d3.forceCenter(W / 2, H / 2))
            .force('x', d3.forceX(W / 2).strength(0.06))
            .force('y', d3.forceY(H / 2).strength(0.06))
            .force('collision', d3.forceCollide().radius((d: any) => nodeRadius(d) + 8));

        // Links
        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', '#1A1A18')
            .attr('stroke-width', 0.8)
            .attr('stroke-opacity', 0.08); // 8% opacity as required

        // Nodes
        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .join('g')
            .attr('cursor', 'pointer')
            .call((d3.drag<SVGGElement, any>()
                .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
                .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
                .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
            ) as any);

        node.append('circle')
            .attr('r', d => nodeRadius(d))
            .attr('fill', (d: any) => COLORS[d.type] || COLORS.default)
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 0.5);

        node.append('text')
            .attr('class', 'node-label')
            .text((d: any) => trimLabel(d.label))
            .attr('dy', (d: any) => nodeRadius(d) + 16)
            .attr('text-anchor', 'middle')
            .attr('font-size', '11px')
            .attr('font-family', '"DM Sans", system-ui, sans-serif')
            .attr('font-weight', '500')
            .attr('fill', TEXT_DARK)
            .style('opacity', 0) // default invisible until zoomed
            .style('pointer-events', 'none')
            .style('text-shadow', '1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff');

        // Interactions
        node.on('mouseover', (e, d: any) => {
            const deg = degrees[d.id] || 0;
            tooltipRef.current!.style.opacity = '1';
            tooltipRef.current!.innerHTML = `
                <div style="font-weight:600;font-size:13px;color:${TEXT_DARK};margin-bottom:4px;">${d.label}</div>
                <div style="color:${COLORS[d.type]};text-transform:capitalize;">${d.type} Node</div>
                <div style="color:${TEXT_MUTED};margin-top:2px;">Connections: ${deg}</div>
                ${d.date ? `<div style="color:${TEXT_MUTED};">Date: ${d.date.slice(0, 10)}</div>` : ''}
            `;
            // Temporary label pop
            d3.select(e.currentTarget).select('.node-label').style('opacity', 1);
        }).on('mousemove', (e) => {
            tooltipRef.current!.style.left = (e.clientX + 14) + 'px';
            tooltipRef.current!.style.top = (e.clientY + 14) + 'px';
        }).on('mouseout', (e, d) => {
            tooltipRef.current!.style.opacity = '0';
            // Return label to default state based on zoom
            const currentZoom = d3.zoomTransform(svg.node()!).k;
            d3.select(e.currentTarget).select('.node-label').style('opacity', currentZoom > 0.7 ? 1 : 0);
        }).on('click', (e, d: any) => {
            console.log(`Tell me about node "${d.label}" in the knowledge graph`);
        });

        sim.on('tick', () => {
            link
                .attr('x1', (d: any) => d.source.x)
                .attr('y1', (d: any) => d.source.y)
                .attr('x2', (d: any) => d.target.x)
                .attr('y2', (d: any) => d.target.y);
            node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
        });

        return () => { sim.stop(); };
    }, [nodes, links, degrees, onSetControls]);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
            <svg ref={svgRef} style={{ outline: 'none', cursor: 'grab' }} />
            <div ref={tooltipRef} style={{
                position: 'fixed',
                background: '#FFFFFF',
                border: `0.5px solid ${BORDER_COLOR}`,
                borderRadius: '8px',
                padding: '10px 14px',
                fontSize: '12px',
                fontFamily: '"DM Sans", system-ui, sans-serif',
                pointerEvents: 'none',
                opacity: 0,
                zIndex: 9999,
                maxWidth: '240px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                transition: 'opacity 0.1s ease',
            }} />
        </div>
    );
}

export default function GraphPage() {
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
    const [threads, setThreads] = useState<Thread[]>([]);
    const [tab, setTab] = useState<'graph' | 'threads' | 'table'>('graph');
    const [loading, setLoading] = useState(true);

    const [filters, setFilters] = useState<Record<string, boolean>>({
        meeting: true, topic: true, decision: true, problem: true
    });
    const [showFilters, setShowFilters] = useState(false);

    // Zoom controls state
    const [zoomFns, setZoomFns] = useState<{ zoomIn?: () => void, zoomOut?: () => void, zoomReset?: () => void }>({});

    // Table view state
    const [searchQuery, setSearchQuery] = useState('');
    const [sortConfig, setSortConfig] = useState<{ key: string, direction: 'asc' | 'desc' }>({ key: 'connections', direction: 'desc' });

    useEffect(() => {
        const load = async () => {
            try {
                const [g, t] = await Promise.all([fetchGraph(), fetchThreads(0.5)]);
                setGraphData(g);
                setThreads(t.threads);
            } catch { } finally { setLoading(false); }
        };
        load();
    }, []);

    // Filtered data & Precomputed degrees
    const activeNodes = useMemo(() => graphData.nodes.filter(n => filters[n.type] !== false), [graphData.nodes, filters]);

    const activeLinks = useMemo(() => {
        const validNodeIds = new Set(activeNodes.map(n => n.id));
        return graphData.links.filter(l => {
            const source = (l.source as any).id || l.source;
            const target = (l.target as any).id || l.target;
            return validNodeIds.has(source) && validNodeIds.has(target);
        });
    }, [graphData.links, activeNodes]);

    const degrees = useMemo(() => {
        const degs: Record<string, number> = {};
        activeLinks.forEach(l => {
            const src = (l.source as any).id || l.source;
            const tgt = (l.target as any).id || l.target;
            degs[src] = (degs[src] || 0) + 1;
            degs[tgt] = (degs[tgt] || 0) + 1;
        });
        return degs;
    }, [activeLinks]);

    // Table mapping
    const tableData = useMemo(() => {
        let data = activeNodes.map(n => ({
            id: n.id,
            label: n.label,
            type: n.type,
            connections: degrees[n.id] || 0,
            date: n.date || '',
            mentions: n.mentions || 0,
        }));

        if (searchQuery) {
            const sq = searchQuery.toLowerCase();
            data = data.filter(d => d.label.toLowerCase().includes(sq) || d.type.toLowerCase().includes(sq));
        }

        data.sort((a, b) => {
            const mod = sortConfig.direction === 'asc' ? 1 : -1;
            const key = sortConfig.key as keyof typeof a;
            if (a[key] < b[key]) return -1 * mod;
            if (a[key] > b[key]) return 1 * mod;
            return 0;
        });

        return data;
    }, [activeNodes, degrees, searchQuery, sortConfig]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: BG_COLOR, overflow: 'hidden', fontFamily: '"DM Sans", system-ui, sans-serif' }}>

            {/* TOP BAR */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '20px 28px', background: '#FFFFFF', borderBottom: `0.5px solid ${BORDER_COLOR}`, flexShrink: 0 }}>
                <div>
                    <h1 style={{ fontSize: '20px', fontWeight: 600, color: TEXT_DARK, margin: 0 }}>Knowledge graph</h1>
                    <div style={{ fontSize: '13px', color: TEXT_MUTED, marginTop: '2px' }}>Cross-meeting entity relationships</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ display: 'flex', background: BG_COLOR, borderRadius: '8px', padding: '4px', border: `0.5px solid ${BORDER_COLOR}` }}>
                        {(['graph', 'threads', 'table'] as const).map(t => (
                            <button key={t} onClick={() => setTab(t)} style={{
                                padding: '6px 14px', borderRadius: '6px', fontSize: '13px', fontWeight: 500, textTransform: 'capitalize',
                                border: 'none', cursor: 'pointer', background: tab === t ? '#FFFFFF' : 'transparent',
                                color: tab === t ? TEXT_DARK : TEXT_MUTED,
                                boxShadow: tab === t ? '0 1px 3px rgba(0,0,0,0.05)' : 'none',
                                transition: 'all 0.15s ease'
                            }}>{t}</button>
                        ))}
                    </div>
                    <button style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: TEXT_MUTED }} title="Fullscreen">
                        <Maximize2 style={{ width: 18, height: 18 }} />
                    </button>
                </div>
            </div>

            {/* STATS BAR */}
            <div style={{ display: 'flex', alignItems: 'center', padding: '12px 28px', background: '#FFFFFF', borderBottom: `0.5px solid ${BORDER_COLOR}`, flexShrink: 0, gap: '24px' }}>
                <div style={{ fontSize: '13px', fontWeight: 500, color: TEXT_DARK }}>
                    {activeNodes.length} nodes <span style={{ color: BORDER_COLOR, margin: '0 6px' }}>|</span> {activeLinks.length} edges
                </div>
                <div style={{ display: 'flex', gap: '16px', fontSize: '13px', fontWeight: 500 }}>
                    {Object.entries(COLORS).filter(([k]) => k !== 'default').map(([key, col]) => (
                        <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '6px', color: TEXT_MUTED, textTransform: 'capitalize' }}>
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: col }} />
                            {key}
                        </div>
                    ))}
                </div>
            </div>

            {/* GRAPH CANVAS & OVERLAYS */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>

                {tab === 'graph' && (
                    <>
                        <GraphCanvas nodes={activeNodes} links={activeLinks} degrees={degrees} onSetControls={setZoomFns} />

                        {/* FILTER PANEL (Top Right) */}
                        <div style={{ position: 'absolute', top: 20, right: 20 }}>
                            <button onClick={() => setShowFilters(!showFilters)} style={{
                                display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', background: '#FFFFFF',
                                border: `0.5px solid ${BORDER_COLOR}`, borderRadius: '8px', fontSize: '13px', fontWeight: 500,
                                cursor: 'pointer', color: TEXT_DARK, boxShadow: '0 4px 12px rgba(0,0,0,0.04)'
                            }}>
                                <Filter style={{ width: 14, height: 14 }} /> Type Filters <ChevronDown style={{ width: 14, height: 14 }} />
                            </button>

                            {showFilters && (
                                <div style={{
                                    position: 'absolute', top: '100%', right: 0, marginTop: '8px', background: '#FFFFFF',
                                    border: `0.5px solid ${BORDER_COLOR}`, borderRadius: '12px', padding: '16px',
                                    minWidth: '200px', boxShadow: '0 8px 24px rgba(0,0,0,0.08)'
                                }}>
                                    <div style={{ fontSize: '12px', fontWeight: 600, color: TEXT_MUTED, marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Node Types</div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                        {Object.keys(COLORS).filter(k => k !== 'default').map(type => (
                                            <label key={type} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '13px', fontWeight: 500, color: TEXT_DARK, cursor: 'pointer' }}>
                                                <input
                                                    type="checkbox"
                                                    checked={filters[type]}
                                                    onChange={e => setFilters(f => ({ ...f, [type]: e.target.checked }))}
                                                    style={{ width: 16, height: 16, accentColor: COLORS[type] }}
                                                />
                                                <span style={{ textTransform: 'capitalize' }}>{type}</span>
                                            </label>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* ZOOM CONTROLS (Bottom Left) */}
                        <div style={{ position: 'absolute', bottom: 20, left: 20, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            <button onClick={zoomFns.zoomIn} style={zoomBtnStyle}><Plus style={{ width: 16, height: 16 }} /></button>
                            <button onClick={zoomFns.zoomOut} style={zoomBtnStyle}><Minus style={{ width: 16, height: 16 }} /></button>
                            <button onClick={zoomFns.zoomReset} style={{ ...zoomBtnStyle, marginTop: '4px' }} title="Reset View"><RotateCcw style={{ width: 16, height: 16 }} /></button>
                        </div>

                        {/* HINT TEXT (Bottom Right) */}
                        <div style={{ position: 'absolute', bottom: 24, right: 24, fontSize: '12px', color: TEXT_MUTED, fontWeight: 500, opacity: 0.7 }}>
                            Drag to pan · Scroll to zoom · Click node to explore
                        </div>
                    </>
                )}

                {/* THREADS TAB PLACEHOLDER */}
                {tab === 'threads' && (
                    <div style={{ padding: '32px', maxWidth: '800px', margin: '0 auto', overflowY: 'auto', height: '100%' }}>
                        <h2 style={{ fontSize: '18px', fontWeight: 600, color: TEXT_DARK, marginBottom: '20px' }}>Recurring Threads</h2>
                        {threads.map(t => (
                            <div key={t.thread_id} style={{ background: '#FFFFFF', padding: '20px', borderRadius: '12px', border: `0.5px solid ${BORDER_COLOR}`, marginBottom: '16px' }}>
                                <div style={{ fontSize: '15px', fontWeight: 600, color: TEXT_DARK }}>{t.label}</div>
                                <div style={{ fontSize: '13px', color: TEXT_MUTED, marginTop: '4px' }}>Across {t.meeting_count} meetings ({t.first_seen.slice(0, 10)} → {t.last_seen.slice(0, 10)})</div>
                            </div>
                        ))}
                    </div>
                )}

                {/* DATA TABLE VIEW */}
                {tab === 'table' && (
                    <div style={{ padding: '32px', maxWidth: '1000px', margin: '0 auto', overflowY: 'auto', height: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                            <h2 style={{ fontSize: '18px', fontWeight: 600, color: TEXT_DARK, margin: 0 }}>Data Explorer</h2>
                            <div style={{ display: 'flex', alignItems: 'center', background: '#FFFFFF', border: `0.5px solid ${BORDER_COLOR}`, borderRadius: '8px', padding: '6px 12px' }}>
                                <Search style={{ width: 14, height: 14, color: TEXT_MUTED, marginRight: '8px' }} />
                                <input
                                    type="text"
                                    placeholder="Search nodes..."
                                    value={searchQuery}
                                    onChange={e => setSearchQuery(e.target.value)}
                                    style={{ border: 'none', outline: 'none', fontSize: '13px', fontFamily: 'inherit', width: '200px' }}
                                />
                            </div>
                        </div>

                        <div style={{ background: '#FFFFFF', borderRadius: '12px', border: `0.5px solid ${BORDER_COLOR}`, overflow: 'hidden', boxShadow: '0 4px 12px rgba(0,0,0,0.02)' }}>
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                                    <thead style={{ background: BG_COLOR, borderBottom: `0.5px solid ${BORDER_COLOR}` }}>
                                        <tr>
                                            {['Label', 'Type', 'Connections', 'Date'].map(col => {
                                                const key = col.toLowerCase();
                                                const isSorted = sortConfig.key === key;
                                                return (
                                                    <th key={col}
                                                        onClick={() => setSortConfig(s => ({ key, direction: s.key === key && s.direction === 'desc' ? 'asc' : 'desc' }))}
                                                        style={{ padding: '12px 16px', fontSize: '12px', fontWeight: 600, color: TEXT_MUTED, cursor: 'pointer', userSelect: 'none' }}>
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                            {col}
                                                            <ArrowUpDown style={{ width: 12, height: 12, opacity: isSorted ? 1 : 0.3 }} />
                                                        </div>
                                                    </th>
                                                );
                                            })}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {tableData.slice(0, 100).map((row, i) => (
                                            <tr key={row.id} style={{ borderBottom: i < tableData.length - 1 ? `0.5px solid ${BORDER_COLOR}` : 'none' }}>
                                                <td style={{ padding: '12px 16px', fontSize: '13px', fontWeight: 500, color: TEXT_DARK, maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                                    {row.label}
                                                </td>
                                                <td style={{ padding: '12px 16px', fontSize: '13px' }}>
                                                    <span style={{
                                                        padding: '4px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600, textTransform: 'capitalize',
                                                        background: `${COLORS[row.type] || COLORS.default}20`,
                                                        color: COLORS[row.type] || COLORS.default
                                                    }}>
                                                        {row.type}
                                                    </span>
                                                </td>
                                                <td style={{ padding: '12px 16px', fontSize: '13px', color: TEXT_MUTED }}>{row.connections}</td>
                                                <td style={{ padding: '12px 16px', fontSize: '13px', color: TEXT_MUTED }}>{row.date ? row.date.slice(0, 10) : '—'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            {tableData.length > 100 && (
                                <div style={{ padding: '12px 16px', background: BG_COLOR, borderTop: `0.5px solid ${BORDER_COLOR}`, fontSize: '12px', color: TEXT_MUTED, textAlign: 'center' }}>
                                    Showing top 100 results of {tableData.length}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

const zoomBtnStyle = {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: '32px', height: '32px', background: '#FFFFFF', border: `0.5px solid ${BORDER_COLOR}`,
    borderRadius: '8px', cursor: 'pointer', color: TEXT_DARK, boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
};
