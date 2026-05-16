"use client";
import React, { useRef, useState } from 'react';
import { motion, useScroll, useTransform, useSpring, useMotionValueEvent } from 'framer-motion';
import { 
  Bot, Ticket, Share2, Sparkles, Database, 
  Eye, LayoutDashboard, Monitor, Cloud, Server, BrainCircuit, Activity,
  ListOrdered, AudioLines, Code, Hash, FileText, Mail, Lightbulb, BarChart2
} from 'lucide-react';

const layersData = [
  { 
    id: 'presentation', label: 'PRESENTATION',
    tooltip: 'User-facing dashboards and interfaces.',
    items: [ { name: 'Streamlit Admin', icon: LayoutDashboard }, { name: 'Metrics Dashboard', icon: BarChart2 } ]
  },
  { 
    id: 'orchestration', label: 'ORCHESTRATION',
    tooltip: 'Coordinates workflows and task scheduling.',
    items: [ { name: 'Job Queue', icon: ListOrdered }, { name: 'Session Manager', icon: Activity } ]
  },
  { 
    id: 'processing', label: 'PROCESSING',
    tooltip: 'Processes transcripts and extracts meaning.',
    items: [ { name: 'Contextual Fusion', icon: BrainCircuit }, { name: 'Speech Analysis', icon: AudioLines } ]
  },
  { 
    id: 'intelligence', label: 'INTELLIGENCE',
    tooltip: 'Applies reasoning and generates insights.',
    items: [ { name: 'RAG Retrieval', icon: Database }, { name: 'Decision Engine', icon: Lightbulb }, { name: 'Agent Autocoder', icon: Code } ]
  },
  { 
    id: 'integration', label: 'INTEGRATION',
    tooltip: 'Delivers outputs to external systems.',
    items: [ { name: 'Slack Connector', icon: Hash }, { name: 'Notion Sync', icon: FileText }, { name: 'Email Gateway', icon: Mail } ]
  },
];

/** Colors for each layer — gradient from cool blue (top) to warm indigo (bottom) */
const layerColors = [
  { border: '#60a5fa', glow: '#3b82f6', bg: 'from-[#0c1a35]/95 to-[#14305e]/95' },
  { border: '#818cf8', glow: '#6366f1', bg: 'from-[#0f1535]/95 to-[#1e2660]/95' },
  { border: '#a78bfa', glow: '#8b5cf6', bg: 'from-[#120f35]/95 to-[#251860]/95' },
  { border: '#c084fc', glow: '#a855f7', bg: 'from-[#18102e]/95 to-[#2d1a5e]/95' },
  { border: '#e879f9', glow: '#d946ef', bg: 'from-[#1e0f2e]/95 to-[#35185e]/95' },
];

export const AnimatedStack = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const smoothProgress = useSpring(scrollYProgress, { stiffness: 50, damping: 18 });

  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);
  const [scrollActiveLayer, setScrollActiveLayer] = useState<number>(0);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useMotionValueEvent(smoothProgress, "change", (latest) => {
    if (latest < 0.15) setScrollActiveLayer(0);
    else if (latest < 0.35) setScrollActiveLayer(1);
    else if (latest < 0.55) setScrollActiveLayer(2);
    else if (latest < 0.75) setScrollActiveLayer(3);
    else setScrollActiveLayer(4);
  });

  const activeLayer = hoveredLayer !== null ? hoveredLayer : scrollActiveLayer;

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    setMousePosition({ x, y });
  };

  // Base isometric transforms — slightly less aggressive rotation for clarity
  const baseRotateX = 55;
  const baseRotateZ = -45;
  const tiltX = hoveredLayer !== null ? mousePosition.y * -10 : 0;
  const tiltZ = hoveredLayer !== null ? mousePosition.x * 10 : 0;

  // Much wider Z spacing so layers don't overlap
  const z0 = useTransform(smoothProgress, [0, 0.8], [40,  320]);
  const z1 = useTransform(smoothProgress, [0, 0.8], [28,  240]);
  const z2 = useTransform(smoothProgress, [0, 0.8], [18,  160]);
  const z3 = useTransform(smoothProgress, [0, 0.8], [8,   80]);
  const z4 = useTransform(smoothProgress, [0, 0.8], [0,   0]);

  const zValues = [z0, z1, z2, z3, z4];

  // Opacities for the right-side text items
  const getOp = (index: number) => {
    const start = index * 0.15 + 0.1;
    const end = start + 0.15;
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useTransform(smoothProgress, [start, end], [0.2, 1]);
  };

  const textOpacities = [getOp(0), getOp(1), getOp(2), getOp(3), getOp(4)];

  // Global Y offset to keep the expanding stack centered
  const globalYShift = useTransform(smoothProgress, [0, 0.8], [0, 80]);

  const getLayerStyle = (index: number, isActive: boolean, isDimmed: boolean) => {
    const color = layerColors[index];
    return {
      borderColor: isActive ? `${color.border}` : `${color.border}44`,
      boxShadow: isActive 
        ? `0 0 40px ${color.glow}55, inset 0 0 20px rgba(255,255,255,0.04), 0 12px 48px rgba(0,0,0,0.6)` 
        : `inset 0 0 16px rgba(255,255,255,0.03), 0 8px 32px rgba(0,0,0,0.5)`,
      opacity: isDimmed ? 0.15 : 1,
    };
  };

  return (
    <div ref={containerRef} className="h-[300vh] w-full bg-[#000000] relative">
      
      {/* Sticky viewport */}
      <div className="sticky top-0 h-screen w-full flex flex-col items-center justify-start overflow-hidden pt-[20px] lg:pt-[40px]">

        {/* Title */}
        <motion.div 
          className="w-full z-50 pointer-events-none pt-[60px] lg:pt-[80px] shrink-0"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h3 className="text-4xl md:text-5xl font-mono font-bold text-center text-[#F5F5F5] drop-shadow-lg tracking-tight">How It Works</h3>
        </motion.div>

        {/* Background Grid */}
        <div className="absolute inset-0 z-0 pointer-events-none opacity-[0.08]" style={{
            backgroundImage: `linear-gradient(#222 1px, transparent 1px), linear-gradient(90deg, #222 1px, transparent 1px)`,
            backgroundSize: '80px 80px',
            backgroundPosition: 'center center'
        }} />

        {/* Content Area */}
        <div className="w-full max-w-[1200px] mx-auto flex flex-col lg:flex-row items-center justify-center flex-1 px-6 lg:px-12 relative mt-[40px] lg:mt-[60px] pb-12">
          
          {/* Left: 3D Stack Stage */}
          <div className="w-full lg:w-[50%] flex justify-center items-center h-[40vh] lg:h-full relative z-10">
            <motion.div 
              className="relative w-[260px] h-[260px] sm:w-[340px] sm:h-[340px] lg:w-[400px] lg:h-[400px]"
              style={{
                perspective: '1200px',
                transformStyle: "preserve-3d",
                y: globalYShift
              }}
              animate={{
                rotateX: baseRotateX + tiltX,
                rotateZ: baseRotateZ + tiltZ,
              }}
              transition={{ type: "spring", stiffness: 40, damping: 20 }}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => {
                setHoveredLayer(null);
                setMousePosition({ x: 0, y: 0 });
              }}
            >
              {layersData.map((_, index) => {
                const reverseIndex = 4 - index;
                const currentLayer = layersData[reverseIndex];
                const currentZ = zValues[reverseIndex];
                const isTop = reverseIndex === 0;
                const isActive = activeLayer === reverseIndex;
                const isDimmed = activeLayer !== reverseIndex;
                const color = layerColors[reverseIndex];

                return (
                  <motion.div
                    key={currentLayer.id}
                    className="absolute inset-0 cursor-pointer"
                    style={{
                      transformStyle: "preserve-3d",
                      translateZ: currentZ,
                    }}
                    onMouseEnter={() => setHoveredLayer(reverseIndex)}
                  >
                    {/* Floating animation wrapper */}
                    <motion.div
                      className="w-full h-full relative"
                      style={{ transformStyle: "preserve-3d" }}
                      animate={{ z: [0, 6, 0] }}
                      transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: reverseIndex * 0.5 }}
                    >
                      {/* Layer panel */}
                      <div
                        className={`absolute inset-0 rounded-[20px] border flex items-center justify-center overflow-hidden backdrop-blur-md transition-all duration-700 bg-gradient-to-br ${color.bg}`}
                        style={getLayerStyle(reverseIndex, isActive, isDimmed)}
                      >
                        {/* Grid scanline on top/active layer */}
                        {isTop && (
                          <>
                            <div className="absolute inset-0 pointer-events-none opacity-60" style={{
                              backgroundImage: `linear-gradient(${color.border}33 1px, transparent 1px), linear-gradient(90deg, ${color.border}33 1px, transparent 1px)`,
                              backgroundSize: '36px 36px',
                              backgroundPosition: 'center center'
                            }}>
                              <motion.div 
                                className="absolute inset-0 w-[200%] h-[200%]"
                                style={{
                                  background: `linear-gradient(135deg, transparent 30%, ${color.glow}22 50%, transparent 70%)`,
                                }}
                                animate={{ x: ['-100%', '0%'], y: ['-100%', '0%'] }}
                                transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
                              />
                            </div>
                            <motion.div 
                              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 rounded-full pointer-events-none"
                              style={{ background: color.glow, filter: 'blur(70px)', mixBlendMode: 'screen' }}
                              animate={{ opacity: [0.1, 0.3, 0.1], scale: [0.8, 1.1, 0.8] }}
                              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
                            />
                          </>
                        )}

                        {/* Active layer glow pulse */}
                        {isActive && !isTop && (
                          <motion.div 
                            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-40 h-40 rounded-full pointer-events-none"
                            style={{ background: color.glow, filter: 'blur(60px)', mixBlendMode: 'screen' }}
                            animate={{ opacity: [0.08, 0.2, 0.08] }}
                            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                          />
                        )}

                        {/* Layer label in top-left corner */}
                        <span 
                          className="absolute top-3 left-4 font-mono text-[9px] tracking-[0.3em] uppercase z-20 transition-opacity duration-500"
                          style={{ color: color.border, opacity: isActive ? 0.9 : 0.5 }}
                        >
                          {currentLayer.label}
                        </span>

                        {/* Content cards */}
                        <div className={`flex items-center justify-center gap-3 w-full h-full relative z-10 p-5 pt-8 ${currentLayer.items.length > 2 ? 'grid grid-cols-2' : ''}`}>
                          {currentLayer.items.map((item, i) => {
                            const Icon = item.icon;
                            return (
                              <div key={i} className={`flex flex-col items-center flex-1 rounded-[12px] p-3 backdrop-blur-xl transition-all duration-500 ${currentLayer.items.length === 3 && i === 2 ? 'col-span-2' : ''}`}
                                style={{
                                  background: 'rgba(0,0,0,0.55)',
                                  border: `1px solid ${isActive ? `${color.border}40` : `${color.border}18`}`,
                                  boxShadow: isActive ? `0 2px 16px ${color.glow}22` : '0 2px 12px rgba(0,0,0,0.3)',
                                }}
                              >
                                <Icon size={20} className="mb-2 transition-colors duration-300" style={{ color: isActive ? color.border : '#6b7280' }} />
                                <span className="text-[10px] font-mono font-medium text-center leading-snug transition-colors duration-300"
                                  style={{ color: isActive ? '#e2e8f0' : '#94a3b8' }}
                                >
                                  {item.name}
                                </span>
                              </div>
                            );
                          })}
                        </div>

                        {/* Edge highlight line (top) */}
                        <div 
                          className="absolute top-0 left-[10%] right-[10%] h-[1px] transition-opacity duration-500"
                          style={{ 
                            background: `linear-gradient(90deg, transparent, ${color.border}${isActive ? '66' : '22'}, transparent)`,
                            opacity: isActive ? 1 : 0.4
                          }}
                        />
                      </div>
                    </motion.div>
                  </motion.div>
                );
              })}
            </motion.div>
          </div>

          {/* Right: Text Descriptions */}
          <div className="w-full lg:w-[50%] flex flex-col justify-center h-[50vh] lg:h-full relative z-20 pl-4 lg:pl-16 mt-12 lg:mt-0">
            <div className="flex flex-col justify-center gap-6 lg:gap-8 w-full max-w-[480px]">
              {layersData.map((layer, index) => {
                const currentOp = textOpacities[index];
                const isActive = activeLayer === index;
                const color = layerColors[index];
                
                // eslint-disable-next-line react-hooks/rules-of-hooks
                const scrollOpacity = useTransform(currentOp, (op) => {
                  if (isActive) return 1;
                  if (index < activeLayer) return 0.25;
                  return op;
                });

                return (
                  <motion.div 
                    key={layer.id}
                    style={{ opacity: hoveredLayer !== null ? (hoveredLayer === index ? 1 : 0.15) : scrollOpacity }}
                    className="flex flex-col gap-3 relative transition-all duration-500 cursor-pointer group"
                    onMouseEnter={() => setHoveredLayer(index)}
                    onMouseLeave={() => setHoveredLayer(null)}
                  >
                    {/* Connector line */}
                    <div 
                      className={`absolute top-[10px] right-[100%] mr-6 h-[1px] transition-all duration-500 hidden xl:block ${isActive ? 'w-[120px] opacity-100' : 'w-[40px] opacity-20'}`}
                      style={{ background: `linear-gradient(to right, transparent, ${color.border})` }}
                    />

                    <div className="flex items-center gap-4">
                      <div 
                        className="w-2.5 h-2.5 rounded-full transition-all duration-500"
                        style={{
                          backgroundColor: isActive ? color.border : '#334155',
                          boxShadow: isActive ? `0 0 15px ${color.glow}` : 'none',
                          transform: isActive ? 'scale(1.25)' : 'scale(1)',
                        }}
                      />
                      <span 
                        className={`font-mono font-bold tracking-[0.25em] text-[13px] uppercase text-left transition-colors duration-500`}
                        style={{ color: isActive ? '#f5f5f5' : '#64748b' }}
                      >
                        {layer.label}
                      </span>
                    </div>
                    
                    <div 
                      className="pl-6 border-l border-transparent transition-colors duration-500 group-hover:border-[#334155] py-1 ml-[5px]"
                    >
                      <p 
                        className="font-sans text-[15px] leading-relaxed transition-colors duration-500"
                        style={{ color: isActive ? '#e2e8f0' : '#94a3b8' }}
                      >
                        {layer.tooltip}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};
