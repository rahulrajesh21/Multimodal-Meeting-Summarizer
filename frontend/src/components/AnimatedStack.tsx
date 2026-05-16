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

export const AnimatedStack = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const smoothProgress = useSpring(scrollYProgress, { stiffness: 40, damping: 15 });

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

  // Base isometric transforms
  const baseRotateX = 60;
  const baseRotateZ = -45;
  const tiltX = hoveredLayer !== null ? mousePosition.y * -15 : 0;
  const tiltZ = hoveredLayer !== null ? mousePosition.x * 15 : 0;

  // Tighter Z translations for an elegant, constrained expansion
  const z0 = useTransform(smoothProgress, [0, 0.8], [30, 240]);
  const z1 = useTransform(smoothProgress, [0, 0.8], [22, 180]);
  const z2 = useTransform(smoothProgress, [0, 0.8], [15, 120]);
  const z3 = useTransform(smoothProgress, [0, 0.8], [7, 60]);
  const z4 = useTransform(smoothProgress, [0, 0.8], [0, 0]);

  const zValues = [z0, z1, z2, z3, z4];

  // Opacities for the right-side text items
  const getOp = (index: number) => {
    const start = index * 0.15 + 0.1;
    const end = start + 0.15;
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useTransform(smoothProgress, [start, end], [0.2, 1]);
  };

  const textOpacities = [getOp(0), getOp(1), getOp(2), getOp(3), getOp(4)];

  // Global Y offset to keep the expanding stack perfectly centered in its space
  const globalYShift = useTransform(smoothProgress, [0, 0.8], [0, 60]);

  const getLayerClass = (index: number) => {
    const isActive = activeLayer === index;
    const isDimmed = activeLayer !== index;
    return `absolute inset-0 rounded-[28px] border border-[#3b82f6]/30 flex items-center justify-center overflow-hidden backdrop-blur-md transition-all duration-500 shadow-[inset_0_0_20px_rgba(255,255,255,0.05),0_10px_40px_rgba(0,0,0,0.5)] bg-gradient-to-br from-[#0A1428]/90 to-[#112A5A]/90 ${isActive ? 'border-[#4F84FF]/70 shadow-[0_0_40px_rgba(79,132,255,0.4)]' : ''} ${isDimmed ? 'opacity-30' : 'opacity-100'}`;
  };

  return (
    <div ref={containerRef} className="h-[300vh] w-full bg-[#000000] relative">
      
      {/* Sticky viewport container with massive top padding for safety */}
      <div className="sticky top-0 h-screen w-full flex flex-col items-center justify-start overflow-hidden pt-[20px] lg:pt-[40px]">

        {/* Title Area - Reserved space */}
        <motion.div 
          className="w-full z-50 pointer-events-none pt-[60px] lg:pt-[80px] shrink-0"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h3 className="text-4xl md:text-5xl font-mono font-bold text-center text-[#F5F5F5] drop-shadow-lg tracking-tight">How It Works</h3>
        </motion.div>

        {/* Background Grid - Premium subtle touch */}
        <div className="absolute inset-0 z-0 pointer-events-none opacity-20" style={{
            backgroundImage: `linear-gradient(#151515 1px, transparent 1px), linear-gradient(90deg, #151515 1px, transparent 1px)`,
            backgroundSize: '80px 80px',
            backgroundPosition: 'center center'
        }} />

        {/* Main Content Area - Forced down by heavy margin-top */}
        <div className="w-full max-w-[1200px] mx-auto flex flex-col lg:flex-row items-center justify-center flex-1 px-6 lg:px-12 relative mt-[40px] lg:mt-[60px] pb-12">
          
          {/* Left: 3D Stage */}
          <div className="w-full lg:w-[50%] flex justify-center items-center h-[40vh] lg:h-full relative z-10">
            <motion.div 
              className="relative w-[300px] h-[300px] sm:w-[400px] sm:h-[400px] lg:w-[460px] lg:h-[460px]"
              style={{
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
                    {/* Continuous Float Wrapper */}
                    <motion.div
                      className="w-full h-full relative"
                      style={{ transformStyle: "preserve-3d" }}
                      animate={{ z: [0, 8, 0] }}
                      transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: reverseIndex * 0.4 }}
                    >
                      <div className={getLayerClass(reverseIndex)}>
                        {isTop && (
                          <>
                            <div className="absolute inset-0 pointer-events-none opacity-90" style={{
                              backgroundImage: `linear-gradient(rgba(79,132,255,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(79,132,255,0.4) 1px, transparent 1px)`,
                              backgroundSize: '40px 40px',
                              backgroundPosition: 'center center'
                            }}>
                              <motion.div 
                                className="absolute inset-0 bg-gradient-to-tr from-transparent via-[#4F84FF]/30 to-transparent w-[200%] h-[200%]"
                                animate={{ x: ['-100%', '0%'], y: ['-100%', '0%'] }}
                                transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                              />
                            </div>
                            <motion.div 
                              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-56 h-56 bg-[#4F84FF] blur-[80px] rounded-full pointer-events-none"
                              animate={{ opacity: [0.15, 0.4, 0.15], scale: [0.8, 1.1, 0.8] }}
                              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
                              style={{ mixBlendMode: 'screen' }}
                            />
                          </>
                        )}

                        <div className={`flex items-center justify-center gap-4 w-full h-full relative z-10 p-6 ${currentLayer.items.length > 2 ? 'grid grid-cols-2' : ''}`}>
                          {currentLayer.items.map((item, i) => {
                            const Icon = item.icon;
                            return (
                              <div key={i} className={`flex flex-col items-center flex-1 bg-[#000000]/70 rounded-[16px] p-4 border border-[#4F84FF]/20 backdrop-blur-xl shadow-[0_4px_20px_rgba(0,0,0,0.4)] transition-all duration-300 ${currentLayer.items.length === 3 && i === 2 ? 'col-span-2' : ''}`}>
                                <Icon size={22} className="text-[#8EA3C7] mb-3 transition-colors duration-300 group-hover:text-[#4F84FF]" />
                                <span className="text-[11px] font-mono font-medium text-[#E2E8F0] tracking-wide text-center leading-relaxed">
                                  {item.name}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </motion.div>
                  </motion.div>
                );
              })}
            </motion.div>
          </div>

          {/* Right: Text Descriptions - Static 2D Layout */}
          <div className="w-full lg:w-[50%] flex flex-col justify-center h-[50vh] lg:h-full relative z-20 pl-4 lg:pl-16 mt-12 lg:mt-0">
            <div className="flex flex-col justify-center gap-6 lg:gap-8 w-full max-w-[480px]">
              {layersData.map((layer, index) => {
                const currentOp = textOpacities[index];
                const isActive = activeLayer === index;
                
                // eslint-disable-next-line react-hooks/rules-of-hooks
                const scrollOpacity = useTransform(currentOp, (op) => {
                  if (isActive) return 1;
                  if (index < activeLayer) return 0.3;
                  return op;
                });

                return (
                  <motion.div 
                    key={layer.id}
                    style={{ opacity: hoveredLayer !== null ? (hoveredLayer === index ? 1 : 0.2) : scrollOpacity }}
                    className="flex flex-col gap-3 relative transition-all duration-500 cursor-pointer group"
                    onMouseEnter={() => setHoveredLayer(index)}
                    onMouseLeave={() => setHoveredLayer(null)}
                  >
                    {/* Horizontal Connector Line pointing towards stack */}
                    <div className={`absolute top-[10px] right-[100%] mr-6 h-[1px] bg-gradient-to-r from-transparent to-[#4F84FF] transition-all duration-500 hidden xl:block ${isActive ? 'w-[120px] opacity-100' : 'w-[40px] opacity-30'}`} />

                    <div className="flex items-center gap-4">
                      <div className={`w-2.5 h-2.5 rounded-full transition-all duration-500 ${isActive ? 'bg-[#4F84FF] shadow-[0_0_15px_#4F84FF] scale-125' : 'bg-[#334155]'}`} />
                      <span className={`font-mono font-bold tracking-[0.25em] text-[13px] uppercase text-left transition-colors duration-500 ${isActive ? 'text-[#F5F5F5]' : 'text-[#64748B]'}`}>
                        {layer.label}
                      </span>
                    </div>
                    
                    <div className="pl-6 border-l border-transparent transition-colors duration-500 group-hover:border-[#334155] py-1 ml-[5px]">
                      <p className={`font-sans text-[15px] text-[#94A3B8] leading-relaxed transition-colors duration-500 ${isActive ? 'text-[#E2E8F0]' : ''}`}>
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
