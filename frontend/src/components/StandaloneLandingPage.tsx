"use client";

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useRouter } from 'next/navigation';
import { Show, SignInButton, SignUpButton } from '@clerk/nextjs';

import { Text } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { AnimatedStack } from './AnimatedStack';

function GlobeWireframe({ radius }: { radius: number }) {
  const meridians = 24;
  const latitudes = 16;

  const lineGeometries = useMemo(() => {
    const geometries: THREE.BufferGeometry[] = [];

    // Latitudes (horizontal rings)
    for (let i = 1; i < latitudes; i++) {
      const phi = (Math.PI * i) / latitudes;
      const y = radius * Math.cos(phi);
      const ringRadius = radius * Math.sin(phi);

      const pts = [];
      for (let j = 0; j <= 64; j++) {
        const theta = (Math.PI * 2 * j) / 64;
        pts.push(new THREE.Vector3(ringRadius * Math.cos(theta), y, ringRadius * Math.sin(theta)));
      }
      geometries.push(new THREE.BufferGeometry().setFromPoints(pts));
    }

    // Meridians (vertical circles)
    for (let i = 0; i < meridians / 2; i++) {
      const theta = (Math.PI * i) / (meridians / 2);
      const pts = [];
      for (let j = 0; j <= 64; j++) {
        const phi = (Math.PI * 2 * j) / 64;
        const x = radius * Math.cos(theta) * Math.cos(phi);
        const z = radius * Math.sin(theta) * Math.cos(phi);
        const y = radius * Math.sin(phi);
        pts.push(new THREE.Vector3(x, y, z));
      }
      geometries.push(new THREE.BufferGeometry().setFromPoints(pts));
    }

    return geometries;
  }, [radius]);

  const material = useMemo(() => new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.15 }), []);

  return (
    <group>
      {lineGeometries.map((geometry, i) => (
        <primitive key={i} object={new THREE.Line(geometry, material)} />
      ))}
    </group>
  );
}

function SphereModel() {
  const pivotGroupRef = useRef<THREE.Group>(null);
  const globeGroupRef = useRef<THREE.Group>(null);
  const textRefs = useRef<any[]>([]);

  const radius = 25;
  const { particles, glowingNodes } = useMemo(() => {
    const temp = [];
    const count = 700;

    // Z axis vector for quaternion calculation
    const zAxis = new THREE.Vector3(0, 0, 1);

    for (let i = 0; i < count; i++) {
      const phi = Math.acos((Math.random() * 2) - 1);
      const theta = Math.random() * Math.PI * 2;

      // Position slightly above the wireframe (radius * 1.02)
      const particleRadius = radius * 1.02;
      const x = particleRadius * Math.sin(phi) * Math.cos(theta);
      const y = particleRadius * Math.cos(phi);
      const z = particleRadius * Math.sin(phi) * Math.sin(theta);

      const pos = new THREE.Vector3(x, y, z);

      // Orient to face outward from the center of the sphere
      const quaternion = new THREE.Quaternion().setFromUnitVectors(zAxis, pos.clone().normalize());
      const euler = new THREE.Euler().setFromQuaternion(quaternion);

      const isHighlight = Math.random() > 0.95;
      temp.push({
        pos,
        rotation: euler,
        char: Math.random() > 0.5 ? '1' : '0',
        size: isHighlight ? 0.35 : 0.18 + Math.random() * 0.06,
        baseOpacity: isHighlight ? 0.8 + Math.random() * 0.2 : 0.25 + Math.random() * 0.50
      });
    }

    const nodes = [];
    for (let i = 0; i < 6; i++) {
      const phi = Math.acos((Math.random() * 2) - 1);
      const theta = Math.random() * Math.PI * 2;
      const x = (radius - 0.1) * Math.sin(phi) * Math.cos(theta);
      const y = (radius - 0.1) * Math.cos(phi);
      const z = (radius - 0.1) * Math.sin(phi) * Math.sin(theta);
      nodes.push(new THREE.Vector3(x, y, z));
    }

    return { particles: temp, glowingNodes: nodes };
  }, []);

  useFrame(() => {
    if (globeGroupRef.current) {
      globeGroupRef.current.rotation.y += 0.0015;

      const matrixWorld = globeGroupRef.current.matrixWorld;
      const vec = new THREE.Vector3();

      textRefs.current.forEach((textRef, i) => {
        if (!textRef) return;

        vec.copy(particles[i].pos);
        vec.applyMatrix4(matrixWorld);

        const normalizedZ = (vec.z + radius) / (radius * 2);
        const depthOpacity = Math.max(0.02, Math.pow(normalizedZ, 2.0));

        // Dynamic center-right focal boost
        let focalBoost = 1.0;
        if (vec.x > 8 && vec.y > -12 && vec.y < 12 && vec.z > 8) {
          const dist = Math.sqrt(Math.pow(vec.x - 18, 2) + Math.pow(vec.y, 2) + Math.pow(vec.z - 18, 2));
          if (dist < 12) {
            focalBoost = 1.0 + ((12 - dist) / 12) * 1.5;
          }
        }

        const finalOpacity = particles[i].baseOpacity * Math.min(1.2, depthOpacity * 1.5) * focalBoost;
        textRef.fillOpacity = Math.min(1.0, finalOpacity);

        const scaleMultiplier = 0.8 + depthOpacity * 0.4;
        textRef.scale.setScalar(scaleMultiplier * particles[i].size);
      });
    }
  });

  return (
    <group ref={pivotGroupRef} rotation={[0.06, 0, -0.15]}>
      <group ref={globeGroupRef}>
        <GlobeWireframe radius={radius - 0.2} />

        {glowingNodes.map((pos, i) => (
          <mesh key={`glow-${i}`} position={pos}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshBasicMaterial color="#ffffff" transparent opacity={0.15} />
          </mesh>
        ))}

        {particles.map((p, i) => (
          <Text
            key={i}
            ref={(el) => {
              textRefs.current[i] = el;
            }}
            position={p.pos}
            rotation={p.rotation}
            color="white"
            fontSize={1.5}
            anchorX="center"
            anchorY="middle"
          >
            {p.char}
          </Text>
        ))}
      </group>
    </group>
  );
}

function BackgroundParticles() {
  const pivotGroupRef = useRef<THREE.Group>(null);
  const globeGroupRef = useRef<THREE.Group>(null);
  const textRefs = useRef<any[]>([]);

  const particles = useMemo(() => {
    const temp = [];
    const count = 400;
    for (let i = 0; i < count; i++) {
      const phi = Math.acos((Math.random() * 2) - 1);
      const theta = Math.random() * Math.PI * 2;
      const particleRadius = 35 + Math.random() * 25;
      const x = particleRadius * Math.sin(phi) * Math.cos(theta);
      const y = particleRadius * Math.cos(phi);
      const z = particleRadius * Math.sin(phi) * Math.sin(theta);
      temp.push({
        pos: new THREE.Vector3(x, y, z),
        char: Math.random() > 0.5 ? '1' : '0',
      });
    }
    return temp;
  }, []);

  useFrame(({ camera }) => {
    if (globeGroupRef.current) {
      globeGroupRef.current.rotation.y -= 0.0004;

      textRefs.current.forEach((textRef) => {
        if (!textRef) return;
        textRef.quaternion.copy(camera.quaternion);
      });
    }
  });

  return (
    <group ref={pivotGroupRef} rotation={[0.06, 0, -0.15]}>
      <group ref={globeGroupRef}>
        {particles.map((p, i) => (
          <Text
            key={i}
            ref={(el) => {
              textRefs.current[i] = el;
            }}
            position={p.pos}
            color="white"
            fontSize={0.6}
            fillOpacity={0.04}
            anchorX="center"
            anchorY="middle"
          >
            {p.char}
          </Text>
        ))}
      </group>
    </group>
  );
}

const BinarySphere = () => (
  <div className="relative w-full max-w-[1200px] h-[800px] mx-auto flex items-center justify-center opacity-100">
    <Canvas camera={{ position: [0, 0, 48], fov: 45 }}>
      <SphereModel />
      <BackgroundParticles />
      <EffectComposer>
        <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} opacity={0.6} />
      </EffectComposer>
    </Canvas>
    <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,#080808_70%)] pointer-events-none" />
  </div>
);

// ── Wireframe SVG shapes ─────────────────────────────────────────────────
const WireframeCube = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="white" strokeWidth="0.9" strokeLinecap="round" strokeLinejoin="round" className="w-full h-full">
    {/* Front face */}
    <rect x="50" y="62" width="88" height="88" strokeWidth="0.9" />
    {/* Back face */}
    <rect x="64" y="48" width="88" height="88" strokeWidth="0.6" opacity={0.5} />
    {/* Connecting edges */}
    <line x1="50" y1="62" x2="64" y2="48" /><line x1="138" y1="62" x2="152" y2="48" />
    <line x1="138" y1="150" x2="152" y2="136" /><line x1="50" y1="150" x2="64" y2="136" />
    {/* Face diagonals */}
    <line x1="50" y1="62" x2="138" y2="150" strokeWidth="0.4" opacity={0.4} />
    <line x1="138" y1="62" x2="50" y2="150" strokeWidth="0.4" opacity={0.4} />
    <line x1="50" y1="62" x2="152" y2="48" strokeWidth="0.4" opacity={0.35} />
    <line x1="64" y1="48" x2="138" y2="62" strokeWidth="0.4" opacity={0.35} />
    <line x1="138" y1="62" x2="152" y2="136" strokeWidth="0.4" opacity={0.35} />
    <line x1="152" y1="48" x2="138" y2="150" strokeWidth="0.4" opacity={0.35} />
    <circle cx="100" cy="106" r="2" fill="white" opacity={0.3} />
    <circle cx="100" cy="100" r="76" strokeWidth="0.3" opacity={0.2} />
  </svg>
);

const WireframeIcosphere = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="white" strokeWidth="0.8" strokeLinecap="round" strokeLinejoin="round" className="w-full h-full">
    <circle cx="100" cy="100" r="78" strokeWidth="0.5" opacity={0.4} />
    <polygon points="100,22 143,55 100,66" /><polygon points="100,22 57,55 100,66" />
    <polygon points="100,22 143,55 172,95" /><polygon points="100,22 57,55 28,95" />
    <polygon points="143,55 172,95 143,130" /><polygon points="57,55 28,95 57,130" />
    <polygon points="100,66 143,55 143,130" /><polygon points="100,66 57,55 57,130" />
    <polygon points="100,66 143,130 100,145" /><polygon points="100,66 57,130 100,145" />
    <polygon points="143,130 172,95 172,145" /><polygon points="57,130 28,95 28,145" />
    <polygon points="100,145 143,130 172,145" /><polygon points="100,145 57,130 28,145" />
    <polygon points="100,145 172,145 100,178" /><polygon points="100,145 28,145 100,178" />
    <ellipse cx="100" cy="66" rx="43" ry="8" strokeWidth="0.4" opacity={0.3} />
    <ellipse cx="100" cy="130" rx="43" ry="8" strokeWidth="0.4" opacity={0.3} />
  </svg>
);

const WireframePolyhedron = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="white" strokeWidth="0.9" strokeLinecap="round" strokeLinejoin="round" className="w-full h-full">
    <line x1="100" y1="28" x2="148" y2="75" /><line x1="100" y1="28" x2="52" y2="75" /><line x1="100" y1="28" x2="100" y2="72" />
    <polygon points="100,72 148,75 130,118" /><polygon points="100,72 52,75 70,118" />
    <polygon points="100,72 130,118 70,118" /><polygon points="148,75 52,75 100,72" opacity={0.5} />
    <polygon points="148,75 130,118 172,118" /><polygon points="52,75 70,118 28,118" />
    <polygon points="130,118 172,118 148,155" /><polygon points="70,118 28,118 52,155" />
    <polygon points="130,118 70,118 100,155" />
    <polygon points="100,155 148,155 100,175" /><polygon points="100,155 52,155 100,175" />
    <polygon points="148,155 172,118 100,175" opacity={0.6} /><polygon points="52,155 28,118 100,175" opacity={0.6} />
    <circle cx="100" cy="102" r="74" strokeWidth="0.35" opacity={0.25} />
  </svg>
);

const WireframeSphere = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="white" strokeWidth="0.7" strokeLinecap="round" className="w-full h-full">
    <circle cx="100" cy="100" r="72" />
    <ellipse cx="100" cy="100" rx="72" ry="26" />
    <ellipse cx="100" cy="100" rx="72" ry="26" transform="rotate(60 100 100)" />
    <ellipse cx="100" cy="100" rx="72" ry="26" transform="rotate(120 100 100)" />
    <ellipse cx="100" cy="100" rx="40" ry="72" strokeWidth="0.5" opacity={0.5} />
    <line x1="100" y1="28" x2="100" y2="172" strokeWidth="0.4" opacity={0.4} />
    <line x1="28" y1="100" x2="172" y2="100" strokeWidth="0.4" opacity={0.4} />
  </svg>
);

const WireframeCylinder = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="white" strokeWidth="0.8" strokeLinecap="round" className="w-full h-full">
    <ellipse cx="100" cy="55" rx="52" ry="18" />
    <ellipse cx="100" cy="148" rx="52" ry="18" strokeDasharray="3 3" opacity={0.6} />
    <line x1="48" y1="55" x2="48" y2="148" /><line x1="152" y1="55" x2="152" y2="148" />
    <line x1="68" y1="51" x2="68" y2="148" strokeDasharray="3 4" opacity={0.4} />
    <line x1="132" y1="51" x2="132" y2="148" strokeDasharray="3 4" opacity={0.4} />
    <line x1="100" y1="37" x2="100" y2="148" strokeDasharray="2 4" opacity={0.3} />
    <ellipse cx="100" cy="100" rx="52" ry="18" strokeWidth="0.4" opacity={0.3} />
  </svg>
);

const WireframeShield = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="white" strokeWidth="0.9" strokeLinecap="round" strokeLinejoin="round" className="w-full h-full">
    <path d="M100 28 L168 52 L168 108 C168 148 100 172 100 172 C100 172 32 148 32 108 L32 52 Z" />
    <path d="M100 28 L100 172" strokeDasharray="3 4" strokeWidth="0.5" opacity={0.4} />
    <path d="M32 72 L168 72" strokeWidth="0.4" opacity={0.35} />
    <path d="M32 100 L168 100" strokeWidth="0.4" opacity={0.35} />
    <path d="M32 125 L168 125" strokeWidth="0.4" opacity={0.25} />
    <path d="M100 28 L168 72 L100 100 L32 72 Z" strokeWidth="0.5" opacity={0.4} />
    <circle cx="100" cy="105" r="18" strokeWidth="0.5" opacity={0.3} />
  </svg>
);

const iconMap: Record<string, React.ReactNode> = {
  cube: <WireframeCube />,
  network: <WireframeIcosphere />,
  sphere: <WireframeSphere />,
  polyhedron: <WireframePolyhedron />,
  cylinder: <WireframeCylinder />,
  shield: <WireframeShield />,
};

const FeatureCard = ({ title, desc, icon }: { title: string, desc: string, icon: string }) => {
  return (
    <div
      className="group relative overflow-hidden flex flex-col transition-all duration-500 cursor-default"
      style={{
        background: '#000000',
        border: '1px solid rgba(255,255,255,0.14)',
        boxShadow: 'inset 0 0 30px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.04)',
        height: '330px',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)';
        (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.28)';
        (e.currentTarget as HTMLDivElement).style.boxShadow = 'inset 0 0 40px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.06), 0 6px 24px rgba(0,0,0,0.7)';
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
        (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.14)';
        (e.currentTarget as HTMLDivElement).style.boxShadow = 'inset 0 0 30px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.04)';
      }}
    >
      {/* ── Top-right dot texture — small, localized ── */}
      <div
        className="absolute top-0 right-0 pointer-events-none z-[2]"
        style={{
          width: '80px',
          height: '80px',
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.14) 1px, transparent 1px)',
          backgroundSize: '9px 9px',
          WebkitMaskImage: 'radial-gradient(circle at top right, black 20%, transparent 70%)',
          maskImage: 'radial-gradient(circle at top right, black 20%, transparent 70%)',
        }}
      />

      {/* ── Text content (top-left) ── */}
      <div className="relative z-[4] pt-6 px-6">
        <h3
          className="font-mono font-bold text-white tracking-[0.02em] leading-snug"
          style={{ fontSize: '14.5px', marginBottom: '10px' }}
        >
          {title}
        </h3>
        <p
          className="font-mono text-white/40 leading-[1.8]"
          style={{ fontSize: '11.5px', maxWidth: '76%' }}
        >
          {desc}
        </p>
      </div>

      {/* ── Centered wireframe illustration — larger, anchored low ── */}
      <div
        className="absolute z-[3] pointer-events-none"
        style={{
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '180px',
          height: '180px',
          opacity: 0.34,
          transition: 'opacity 0.5s ease',
        }}
        onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.opacity = '0.56'; }}
        onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.opacity = '0.34'; }}
      >
        {iconMap[icon] ?? <WireframeCube />}
      </div>

      {/* ── Narrow vertical mist — concentrated beam from bottom-center ── */}
      <div
        className="absolute bottom-0 left-0 right-0 pointer-events-none z-[1]"
        style={{
          height: '70%',
          background: [
            'radial-gradient(ellipse 28% 70% at 50% 100%, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.10) 30%, rgba(255,255,255,0.02) 60%, transparent 100%)',
            'radial-gradient(ellipse 55% 35% at 50% 100%, rgba(255,255,255,0.07) 0%, transparent 70%)',
          ].join(', '),
        }}
      />
      {/* Hover-intensified mist ── */}
      <div
        className="absolute bottom-0 left-0 right-0 pointer-events-none z-[1] opacity-0 group-hover:opacity-100 transition-opacity duration-500"
        style={{
          height: '70%',
          background: 'radial-gradient(ellipse 32% 75% at 50% 100%, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.05) 40%, transparent 80%)',
        }}
      />

      {/* ── Halftone dotted bottom band — denser, tighter ── */}
      <div
        className="absolute bottom-0 left-0 right-0 pointer-events-none z-[2]"
        style={{
          height: '48px',
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.30) 1px, transparent 1px)',
          backgroundSize: '7px 7px',
          WebkitMaskImage: 'linear-gradient(to top, rgba(0,0,0,0.65) 0%, rgba(0,0,0,0.20) 50%, transparent 100%)',
          maskImage: 'linear-gradient(to top, rgba(0,0,0,0.65) 0%, rgba(0,0,0,0.20) 50%, transparent 100%)',
        }}
      />
    </div>
  );
};

/** ── Procedural halftone cloud via offscreen canvas ─────────────────────────
 *  1. Build a grayscale "brightness map" by painting many elliptical radial
 *     gradients in "lighter" composite mode onto an offscreen canvas.
 *  2. Optionally run a simple box blur over the map to smooth blob boundaries.
 *  3. Sample the map at each dot-grid cell centre and draw a proportional
 *     circle — big solid dots in bright cores, sparse tiny dots in wisps.
 */
const HalftoneCloud = ({ isLeft }: { isLeft: boolean }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext('2d')!;

    // ── Step 1: build brightness map on offscreen canvas ────────────────────
    const bmp = document.createElement('canvas');
    bmp.width = W;
    bmp.height = H;
    const bc = bmp.getContext('2d')!;
    bc.fillStyle = '#000';
    bc.fillRect(0, 0, W, H);
    bc.globalCompositeOperation = 'lighter';

    // Helper: draw an elliptical radial gradient blob
    const blob = (cx: number, cy: number, rx: number, ry: number, alpha: number) => {
      bc.save();
      bc.translate(cx, cy);
      bc.scale(rx, ry);
      const g = bc.createRadialGradient(0, 0, 0, 0, 0, 1);
      g.addColorStop(0, `rgba(255,255,255,${alpha})`);
      g.addColorStop(0.4, `rgba(255,255,255,${(alpha * 0.55).toFixed(3)})`);
      g.addColorStop(0.75, `rgba(255,255,255,${(alpha * 0.15).toFixed(3)})`);
      g.addColorStop(1, 'rgba(255,255,255,0)');
      bc.fillStyle = g;
      bc.beginPath();
      bc.arc(0, 0, 1, 0, Math.PI * 2);
      bc.fill();
      bc.restore();
    };

    if (isLeft) {
      // ── LEFT side — asymmetric storm formation ──────────────────────────
      // Dense upper-corner mass
      blob(W * 0.04, H * 0.07, W * 0.22, H * 0.30, 0.95);
      blob(W * 0.18, H * 0.12, W * 0.17, H * 0.24, 0.75);
      blob(W * 0.10, H * 0.22, W * 0.14, H * 0.18, 0.70);
      blob(W * 0.28, H * 0.05, W * 0.13, H * 0.18, 0.55);
      blob(W * 0.06, H * 0.32, W * 0.10, H * 0.14, 0.55);
      // Secondary mid plume — offset lower and inward
      blob(W * 0.09, H * 0.50, W * 0.20, H * 0.28, 0.72);
      blob(W * 0.24, H * 0.44, W * 0.14, H * 0.20, 0.50);
      blob(W * 0.16, H * 0.60, W * 0.11, H * 0.16, 0.42);
      // Lower dense mass
      blob(W * 0.04, H * 0.84, W * 0.18, H * 0.25, 0.88);
      blob(W * 0.20, H * 0.78, W * 0.16, H * 0.22, 0.62);
      blob(W * 0.32, H * 0.90, W * 0.12, H * 0.18, 0.45);
      blob(W * 0.12, H * 0.94, W * 0.09, H * 0.12, 0.55);
      // Wisps & tendrils extending inward
      blob(W * 0.35, H * 0.22, W * 0.09, H * 0.13, 0.28);
      blob(W * 0.38, H * 0.52, W * 0.08, H * 0.12, 0.25);
      blob(W * 0.42, H * 0.72, W * 0.07, H * 0.10, 0.20);
      blob(W * 0.30, H * 0.65, W * 0.10, H * 0.14, 0.30);
      // Torn inner edges — irregular sub-lumps
      blob(W * 0.14, H * 0.38, W * 0.07, H * 0.09, 0.45);
      blob(W * 0.26, H * 0.30, W * 0.06, H * 0.10, 0.35);
      blob(W * 0.08, H * 0.70, W * 0.08, H * 0.11, 0.48);
      blob(W * 0.22, H * 0.86, W * 0.07, H * 0.09, 0.38);
    } else {
      // ── RIGHT side — different composition, not a mirror ────────────────
      // Dense upper-corner mass — sits higher and more jagged
      blob(W * 0.96, H * 0.10, W * 0.24, H * 0.35, 0.92);
      blob(W * 0.80, H * 0.05, W * 0.16, H * 0.22, 0.70);
      blob(W * 0.90, H * 0.26, W * 0.18, H * 0.24, 0.68);
      blob(W * 0.72, H * 0.14, W * 0.12, H * 0.17, 0.52);
      blob(W * 0.94, H * 0.40, W * 0.10, H * 0.15, 0.58);
      // Secondary mid plume — higher and broader than left
      blob(W * 0.91, H * 0.55, W * 0.22, H * 0.30, 0.75);
      blob(W * 0.75, H * 0.46, W * 0.15, H * 0.21, 0.52);
      blob(W * 0.82, H * 0.66, W * 0.12, H * 0.17, 0.40);
      // Lower mass — wider base than left side
      blob(W * 0.96, H * 0.88, W * 0.20, H * 0.28, 0.85);
      blob(W * 0.78, H * 0.80, W * 0.18, H * 0.24, 0.65);
      blob(W * 0.68, H * 0.92, W * 0.14, H * 0.20, 0.48);
      blob(W * 0.88, H * 0.96, W * 0.10, H * 0.14, 0.55);
      // Wisps — spill further inward on right
      blob(W * 0.64, H * 0.18, W * 0.09, H * 0.14, 0.26);
      blob(W * 0.60, H * 0.50, W * 0.08, H * 0.12, 0.22);
      blob(W * 0.58, H * 0.76, W * 0.07, H * 0.10, 0.18);
      blob(W * 0.68, H * 0.62, W * 0.10, H * 0.14, 0.28);
      // Torn sub-lumps
      blob(W * 0.86, H * 0.36, W * 0.07, H * 0.09, 0.44);
      blob(W * 0.74, H * 0.28, W * 0.06, H * 0.10, 0.33);
      blob(W * 0.92, H * 0.72, W * 0.08, H * 0.11, 0.46);
      blob(W * 0.76, H * 0.88, W * 0.07, H * 0.09, 0.36);
    }

    // ── Step 2: simple 2-pass box blur to smooth hard blob edges ───────────
    const blurPasses = 3;
    const radius = 12;
    for (let pass = 0; pass < blurPasses; pass++) {
      const src = bc.getImageData(0, 0, W, H);
      const dst = bc.createImageData(W, H);
      const d = src.data;
      const o = dst.data;
      // Horizontal pass
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let sum = 0, cnt = 0;
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = Math.min(W - 1, Math.max(0, x + dx));
            sum += d[(y * W + nx) * 4];
            cnt++;
          }
          const idx = (y * W + x) * 4;
          const v = sum / cnt;
          o[idx] = o[idx + 1] = o[idx + 2] = v;
          o[idx + 3] = 255;
        }
      }
      bc.putImageData(dst, 0, 0);
      // Vertical pass
      const src2 = bc.getImageData(0, 0, W, H);
      const dst2 = bc.createImageData(W, H);
      const d2 = src2.data;
      const o2 = dst2.data;
      for (let x = 0; x < W; x++) {
        for (let y = 0; y < H; y++) {
          let sum = 0, cnt = 0;
          for (let dy = -radius; dy <= radius; dy++) {
            const ny = Math.min(H - 1, Math.max(0, y + dy));
            sum += d2[(ny * W + x) * 4];
            cnt++;
          }
          const idx = (y * W + x) * 4;
          const v = sum / cnt;
          o2[idx] = o2[idx + 1] = o2[idx + 2] = v;
          o2[idx + 3] = 255;
        }
      }
      bc.putImageData(dst2, 0, 0);
    }

    // ── Step 3: sample brightness map → draw halftone dots ─────────────────
    const brightPixels = bc.getImageData(0, 0, W, H).data;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);

    const spacing = 6;   // dot grid spacing in px
    const maxRadius = spacing * 0.48; // largest dot radius

    for (let cy = spacing / 2; cy < H; cy += spacing) {
      for (let cx = spacing / 2; cx < W; cx += spacing) {
        const px = Math.min(W - 1, Math.round(cx));
        const py = Math.min(H - 1, Math.round(cy));
        const brightness = brightPixels[(py * W + px) * 4] / 255;

        // Gamma-correct: boost mid-dark range for more visible wisps
        const adjusted = Math.pow(brightness, 0.65);
        const r = adjusted * maxRadius;
        if (r < 0.35) continue;

        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ── Step 4: fade toward center edge ───────────────────────────────────
    const fadeGrad = ctx.createLinearGradient(
      isLeft ? 0 : W, 0,
      isLeft ? W : 0, 0
    );
    fadeGrad.addColorStop(0, 'rgba(0,0,0,0)');
    fadeGrad.addColorStop(0.55, 'rgba(0,0,0,0)');
    fadeGrad.addColorStop(0.82, 'rgba(0,0,0,0.72)');
    fadeGrad.addColorStop(1, 'rgba(0,0,0,1)');
    ctx.fillStyle = fadeGrad;
    ctx.fillRect(0, 0, W, H);

    // Fade top + bottom edges
    const topGrad = ctx.createLinearGradient(0, 0, 0, H * 0.12);
    topGrad.addColorStop(0, 'rgba(0,0,0,1)');
    topGrad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = topGrad;
    ctx.fillRect(0, 0, W, H * 0.12);

    const botGrad = ctx.createLinearGradient(0, H * 0.88, 0, H);
    botGrad.addColorStop(0, 'rgba(0,0,0,0)');
    botGrad.addColorStop(1, 'rgba(0,0,0,1)');
    ctx.fillStyle = botGrad;
    ctx.fillRect(0, H * 0.88, W, H * 0.12);

  }, [isLeft]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={400}
      className="absolute inset-y-0 pointer-events-none z-[1]"
      style={{
        width: '50%',
        height: '100%',
        ...(isLeft ? { left: 0 } : { right: 0 }),
        mixBlendMode: 'screen',
      }}
    />
  );
};

const Step = ({ number, title, desc }: { number: string, title: string, desc: string }) => (
  <div className="flex flex-col items-center md:items-start text-center md:text-left relative z-10 bg-[#080808] px-4">
    <div className="w-12 h-12 rounded-full border border-white/20 bg-[#111] flex items-center justify-center font-mono text-sm mb-6 text-white/80">
      {number}
    </div>
    <h4 className="text-lg font-mono font-bold mb-3">{title}</h4>
    <p className="text-[13px] text-white/50 font-mono leading-relaxed max-w-[250px]">{desc}</p>
  </div>
);

const BenefitCard = ({ title, description }: { title: string, description: string }) => (
  <div
    className="border border-white/10 p-8 flex flex-col justify-start bg-white/[0.01] relative overflow-hidden transition-all duration-500 cursor-default"
    style={{ minHeight: '240px' }}
    onMouseEnter={e => {
      (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-3px)';
      (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.22)';
      (e.currentTarget as HTMLDivElement).style.boxShadow = '0 6px 28px rgba(0,0,0,0.5)';
    }}
    onMouseLeave={e => {
      (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
      (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.10)';
      (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
    }}
  >
    {/* Subtle radial glow */}
    <div
      className="absolute inset-0 pointer-events-none z-[0]"
      style={{
        background: 'radial-gradient(ellipse 70% 60% at 20% 20%, rgba(255,255,255,0.025) 0%, transparent 70%)',
      }}
    />
    {/* Dot noise texture */}
    <div
      className="absolute inset-0 pointer-events-none z-[0]"
      style={{
        backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px)',
        backgroundSize: '6px 6px',
      }}
    />
    <h4 className="font-mono font-bold text-[16px] text-white/90 tracking-tight mb-5 relative z-[1]">{title}</h4>
    <p className="text-[13px] text-white/45 font-mono leading-[2] relative z-[1]" style={{ maxWidth: '90%' }}>{description}</p>
  </div>
);

const CapabilityCard = ({ category, heading, features, buttonLabel, isHighlighted = false }: { category: string, heading: string, features: string[], buttonLabel: string, isHighlighted?: boolean }) => (
  <div
    className={`border p-8 flex flex-col bg-[#0a0a0a] relative overflow-hidden transition-all duration-500 cursor-default group ${isHighlighted
        ? 'border-white/40 shadow-[0_0_30px_rgba(255,255,255,0.05)] scale-[1.02] z-10'
        : 'border-white/10 hover:border-white/20'
      }`}
    style={{ willChange: 'transform, box-shadow' }}
    onMouseEnter={e => {
      const el = e.currentTarget as HTMLDivElement;
      el.style.transform = isHighlighted ? 'scale(1.02) translateY(-4px)' : 'translateY(-4px)';
      el.style.borderColor = isHighlighted ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.25)';
      el.style.boxShadow = isHighlighted
        ? '0 0 40px rgba(255,255,255,0.08), 0 8px 32px rgba(0,0,0,0.6)'
        : '0 8px 28px rgba(0,0,0,0.5)';
    }}
    onMouseLeave={e => {
      const el = e.currentTarget as HTMLDivElement;
      el.style.transform = isHighlighted ? 'scale(1.02) translateY(0)' : 'translateY(0)';
      el.style.borderColor = isHighlighted ? 'rgba(255,255,255,0.40)' : 'rgba(255,255,255,0.10)';
      el.style.boxShadow = isHighlighted ? '0 0 30px rgba(255,255,255,0.05)' : 'none';
    }}
  >
    {/* Featured card top accent line */}
    {isHighlighted && <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-white/50 to-transparent" />}
    {/* Subtle radial glow for featured card */}
    {isHighlighted && (
      <div className="absolute inset-0 pointer-events-none z-[0]" style={{
        background: 'radial-gradient(ellipse 80% 60% at 50% 0%, rgba(255,255,255,0.03) 0%, transparent 70%)',
      }} />
    )}
    {/* Dot noise texture */}
    <div className="absolute inset-0 pointer-events-none z-[0]" style={{
      backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.02) 1px, transparent 1px)',
      backgroundSize: '6px 6px',
    }} />

    <h4 className="text-lg font-mono text-white/50 mb-2 uppercase tracking-[0.2em] relative z-[1]">{category}</h4>
    <div className="text-[28px] font-mono font-bold mb-8 text-white tracking-tight leading-tight relative z-[1]">{heading}</div>
    <ul className="flex flex-col gap-4 mb-10 flex-grow relative z-[1]">
      {features.map((f, i) => (
        <li key={i} className="flex items-center gap-3 text-[13px] text-white/70 font-mono">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0"><path d="M20 6L9 17l-5-5" /></svg>
          {f}
        </li>
      ))}
    </ul>
    <button className={`w-full py-3 rounded-sm font-mono text-[12px] font-bold transition-colors relative z-[1] ${isHighlighted
        ? 'bg-white text-black hover:bg-gray-200'
        : 'bg-[#111] text-white hover:bg-[#1a1a1a] border border-white/10'
      }`}>
      {buttonLabel}
    </button>
  </div>
);

export default function StandaloneLandingPage() {
  const [mounted, setMounted] = useState(false);
  const router = useRouter();
  
  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="relative min-h-screen bg-[#080808] text-white font-mono overflow-x-clip selection:bg-white/30 selection:text-white">
      {/* Main Content Area */}
      <div className="relative z-10 max-w-[85vw] mx-auto flex flex-col min-h-screen">
        {/* Navbar */}
        <nav className="flex items-center justify-between h-[80px] flex-shrink-0">
          <div 
            onClick={() => router.push('/')}
            className="font-sans font-medium tracking-tight text-lg flex items-center gap-2.5 cursor-pointer hover:opacity-80 transition-opacity"
          >
            <img src="/vela_logo.svg" alt="VelaAI" className="w-7 h-7" />
            VelaAI
          </div>

          <div className="hidden md:flex items-center gap-10 text-[11px] text-white/40 tracking-widest font-mono">
            {['Home', 'Features', 'Capabilities', 'About'].map(item => (
              <a key={item} href="#" className="hover:text-white transition-colors">
                [{item}]
              </a>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <Show when="signed-in">
              <button 
                onClick={() => router.push('/')}
                className="flex items-center justify-center bg-white text-black px-4 py-1.5 rounded-md text-[11px] font-mono font-bold hover:bg-gray-200 transition-colors cursor-pointer"
              >
                Dashboard
              </button>
            </Show>
            
            <Show when="signed-out">
              <SignInButton mode="modal">
                <button className="flex items-center justify-center bg-transparent text-white px-4 py-1.5 rounded-md text-[11px] font-mono font-bold hover:bg-white/10 transition-colors cursor-pointer border border-white/20">
                  Sign In
                </button>
              </SignInButton>
              
              <SignUpButton mode="modal">
                <button className="flex items-center justify-center bg-white text-black px-4 py-1.5 rounded-md text-[11px] font-mono font-bold hover:bg-gray-200 transition-colors cursor-pointer">
                  Get Started
                </button>
              </SignUpButton>
            </Show>
          </div>
        </nav>

        {/* Hero Section */}
        <div className="relative w-full flex-grow flex flex-col justify-end pb-16 pt-[20vh]">
          {/* Sphere absolute behind */}
          <div className="absolute top-1/2 left-[82%] -translate-x-1/2 -translate-y-1/2 w-full max-w-[1200px] pointer-events-none -z-10">
            {mounted && <BinarySphere />}
          </div>

          {/* Hero Bottom: Text & Split Area */}
          <div className="flex flex-col md:flex-row justify-between items-end gap-10">
            <div className="max-w-2xl">
              <div className="inline-flex items-center gap-2 border border-white/20 bg-[#111] rounded-md px-1.5 py-1 mb-6">
                <span className="bg-white text-black text-[9px] font-bold px-1.5 py-0.5 rounded-sm font-mono uppercase">All</span>
                <span className="text-[10px] text-white/70 px-1 font-mono">Become A Beta Partner</span>
              </div>

              <h1 className="text-5xl md:text-[5rem] lg:text-[6rem] font-mono font-bold tracking-[-0.03em] leading-[1.05]">
                Your AI<br />Meeting Brain
              </h1>
            </div>

            <div className="max-w-[400px] flex flex-col items-start md:items-end text-left md:text-right">
              <p className="text-[14px] text-white/80 mb-6 font-mono leading-[1.8] tracking-tight text-left">
                Secure code, dependencies, containers, and<br />infrastructure from one platform.
              </p>
              <Show when="signed-out">
                <SignUpButton mode="modal">
                  <button className="flex items-center justify-center bg-white text-black px-6 py-2.5 rounded-full text-[13px] font-mono font-bold hover:bg-gray-200 transition-colors min-w-[140px] cursor-pointer">
                    Join Waitlist
                  </button>
                </SignUpButton>
              </Show>
              <Show when="signed-in">
                <button 
                  onClick={() => router.push('/')}
                  className="flex items-center justify-center bg-white text-black px-6 py-2.5 rounded-full text-[13px] font-mono font-bold hover:bg-gray-200 transition-colors min-w-[140px] cursor-pointer"
                >
                  Go to Dashboard
                </button>
              </Show>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="py-28 relative border-t border-white/10 overflow-hidden flex justify-center">
          {/* Background Effects */}
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(255,255,255,0.03)_0%,transparent_70%)] pointer-events-none" />

          <div className="relative z-10 flex flex-wrap justify-center gap-x-5 gap-y-10 max-w-[1600px] px-4">
            {[
              { value: "+7.3%", label: "ROUGE SCORE" },
              { value: "0.3265", label: "ROUGE-1" },
              { value: "+3.65%", label: "VS GPT-4" },
              { value: "+27.4%", label: "MRR SCORE" }
            ].map((stat, i) => (
              <div
                key={i}
                className="group relative flex items-center gap-6 px-8 py-6 rounded-md border border-dashed border-white/[0.08] bg-white/[0.01] bg-gradient-to-b from-white/[0.02] to-transparent shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] hover:border-white/[0.15] hover:bg-white/[0.03] hover:-translate-y-[2px] hover:shadow-[0_0_20px_rgba(255,255,255,0.03)] transition-all duration-300 w-full sm:w-[330px] h-[88px]"
              >
                <div className="w-[20px] h-[20px] border border-white/30 rounded-[2px] group-hover:shadow-[0_0_8px_rgba(255,255,255,0.3)] group-hover:border-white/60 transition-all duration-300 flex-shrink-0" />
                <div className="flex items-baseline gap-3 font-mono">
                  <span className="text-2xl md:text-3xl text-white font-bold tracking-tight">{stat.value}</span>
                  <span className="text-xs md:text-sm text-white/50 uppercase tracking-[0.15em]">{stat.label}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Intro Section */}
        <div className="py-32 flex flex-col items-center text-center relative border-t border-white/10">
          <div className="inline-flex items-center gap-2 border border-white/20 bg-[#111] rounded-full px-4 py-1.5 mb-8">
            <span className="text-[10px] text-white/70 font-mono uppercase tracking-widest">AI-Powered Meeting Intelligence</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-mono font-bold tracking-tight mb-6">
            Meet With Purpose
          </h2>
          <p className="text-[14px] text-white/60 max-w-xl mx-auto font-mono leading-[1.8] tracking-tight">
            Let AI handle notes and action items—so your team can focus on the conversation. Transform every discussion into structured, searchable knowledge instantly.
          </p>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 relative w-full" style={{ marginTop: '0', gap: 0 }}>
          <FeatureCard
            title="Personalized Proactive Summaries"
            desc="Get concise summaries tailored to your role, highlighting what matters most to you without digging through transcripts."
            icon="cube"
          />
          <FeatureCard
            title="Quickly Search Across Meetings"
            desc="Instantly locate essential discussions, quotes, or decisions using advanced AI semantic search across your entire meeting history."
            icon="network"
          />
          <FeatureCard
            title="Action Items Extracted Automatically"
            desc="Tasks, owners, and deadlines are intelligently derived and tracked, ensuring no follow-up is forgotten."
            icon="sphere"
          />
          <FeatureCard
            title="Functions Universally"
            desc="Works seamlessly with Zoom, Google Meet, Teams, and offline recordings. The application adjusts to your specific workflow."
            icon="polyhedron"
          />
          <FeatureCard
            title="Designed for Desktop Productivity"
            desc="A fast, reliable, distraction-free workflow designed specifically for desktop users who need peak productivity."
            icon="cylinder"
          />
          <FeatureCard
            title="Secure and Privacy-Focused Infrastructure"
            desc="Privacy-focused architecture ensuring your sensitive meeting data is encrypted, secure, and fully compliant."
            icon="shield"
          />
        </div>

        {/* Workflow Section */}
        <div className="relative border-t border-white/10 mt-16 w-full bg-black">
          <AnimatedStack />
        </div>

        {/* Benefits */}
        <div className="py-24 grid grid-cols-1 md:grid-cols-3 gap-6 relative border-t border-white/10">
          <BenefitCard
            title="Preserves Context"
            description="RoME's Temporal Graph Memory connects discussions across weeks and months, tracking unresolved issues, recurring topics, and past decisions so nothing important is forgotten."
          />
          <BenefitCard
            title="Extracts Action Items"
            description="Automatically identifies tasks, owners, deadlines, and dependencies from conversations and converts them into structured action items ready for execution."
          />
          <BenefitCard
            title="Searches Across Meetings"
            description="Instantly locate specific discussions, decisions, blockers, and quotes using semantic search across your complete meeting history."
          />
        </div>

        {/* Core Capabilities */}
        <div className="py-32 relative border-t border-white/10">
          <div className="flex flex-col items-center mb-16">
            <img src="/vela_logo.svg" alt="VelaAI" className="w-10 h-10 mb-5 opacity-80" />
            <h3 className="text-3xl font-mono font-bold mb-4 text-center">Core Capabilities</h3>
            <p className="text-[13px] text-white/50 font-mono leading-relaxed text-center max-w-lg tracking-tight">
              How VelaAI transforms conversations into structured, searchable, and actionable intelligence.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            <CapabilityCard
              category="Understand"
              heading="Capture Every Detail"
              features={["Multimodal Summaries", "Speaker-Aware Transcripts", "Role-Specific Insights", "Key Decision Detection"]}
              buttonLabel="Explore"
            />
            <CapabilityCard
              category="Organize"
              heading="Build Lasting Context"
              features={["Temporal Graph Memory", "Cross-Meeting Search", "Linked Decisions", "Unlimited History"]}
              buttonLabel="Explore"
              isHighlighted
            />
            <CapabilityCard
              category="Search"
              heading="Find Anything Instantly"
              features={["Semantic Meeting Search", "Decision Retrieval", "Quote and Topic Lookup", "Cross-Meeting Discovery"]}
              buttonLabel="Explore"
            />
          </div>
        </div>

        {/* ══ Final CTA ══ */}
        <div className="relative border border-white/10 overflow-hidden" style={{ background: '#000', minHeight: '320px' }}>

          {/* Subtle global halftone */}
          <div className="absolute inset-0 pointer-events-none z-[0]"
            style={{
              backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px)',
              backgroundSize: '5px 5px',
            }}
          />

          {/* Left halftone cloud */}
          <HalftoneCloud isLeft={true} />

          {/* Right halftone cloud */}
          <HalftoneCloud isLeft={false} />

          {/* Centre content */}
          <div className="relative z-[2] flex flex-col items-center justify-center text-center px-6 py-20">
            <h2 className="font-mono font-bold text-white leading-[1.05] tracking-[-0.02em] mb-4"
              style={{ fontSize: 'clamp(28px, 4vw, 42px)', maxWidth: '500px' }}>
              Meetings<br />Actionable results.
            </h2>
            <p className="font-mono text-white/50 mb-6 leading-[1.6]" style={{ fontSize: '12.5px', maxWidth: '380px' }}>
              Automated summaries, immediate sharing, and intelligent organization to assist you in advancing projects.
            </p>
            <Show when="signed-out">
              <SignUpButton mode="modal">
                <button className="flex items-center gap-3 bg-white text-black px-5 py-2.5 rounded-md font-mono font-bold hover:bg-gray-100 transition-colors cursor-pointer" style={{ fontSize: '13px' }}>
                  Get Started
                </button>
              </SignUpButton>
            </Show>
            <Show when="signed-in">
              <button 
                onClick={() => router.push('/')}
                className="flex items-center gap-3 bg-white text-black px-5 py-2.5 rounded-md font-mono font-bold hover:bg-gray-100 transition-colors cursor-pointer" 
                style={{ fontSize: '13px' }}
              >
                Go to Dashboard
              </button>
            </Show>
          </div>

          {/* Top-right crosshair decoration */}
          <span className="absolute top-4 right-6 text-white/25 font-mono text-xs pointer-events-none z-[3] select-none">+</span>
          <span className="absolute bottom-4 right-6 text-white/25 font-mono text-xs pointer-events-none z-[3] select-none">+</span>
        </div>

        {/* ══ Footer ══ */}
        <footer className="border-t border-white/10 pt-16 pb-10 font-mono">
          <div className="grid grid-cols-1 md:grid-cols-[200px_1fr_1fr_1fr_1fr] gap-12 md:gap-8 mb-14">

            {/* Brand column */}
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-2.5 text-white/90 text-sm font-bold">
                <img src="/vela_logo.svg" alt="VelaAI" className="w-7 h-7" />
                VelaAI
              </div>
              <p className="text-white/30 text-[11px] leading-relaxed" style={{ maxWidth: '160px' }}>
                AI-powered meeting intelligence for modern teams.
              </p>
              <p className="text-white/20 text-[10px] mt-2">© 2026 VelaAI Inc.</p>
            </div>

            {/* Quick Links */}
            <div className="flex flex-col gap-3">
              <h5 className="text-white/90 text-[11px] font-bold uppercase tracking-widest mb-1">Quick Links</h5>
              {['Integrations', 'Features', 'Benefits', 'Changelog'].map(link => (
                <a key={link} href="#" className="text-white/35 text-[11px] hover:text-white/80 transition-colors">{link}</a>
              ))}
            </div>

            {/* Product */}
            <div className="flex flex-col gap-3">
              <h5 className="text-white/90 text-[11px] font-bold uppercase tracking-widest mb-1">Product</h5>
              {['Home', 'Pricing', 'Contact', 'Status'].map(link => (
                <a key={link} href="#" className="text-white/35 text-[11px] hover:text-white/80 transition-colors">{link}</a>
              ))}
            </div>

            {/* Company / Legal */}
            <div className="flex flex-col gap-3">
              <h5 className="text-white/90 text-[11px] font-bold uppercase tracking-widest mb-1">Pages</h5>
              {['Terms of Service', 'Privacy Policy', 'Cookie Policy', 'Security'].map(link => (
                <a key={link} href="#" className="text-white/35 text-[11px] hover:text-white/80 transition-colors">{link}</a>
              ))}
            </div>

            {/* Stay in touch */}
            <div className="flex flex-col gap-3">
              <h5 className="text-white/90 text-[11px] font-bold uppercase tracking-widest mb-1">Stay in touch</h5>
              <p className="text-white/35 text-[11px] leading-relaxed">
                Follow us for updates, tips, and early access to new features.
              </p>
              <div className="flex gap-4 mt-2">
                {['Twitter', 'GitHub', 'Discord'].map(s => (
                  <a key={s} href="#" className="text-white/30 text-[11px] hover:text-white/70 transition-colors">{s}</a>
                ))}
              </div>
            </div>
          </div>

          {/* Footer bottom bar */}
          <div className="border-t border-white/[0.07] pt-6 flex flex-col md:flex-row justify-between items-center gap-4">
            <span className="text-white/20 text-[10px]">Built with TypeScript · Next.js · Three.js</span>
            <span className="text-white/20 text-[10px]">All rights reserved. 2026.</span>
          </div>
        </footer>

      </div>
    </div>
  );
}
