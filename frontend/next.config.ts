import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  transpilePackages: ['@shadergradient/react', 'three', '@react-three/fiber', '@react-three/drei'],
};

export default nextConfig;
