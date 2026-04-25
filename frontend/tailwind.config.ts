import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                // ── Warm light palette ──
                base: '#FFFFFF',
                warm: '#F7F6F3',
                muted: '#F0EEE9',
                border: '#E8E6E1',
                'border-bright': '#D4D2CC',

                // ── Text ──
                ink: '#1A1A18',
                'ink-secondary': '#6B6A66',
                'ink-muted': '#9B9891',
                'ink-faint': '#B0AEA8',

                // ── Accent ──
                indigo: {
                    DEFAULT: '#4F46E5',
                    light: '#EEF2FF',
                    dark: '#3730A3',
                },
                lime: {
                    DEFAULT: '#CCFF00',
                    dark: '#84CC16',
                },

                // ── Semantic ──
                positive: '#16A34A',
                'positive-light': '#F0FDF4',
                negative: '#DC2626',
                'negative-light': '#FEF2F2',
                amber: '#D97706',
                'amber-light': '#FFFBEB',
            },
            fontFamily: {
                sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
                display: ['"Instrument Serif"', '"Playfair Display"', 'Georgia', 'serif'],
                mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
            },
            fontSize: {
                'hero': ['72px', { lineHeight: '1', letterSpacing: '-0.02em', fontWeight: '300' }],
                'display': ['36px', { lineHeight: '1.1', letterSpacing: '-0.01em', fontWeight: '300' }],
                'section': ['10px', { lineHeight: '1', letterSpacing: '0.1em', fontWeight: '600' }],
                'label': ['11px', { lineHeight: '1', letterSpacing: '0.08em', fontWeight: '600' }],
            },
            borderRadius: {
                'card': '12px',
            },
            boxShadow: {
                'card': '0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03)',
                'card-hover': '0 4px 12px rgba(0,0,0,0.08), 0 12px 40px rgba(0,0,0,0.05)',
                'btn': '0 1px 2px rgba(0,0,0,0.05)',
                'btn-hover': '0 2px 8px rgba(0,0,0,0.1)',
            },
            keyframes: {
                shimmer: {
                    '0%': { backgroundPosition: '-200% 0' },
                    '100%': { backgroundPosition: '200% 0' },
                },
                'pulse-soft': {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.4' },
                },
            },
            animation: {
                shimmer: 'shimmer 1.5s ease-in-out infinite',
                'pulse-soft': 'pulse-soft 2s ease-in-out infinite',
            },
        },
    },
    plugins: [],
};
export default config;
