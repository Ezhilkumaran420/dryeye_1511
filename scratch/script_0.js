
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            display: ['Outfit', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            deepNavy: '#070E1A',
            panelNavy: '#0D1B2E',
            cardNavy: '#13243D',
            brandTeal: '#00B4D8',
            brandCyan: '#48CAE4',
            accentCyan: '#38BDF8',
            alertAmber: '#F59E0B',
            alertRose: '#F43F5E',
            clinicGreen: '#10B981',
          },
          boxShadow: {
            'glow-cyan': '0 0 30px -4px rgba(56, 189, 248, 0.45)',
            'glow-teal': '0 0 35px -5px rgba(0, 180, 216, 0.5)',
            'glow-emerald': '0 0 25px -3px rgba(16, 185, 129, 0.4)',
            'glow-rose': '0 0 25px -3px rgba(244, 63, 94, 0.4)',
            'glass': '0 12px 40px 0 rgba(0, 0, 0, 0.45)',
          }
        }
      }
    }
  