# Design Philosophy: Precision Lab

## Visual Identity

Clean, technical, data-forward. Not flashy, not generic. The aesthetic communicates
competence through restraint — every element earns its place.

## Typography

- **Display / Metrics:** JetBrains Mono — monospaced, technical, precise
- **Body / UI:** Source Sans 3 — clean, readable, professional
- NO Inter, Roboto, or Arial

## Color System

All colors defined as CSS custom properties in `:root`.

### Primary
- Teal-700: `#0f766e` (primary actions, links, active states)
- Teal-600: `#0d9488` (hover states)
- Teal-50: `#f0fdfa` (subtle backgrounds)

### Neutrals (Warm Stone)
- Stone-50: `#fafaf9` (page background)
- Stone-100: `#f5f5f4` (card backgrounds)
- Stone-200: `#e7e5e4` (borders)
- Stone-400: `#a8a29e` (muted text)
- Stone-600: `#57534e` (secondary text)
- Stone-800: `#292524` (primary text)
- Stone-900: `#1c1917` (headings)

### Category Accents
- Vision: `#2563eb` (Blue-600)
- NLP: `#d97706` (Amber-600)
- Audio: `#7c3aed` (Violet-600)
- Tabular: `#059669` (Emerald-600)
- Multimodal: `#e11d48` (Rose-600)

### Data Visualization (Colorblind-safe)
```
#2563eb, #d97706, #059669, #7c3aed, #e11d48, #0891b2, #4f46e5, #ca8a04
```

## Spatial Rhythm
- Base grid: 4px
- Card padding: 24px
- Section gaps: 48px
- Border radius: 2-4px (sharp, not rounded)

## Components
- Minimal shadows, left-accent borders on cards by category
- 200ms ease transitions on hovers
- Staggered card fade-in on page load

## Principles
1. Data speaks first — metrics are hero elements
2. Category colors create visual grouping without labels
3. White space is a feature, not waste
4. Every chart uses the portfolio palette, never defaults

---

## Quant Finance Adaptation

The quant finance portfolio uses the same Precision Lab design system with one change: the
five category accent colors are remapped to quant domains.

### Quant Category Palette
| Category | CSS class | Color | Hex |
|---|---|---|---|
| Backtesting Engine | `backtesting` | Blue-400 | `#60a5fa` |
| ML Alpha | `mlalpha` | Amber-400 | `#fbbf24` |
| Macro / Regimes | `macro` | Emerald-400 | `#34d399` |
| Options / Volatility | `options` | Violet-400 | `#a78bfa` |
| Crypto / DeFi | `crypto` | Rose-400 | `#fb7185` |

All other design tokens (typography, spacing, dark neutrals, primary teal, plotly palette)
are identical to the ML app. The `format_metric_value` function replaces ML-specific metric
names (accuracy, loss, WER) with quant conventions: Sharpe/Sortino/ICIR at 2 decimals,
percent-formatted drawdowns/returns, integer counts, and dollar-sign PnL values.
