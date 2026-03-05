# Signal Architecture — Design Philosophy

## Visual Identity

**Signal Architecture** is a precision-engineered aesthetic built for quantitative research. The design language draws from trading terminal culture — data-dense, monospaced metrics, dark environments that reduce eye strain during extended analysis sessions — while rejecting the cluttered, noisy feel of legacy systems. Every element earns its pixels.

The visual tone is **clinical confidence**: clean lines, sharp accent colors against deep backgrounds, generous whitespace that lets data breathe. This is not a consumer product. It is a research instrument, and the interface communicates that the person behind it thinks in systems, distributions, and risk-adjusted returns.

## Color System

The palette anchors on a deep navy-black (`#0E1117`) that recedes, pushing data forward. Cards and containers use a slightly elevated `#1A1F2E` to create depth without borders. The primary accent — a teal-mint `#00D4AA` — marks interactive elements and positive values. Seven category colors (burnt orange, slate blue, teal, coral, amber, dodger blue, purple) provide instant visual grouping without labels.

Semantic color is non-negotiable in finance: profit is green (`#00D4AA`), loss is red (`#FF4757`). These are differentiated by brightness as well as hue, ensuring colorblind accessibility. The data visualization palette uses 6-8 hues chosen for perceptual distinctness at small size.

## Typography

Three font families create a clear hierarchy: **Space Grotesk** brings geometric confidence to page titles and hero text. **IBM Plex Sans** handles body copy, labels, and descriptions with high x-height readability. **JetBrains Mono** renders metric numbers, code-like data, and tick values with the aligned columns and tabular figures that quant work demands.

## Spatial Rhythm

Metrics panels use large monospace numbers (1.75rem) with small uppercase labels beneath. Cards breathe with 1.25rem padding and 2px left-border accents. Charts sit in dark containers with consistent 20px padding. The maximum content width avoids the sprawl of full-width layouts while maintaining data density. Vertical rhythm follows an 8px base grid.

## Component Identity

Strategy cards announce their category through a colored left border, present a headline metric in large mono text, and reveal depth on hover with a subtle translateY and accent glow. Charts use transparent or matching backgrounds with bright data lines and minimal gridlines. Parameter controls adopt a terminal-like aesthetic — slim inputs, monospace values, muted borders that sharpen on focus. Transitions are snappy (150-200ms) to feel responsive rather than animated.
