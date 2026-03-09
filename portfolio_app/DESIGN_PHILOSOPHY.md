# Signal Lab — Design Philosophy

## Visual Identity

Signal Lab is a dark, data-dense quantitative research dashboard built for an audience that reads order books, not marketing copy. The aesthetic draws from trading terminals and laboratory instrumentation — precise, functional, and unapologetically technical. Every pixel serves a purpose: communicating data, establishing hierarchy, or creating breathing room between dense information blocks.

The interface should feel like opening a proprietary research tool, not a consumer web app. It signals that the author thinks in systems, measures in basis points, and builds infrastructure that runs at microsecond resolution.

## Color System

The palette is anchored in near-black navy (`#0a0e17`) — deep enough to make data pop, warm enough to avoid clinical coldness. Cards and panels use `#111827` with subtle borders, creating layered depth without heavy shadows. Amber (`#f59e0b`) is the primary accent: it marks interactive elements, active states, and category highlights. It reads as "alert" and "signal" without the aggression of red.

Semantic colors are non-negotiable in finance: emerald green (`#10b981`) means profit, red (`#ef4444`) means loss. These must be distinguishable by brightness alone for colorblind accessibility — green is lighter, red is darker against the dark background. The data visualization palette uses 7 distinct hues at controlled saturation to remain readable on dark backgrounds without vibrating.

## Typography & Data Presentation

Three font families create clear hierarchy. DM Serif Display anchors page titles and hero text with editorial confidence — it says "research publication", not "SaaS landing page." DM Sans handles body text, labels, and descriptions with clean readability. JetBrains Mono is the workhorse for everything that matters most: metric values, Sharpe ratios, returns, dates, and code. Monospace numbers align naturally in columns and convey precision.

Metric values are displayed large (1.75rem+) in JetBrains Mono with profit/loss coloring. Labels sit small and muted above or below. This inverted hierarchy — numbers first, context second — mirrors how quant professionals actually scan data.

## Spatial Rhythm & Composition

The layout is data-dense but never cramped. A consistent spacing scale (multiples of 0.5rem) creates rhythm. Category sections breathe with generous vertical gaps between them, while cards within a category pack tightly in 3-column grids. The sidebar is minimal — navigation only, no decoration.

Strategy detail pages use asymmetric two-column layouts (60/40) that give performance charts room to breathe on the left while keeping interactive controls compact on the right. This mirrors the mental model of "analysis on one screen, controls on another" familiar from any trading desk.

## Motion & Interaction

Animations are functional, not decorative. Cards reveal with a subtle staggered fade-up on page load (35ms delay per card, 350ms duration) — fast enough to feel snappy, slow enough to establish reading order. Hover states use 200ms transitions with a slight upward shift and accent border glow. Nothing bounces, pulses, or draws attention away from data. The only motion that matters is the loading spinner when a simulation runs.
