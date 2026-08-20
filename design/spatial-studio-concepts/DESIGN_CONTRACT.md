# Saddle Spatial Studio design contract

Approval status: **approved**  
Approval response: `好的 开始吧`  
Approval date: 2026-08-09  
Approval scope: desktop, mobile portrait, mobile landscape, and implementation direction

## Approved references

- `desktop-v1.png` — 1536 × 1024 desktop workspace
- `mobile-portrait-v1.png` — 430 × 932 mobile portrait workspace
- `mobile-landscape-v1.png` — 932 × 430 mobile landscape workspace

## Evidence lock

The primary claim is that the system converts a top-down image into a traversability model and presents multiple valid, visibly distinct routes between a user-selected start and goal. The interface must never imply that semantic labels, geometric certainty, or learned world-model scores exist when their corresponding model is disabled.

Truth invariants:

- walls and blocked occupancy remain visually distinct from routes;
- route identity is encoded by R1/R2/R3 labels as well as color;
- start and goal remain explicitly labeled S/G;
- length, turns, minimum clearance, risk, and optional learned return stay data-bound;
- unknown regions and caveats stay visible in the uncertainty surface;
- Qwen-VL semantics and RSSM scoring report their actual enabled/disabled state;
- a perspective image is not presented as a complete navigation map without an explicit approximation warning.

## Locked visual and interaction elements

- True white and cool light-gray surfaces; deep navy text.
- Cobalt blue primary route, orange second route, green third route, restrained red occupancy.
- Thin borders, 8–12 px radii, restrained elevation, consistent outline icons.
- Desktop reading order: app bar → workflow/settings rail → dominant map canvas → route/space inspector → status strip.
- Mobile portrait reading order: app bar → compact command bar → map → partially open route sheet → primary planning action.
- Mobile landscape reading order: app bar → map and right route inspector split.
- The map stays visible when mobile controls or the route sheet are open.
- Route selection emphasizes one path without hiding alternatives.
- Upload, start/goal placement, planning, layer visibility, zoom/reset, inspector tabs, route selection, and export are real controls.
- Minimum 44 px coarse-pointer target for primary mobile controls.
- Essential values never depend on hover or color alone.

## Flexible implementation details

- Exact breakpoints, spacing increments, typography fallback stack, SVG line width, and internal component boundaries.
- Small copy corrections needed for factual state, localization, accessibility, or unavailable model features.
- Inspector presentation may use a drawer or sheet when the visual viewport is constrained, provided the map remains quickly visible/restorable.

## Code-owned layers

All uploaded imagery, occupancy masks, semantic boxes, route paths, markers, labels, metrics, controls, warnings, and export links remain code-native and data-bound. Concept images are implementation references only and are never shipped as interface screenshots.

## State and persistence

- URL state: selected route and active inspector tab.
- Local preference state: visible layers and last non-sensitive planning options, with a versioned schema.
- Ephemeral state: uploaded image bytes, start/goal selection mode, pending request, transient errors.
- Server state: generated job artifacts and optional loaded Qwen-VL/RSSM capability status.

## Accessibility and fallback

- SVG has an accessible title/description and a tabular route alternative.
- Keyboard controls can select route, set explicit coordinates, reset view, open settings, plan, and export.
- Reduced motion removes nonessential transitions.
- API failure preserves the last known result and labels it stale rather than blanking the map.
- Mobile portrait, mobile landscape, and desktop share the same route facts and caveats.
