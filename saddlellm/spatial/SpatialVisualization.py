"""Self-contained HTML visualization for spatial composition and routes."""

from __future__ import annotations

import base64
import html
import json
import os
from typing import Any

import numpy as np

from .SpatialPlanner import OccupancyGrid, RouteCandidate
from .SpatialWorldModel import SpatialPlanResult


_ROUTE_COLORS = [
    (0, 105, 194),
    (196, 81, 0),
    (8, 127, 91),
    (122, 79, 179),
    (166, 30, 77),
    (102, 112, 0),
]


def save_spatial_plan_json(
    result: SpatialPlanResult,
    output_path: str,
    include_grid: bool = False,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            result.to_dict(include_grid=include_grid),
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    return os.path.abspath(output_path)


def render_spatial_plan_html(result: SpatialPlanResult, output_path: str) -> str:
    """Write an offline, responsive route overlay with an accessible table."""

    image_uri = _file_data_uri(result.image_path)
    mask_uri = _occupancy_mask_data_uri(result.grid)
    route_markup = "\n".join(
        _route_svg(index, route) for index, route in enumerate(result.routes)
    )
    route_controls = "\n".join(
        _route_control(index, route) for index, route in enumerate(result.routes)
    )
    route_rows = "\n".join(
        _route_table_row(index, route) for index, route in enumerate(result.routes)
    )
    entity_markup = "\n".join(
        _entity_svg(entity, result.grid.width, result.grid.height)
        for entity in result.analysis.entities
    )
    caveats = result.analysis.caveats + [
        f"Unknown region: {value}" for value in result.analysis.unknown_regions
    ]
    caveat_markup = "".join(
        f"<li>{html.escape(str(value))}</li>" for value in caveats
    ) or "<li>No additional caveats were reported.</li>"
    start_x, start_y = result.start
    goal_x, goal_y = result.goal
    summary = html.escape(result.analysis.summary or "Spatial route plan")
    instruction = html.escape(result.instruction or "No natural-language constraint")
    confidence = f"{result.analysis.confidence * 100:.0f}%"
    title = "Spatial world-model route plan"
    output = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --page: #f7f8fb;
      --surface: #ffffff;
      --text: #172033;
      --muted: #5c667a;
      --border: #cbd2df;
      --route-1: #0069c2;
      --route-2: #c45100;
      --route-3: #087f5b;
      --route-4: #7a4fb3;
      --route-5: #a61e4d;
      --route-6: #667000;
      --start: #087f5b;
      --goal: #b42318;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --page: #10141d;
        --surface: #181e2a;
        --text: #edf1f8;
        --muted: #b4bed0;
        --border: #465168;
        --route-1: #5db3ff;
        --route-2: #ff9a52;
        --route-3: #55d6a9;
        --route-4: #c5a2ff;
        --route-5: #ff8daf;
        --route-6: #d3df63;
        --start: #55d6a9;
        --goal: #ff8075;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--page); color: var(--text); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ font-weight: 500; margin: 0; }}
    h1 {{ font-size: clamp(1.45rem, 3vw, 2rem); }}
    h2 {{ font-size: 1.05rem; margin-bottom: 12px; }}
    p {{ margin: 6px 0; }}
    .subtitle {{ color: var(--muted); max-width: 76ch; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric, .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }}
    .metric {{ padding: 14px; }}
    .metric span {{ display: block; color: var(--muted); font-size: .82rem; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 1.25rem; font-weight: 500; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 2fr) minmax(250px, .75fr); gap: 16px; align-items: start; }}
    .panel {{ padding: 14px; }}
    .map-frame {{ position: relative; width: 100%; aspect-ratio: {result.grid.width} / {result.grid.height}; border: 1px solid var(--border); background: var(--surface); overflow: hidden; }}
    .map-frame svg {{ display: block; width: 100%; height: 100%; }}
    .map-frame image {{ pointer-events: none; }}
    .occupancy-mask {{ opacity: .40; }}
    .route {{ fill: none; stroke-width: 3px; stroke-linecap: round; stroke-linejoin: round; vector-effect: non-scaling-stroke; }}
    .route-1 {{ stroke: var(--route-1); }}
    .route-2 {{ stroke: var(--route-2); stroke-dasharray: 9 5; }}
    .route-3 {{ stroke: var(--route-3); stroke-dasharray: 3 4; }}
    .route-4 {{ stroke: var(--route-4); stroke-dasharray: 12 4 3 4; }}
    .route-5 {{ stroke: var(--route-5); stroke-dasharray: 6 6; }}
    .route-6 {{ stroke: var(--route-6); stroke-dasharray: 2 5; }}
    .route-label {{ fill: var(--text); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; font-size: max(11px, 2%); font-weight: 500; }}
    .entity {{ fill: none; stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; }}
    .entity-label {{ fill: var(--text); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; font-size: max(10px, 1.6%); }}
    .point {{ stroke: var(--surface); stroke-width: 2; vector-effect: non-scaling-stroke; }}
    .start {{ fill: var(--start); }}
    .goal {{ fill: var(--goal); }}
    .controls {{ display: grid; gap: 10px; }}
    .control {{ display: grid; grid-template-columns: auto 1fr; gap: 9px; align-items: start; padding: 4px 0; }}
    .control input {{ width: 20px; height: 20px; margin: 0; }}
    .control strong {{ font-weight: 500; }}
    .control small {{ display: block; color: var(--muted); margin-top: 2px; }}
    .swatch {{ display: inline-block; width: 22px; height: 4px; margin-right: 6px; vertical-align: middle; background: var(--route-1); }}
    .swatch-2 {{ background: var(--route-2); }}
    .swatch-3 {{ background: var(--route-3); }}
    .swatch-4 {{ background: var(--route-4); }}
    .swatch-5 {{ background: var(--route-5); }}
    .swatch-6 {{ background: var(--route-6); }}
    .table-wrap {{ overflow-x: auto; margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 500; }}
    details {{ margin-top: 16px; }}
    summary {{ cursor: pointer; font-weight: 500; }}
    li {{ margin: 5px 0; }}
    .sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }}
    @media (max-width: 760px) {{
      main {{ padding: 16px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .layout {{ grid-template-columns: 1fr; }}
      .route {{ stroke-width: 3px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{title}</h1>
    <p class="subtitle">{summary}</p>
    <p class="subtitle"><strong>Constraint:</strong> {instruction}</p>
  </header>
  <section class="metrics" aria-label="Plan summary">
    <div class="metric"><span>Candidate routes</span><strong>{len(result.routes)}</strong></div>
    <div class="metric"><span>Map grid</span><strong>{result.grid.width} × {result.grid.height}</strong></div>
    <div class="metric"><span>Semantic confidence</span><strong>{confidence}</strong></div>
  </section>
  <section class="layout">
    <div class="panel">
      <h2>Spatial composition and routes</h2>
      <div class="map-frame">
        <svg viewBox="0 0 {result.grid.width} {result.grid.height}" role="img" aria-labelledby="map-title map-desc" preserveAspectRatio="none">
          <title id="map-title">Navigation map with {len(result.routes)} candidate routes</title>
          <desc id="map-desc">Start at {result.start}; goal at {result.goal}. Routes are also described in the table below.</desc>
          <image href="{image_uri}" x="0" y="0" width="{result.grid.width}" height="{result.grid.height}" preserveAspectRatio="none" />
          <image class="occupancy-mask" data-mask href="{mask_uri}" x="0" y="0" width="{result.grid.width}" height="{result.grid.height}" preserveAspectRatio="none" />
          <g data-entities>{entity_markup}</g>
          <g data-routes>{route_markup}</g>
          <circle class="point start" cx="{start_x}" cy="{start_y}" r="{max(2, min(result.grid.width, result.grid.height) * .012):.2f}" />
          <circle class="point goal" cx="{goal_x}" cy="{goal_y}" r="{max(2, min(result.grid.width, result.grid.height) * .012):.2f}" />
        </svg>
      </div>
    </div>
    <aside class="panel">
      <h2>Visible layers</h2>
      <div class="controls">
        <label class="control"><input type="checkbox" data-toggle-mask checked><span><strong>Occupancy mask</strong><small>Blocked red; unknown amber</small></span></label>
        <label class="control"><input type="checkbox" data-toggle-entities checked><span><strong>Semantic entities</strong><small>Vision-model grounding</small></span></label>
        {route_controls}
      </div>
    </aside>
  </section>
  <section class="panel table-wrap" aria-labelledby="route-table-title">
    <h2 id="route-table-title">Route comparison</h2>
    <table>
      <thead><tr><th>Route</th><th>Type</th><th>Length</th><th>Turns</th><th>Min clearance</th><th>Risk</th><th>World-model return</th></tr></thead>
      <tbody>{route_rows}</tbody>
    </table>
  </section>
  <details class="panel">
    <summary>Uncertainty and caveats</summary>
    <ul>{caveat_markup}</ul>
  </details>
</main>
<script>
(() => {{
  const setVisible = (selector, visible) => {{
    document.querySelectorAll(selector).forEach(node => {{ node.style.display = visible ? '' : 'none'; }});
  }};
  document.querySelectorAll('[data-route-toggle]').forEach(input => {{
    input.addEventListener('change', () => setVisible(`[data-route="${{input.dataset.routeToggle}}"]`, input.checked));
  }});
  document.querySelector('[data-toggle-mask]').addEventListener('change', event => setVisible('[data-mask]', event.target.checked));
  document.querySelector('[data-toggle-entities]').addEventListener('change', event => setVisible('[data-entities]', event.target.checked));
}})();
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(output)
    return os.path.abspath(output_path)


def render_spatial_plan_png(
    result: SpatialPlanResult,
    output_path: str,
    width: int = 1280,
) -> str:
    """Render a shareable static preview of the spatial plan.

    The HTML report remains the interactive artifact; this PNG is deliberately
    dependency-light so command-line and headless environments can still show
    the actual geometry and candidate routes.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise ImportError("Pillow is required for spatial visualization") from error
    if width < 720:
        raise ValueError("Spatial preview width must be at least 720 pixels")

    canvas_height = max(720, int(round(width * 0.625)))
    canvas = Image.new("RGB", (width, canvas_height), (247, 248, 251))
    draw = ImageDraw.Draw(canvas)
    margin = max(24, width // 32)
    header_height = 104
    gutter = max(20, width // 55)
    sidebar_width = max(280, int(width * 0.27))
    map_width = width - margin * 2 - gutter - sidebar_width
    map_top = header_height + 24
    map_available_height = canvas_height - map_top - margin
    aspect = result.grid.width / result.grid.height
    map_height = min(map_available_height, int(round(map_width / aspect)))
    map_width = min(map_width, int(round(map_height * aspect)))
    map_left = margin
    map_top += max(0, (map_available_height - map_height) // 2)
    sidebar_left = width - margin - sidebar_width

    title_font = _load_font(ImageFont, max(25, width // 42), bold=True)
    body_font = _load_font(ImageFont, max(15, width // 80))
    small_font = _load_font(ImageFont, max(13, width // 96))
    metric_font = _load_font(ImageFont, max(20, width // 58), bold=True)

    draw.text((margin, 24), "Spatial world-model route plan", fill=(23, 32, 51), font=title_font)
    subtitle = result.analysis.summary or "Spatial composition and candidate routes"
    draw.text((margin, 66), _clip_text(draw, subtitle, body_font, width - margin * 2), fill=(92, 102, 122), font=body_font)

    with Image.open(result.image_path) as source:
        floorplan = source.convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    floorplan = floorplan.resize((map_width, map_height), resampling.NEAREST)
    canvas.paste(floorplan, (map_left, map_top))

    mask = np.zeros((result.grid.height, result.grid.width, 4), dtype=np.uint8)
    mask[result.grid.cells == OccupancyGrid.BLOCKED] = [205, 40, 40, 92]
    mask[result.grid.cells == OccupancyGrid.UNKNOWN] = [224, 156, 31, 100]
    mask_image = Image.fromarray(mask, mode="RGBA").resize(
        (map_width, map_height), resampling.NEAREST
    )
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(mask_image, (map_left, map_top))
    canvas = canvas_rgba.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (map_left, map_top, map_left + map_width, map_top + map_height),
        outline=(203, 210, 223),
        width=2,
    )

    scale_x = map_width / result.grid.width
    scale_y = map_height / result.grid.height

    def screen_point(point: Any) -> tuple[float, float]:
        return (
            map_left + (float(point[0]) + 0.5) * scale_x,
            map_top + (float(point[1]) + 0.5) * scale_y,
        )

    for index, route in reversed(list(enumerate(result.routes))):
        points = [screen_point(point) for point in route.points]
        color = _ROUTE_COLORS[index % len(_ROUTE_COLORS)]
        draw.line(points, fill=(255, 255, 255), width=max(8, width // 110), joint="curve")
        draw.line(points, fill=color, width=max(4, width // 210), joint="curve")
        label_point = points[len(points) // 2]
        radius = max(10, width // 100)
        draw.ellipse(
            (
                label_point[0] - radius,
                label_point[1] - radius,
                label_point[0] + radius,
                label_point[1] + radius,
            ),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        label = f"R{index + 1}"
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (label_point[0] - (box[2] - box[0]) / 2, label_point[1] - (box[3] - box[1]) / 2 - 1),
            label,
            fill=(255, 255, 255),
            font=small_font,
        )

    for point, color, label in (
        (result.start, (8, 127, 91), "S"),
        (result.goal, (180, 35, 24), "G"),
    ):
        x, y = screen_point(point)
        radius = max(11, width // 90)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=3)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 1), label, fill=(255, 255, 255), font=small_font)

    panel_bottom = map_top + map_height
    draw.rounded_rectangle(
        (sidebar_left, map_top, width - margin, panel_bottom),
        radius=12,
        fill=(255, 255, 255),
        outline=(203, 210, 223),
        width=2,
    )
    panel_x = sidebar_left + 20
    panel_y = map_top + 20
    draw.text((panel_x, panel_y), "Plan summary", fill=(23, 32, 51), font=metric_font)
    panel_y += 44
    for name, value in (
        ("Candidate routes", str(len(result.routes))),
        ("Occupancy grid", f"{result.grid.width} x {result.grid.height}"),
        ("Semantic confidence", f"{result.analysis.confidence * 100:.0f}%"),
    ):
        draw.text((panel_x, panel_y), name, fill=(92, 102, 122), font=small_font)
        draw.text((width - margin - 20, panel_y), value, fill=(23, 32, 51), font=body_font, anchor="ra")
        panel_y += 34
    panel_y += 8
    draw.line((panel_x, panel_y, width - margin - 20, panel_y), fill=(220, 224, 232), width=1)
    panel_y += 20
    draw.text((panel_x, panel_y), "Route comparison", fill=(23, 32, 51), font=metric_font)
    panel_y += 42
    for index, route in enumerate(result.routes):
        color = _ROUTE_COLORS[index % len(_ROUTE_COLORS)]
        draw.rounded_rectangle((panel_x, panel_y + 4, panel_x + 28, panel_y + 10), radius=3, fill=color)
        draw.text((panel_x + 40, panel_y), f"R{index + 1}  {route.label}", fill=(23, 32, 51), font=body_font)
        panel_y += 25
        detail = f"{route.length:.2f} units  |  {route.turns} turns  |  risk {route.risk:.3f}"
        draw.text((panel_x + 40, panel_y), detail, fill=(92, 102, 122), font=small_font)
        panel_y += 45

    legend_y = min(panel_bottom - 62, panel_y + 4)
    draw.rectangle((panel_x, legend_y + 3, panel_x + 16, legend_y + 19), fill=(205, 40, 40))
    draw.text((panel_x + 25, legend_y), "Blocked", fill=(92, 102, 122), font=small_font)
    draw.ellipse((panel_x + 110, legend_y + 3, panel_x + 126, legend_y + 19), fill=(8, 127, 91))
    draw.text((panel_x + 135, legend_y), "Start", fill=(92, 102, 122), font=small_font)
    draw.ellipse((panel_x + 202, legend_y + 3, panel_x + 218, legend_y + 19), fill=(180, 35, 24))
    draw.text((panel_x + 227, legend_y), "Goal", fill=(92, 102, 122), font=small_font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return os.path.abspath(output_path)


def save_occupancy_mask_png(grid: OccupancyGrid, output_path: str) -> str:
    """Save a transparent blocked/unknown overlay for browser map rendering."""

    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Pillow is required for spatial visualization") from error
    rgba = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    rgba[grid.cells == grid.BLOCKED] = [150, 20, 28, 184]
    rgba[grid.cells == grid.UNKNOWN] = [224, 156, 31, 150]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG")
    return os.path.abspath(output_path)


def _load_font(image_font: Any, size: int, bold: bool = False) -> Any:
    names = ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in names:
        try:
            return image_font.truetype(name, size=size)
        except OSError:
            continue
    return image_font.load_default()


def _clip_text(draw: Any, value: str, font: Any, maximum_width: int) -> str:
    value = str(value)
    if draw.textlength(value, font=font) <= maximum_width:
        return value
    suffix = "..."
    while value and draw.textlength(value + suffix, font=font) > maximum_width:
        value = value[:-1]
    return value + suffix


def _route_svg(index: int, route: RouteCandidate) -> str:
    route_number = index + 1
    css_index = index % 6 + 1
    points = " ".join(f"{x},{y}" for x, y in route.points)
    midpoint = route.points[len(route.points) // 2]
    return (
        f'<g data-route="{route_number}">'
        f'<polyline class="route route-{css_index}" points="{points}" />'
        f'<text class="route-label" x="{midpoint[0]}" y="{midpoint[1]}">R{route_number}</text>'
        "</g>"
    )


def _route_control(index: int, route: RouteCandidate) -> str:
    number = index + 1
    css_index = index % 6 + 1
    predicted = (
        f" · return {route.predicted_return:.2f}"
        if route.predicted_return is not None
        else ""
    )
    return (
        '<label class="control">'
        f'<input type="checkbox" data-route-toggle="{number}" checked>'
        f'<span><strong><i class="swatch swatch-{css_index}"></i>Route {number}</strong>'
        f'<small>{html.escape(route.label)} · {route.length:.2f} units · risk {route.risk:.3f}{predicted}</small></span>'
        "</label>"
    )


def _route_table_row(index: int, route: RouteCandidate) -> str:
    predicted = "—" if route.predicted_return is None else f"{route.predicted_return:.3f}"
    return (
        f"<tr><td>R{index + 1}</td><td>{html.escape(route.label)}</td>"
        f"<td>{route.length:.2f}</td><td>{route.turns}</td>"
        f"<td>{route.minimum_clearance:.2f}</td><td>{route.risk:.3f}</td>"
        f"<td>{predicted}</td></tr>"
    )


def _entity_svg(entity: Any, width: int, height: int) -> str:
    x1, y1, x2, y2 = entity.bbox
    x = x1 / 1000.0 * width
    y = y1 / 1000.0 * height
    box_width = max(0.0, (x2 - x1) / 1000.0 * width)
    box_height = max(0.0, (y2 - y1) / 1000.0 * height)
    label = html.escape(f"{entity.name} ({entity.category})")
    return (
        f'<g><rect class="entity" x="{x:.2f}" y="{y:.2f}" width="{box_width:.2f}" height="{box_height:.2f}" />'
        f'<text class="entity-label" x="{x:.2f}" y="{max(1.0, y - 1):.2f}">{label}</text></g>'
    )


def _file_data_uri(path: str) -> str:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Pillow is required for spatial visualization") from error
    from io import BytesIO

    with Image.open(path) as image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _occupancy_mask_data_uri(grid: OccupancyGrid) -> str:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Pillow is required for spatial visualization") from error
    rgba = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    blocked = grid.cells == grid.BLOCKED
    unknown = grid.cells == grid.UNKNOWN
    rgba[blocked] = [205, 40, 40, 155]
    rgba[unknown] = [224, 156, 31, 135]
    image = Image.fromarray(rgba, mode="RGBA")
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
