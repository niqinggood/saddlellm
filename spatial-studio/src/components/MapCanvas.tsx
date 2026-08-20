import { Crosshair, LocateFixed, Maximize, Minus, Plus } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { apiUrl } from '../api/spatialApi'
import { routeDisplayName, routePolyline, routeStyle } from '../lib/routes'
import type {
  Coordinate,
  ImageSource,
  LayerVisibility,
  PlacementMode,
  SpatialJobResponse,
} from '../types/spatial'
import { LayerControls } from './LayerControls'

interface MapCanvasProps {
  image: ImageSource | null
  response: SpatialJobResponse | null
  layers: LayerVisibility
  selectedRoute: number
  start: Coordinate
  goal: Coordinate
  placementMode: PlacementMode
  stale: boolean
  onLayers: (layers: LayerVisibility) => void
  onSelectRoute: (route: number) => void
  onPoint: (point: Coordinate) => void
  onPlacementMode: (mode: PlacementMode) => void
}

interface ViewBoxState {
  x: number
  y: number
  width: number
  height: number
}

export function MapCanvas({
  image,
  response,
  layers,
  selectedRoute,
  start,
  goal,
  placementMode,
  stale,
  onLayers,
  onSelectRoute,
  onPoint,
  onPlacementMode,
}: MapCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)

  useEffect(() => setZoom(1), [image?.url])

  const viewBox = useMemo<ViewBoxState>(() => {
    const width = image?.width ?? 1
    const height = image?.height ?? 1
    const visibleWidth = width / zoom
    const visibleHeight = height / zoom
    return {
      x: (width - visibleWidth) / 2,
      y: (height - visibleHeight) / 2,
      width: visibleWidth,
      height: visibleHeight,
    }
  }, [image?.height, image?.width, zoom])

  const routes = response?.result.routes ?? []
  const worldForecastPolyline = useMemo(() => {
    const forecast = response?.result.world_model?.forecast.predicted_points
    const map = response?.result.map
    if (!forecast?.length || !map || !image) return ''
    return forecast
      .map(([x, y]) => `${(x / Math.max(map.width, 1)) * image.width},${(y / Math.max(map.height, 1)) * image.height}`)
      .join(' ')
  }, [image, response])
  const markerRadius = image ? Math.max(0.55, Math.min(image.width, image.height) * 0.025) : 0.55
  const mapFontSize = markerRadius * 1.15
  const orderedRoutes = useMemo(
    () => [...routes].sort((first, second) => Number(first.id === selectedRoute) - Number(second.id === selectedRoute)),
    [routes, selectedRoute],
  )

  function pointerCoordinate(event: React.PointerEvent<SVGSVGElement>): Coordinate | null {
    if (!image) return null
    const svg = svgRef.current
    if (!svg) return null
    const matrix = svg.getScreenCTM()
    if (matrix && typeof DOMPoint !== 'undefined') {
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
      return [
        Math.min(Math.max(point.x, 0), image.width - 0.001),
        Math.min(Math.max(point.y, 0), image.height - 0.001),
      ]
    }
    const bounds = svg.getBoundingClientRect()
    if (!bounds.width || !bounds.height) return null
    return [
      viewBox.x + ((event.clientX - bounds.left) / bounds.width) * viewBox.width,
      viewBox.y + ((event.clientY - bounds.top) / bounds.height) * viewBox.height,
    ]
  }

  function handleCanvasPointer(event: React.PointerEvent<SVGSVGElement>) {
    if (!placementMode) return
    const coordinate = pointerCoordinate(event)
    if (coordinate) onPoint(coordinate)
  }

  const title = response?.result.analysis.summary || '上传平面图并设置起点与终点'
  const entities = response?.result.analysis.entities ?? []

  return (
    <section className="map-workspace" id="map-workspace" aria-labelledby="map-heading">
      <div className="map-toolbar-row">
        <div>
          <h1 id="map-heading">空间构成与候选路线</h1>
          <p>{title}</p>
        </div>
        <LayerControls layers={layers} hasPlan={Boolean(response)} onChange={onLayers} />
      </div>

      <div className={placementMode ? 'map-frame placing' : 'map-frame'}>
        {image ? (
          <svg
            ref={svgRef}
            className="route-map"
            viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
            role="img"
            aria-labelledby="route-map-title route-map-description"
            onPointerUp={handleCanvasPointer}
          >
            <title id="route-map-title">平面图与候选路线</title>
            <desc id="route-map-description">
              起点坐标 {start[0].toFixed(1)}, {start[1].toFixed(1)}；终点坐标 {goal[0].toFixed(1)}, {goal[1].toFixed(1)}；
              当前显示 {routes.length} 条候选路线。详细指标位于路线检查器中。
            </desc>
            <rect width={image.width} height={image.height} fill="#ffffff" />
            {layers.map ? (
              <image
                href={image.url}
                width={image.width}
                height={image.height}
                preserveAspectRatio="none"
              />
            ) : null}
            {response && layers.occupancy ? (
              <image
                className="occupancy-layer"
                href={apiUrl(response.artifacts.mask)}
                width={image.width}
                height={image.height}
                preserveAspectRatio="none"
              />
            ) : null}
            {response && layers.semantics
              ? entities.map((entity) => {
                  const [x1, y1, x2, y2] = entity.bbox
                  const x = (x1 / 1000) * image.width
                  const y = (y1 / 1000) * image.height
                  const width = ((x2 - x1) / 1000) * image.width
                  const height = ((y2 - y1) / 1000) * image.height
                  return (
                    <g key={`${entity.name}-${x1}-${y1}`} className="semantic-entity">
                      <rect x={x} y={y} width={width} height={height} />
                      <text x={x} y={Math.max(mapFontSize, y - markerRadius)} fontSize={mapFontSize}>
                        {entity.name}
                      </text>
                    </g>
                  )
                })
              : null}
            {worldForecastPolyline && layers.routes ? (
              <g className="world-model-forecast" aria-label="world model imagined rollout">
                <polyline points={worldForecastPolyline} />
              </g>
            ) : null}
            {response && layers.routes
              ? orderedRoutes.map((route) => {
                  const index = routes.findIndex((value) => value.id === route.id)
                  const style = routeStyle(index)
                  const selected = route.id === selectedRoute
                  const midpoint = route.source_points[Math.floor(route.source_points.length / 2)]
                  return (
                    <g
                      key={route.id}
                      className={selected ? 'route-mark selected' : 'route-mark'}
                      onPointerUp={(event) => {
                        event.stopPropagation()
                        onSelectRoute(route.id)
                      }}
                    >
                      <polyline className="route-halo" points={routePolyline(route)} />
                      <polyline
                        className="route-line"
                        points={routePolyline(route)}
                        stroke={style.color}
                        strokeDasharray={style.dash}
                      />
                      {midpoint ? (
                        <g className="route-map-label" transform={`translate(${midpoint[0]} ${midpoint[1]})`}>
                          <circle r={selected ? markerRadius * 1.1 : markerRadius} fill={style.color} />
                          <text fontSize={mapFontSize} aria-label={`路线 ${route.id}，${routeDisplayName(route, index)}`}>R{route.id}</text>
                        </g>
                      ) : null}
                    </g>
                  )
                })
              : null}
            <MapPoint point={start} kind="start" label="S" radius={markerRadius} fontSize={mapFontSize} />
            <MapPoint point={goal} kind="goal" label="G" radius={markerRadius} fontSize={mapFontSize} />
          </svg>
        ) : (
          <button className="empty-map" type="button" onClick={() => onPlacementMode(null)}>
            <Crosshair aria-hidden="true" size={34} />
            <strong>选择一张俯视图或户型图</strong>
            <span>上传后可直接在图上点击起点和终点</span>
          </button>
        )}

        {image ? (
          <div className="map-zoom-controls" aria-label="地图缩放">
            <button type="button" onClick={() => setZoom(1)} aria-label="重置视图">
              <LocateFixed aria-hidden="true" />
            </button>
            <button type="button" onClick={() => setZoom((value) => Math.min(4, value * 1.25))} aria-label="放大">
              <Plus aria-hidden="true" />
            </button>
            <button type="button" onClick={() => setZoom((value) => Math.max(1, value / 1.25))} aria-label="缩小">
              <Minus aria-hidden="true" />
            </button>
            <button type="button" onClick={() => setZoom(1)} aria-label="适合窗口">
              <Maximize aria-hidden="true" />
            </button>
          </div>
        ) : null}

        {placementMode ? (
          <div className="placement-banner" role="status">
            <Crosshair aria-hidden="true" size={17} />
            点击地图设置{placementMode === 'start' ? '起点' : '终点'}
            <button type="button" onClick={() => onPlacementMode(null)}>取消</button>
          </div>
        ) : null}
        {stale ? <div className="stale-banner">显示的是上一次成功结果</div> : null}
      </div>
    </section>
  )
}

function MapPoint({
  point,
  kind,
  label,
  radius,
  fontSize,
}: {
  point: Coordinate
  kind: 'start' | 'goal'
  label: string
  radius: number
  fontSize: number
}) {
  return (
    <g className={`map-point ${kind}`} transform={`translate(${point[0]} ${point[1]})`}>
      <circle r={radius} />
      <text fontSize={fontSize}>{label}</text>
    </g>
  )
}
