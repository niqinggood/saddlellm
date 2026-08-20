import type { SpatialRoute } from '../types/spatial'

export const ROUTE_STYLES = [
  { color: '#0868d7', dash: undefined, name: '最短' },
  { color: '#e85d04', dash: '12 7', name: '上方备选' },
  { color: '#07875c', dash: '4 6', name: '下方备选' },
  { color: '#7548a8', dash: '14 5 3 5', name: '备选' },
  { color: '#a61e4d', dash: '7 7', name: '备选' },
  { color: '#697500', dash: '3 6', name: '备选' },
] as const

export function routeStyle(index: number) {
  return ROUTE_STYLES[index % ROUTE_STYLES.length]
}

export function routeDisplayName(route: SpatialRoute, index: number): string {
  if (route.label === 'world-model-best') return '世界模型优选'
  if (route.label === 'world-model-alternative') return '世界模型备选'
  return routeStyle(index).name
}

export function routePolyline(route: SpatialRoute): string {
  return route.source_points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
}
