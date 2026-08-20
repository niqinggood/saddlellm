import type { InspectorTab } from '../types/spatial'

export interface ViewState {
  route: number
  tab: InspectorTab
}

const TABS = new Set<InspectorTab>(['routes', 'structure', 'uncertainty'])

export function parseViewState(search: string): ViewState {
  const params = new URLSearchParams(search)
  const route = Number.parseInt(params.get('route') ?? '1', 10)
  const tabValue = params.get('tab') as InspectorTab | null
  return {
    route: Number.isFinite(route) && route > 0 ? route : 1,
    tab: tabValue && TABS.has(tabValue) ? tabValue : 'routes',
  }
}

export function writeViewState(state: ViewState): void {
  const url = new URL(window.location.href)
  url.searchParams.set('route', String(state.route))
  url.searchParams.set('tab', state.tab)
  window.history.replaceState(null, '', url)
}
