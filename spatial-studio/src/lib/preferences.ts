import type { LayerVisibility, PlanningOptions } from '../types/spatial'

const STORAGE_KEY = 'saddle-spatial-studio/preferences'
const VERSION = 1

export interface StudioPreferences {
  layers: LayerVisibility
  options: Pick<PlanningOptions, 'routeCount' | 'allowUnknown' | 'diagonal'>
}

interface StoredPreferences extends StudioPreferences {
  version: number
}

export const DEFAULT_LAYERS: LayerVisibility = {
  map: true,
  occupancy: true,
  semantics: true,
  routes: true,
}

export const DEFAULT_OPTIONS: PlanningOptions = {
  routeCount: 3,
  resolution: 0.5,
  clearanceWeight: 0.25,
  allowUnknown: false,
  diagonal: true,
  obstacleDilation: 1,
  freeThreshold: 0.72,
  semanticBackend: 'disabled',
  useWorldModel: false,
}

export function loadPreferences(): StudioPreferences {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) throw new Error('missing')
    const parsed = JSON.parse(raw) as StoredPreferences
    if (parsed.version !== VERSION) throw new Error('stale')
    return {
      layers: { ...DEFAULT_LAYERS, ...parsed.layers },
      options: {
        routeCount: parsed.options.routeCount,
        allowUnknown: parsed.options.allowUnknown,
        diagonal: parsed.options.diagonal,
      },
    }
  } catch {
    return {
      layers: DEFAULT_LAYERS,
      options: {
        routeCount: DEFAULT_OPTIONS.routeCount,
        allowUnknown: DEFAULT_OPTIONS.allowUnknown,
        diagonal: DEFAULT_OPTIONS.diagonal,
      },
    }
  }
}

export function savePreferences(
  layers: LayerVisibility,
  options: PlanningOptions,
): void {
  const payload: StoredPreferences = {
    version: VERSION,
    layers,
    options: {
      routeCount: options.routeCount,
      allowUnknown: options.allowUnknown,
      diagonal: options.diagonal,
    },
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
}
