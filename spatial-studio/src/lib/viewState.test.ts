import { beforeEach, describe, expect, it } from 'vitest'

import { DEFAULT_LAYERS, DEFAULT_OPTIONS, loadPreferences, savePreferences } from './preferences'
import { parseViewState, writeViewState } from './viewState'

describe('view state', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/studio?project=alpha')
  })

  it('normalizes invalid route and tab values', () => {
    expect(parseViewState('?route=-3&tab=missing')).toEqual({ route: 1, tab: 'routes' })
    expect(parseViewState('?route=2&tab=uncertainty')).toEqual({ route: 2, tab: 'uncertainty' })
  })

  it('writes shareable inspector state without dropping unrelated parameters', () => {
    writeViewState({ route: 3, tab: 'structure' })
    expect(window.location.search).toContain('project=alpha')
    expect(window.location.search).toContain('route=3')
    expect(window.location.search).toContain('tab=structure')
  })
})

describe('preferences', () => {
  beforeEach(() => window.localStorage.clear())

  it('falls back to stable defaults for missing or corrupt data', () => {
    expect(loadPreferences()).toEqual({
      layers: DEFAULT_LAYERS,
      options: {
        routeCount: DEFAULT_OPTIONS.routeCount,
        allowUnknown: DEFAULT_OPTIONS.allowUnknown,
        diagonal: DEFAULT_OPTIONS.diagonal,
      },
    })
    window.localStorage.setItem('saddle-spatial-studio/preferences', '{broken')
    expect(loadPreferences().options.routeCount).toBe(3)
  })

  it('persists user-controlled layers and planning choices', () => {
    const layers = { ...DEFAULT_LAYERS, semantics: false }
    const options = { ...DEFAULT_OPTIONS, routeCount: 5, allowUnknown: true, diagonal: false }
    savePreferences(layers, options)
    expect(loadPreferences()).toEqual({
      layers,
      options: { routeCount: 5, allowUnknown: true, diagonal: false },
    })
  })
})
