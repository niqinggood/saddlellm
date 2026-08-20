import { afterEach, describe, expect, it, vi } from 'vitest'

import { fetchDemo, planSpatialImage, SpatialApiError } from './spatialApi'
import { jobFixture, jsonResponse } from '../test/fixtures'

afterEach(() => vi.unstubAllGlobals())

describe('spatial API client', () => {
  it('preserves structured server errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      error: { code: 'model_unavailable', message: 'Qwen-VL 尚未配置' },
    }, 409)))

    await expect(fetchDemo()).rejects.toEqual(
      expect.objectContaining<Partial<SpatialApiError>>({
        name: 'SpatialApiError',
        code: 'model_unavailable',
        status: 409,
        message: 'Qwen-VL 尚未配置',
      }),
    )
  })

  it('submits the image and request as multipart form data', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(jobFixture))
    vi.stubGlobal('fetch', fetchMock)
    const file = new File(['pixels'], 'plan.png', { type: 'image/png' })
    const request = {
      start: [2, 2] as [number, number],
      goal: [21, 13] as [number, number],
      route_count: 3,
      instruction: '测试路线',
      free_threshold: 0.72,
      free_is_bright: true,
      uncertainty_band: 0,
      obstacle_dilation: 1,
      max_dimension: 768,
      resolution: 0.5,
      diagonal: true,
      allow_unknown: false,
      clearance_weight: 0.25,
      diversity_weight: 2,
      diversity_radius: 2,
      max_detour_ratio: 2.5,
      semantic_backend: 'disabled' as const,
      use_world_model: false,
      allow_perspective: false,
    }

    await expect(planSpatialImage(file, request)).resolves.toEqual(jobFixture)
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(init.method).toBe('POST')
    expect(init.body).toBeInstanceOf(FormData)
    const form = init.body as FormData
    expect(form.get('image')).toBe(file)
    expect(JSON.parse(String(form.get('request'))).route_count).toBe(3)
  })
})
