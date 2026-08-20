import type {
  SpatialCapabilities,
  SpatialJobResponse,
  SpatialPlanRequest,
  StudioError,
} from '../types/spatial'

const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

export class SpatialApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(error: StudioError, status: number) {
    super(error.message)
    this.name = 'SpatialApiError'
    this.code = error.code
    this.status = status
  }
}

export function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init)
  const contentType = response.headers.get('content-type') ?? ''
  const payload = contentType.includes('application/json')
    ? await response.json()
    : null
  if (!response.ok) {
    const structured = payload?.error as StudioError | undefined
    const detail = typeof payload?.detail === 'string' ? payload.detail : undefined
    throw new SpatialApiError(
      structured ?? {
        code: 'request_failed',
        message: detail ?? `请求失败（HTTP ${response.status}）`,
      },
      response.status,
    )
  }
  return payload as T
}

export function fetchCapabilities(signal?: AbortSignal): Promise<SpatialCapabilities> {
  return requestJson('/api/capabilities', { signal })
}

export function fetchDemo(signal?: AbortSignal): Promise<SpatialJobResponse> {
  return requestJson('/api/demo', { signal })
}

export function planSpatialImage(
  file: File,
  request: SpatialPlanRequest,
  signal?: AbortSignal,
): Promise<SpatialJobResponse> {
  const form = new FormData()
  form.set('image', file)
  form.set('request', JSON.stringify(request))
  return requestJson('/api/plan', {
    method: 'POST',
    body: form,
    signal,
  })
}

export async function responseImageAsFile(response: SpatialJobResponse): Promise<File> {
  const image = await fetch(apiUrl(response.image.url))
  if (!image.ok) throw new Error('无法读取当前平面图')
  const blob = await image.blob()
  return new File([blob], response.source_name || 'floorplan.png', {
    type: blob.type || 'image/png',
  })
}
