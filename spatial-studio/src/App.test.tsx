import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { App } from './App'
import { capabilitiesFixture, jobFixture, jsonResponse } from './test/fixtures'

function installApiMock() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (path.endsWith('/api/capabilities')) return jsonResponse(capabilitiesFixture)
    if (path.endsWith('/api/demo')) return jsonResponse(jobFixture)
    if (path.includes('/artifacts/source')) {
      return {
        ok: true,
        blob: async () => new Blob(['png'], { type: 'image/png' }),
      } as Response
    }
    if (path.endsWith('/api/plan') && init?.method === 'POST') {
      return jsonResponse({ ...jobFixture, job_id: 'planned', source_name: 'planned-result.png' })
    }
    throw new Error(`Unexpected request: ${path}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('spatial studio', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState(null, '', '/?route=1&tab=routes')
  })

  afterEach(() => vi.unstubAllGlobals())

  it('loads the built-in plan and keeps route selection in the URL', async () => {
    installApiMock()
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByRole('img')).toBeTruthy()
    expect(screen.getByText('平面图与候选路线', { selector: 'title' })).toBeTruthy()
    expect(screen.getByText('几何占用地图：两条走廊连接三个主要空间。')).toBeTruthy()
    const routeTwo = screen.getByRole('button', { name: /路线 2/ })
    await user.click(routeTwo)
    expect(routeTwo.getAttribute('aria-pressed')).toBe('true')
    expect(window.location.search).toContain('route=2')

    await user.click(screen.getByRole('tab', { name: '空间结构' }))
    expect(screen.getByText('occupancy_map')).toBeTruthy()
    expect(window.location.search).toContain('tab=structure')
  })

  it('opens upload validation and can re-plan the current demo image', async () => {
    const fetchMock = installApiMock()
    const user = userEvent.setup()
    render(<App />)
    await screen.findByRole('img')

    await user.click(screen.getByRole('button', { name: '项目' }))
    const dialog = screen.getByRole('dialog', { name: '上传平面图' })
    const input = dialog.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, {
      target: { files: [new File(['bad'], 'notes.txt', { type: 'text/plain' })] },
    })
    expect(screen.getByRole('alert').textContent).toContain('请选择')
    await user.click(screen.getByRole('button', { name: '关闭上传窗口' }))

    await user.click(screen.getByRole('button', { name: '开始规划' }))
    await waitFor(() => {
      expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(3)
    })
    expect(fetchMock.mock.calls.map(([url, init]) => `${String(url)}:${init?.method ?? 'GET'}`))
      .toContain('/api/plan:POST')
    expect(await screen.findByText('planned-result.png')).toBeTruthy()
  })
})
