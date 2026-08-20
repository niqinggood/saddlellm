import { AlertCircle, CheckCircle2, CloudOff, LoaderCircle, Map } from 'lucide-react'

import type { SpatialJobResponse } from '../types/spatial'

interface StatusBarProps {
  response: SpatialJobResponse | null
  planning: boolean
  online: boolean
  stale: boolean
  error: string | null
}

export function StatusBar({ response, planning, online, stale, error }: StatusBarProps) {
  let message = '等待平面图'
  let Icon = Map
  let tone = 'neutral'
  if (planning) {
    message = '正在构建占用图并搜索候选路线'
    Icon = LoaderCircle
    tone = 'working'
  } else if (error) {
    message = error
    Icon = AlertCircle
    tone = 'error'
  } else if (!online) {
    message = '网络离线 · 保留上一次成功结果'
    Icon = CloudOff
    tone = 'stale'
  } else if (response) {
    message = `${stale ? '结果已过期' : '规划完成'} · ${response.result.routes.length} 条可行路线`
    Icon = CheckCircle2
    tone = stale ? 'stale' : 'success'
  }
  return (
    <footer className={`status-bar ${tone}`} aria-live="polite">
      <div>
        <Icon aria-hidden="true" className={planning ? 'spin' : undefined} />
        <span>{message}</span>
      </div>
      {response ? (
        <dl>
          <div><dt>地图尺寸</dt><dd>{response.result.map.width} × {response.result.map.height}</dd></div>
          <div><dt>分辨率</dt><dd>{response.result.map.resolution.toFixed(2)} m/px</dd></div>
          <div><dt>排序</dt><dd>{response.models.route_ranking}</dd></div>
        </dl>
      ) : null}
    </footer>
  )
}
