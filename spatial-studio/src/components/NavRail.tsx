import { BarChart3, Box, Globe2, ImageUp, Map, Settings } from 'lucide-react'

import type { InspectorTab } from '../types/spatial'

interface NavRailProps {
  onUpload: () => void
  onSettings: () => void
  onInspectorTab: (tab: InspectorTab) => void
  onFocusMap: () => void
}

export function NavRail({ onUpload, onSettings, onInspectorTab, onFocusMap }: NavRailProps) {
  return (
    <nav className="nav-rail" aria-label="工作区导航">
      <button type="button" onClick={onUpload} aria-label="上传与规划">
        <ImageUp aria-hidden="true" />
        <span>上传与规划</span>
      </button>
      <button className="active" type="button" onClick={onFocusMap} aria-label="地图工作区">
        <Map aria-hidden="true" />
      </button>
      <button type="button" onClick={onSettings} aria-label="世界模型设置">
        <Box aria-hidden="true" />
      </button>
      <button type="button" onClick={() => onInspectorTab('structure')} aria-label="空间结构">
        <Globe2 aria-hidden="true" />
      </button>
      <button type="button" onClick={() => onInspectorTab('uncertainty')} aria-label="不确定性与指标">
        <BarChart3 aria-hidden="true" />
      </button>
      <button className="rail-settings" type="button" onClick={onSettings} aria-label="设置">
        <Settings aria-hidden="true" />
      </button>
    </nav>
  )
}
