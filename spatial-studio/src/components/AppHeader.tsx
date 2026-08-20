import { Download, HelpCircle, ImagePlus, Settings2 } from 'lucide-react'

import { apiUrl } from '../api/spatialApi'
import type { SpatialJobResponse } from '../types/spatial'
import { LogoMark } from './LogoMark'

interface AppHeaderProps {
  response: SpatialJobResponse | null
  onUpload: () => void
  onSettings: () => void
  onHelp: () => void
}

export function AppHeader({ response, onUpload, onSettings, onHelp }: AppHeaderProps) {
  return (
    <header className="app-header">
      <a className="brand" href="/" aria-label="Saddle 空间世界模型首页">
        <LogoMark />
        <span>Saddle 空间世界模型</span>
      </a>
      <nav className="desktop-nav" aria-label="主导航">
        <button type="button" onClick={onUpload}>
          项目
        </button>
        <button type="button" onClick={onSettings}>
          模型
        </button>
        <button type="button" onClick={onHelp}>
          帮助
        </button>
      </nav>
      <div className="header-actions">
        <button className="mobile-icon-button" type="button" onClick={onUpload} aria-label="更换图片">
          <ImagePlus aria-hidden="true" size={21} />
        </button>
        <button className="mobile-icon-button" type="button" onClick={onSettings} aria-label="打开设置">
          <Settings2 aria-hidden="true" size={21} />
        </button>
        <button className="mobile-icon-button" type="button" onClick={onHelp} aria-label="打开帮助">
          <HelpCircle aria-hidden="true" size={21} />
        </button>
        <details className="export-menu">
          <summary className={response ? 'button secondary' : 'button secondary disabled'}>
            <Download aria-hidden="true" size={18} />
            <span>导出结果</span>
          </summary>
          {response ? (
            <div className="export-popover">
              <a href={apiUrl(response.artifacts.png)} download="spatial-plan.png">
                PNG 路线图
              </a>
              <a href={apiUrl(response.artifacts.json)} download="spatial-plan.json">
                JSON 数据
              </a>
              <a href={apiUrl(response.artifacts.html)} target="_blank" rel="noreferrer">
                交互 HTML
              </a>
            </div>
          ) : null}
        </details>
      </div>
    </header>
  )
}
