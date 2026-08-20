import { Image, MapPin, Settings } from 'lucide-react'

interface MobileCommandBarProps {
  onUpload: () => void
  onPoints: () => void
  onSettings: () => void
}

export function MobileCommandBar({ onUpload, onPoints, onSettings }: MobileCommandBarProps) {
  return (
    <nav className="mobile-command-bar" aria-label="移动工作区命令">
      <button type="button" onClick={onUpload}>
        <Image aria-hidden="true" />
        图片
      </button>
      <button type="button" onClick={onPoints}>
        <MapPin aria-hidden="true" />
        起终点
      </button>
      <button type="button" onClick={onSettings}>
        <Settings aria-hidden="true" />
        设置
      </button>
    </nav>
  )
}
