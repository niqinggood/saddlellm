import { Layers3, Map, Route, ScanSearch } from 'lucide-react'

import type { LayerVisibility } from '../types/spatial'

interface LayerControlsProps {
  layers: LayerVisibility
  hasPlan: boolean
  onChange: (layers: LayerVisibility) => void
}

const CONTROLS = [
  { key: 'map', label: '地图', Icon: Map },
  { key: 'occupancy', label: '占用层', Icon: Layers3 },
  { key: 'semantics', label: '语义层', Icon: ScanSearch },
  { key: 'routes', label: '路线', Icon: Route },
] as const

export function LayerControls({ layers, hasPlan, onChange }: LayerControlsProps) {
  return (
    <div className="layer-controls" aria-label="地图可见图层">
      {CONTROLS.map(({ key, label, Icon }) => {
        const disabled = key !== 'map' && !hasPlan
        return (
          <label key={key} className={disabled ? 'layer-toggle disabled' : 'layer-toggle'}>
            <Icon aria-hidden="true" size={18} />
            <span>{label}</span>
            <input
              type="checkbox"
              checked={layers[key]}
              disabled={disabled}
              onChange={(event) => onChange({ ...layers, [key]: event.target.checked })}
            />
            <span className="switch small" aria-hidden="true" />
          </label>
        )
      })}
    </div>
  )
}
