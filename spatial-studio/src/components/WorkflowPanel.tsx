import { ChevronDown, CircleDot, ImagePlus, Info, Play, ScanLine } from 'lucide-react'

import type {
  Coordinate,
  PlacementMode,
  PlanningOptions,
  SpatialCapabilities,
} from '../types/spatial'

interface WorkflowPanelProps {
  imageName: string | null
  start: Coordinate
  goal: Coordinate
  placementMode: PlacementMode
  options: PlanningOptions
  capabilities: SpatialCapabilities | null
  planning: boolean
  compact?: boolean
  onUpload: () => void
  onPlacementMode: (mode: PlacementMode) => void
  onCoordinate: (kind: 'start' | 'goal', axis: 0 | 1, value: number) => void
  onOptions: (options: PlanningOptions) => void
  onPlan: () => void
}

function CoordinateControl({
  label,
  point,
  active,
  color,
  onActivate,
  onChange,
}: {
  label: string
  point: Coordinate
  active: boolean
  color: 'green' | 'red'
  onActivate: () => void
  onChange: (axis: 0 | 1, value: number) => void
}) {
  return (
    <fieldset className={active ? 'coordinate-control active' : 'coordinate-control'}>
      <legend>
        <span className={`point-dot ${color}`} aria-hidden="true" />
        {label}
      </legend>
      <div className="coordinate-fields">
        <label>
          <span>X</span>
          <input
            aria-label={`${label} X 坐标`}
            inputMode="decimal"
            type="number"
            value={Number(point[0].toFixed(2))}
            onChange={(event) => onChange(0, Number(event.target.value))}
          />
        </label>
        <label>
          <span>Y</span>
          <input
            aria-label={`${label} Y 坐标`}
            inputMode="decimal"
            type="number"
            value={Number(point[1].toFixed(2))}
            onChange={(event) => onChange(1, Number(event.target.value))}
          />
        </label>
        <button
          className="icon-button"
          type="button"
          aria-label={`在地图上选择${label}`}
          aria-pressed={active}
          onClick={onActivate}
        >
          <ScanLine aria-hidden="true" size={18} />
        </button>
      </div>
    </fieldset>
  )
}

export function WorkflowPanel({
  imageName,
  start,
  goal,
  placementMode,
  options,
  capabilities,
  planning,
  compact = false,
  onUpload,
  onPlacementMode,
  onCoordinate,
  onOptions,
  onPlan,
}: WorkflowPanelProps) {
  const qwen = capabilities?.semantic.qwen_vl
  const worldModel = capabilities?.world_model
  const update = <K extends keyof PlanningOptions>(key: K, value: PlanningOptions[K]) =>
    onOptions({ ...options, [key]: value })

  return (
    <aside className={compact ? 'workflow-panel compact' : 'workflow-panel'} aria-label="上传与规划设置">
      <div className="panel-heading">
        <div>
          <h2>平面图</h2>
          <p title={imageName ?? undefined}>{imageName ?? '尚未选择图片'}</p>
        </div>
        <button className="image-button" type="button" onClick={onUpload}>
          <ImagePlus aria-hidden="true" size={18} />
          {imageName ? '更换图片' : '选择图片'}
        </button>
      </div>

      <CoordinateControl
        label="起点"
        point={start}
        active={placementMode === 'start'}
        color="green"
        onActivate={() => onPlacementMode(placementMode === 'start' ? null : 'start')}
        onChange={(axis, value) => onCoordinate('start', axis, value)}
      />
      <CoordinateControl
        label="终点"
        point={goal}
        active={placementMode === 'goal'}
        color="red"
        onActivate={() => onPlacementMode(placementMode === 'goal' ? null : 'goal')}
        onChange={(axis, value) => onCoordinate('goal', axis, value)}
      />

      <div className="setting-row">
        <label htmlFor={compact ? 'mobile-route-count' : 'route-count'}>路线数量</label>
        <select
          id={compact ? 'mobile-route-count' : 'route-count'}
          value={options.routeCount}
          onChange={(event) => update('routeCount', Number(event.target.value))}
        >
          {[1, 2, 3, 4, 5, 6].map((count) => (
            <option key={count} value={count}>
              {count}
            </option>
          ))}
        </select>
      </div>

      <div className="setting-row">
        <label htmlFor={compact ? 'mobile-clearance' : 'clearance'}>安全净空</label>
        <div className="unit-input">
          <input
            id={compact ? 'mobile-clearance' : 'clearance'}
            min="0"
            max="10"
            step="0.05"
            type="number"
            value={options.clearanceWeight}
            onChange={(event) => update('clearanceWeight', Number(event.target.value))}
          />
          <span>权重</span>
        </div>
      </div>

      <label className="switch-row">
        <span>允许未知区域</span>
        <input
          type="checkbox"
          checked={options.allowUnknown}
          onChange={(event) => update('allowUnknown', event.target.checked)}
        />
        <span className="switch" aria-hidden="true" />
      </label>

      <section className="model-section" aria-labelledby={compact ? 'mobile-vision-title' : 'vision-title'}>
        <div className="model-label">
          <CircleDot aria-hidden="true" size={17} />
          <h3 id={compact ? 'mobile-vision-title' : 'vision-title'}>视觉模型</h3>
          <span>{qwen?.configured ? (qwen.loaded ? '已加载' : '可用') : '未配置'}</span>
        </div>
        <label className="select-wrap">
          <span className="sr-only">视觉模型</span>
          <select
            value={options.semanticBackend}
            onChange={(event) =>
              update('semanticBackend', event.target.value as PlanningOptions['semanticBackend'])
            }
          >
            <option value="disabled">未启用</option>
            <option value="qwen-vl" disabled={!qwen?.configured}>
              {qwen?.model ? `Qwen-VL · ${qwen.model}` : 'Qwen-VL（服务器未配置）'}
            </option>
          </select>
          <ChevronDown aria-hidden="true" size={16} />
        </label>
      </section>

      <section className="model-section" aria-labelledby={compact ? 'mobile-world-title' : 'world-title'}>
        <div className="model-label">
          <CircleDot aria-hidden="true" size={17} />
          <h3 id={compact ? 'mobile-world-title' : 'world-title'}>世界模型</h3>
          <span>{worldModel?.configured ? (worldModel.loaded ? '已加载' : '可用') : '几何评分'}</span>
        </div>
        <label className="switch-row embedded">
          <span>RSSM 路线重评分</span>
          <input
            type="checkbox"
            checked={options.useWorldModel}
            disabled={!worldModel?.configured}
            onChange={(event) => update('useWorldModel', event.target.checked)}
          />
          <span className="switch" aria-hidden="true" />
        </label>
      </section>

      <details className="advanced-settings">
        <summary>
          高级建图
          <ChevronDown aria-hidden="true" size={16} />
        </summary>
        <label>
          <span>地图分辨率</span>
          <input
            min="0.01"
            max="100"
            step="0.05"
            type="number"
            value={options.resolution}
            onChange={(event) => update('resolution', Number(event.target.value))}
          />
        </label>
        <label>
          <span>障碍膨胀</span>
          <input
            min="0"
            max="24"
            step="1"
            type="number"
            value={options.obstacleDilation}
            onChange={(event) => update('obstacleDilation', Number(event.target.value))}
          />
        </label>
        <label>
          <span>自由阈值</span>
          <input
            min="0"
            max="1"
            step="0.01"
            type="number"
            value={options.freeThreshold}
            onChange={(event) => update('freeThreshold', Number(event.target.value))}
          />
        </label>
        <label className="switch-row embedded">
          <span>允许斜向移动</span>
          <input
            type="checkbox"
            checked={options.diagonal}
            onChange={(event) => update('diagonal', event.target.checked)}
          />
          <span className="switch" aria-hidden="true" />
        </label>
      </details>

      <p className="settings-note">
        <Info aria-hidden="true" size={15} />
        视觉语义和世界模型仅在服务器配置后启用。
      </p>
      <button className="button primary plan-button" type="button" disabled={!imageName || planning} onClick={onPlan}>
        <Play aria-hidden="true" fill="currentColor" size={18} />
        {planning ? '正在规划…' : '开始规划'}
      </button>
    </aside>
  )
}
