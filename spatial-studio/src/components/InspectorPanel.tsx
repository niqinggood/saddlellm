import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Gauge,
  GitBranch,
  Ruler,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'

import { routeDisplayName, routeStyle } from '../lib/routes'
import type { InspectorTab, SpatialJobResponse } from '../types/spatial'

interface InspectorPanelProps {
  response: SpatialJobResponse | null
  selectedRoute: number
  tab: InspectorTab
  expanded: boolean
  onSelectRoute: (route: number) => void
  onTab: (tab: InspectorTab) => void
  onExpanded: (expanded: boolean) => void
}

const TABS: Array<{ id: InspectorTab; label: string }> = [
  { id: 'routes', label: '路线' },
  { id: 'structure', label: '空间结构' },
  { id: 'uncertainty', label: '不确定性' },
]

export function InspectorPanel({
  response,
  selectedRoute,
  tab,
  expanded,
  onSelectRoute,
  onTab,
  onExpanded,
}: InspectorPanelProps) {
  const routes = response?.result.routes ?? []
  return (
    <aside className={expanded ? 'inspector-panel expanded' : 'inspector-panel collapsed'} aria-label="空间规划检查器">
      <button
        className="sheet-handle"
        type="button"
        aria-label={expanded ? '收起检查器' : '展开检查器'}
        aria-expanded={expanded}
        onClick={() => onExpanded(!expanded)}
      >
        <span aria-hidden="true" />
      </button>
      <div className="inspector-mobile-heading">
        <div>
          <strong>路线与指标</strong>
          <span>{routes.length} 条可行路线</span>
        </div>
        <button type="button" aria-label="收起检查器" onClick={() => onExpanded(false)}>
          <X aria-hidden="true" />
        </button>
      </div>
      <div className="inspector-tabs" role="tablist" aria-label="检查器视图">
        {TABS.map((item) => (
          <button
            key={item.id}
            id={`tab-${item.id}`}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            aria-controls={`panel-${item.id}`}
            className={tab === item.id ? 'active' : undefined}
            onClick={() => onTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="inspector-content">
        {tab === 'routes' ? (
          <section id="panel-routes" role="tabpanel" aria-labelledby="tab-routes">
            {routes.length ? (
              <ol className="route-list">
                {routes.map((route, index) => {
                  const style = routeStyle(index)
                  const selected = route.id === selectedRoute
                  return (
                    <li key={route.id}>
                      <button
                        className={selected ? 'route-card selected' : 'route-card'}
                        type="button"
                        aria-pressed={selected}
                        onClick={() => onSelectRoute(route.id)}
                      >
                        <span className="route-accent" style={{ backgroundColor: style.color }} />
                        <span className="route-card-header">
                          <strong>
                            <span className="route-id" style={{ backgroundColor: style.color }}>R{route.id}</span>
                            路线 {route.id} · <em>{routeDisplayName(route, index)}</em>
                          </strong>
                          <span className={selected ? 'selection-radio checked' : 'selection-radio'} aria-hidden="true" />
                        </span>
                        <span className="route-metrics">
                          <Metric icon={<Ruler />} label="长度" value={`${route.length.toFixed(2)} m`} />
                          <Metric icon={<GitBranch />} label="转弯" value={`${route.turns} 次`} />
                          <Metric icon={<ShieldCheck />} label="最小净空" value={`${route.minimum_clearance.toFixed(2)} m`} />
                          <Metric icon={<Gauge />} label="风险" value={route.risk.toFixed(3)} />
                        </span>
                        {route.predicted_return !== null ? (
                          <span className="world-score">
                            <Sparkles aria-hidden="true" size={15} />
                            世界模型回报 {route.predicted_return.toFixed(3)}
                          </span>
                        ) : null}
                        <ChevronRight className="route-chevron" aria-hidden="true" size={18} />
                      </button>
                    </li>
                  )
                })}
              </ol>
            ) : (
              <EmptyInspector title="尚无路线" detail="上传平面图并完成规划后，这里会显示候选路线和指标。" />
            )}
          </section>
        ) : null}

        {tab === 'structure' ? (
          <section id="panel-structure" role="tabpanel" aria-labelledby="tab-structure" className="structure-panel">
            {response ? (
              <>
                <div className="analysis-summary">
                  <span>空间摘要</span>
                  <strong>{response.result.analysis.summary}</strong>
                  <small>语义置信度 {(response.result.analysis.confidence * 100).toFixed(0)}%</small>
                </div>
                <dl className="map-facts">
                  <div><dt>图像类型</dt><dd>{response.result.analysis.image_type}</dd></div>
                  <div><dt>栅格尺寸</dt><dd>{response.result.map.width} × {response.result.map.height}</dd></div>
                  <div><dt>自由单元</dt><dd>{response.result.map.cell_counts.free}</dd></div>
                  <div><dt>障碍单元</dt><dd>{response.result.map.cell_counts.blocked}</dd></div>
                </dl>
                <h3>识别实体</h3>
                {response.result.analysis.entities.length ? (
                  <ul className="entity-list">
                    {response.result.analysis.entities.map((entity) => (
                      <li key={`${entity.name}-${entity.category}`}>
                        <div><strong>{entity.name}</strong><span>{entity.category}</span></div>
                        <small>{(entity.confidence * 100).toFixed(0)}%</small>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted-copy">视觉模型未启用，因此只显示几何占用结构。</p>
                )}
              </>
            ) : (
              <EmptyInspector title="尚无空间结构" detail="完成一次规划后可查看占用栅格和语义实体。" />
            )}
          </section>
        ) : null}

        {tab === 'uncertainty' ? (
          <section id="panel-uncertainty" role="tabpanel" aria-labelledby="tab-uncertainty" className="uncertainty-panel">
            {response ? (
              <>
                <div className="model-truth">
                  <div><span>语义感知</span><strong>{response.models.semantic_backend}</strong></div>
                  <div><span>路线排序</span><strong>{response.models.route_ranking}</strong></div>
                  <div><span>世界模型</span><strong>{response.models.world_model}</strong></div>
                </div>
                <h3>限制与未知</h3>
                <ul className="caveat-list">
                  {[
                    ...response.result.analysis.caveats,
                    ...response.result.analysis.unknown_regions,
                    ...response.result.analysis.hazards,
                  ].map((item, index) => (
                    <li key={`${item}-${index}`}>
                      <AlertTriangle aria-hidden="true" size={17} />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
                {!response.result.analysis.caveats.length ? (
                  <p className="muted-copy">当前分析没有报告额外限制。</p>
                ) : null}
                <details className="raw-method">
                  <summary>方法说明 <ChevronDown aria-hidden="true" size={15} /></summary>
                  <p>路线首先满足占用网格约束，再按长度、净空和风险生成多样化候选；只有启用并加载 RSSM 时才使用学习回报重排序。</p>
                </details>
              </>
            ) : (
              <EmptyInspector title="尚无不确定性报告" detail="完成规划后系统会明确列出未启用模型与未知区域。" />
            )}
          </section>
        ) : null}
      </div>
    </aside>
  )
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <span>
      <span className="metric-icon" aria-hidden="true">{icon}</span>
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  )
}

function EmptyInspector({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-inspector">
      <Gauge aria-hidden="true" />
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  )
}
