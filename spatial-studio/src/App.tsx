import { useCallback, useEffect, useRef, useState } from 'react'

import {
  apiUrl,
  fetchCapabilities,
  fetchDemo,
  planSpatialImage,
  responseImageAsFile,
  SpatialApiError,
} from './api/spatialApi'
import { AppHeader } from './components/AppHeader'
import { InspectorPanel } from './components/InspectorPanel'
import { MapCanvas } from './components/MapCanvas'
import { MobileCommandBar } from './components/MobileCommandBar'
import { NavRail } from './components/NavRail'
import { OverlayDialog } from './components/OverlayDialog'
import { StatusBar } from './components/StatusBar'
import { UploadDialog } from './components/UploadDialog'
import { WorkflowPanel } from './components/WorkflowPanel'
import {
  DEFAULT_LAYERS,
  DEFAULT_OPTIONS,
  loadPreferences,
  savePreferences,
} from './lib/preferences'
import { parseViewState, writeViewState } from './lib/viewState'
import type {
  Coordinate,
  ImageSource,
  InspectorTab,
  LayerVisibility,
  PlacementMode,
  PlanningOptions,
  SpatialCapabilities,
  SpatialJobResponse,
  SpatialPlanRequest,
} from './types/spatial'

export function App() {
  const initialPreferences = useRef(loadPreferences()).current
  const initialView = useRef(parseViewState(window.location.search)).current
  const [capabilities, setCapabilities] = useState<SpatialCapabilities | null>(null)
  const [response, setResponse] = useState<SpatialJobResponse | null>(null)
  const [image, setImage] = useState<ImageSource | null>(null)
  const [start, setStart] = useState<Coordinate>([2, 2])
  const [goal, setGoal] = useState<Coordinate>([21, 13])
  const [placementMode, setPlacementMode] = useState<PlacementMode>(null)
  const [layers, setLayers] = useState<LayerVisibility>(initialPreferences.layers ?? DEFAULT_LAYERS)
  const [options, setOptions] = useState<PlanningOptions>({
    ...DEFAULT_OPTIONS,
    ...initialPreferences.options,
  })
  const [selectedRoute, setSelectedRoute] = useState(initialView.route)
  const [tab, setTab] = useState<InspectorTab>(initialView.tab)
  const [planning, setPlanning] = useState(false)
  const [stale, setStale] = useState(false)
  const [online, setOnline] = useState(navigator.onLine)
  const [error, setError] = useState<string | null>(null)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const [inspectorExpanded, setInspectorExpanded] = useState(true)
  const planningController = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    Promise.allSettled([
      fetchCapabilities(controller.signal),
      fetchDemo(controller.signal),
    ]).then(([capabilityResult, demoResult]) => {
      if (controller.signal.aborted) return
      if (capabilityResult.status === 'fulfilled') {
        setCapabilities(capabilityResult.value)
      }
      if (demoResult.status === 'fulfilled') {
        applyResponse(demoResult.value, null)
      } else {
        setError(errorMessage(demoResult.reason, '无法加载内置演示，可直接上传图片'))
      }
    })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    savePreferences(layers, options)
  }, [layers, options])

  useEffect(() => {
    writeViewState({ route: selectedRoute, tab })
  }, [selectedRoute, tab])

  useEffect(() => {
    const restore = () => {
      const state = parseViewState(window.location.search)
      setSelectedRoute(state.route)
      setTab(state.tab)
    }
    window.addEventListener('popstate', restore)
    return () => window.removeEventListener('popstate', restore)
  }, [])

  useEffect(() => {
    const goOnline = () => setOnline(true)
    const goOffline = () => {
      setOnline(false)
      setStale(Boolean(response))
    }
    window.addEventListener('online', goOnline)
    window.addEventListener('offline', goOffline)
    return () => {
      window.removeEventListener('online', goOnline)
      window.removeEventListener('offline', goOffline)
    }
  }, [response])

  useEffect(
    () => () => {
      planningController.current?.abort()
      if (image?.url.startsWith('blob:')) URL.revokeObjectURL(image.url)
    },
    [image?.url],
  )

  const applyResponse = useCallback((next: SpatialJobResponse, sourceFile: File | null) => {
    setResponse(next)
    setImage((previous) => {
      if (previous?.url.startsWith('blob:')) URL.revokeObjectURL(previous.url)
      return {
        url: apiUrl(next.image.url),
        width: next.image.width,
        height: next.image.height,
        name: next.source_name,
        file: sourceFile,
      }
    })
    setStart(next.result.start_source)
    setGoal(next.result.goal_source)
    setSelectedRoute(next.result.routes[0]?.id ?? 1)
    setStale(false)
    setError(null)
  }, [])

  const markChanged = useCallback(() => {
    if (response) setStale(true)
    setError(null)
  }, [response])

  const handleUpload = useCallback(async (file: File) => {
    try {
      const next = await inspectImage(file)
      setImage((previous) => {
        if (previous?.url.startsWith('blob:')) URL.revokeObjectURL(previous.url)
        return next
      })
      setResponse(null)
      setStart([next.width * 0.15, next.height * 0.2])
      setGoal([next.width * 0.85, next.height * 0.8])
      setPlacementMode('start')
      setSelectedRoute(1)
      setStale(false)
      setError(null)
      setUploadOpen(false)
      setInspectorExpanded(false)
    } catch (reason) {
      setError(errorMessage(reason, '无法读取这张图片'))
    }
  }, [])

  const handleOptions = useCallback((next: PlanningOptions) => {
    setOptions(next)
    markChanged()
  }, [markChanged])

  const handleCoordinate = useCallback((kind: 'start' | 'goal', axis: 0 | 1, value: number) => {
    if (!Number.isFinite(value) || !image) return
    const maximum = axis === 0 ? image.width - 0.001 : image.height - 0.001
    const normalized = Math.min(Math.max(value, 0), maximum)
    const setter = kind === 'start' ? setStart : setGoal
    setter((point) => {
      const next: Coordinate = [...point]
      next[axis] = normalized
      return next
    })
    markChanged()
  }, [image, markChanged])

  const handleMapPoint = useCallback((point: Coordinate) => {
    if (placementMode === 'start') {
      setStart(point)
      setPlacementMode('goal')
    } else if (placementMode === 'goal') {
      setGoal(point)
      setPlacementMode(null)
    }
    markChanged()
  }, [markChanged, placementMode])

  const handlePlan = useCallback(async () => {
    if (!image || planning) return
    planningController.current?.abort()
    const controller = new AbortController()
    planningController.current = controller
    setPlanning(true)
    setError(null)
    try {
      const file = image.file ?? (response ? await responseImageAsFile(response) : null)
      if (!file) throw new Error('当前图片不可用于重新规划，请重新上传')
      const request: SpatialPlanRequest = {
        start,
        goal,
        route_count: options.routeCount,
        instruction: '避开墙体，生成差异明显且可解释的候选路线',
        free_threshold: options.freeThreshold,
        free_is_bright: true,
        uncertainty_band: 0,
        obstacle_dilation: options.obstacleDilation,
        max_dimension: 768,
        resolution: options.resolution,
        diagonal: options.diagonal,
        allow_unknown: options.allowUnknown,
        clearance_weight: options.clearanceWeight,
        diversity_weight: 2,
        diversity_radius: 2,
        max_detour_ratio: 2.5,
        semantic_backend: options.semanticBackend,
        use_world_model: options.useWorldModel,
        allow_perspective: false,
      }
      const next = await planSpatialImage(file, request, controller.signal)
      applyResponse(next, image.file ?? file)
      setInspectorExpanded(true)
      setTab('routes')
      setSettingsOpen(false)
    } catch (reason) {
      if (controller.signal.aborted) return
      setError(errorMessage(reason, '空间规划失败'))
      setStale(Boolean(response))
    } finally {
      if (!controller.signal.aborted) setPlanning(false)
    }
  }, [applyResponse, goal, image, options, planning, response, start])

  const focusMap = useCallback(() => {
    document.getElementById('map-workspace')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [])

  return (
    <div className="app-shell">
      <AppHeader
        response={response}
        onUpload={() => setUploadOpen(true)}
        onSettings={() => setSettingsOpen(true)}
        onHelp={() => setHelpOpen(true)}
      />
      <MobileCommandBar
        onUpload={() => setUploadOpen(true)}
        onPoints={() => setPlacementMode(placementMode === 'start' ? 'goal' : 'start')}
        onSettings={() => setSettingsOpen(true)}
      />
      <div className="workspace-shell">
        <NavRail
          onUpload={() => setUploadOpen(true)}
          onSettings={() => setSettingsOpen(true)}
          onInspectorTab={(next) => {
            setTab(next)
            setInspectorExpanded(true)
          }}
          onFocusMap={focusMap}
        />
        <WorkflowPanel
          imageName={image?.name ?? null}
          start={start}
          goal={goal}
          placementMode={placementMode}
          options={options}
          capabilities={capabilities}
          planning={planning}
          onUpload={() => setUploadOpen(true)}
          onPlacementMode={setPlacementMode}
          onCoordinate={handleCoordinate}
          onOptions={handleOptions}
          onPlan={handlePlan}
        />
        <MapCanvas
          image={image}
          response={response}
          layers={layers}
          selectedRoute={selectedRoute}
          start={start}
          goal={goal}
          placementMode={placementMode}
          stale={stale}
          onLayers={setLayers}
          onSelectRoute={(route) => {
            setSelectedRoute(route)
            setTab('routes')
          }}
          onPoint={handleMapPoint}
          onPlacementMode={setPlacementMode}
        />
        <InspectorPanel
          response={response}
          selectedRoute={selectedRoute}
          tab={tab}
          expanded={inspectorExpanded}
          onSelectRoute={setSelectedRoute}
          onTab={setTab}
          onExpanded={setInspectorExpanded}
        />
      </div>
      <StatusBar response={response} planning={planning} online={online} stale={stale} error={error} />

      <UploadDialog
        open={uploadOpen}
        maxUploadMb={capabilities?.limits.max_upload_mb ?? 25}
        onClose={() => setUploadOpen(false)}
        onFile={handleUpload}
      />

      <OverlayDialog open={settingsOpen} title="规划与模型设置" onClose={() => setSettingsOpen(false)}>
        <WorkflowPanel
          compact
          imageName={image?.name ?? null}
          start={start}
          goal={goal}
          placementMode={placementMode}
          options={options}
          capabilities={capabilities}
          planning={planning}
          onUpload={() => {
            setSettingsOpen(false)
            setUploadOpen(true)
          }}
          onPlacementMode={(mode) => {
            setPlacementMode(mode)
            if (mode) setSettingsOpen(false)
          }}
          onCoordinate={handleCoordinate}
          onOptions={handleOptions}
          onPlan={handlePlan}
        />
      </OverlayDialog>

      <OverlayDialog open={helpOpen} title="如何使用" onClose={() => setHelpOpen(false)}>
        <div className="help-content">
          <ol>
            <li><strong>上传俯视图</strong><span>选择干净的户型图、楼层图或占用地图。</span></li>
            <li><strong>设置起终点</strong><span>点击坐标按钮，再在地图上依次放置 S 和 G。</span></li>
            <li><strong>开始规划</strong><span>系统先构建占用栅格，再搜索多条差异明显的路线。</span></li>
            <li><strong>检查不确定性</strong><span>语义模型或 RSSM 未启用时，界面会明确标注。</span></li>
          </ol>
          <p>单张透视照片无法证明被遮挡空间。实景导航还需要深度、定位或 SLAM 数据。</p>
        </div>
      </OverlayDialog>
    </div>
  )
}

async function inspectImage(file: File): Promise<ImageSource> {
  const url = URL.createObjectURL(file)
  try {
    const dimensions = await new Promise<{ width: number; height: number }>((resolve, reject) => {
      const preview = new Image()
      preview.onload = () => resolve({ width: preview.naturalWidth, height: preview.naturalHeight })
      preview.onerror = () => reject(new Error('浏览器无法预览这张图片'))
      preview.src = url
    })
    return { url, ...dimensions, name: file.name, file }
  } catch (error) {
    URL.revokeObjectURL(url)
    throw error
  }
}

function errorMessage(reason: unknown, fallback: string): string {
  if (reason instanceof SpatialApiError) return reason.message
  if (reason instanceof Error && reason.message) return reason.message
  return fallback
}
