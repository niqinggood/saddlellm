export type Coordinate = [number, number]
export type InspectorTab = 'routes' | 'structure' | 'uncertainty'
export type PlacementMode = 'start' | 'goal' | null

export interface SpatialEntity {
  name: string
  category: string
  bbox: [number, number, number, number]
  confidence: number
  traversable: boolean | null
  attributes: Record<string, unknown>
}

export interface SpatialAnalysis {
  image_type: string
  summary: string
  entities: SpatialEntity[]
  connectivity: Array<Record<string, unknown>>
  hazards: string[]
  unknown_regions: string[]
  confidence: number
  caveats: string[]
}

export interface SpatialMap {
  width: number
  height: number
  resolution: number
  source_size: [number, number]
  cell_counts: {
    free: number
    blocked: number
    unknown: number
  }
  cells: number[][]
}

export interface SpatialRoute {
  id: number
  label: string
  points: Coordinate[]
  source_points: Coordinate[]
  length: number
  cost: number
  turns: number
  minimum_clearance: number
  average_clearance: number
  risk: number
  predicted_return: number | null
  combined_score: number | null
}

export interface SpatialWorldModelForecast {
  action: number[]
  actions: number[][]
  score: number
  predicted_points: Coordinate[]
  geometry_collisions: number
  predicted_rewards: number[]
  learned_collision_probability: number[] | null
  future_occupancy?: {
    classes: number[][][]
    confidence: number[][][]
  }
}

export interface SpatialWorldModelResult {
  schema: string
  planner: string
  forecast: SpatialWorldModelForecast
}

export interface SpatialPlanResult {
  image_path: string
  instruction: string
  map: SpatialMap
  analysis: SpatialAnalysis
  start: Coordinate
  goal: Coordinate
  start_source: Coordinate
  goal_source: Coordinate
  routes: SpatialRoute[]
  world_model?: SpatialWorldModelResult
}

export interface SpatialJobResponse {
  status: 'completed'
  job_id: string
  source_name: string
  image: {
    width: number
    height: number
    url: string
  }
  artifacts: {
    json: string
    html: string
    png: string
    mask: string
  }
  models: {
    semantic_backend: string
    world_model: string
    route_ranking: string
    observation_schema?: string
    planning?: string
  }
  result: SpatialPlanResult
}

export interface SpatialCapabilities {
  api_version: number
  semantic: {
    qwen_vl: {
      configured: boolean
      loaded: boolean
      model: string | null
    }
  }
  world_model: {
    configured: boolean
    loaded: boolean
    checkpoint: string | null
    observation_schema: {
      schema: string
      minimum_size: number
      features: string[]
    }
    observation_schemas?: Array<{
      schema: string
      minimum_size?: number
      features?: string[]
      observation_shape?: number[]
      channels?: string[]
      sensor_radius?: number
    }>
  }
  limits: {
    max_upload_mb: number
    max_image_pixels: number
    max_routes: number
  }
}

export interface SpatialPlanRequest {
  start: Coordinate
  goal: Coordinate
  route_count: number
  instruction: string
  free_threshold: number
  free_is_bright: boolean
  uncertainty_band: number
  obstacle_dilation: number
  max_dimension: number
  resolution: number
  diagonal: boolean
  allow_unknown: boolean
  clearance_weight: number
  diversity_weight: number
  diversity_radius: number
  max_detour_ratio: number
  semantic_backend: 'disabled' | 'qwen-vl'
  use_world_model: boolean
  allow_perspective: boolean
}

export interface PlanningOptions {
  routeCount: number
  resolution: number
  clearanceWeight: number
  allowUnknown: boolean
  diagonal: boolean
  obstacleDilation: number
  freeThreshold: number
  semanticBackend: 'disabled' | 'qwen-vl'
  useWorldModel: boolean
}

export interface LayerVisibility {
  map: boolean
  occupancy: boolean
  semantics: boolean
  routes: boolean
}

export interface ImageSource {
  url: string
  width: number
  height: number
  name: string
  file: File | null
}

export interface StudioError {
  code: string
  message: string
}
