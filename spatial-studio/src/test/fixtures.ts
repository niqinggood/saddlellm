import type { SpatialCapabilities, SpatialJobResponse } from '../types/spatial'

export const capabilitiesFixture: SpatialCapabilities = {
  api_version: 1,
  semantic: {
    qwen_vl: {
      configured: false,
      loaded: false,
      model: null,
    },
  },
  world_model: {
    configured: false,
    loaded: false,
    checkpoint: null,
    observation_schema: {
      schema: 'spatial_features_v1',
      minimum_size: 16,
      features: ['progress', 'remaining_distance'],
    },
  },
  limits: {
    max_upload_mb: 25,
    max_image_pixels: 16_000_000,
    max_routes: 6,
  },
}

export const jobFixture: SpatialJobResponse = {
  status: 'completed',
  job_id: 'demo',
  source_name: 'demo-floorplan.png',
  image: {
    width: 24,
    height: 16,
    url: '/api/jobs/demo/artifacts/source',
  },
  artifacts: {
    json: '/api/jobs/demo/artifacts/json',
    html: '/api/jobs/demo/artifacts/html',
    png: '/api/jobs/demo/artifacts/png',
    mask: '/api/jobs/demo/artifacts/mask',
  },
  models: {
    semantic_backend: 'disabled',
    world_model: 'disabled',
    route_ranking: 'geometric',
    observation_schema: 'spatial_features_v1',
  },
  result: {
    image_path: 'demo-floorplan.png',
    instruction: '从入口前往会议室',
    map: {
      width: 24,
      height: 16,
      resolution: 0.5,
      source_size: [24, 16],
      cell_counts: { free: 280, blocked: 96, unknown: 8 },
      cells: [],
    },
    analysis: {
      image_type: 'occupancy_map',
      summary: '几何占用地图：两条走廊连接三个主要空间。',
      entities: [],
      connectivity: [],
      hazards: [],
      unknown_regions: ['右上角区域边界不明确'],
      confidence: 0.72,
      caveats: ['未启用视觉语义模型'],
    },
    start: [2, 2],
    goal: [21, 13],
    start_source: [2, 2],
    goal_source: [21, 13],
    routes: [
      {
        id: 1,
        label: 'shortest',
        points: [[2, 2], [12, 2], [21, 13]],
        source_points: [[2, 2], [12, 2], [21, 13]],
        length: 14.24,
        cost: 14.5,
        turns: 2,
        minimum_clearance: 1.5,
        average_clearance: 2.1,
        risk: 0.08,
        predicted_return: null,
        combined_score: null,
      },
      {
        id: 2,
        label: 'clearance',
        points: [[2, 2], [2, 12], [21, 13]],
        source_points: [[2, 2], [2, 12], [21, 13]],
        length: 16.5,
        cost: 16.9,
        turns: 1,
        minimum_clearance: 2.25,
        average_clearance: 2.8,
        risk: 0.04,
        predicted_return: null,
        combined_score: null,
      },
      {
        id: 3,
        label: 'diverse',
        points: [[2, 2], [18, 8], [21, 13]],
        source_points: [[2, 2], [18, 8], [21, 13]],
        length: 17.1,
        cost: 17.8,
        turns: 3,
        minimum_clearance: 1.8,
        average_clearance: 2.3,
        risk: 0.06,
        predicted_return: null,
        combined_score: null,
      },
    ],
  },
}

export function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}
