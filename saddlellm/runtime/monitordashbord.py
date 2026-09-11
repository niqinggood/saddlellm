import asyncio
import json
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import psutil
import pynvml
import prometheus_client
from prometheus_client import Gauge, Counter, Summary
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime

app = FastAPI()

# Prometheus指标定义
GPU_UTIL = Gauge('gpu_utilization', 'GPU utilization percentage', ['device_id'])
GPU_MEM = Gauge('gpu_memory', 'GPU memory usage percentage', ['device_id'])
GPU_TEMP = Gauge('gpu_temperature', 'GPU temperature in Celsius', ['device_id'])
CPU_UTIL = Gauge('cpu_utilization', 'CPU utilization percentage')
CPU_TEMP = Gauge('cpu_temperature', 'CPU temperature in Celsius')
MEM_USAGE = Gauge('memory_usage', 'Memory usage in bytes')
INFERENCE_LATENCY = Summary('inference_latency', 'Inference latency in milliseconds')
TRAINING_LOSS = Gauge('training_loss', 'Current training loss value')
THROUGHPUT = Counter('inference_throughput', 'Total number of inference requests')

# 初始化NVML
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


class MetricData(BaseModel):
    metric: str
    value: float
    timestamp: float
    tags: Optional[Dict[str, str]] = None


class SystemStatus(BaseModel):
    cpu_util: float
    cpu_temp: float
    cpu_cores: int
    mem_used: float
    mem_total: float
    mem_percent: float
    gpu_util: Optional[float] = None
    gpu_mem: Optional[float] = None
    gpu_temp: Optional[float] = None


class AlertData(BaseModel):
    level: str  # info, warning, critical
    type: str  # gpu, cpu, memory, inference, training
    message: str
    timestamp: float


class MonitorServer:
    def __init__(self):
        self.active_connections = {}
        self.metric_history = {
            'training': [],
            'inference': []
        }
        self.alerts = []

    async def collect_system_metrics(self):
        """收集系统指标"""
        while True:
            # CPU指标
            cpu_util = psutil.cpu_percent()
            cpu_temp = self._get_cpu_temp()
            cpu_cores = psutil.cpu_count()

            # 内存指标
            mem = psutil.virtual_memory()

            # GPU指标
            gpu_util = gpu_mem = gpu_temp = None
            if GPU_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_mem = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # 更新Prometheus指标
                GPU_UTIL.labels(device_id=0).set(gpu_util)
                GPU_MEM.labels(device_id=0).set(gpu_mem)
                GPU_TEMP.labels(device_id=0).set(gpu_temp)

            CPU_UTIL.set(cpu_util)
            if cpu_temp: CPU_TEMP.set(cpu_temp)
            MEM_USAGE.set(mem.used)

            status = SystemStatus(
                cpu_util=cpu_util,
                cpu_temp=cpu_temp or 0,
                cpu_cores=cpu_cores,
                mem_used=mem.used / (1024 ** 3),
                mem_total=mem.total / (1024 ** 3),
                mem_percent=mem.percent,
                gpu_util=gpu_util,
                gpu_mem=gpu_mem,
                gpu_temp=gpu_temp
            )

            # 广播系统状态
            await self.broadcast_system_status(status)

            # 检查异常情况
            await self.check_anomalies(status)

            await asyncio.sleep(2)

    def _get_cpu_temp(self) -> Optional[float]:
        """获取CPU温度(Linux系统)"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000
                return temp
        except:
            return None

    async def check_anomalies(self, status: SystemStatus):
        """检查系统异常"""
        alerts = []

        # GPU检查
        if status.gpu_util and status.gpu_util > 90:
            alerts.append(AlertData(
                level="warning",
                type="gpu",
                message=f"GPU利用率过高: {status.gpu_util}%",
                timestamp=time.time()
            ))

        if status.gpu_temp and status.gpu_temp > 85:
            alerts.append(AlertData(
                level="critical",
                type="gpu",
                message=f"GPU温度过高: {status.gpu_temp}°C",
                timestamp=time.time()
            ))

        # CPU检查
        if status.cpu_util > 90:
            alerts.append(AlertData(
                level="warning",
                type="cpu",
                message=f"CPU利用率过高: {status.cpu_util}%",
                timestamp=time.time()
            ))

        # 内存检查
        if status.mem_percent > 90:
            alerts.append(AlertData(
                level="critical",
                type="memory",
                message=f"内存使用率过高: {status.mem_percent}%",
                timestamp=time.time()
            ))

        # 广播告警
        for alert in alerts:
            await self.broadcast_alert(alert)
            self.alerts.append(alert)

    async def broadcast_system_status(self, status: SystemStatus):
        """广播系统状态"""
        message = {
            'type': 'system',
            'data': status.dict()
        }
        await self._broadcast(message)

    async def broadcast_metrics(self, mode: str, metrics: List[MetricData]):
        """广播指标数据"""
        message = {
            'type': 'metrics',
            'mode': mode,
            'data': [m.dict() for m in metrics]
        }
        await self._broadcast(message)

        # 保存到历史记录
        self.metric_history[mode].extend(metrics)

        # 限制历史记录大小
        if len(self.metric_history[mode]) > 1000:
            self.metric_history[mode] = self.metric_history[mode][-1000:]

    async def broadcast_alert(self, alert: AlertData):
        """广播告警"""
        message = {
            'type': 'alert',
            'data': alert.dict()
        }
        await self._broadcast(message)

    async def _broadcast(self, message: Dict):
        """广播消息给所有客户端"""
        for connection in self.active_connections.values():
            try:
                await connection.send_text(json.dumps(message))
            except:
                continue

    async def connect(self, websocket: WebSocket, client_id: str):
        """处理新客户端连接"""
        await websocket.accept()
        self.active_connections[client_id] = websocket

        # 发送历史数据
        history = {
            'training': self.metric_history['training'][-100:],
            'inference': self.metric_history['inference'][-100:]
        }
        await websocket.send_text(json.dumps({
            'type': 'history',
            'data': history
        }))

    def disconnect(self, client_id: str):
        """处理客户端断开连接"""
        self.active_connections.pop(client_id, None)


monitor_server = MonitorServer()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_server.collect_system_metrics())


@app.on_event("shutdown")
async def shutdown_event():
    if GPU_AVAILABLE:
        pynvml.nvmlShutdown()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await monitor_server.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        monitor_server.disconnect(client_id)


@app.get("/metrics")
async def get_metrics():
    return {
        "cpu_util": CPU_UTIL._value.get(),
        "memory_usage": MEM_USAGE._value.get(),
        "training_loss": TRAINING_LOSS._value.get(),
        "throughput": THROUGHPUT._value.get(),
    }


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    alerts = [a.dict() for a in monitor_server.alerts[-limit:]]
    return {"alerts": alerts, "total": len(monitor_server.alerts)}


@app.post("/metrics/push")
async def push_metric(metric: MetricData):
    if metric.metric == "training_loss":
        TRAINING_LOSS.set(metric.value)
    elif metric.metric == "throughput":
        THROUGHPUT.inc()
    return {"status": "ok"}


@app.get("/status")
async def get_status():
    mem = psutil.virtual_memory()
    cpu_util = psutil.cpu_percent()
    status = {
        "cpu_util": cpu_util,
        "cpu_temp": monitor_server._get_cpu_temp() or 0,
        "cpu_cores": psutil.cpu_count(),
        "mem_used_gb": mem.used / (1024 ** 3),
        "mem_total_gb": mem.total / (1024 ** 3),
        "mem_percent": mem.percent,
    }
    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        status["gpu_util"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        status["gpu_mem"] = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
        status["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    return status


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SaddleLLM Monitor</title>
        <meta charset="utf-8">
        <style>
            body { font-family: monospace; max-width: 800px; margin: 40px auto; padding: 20px; background: #1a1a1a; color: #eee; }
            .card { border: 1px solid #333; border-radius: 8px; padding: 16px; margin: 12px 0; background: #222; }
            .metric { display: flex; justify-content: space-between; padding: 4px 0; }
            .bar { background: #333; height: 8px; border-radius: 4px; margin: 4px 0; }
            .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
            .warning { color: #fa0; } .critical { color: #f44; }
        </style>
    </head>
    <body>
        <h1>SaddleLLM Monitor</h1>
        <div class="card"><h3>System Status</h3>
            <div id="sys-status">Loading...</div>
        </div>
        <div class="card"><h3>Alerts</h3>
            <div id="alerts">No alerts</div>
        </div>
        <div class="card"><h3>Training Metrics</h3>
            <div id="train-metrics">No data</div>
        </div>
        <script>
            const ws = new WebSocket(`ws://${location.host}/ws/dashboard`);
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'system') {
                    const s = msg.data;
                    document.getElementById('sys-status').innerHTML =
                        `CPU: ${s.cpu_util.toFixed(1)}% | Mem: ${s.mem_used_gb?.toFixed(1) || '?'}/${s.mem_total_gb?.toFixed(1) || '?'}GB (${s.mem_percent}%)` +
                        (s.gpu_util != null ? ` | GPU: ${s.gpu_util}% | GPU Mem: ${s.gpu_mem}% | GPU Temp: ${s.gpu_temp}C` : '');
                } else if (msg.type === 'alert') {
                    const a = msg.data;
                    const div = document.getElementById('alerts');
                    div.innerHTML = `<span class="${a.level}">[${a.level.toUpperCase()}] ${a.type}: ${a.message}</span><br>` + div.innerHTML;
                }
            };
            fetch('/metrics').then(r => r.json()).then(m => {
                document.getElementById('train-metrics').innerHTML =
                    `Loss: ${m.training_loss ?? 'N/A'} | Throughput: ${m.throughput ?? 'N/A'}`;
            });
        </script>
    </body>
    </html>
    """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
