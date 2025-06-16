from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
from datetime import datetime
import os
from pathlib import Path
import psutil
import GPUtil
import platform
import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import signal

app = FastAPI()

# 로그 파일 경로
LOG_DIR = "logs"
latest_log_file = None

def get_latest_log_file():
    """가장 최근의 로그 파일을 찾습니다."""
    log_files = list(Path(LOG_DIR).glob("api_server_*.log"))
    if not log_files:
        return None
    return max(log_files, key=lambda x: x.stat().st_mtime)

@app.get("/")
async def get():
    """모니터링 웹 인터페이스를 제공합니다."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>API 서버 모니터링</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f0f0f0;
                }
                .main-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    height: 90vh;
                }
                .charts-container {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .chart-container {
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    height: 200px;
                }
                .chart-container canvas {
                    width: 100% !important;
                    height: 100% !important;
                }
                .log-section {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .log-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 10px;
                }
                .log-selector {
                    flex: 1;
                    background-color: #2c3e50;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: 'Courier New', monospace;
                }
                .log-selector select {
                    width: 100%;
                    padding: 5px;
                    background-color: #34495e;
                    color: white;
                    border: 1px solid #7f8c8d;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }
                .log-selector select:focus {
                    outline: none;
                    border-color: #3498db;
                }
                .log-info {
                    background-color: #2c3e50;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: 'Courier New', monospace;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                #log-container {
                    background-color: #1e1e1e;
                    color: #fff;
                    padding: 20px;
                    border-radius: 5px;
                    height: 100%;
                    overflow-y: auto;
                    font-family: 'Courier New', monospace;
                }
                .log-entry {
                    margin: 5px 0;
                    padding: 5px;
                    border-bottom: 1px solid #333;
                }
                .error { color: #ff6b6b; }
                .warning { color: #ffd93d; }
                .info { color: #6bff6b; }
                .debug { color: #6b6bff; }
            </style>
        </head>
        <body>
            <h1>API 서버 모니터링</h1>
            <div class="main-container">
                <div class="charts-container">
                    <div class="chart-container">
                        <canvas id="cpuChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="memoryChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="gpuChart"></canvas>
                    </div>
                </div>
                <div class="log-section">
                    <div class="log-header">
                        <div class="log-selector">
                            <select id="logFileSelect" onchange="changeLogFile()">
                                <option value="">로그 파일 선택 중...</option>
                            </select>
                        </div>
                        <div class="log-info" id="log-info">
                            현재 로그 파일: 로딩 중...
                        </div>
                    </div>
                    <div id="log-container"></div>
                </div>
            </div>
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                const logContainer = document.getElementById('log-container');
                
                // 차트 초기화
                const cpuChart = new Chart(document.getElementById('cpuChart'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'CPU 사용량 (%)',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1,
                            pointRadius: 2,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: {
                                    stepSize: 20
                                }
                            },
                            x: {
                                ticks: {
                                    maxRotation: 0,
                                    autoSkip: true,
                                    maxTicksLimit: 5
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: {
                                    boxWidth: 10,
                                    font: {
                                        size: 11
                                    }
                                }
                            }
                        }
                    }
                });

                const memoryChart = new Chart(document.getElementById('memoryChart'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: '메모리 사용량 (%)',
                            data: [],
                            borderColor: 'rgb(255, 99, 132)',
                            tension: 0.1,
                            pointRadius: 2,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: {
                                    stepSize: 20
                                }
                            },
                            x: {
                                ticks: {
                                    maxRotation: 0,
                                    autoSkip: true,
                                    maxTicksLimit: 5
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: {
                                    boxWidth: 10,
                                    font: {
                                        size: 11
                                    }
                                }
                            }
                        }
                    }
                });

                const gpuChart = new Chart(document.getElementById('gpuChart'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'GPU 사용량 (%)',
                            data: [],
                            borderColor: 'rgb(153, 102, 255)',
                            tension: 0.1,
                            pointRadius: 2,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: {
                                    stepSize: 20
                                }
                            },
                            x: {
                                ticks: {
                                    maxRotation: 0,
                                    autoSkip: true,
                                    maxTicksLimit: 5
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: {
                                    boxWidth: 10,
                                    font: {
                                        size: 11
                                    }
                                }
                            }
                        }
                    }
                });

                function updateCharts(data) {
                    const timestamp = new Date().toLocaleTimeString();
                    
                    // CPU 차트 업데이트
                    cpuChart.data.labels.push(timestamp);
                    cpuChart.data.datasets[0].data.push(data.cpu);
                    if (cpuChart.data.labels.length > 20) {
                        cpuChart.data.labels.shift();
                        cpuChart.data.datasets[0].data.shift();
                    }
                    cpuChart.update();

                    // 메모리 차트 업데이트
                    memoryChart.data.labels.push(timestamp);
                    memoryChart.data.datasets[0].data.push(data.memory);
                    if (memoryChart.data.labels.length > 20) {
                        memoryChart.data.labels.shift();
                        memoryChart.data.datasets[0].data.shift();
                    }
                    memoryChart.update();

                    // GPU 차트 업데이트
                    gpuChart.data.labels.push(timestamp);
                    gpuChart.data.datasets[0].data.push(data.gpu);
                    if (gpuChart.data.labels.length > 20) {
                        gpuChart.data.labels.shift();
                        gpuChart.data.datasets[0].data.shift();
                    }
                    gpuChart.update();
                }
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'metrics') {
                        updateCharts(data);
                    } else if (data.type === 'log_file') {
                        document.getElementById('log-info').textContent = `현재 로그 파일: ${data.file_name}`;
                    } else {
                        const logEntry = document.createElement('div');
                        logEntry.className = `log-entry ${data.level.toLowerCase()}`;
                        logEntry.textContent = `[${data.timestamp}] ${data.message}`;
                        logContainer.appendChild(logEntry);
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }
                };
                
                ws.onclose = function() {
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry error';
                    logEntry.textContent = '연결이 끊어졌습니다. 페이지를 새로고침하세요.';
                    logContainer.appendChild(logEntry);
                };

                // 로그 파일 목록 가져오기
                async function loadLogFiles() {
                    try {
                        const response = await fetch('/api/logs');
                        const data = await response.json();
                        const select = document.getElementById('logFileSelect');
                        select.innerHTML = '';
                        
                        data.files.forEach(file => {
                            const option = document.createElement('option');
                            option.value = file.name;
                            option.textContent = `${file.name} (${formatFileSize(file.size)}, ${file.modified})`;
                            select.appendChild(option);
                        });
                    } catch (error) {
                        console.error('로그 파일 목록을 가져오는데 실패했습니다:', error);
                    }
                }

                // 파일 크기 포맷팅
                function formatFileSize(bytes) {
                    if (bytes === 0) return '0 Bytes';
                    const k = 1024;
                    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                    const i = Math.floor(Math.log(bytes) / Math.log(k));
                    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
                }

                // 로그 파일 변경
                function changeLogFile() {
                    const select = document.getElementById('logFileSelect');
                    const selectedFile = select.value;
                    if (selectedFile) {
                        ws.send(JSON.stringify({
                            type: 'select_log',
                            file_name: selectedFile
                        }));
                    }
                }

                // 페이지 로드 시 로그 파일 목록 가져오기
                loadLogFiles();
                
                // 30초마다 로그 파일 목록 새로고침
                setInterval(loadLogFiles, 30000);
            </script>
        </body>
    </html>
    """)

@app.get("/api/logs")
async def get_log_files():
    """로그 파일 목록을 반환합니다."""
    try:
        log_files = list(Path(LOG_DIR).glob("api_server_*.log"))
        return JSONResponse({
            "files": [{
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            } for f in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결을 처리합니다."""
    await websocket.accept()
    current_log_file = None
    
    try:
        # 초기 연결 시 가장 최근 로그 파일 선택
        current_log_file = get_latest_log_file()
        if not current_log_file:
            await websocket.send_json({
                "level": "ERROR",
                "message": "로그 파일을 찾을 수 없습니다.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return
        
        # 로그 파일 정보 전송
        await websocket.send_json({
            "type": "log_file",
            "file_name": current_log_file.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 파일 모니터링 시작
        await monitor_log_file(websocket, current_log_file)
        
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_json()
            
            if data.get("type") == "select_log":
                # 새로운 로그 파일 선택
                selected_file = data.get("file_name")
                if selected_file:
                    log_file = Path(LOG_DIR) / selected_file
                    if log_file.exists():
                        current_log_file = log_file
                        # 로그 파일 정보 전송
                        await websocket.send_json({
                            "type": "log_file",
                            "file_name": current_log_file.name,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        # 파일 모니터링 시작
                        await monitor_log_file(websocket, current_log_file)
                    
    except Exception as e:
        await websocket.send_json({
            "level": "ERROR",
            "message": f"오류 발생: {str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    finally:
        await websocket.close()

async def monitor_log_file(websocket: WebSocket, log_file: Path):
    """로그 파일을 모니터링하고 변경사항을 전송합니다."""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            # 파일 끝으로 이동
            f.seek(0, 2)
            
            while True:
                # 시스템 메트릭스 수집
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                try:
                    gpus = GPUtil.getGPUs()
                    gpu_percent = gpus[0].load * 100 if gpus else 0
                except:
                    gpu_percent = 0
                
                # 메트릭스 전송
                await websocket.send_json({
                    "type": "metrics",
                    "cpu": cpu_percent,
                    "memory": memory_percent,
                    "gpu": gpu_percent,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # 로그 파일 읽기
                line = f.readline()
                if line:
                    level = "INFO"
                    if "ERROR" in line:
                        level = "ERROR"
                    elif "WARNING" in line:
                        level = "WARNING"
                    elif "DEBUG" in line:
                        level = "DEBUG"
                        
                    await websocket.send_json({
                        "level": level,
                        "message": line.strip(),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                await asyncio.sleep(1)  # 1초마다 업데이트
                
    except Exception as e:
        await websocket.send_json({
            "level": "ERROR",
            "message": f"로그 파일 모니터링 중 오류 발생: {str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = datetime.now()
        self.restart_pending = False

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            current_time = datetime.now()
            # 중복 이벤트 방지를 위해 1초 대기
            if (current_time - self.last_modified).total_seconds() > 1:
                self.last_modified = current_time
                print(f"파일 변경 감지: {event.src_path}")
                self.restart_server()

    def restart_server(self):
        if not self.restart_pending:
            self.restart_pending = True
            print("서버 재시작 중...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

def start_file_watcher():
    event_handler = CodeChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    return observer

if __name__ == "__main__":
    import uvicorn
    
    # 파일 감시 시작
    observer = start_file_watcher()
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8003)
    except KeyboardInterrupt:
        observer.stop()
    observer.join() 