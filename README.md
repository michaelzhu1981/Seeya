# Seeya

Seeya 是一个本地运行的实时视觉识别工作台。它通过浏览器摄像头采集画面，使用 React/Vite 前端展示视频与检测框，再通过本地 FastAPI 后端完成 YOLO/ONNX 物体检测，并可把特定人物事件截图发送给 OpenAI 兼容的 LM Studio 视觉模型生成中文描述。

项目当前重点是 Mac 本机实时识别体验：摄像头权限、低延迟 WebSocket 推理、模型切换、检测结果叠加、LM Studio 视觉事件、历史记录与本地数据保留。

## 功能特性

- 浏览器摄像头实时预览，支持选择摄像头和关闭摄像头。
- 前端按默认 `5 FPS` 抽帧，通过 WebSocket 发送到后端推理。
- 画面上叠加 bounding box、对象标签和置信度。
- 支持后端模型列表、模型可用性提示和模型切换。
- 内置 `demo-local-detector`，即使没有真实 ONNX 模型也能验证前后端实时链路。
- 支持 YOLO ONNX CPU 推理，以及可用时的 ONNX Runtime CoreML Execution Provider。
- 支持中英文界面、深色/浅色/跟随系统外观、置信度阈值等设置。
- 设置通过 SQLite 持久化，包括摄像头、模型、语言、外观、阈值、LM Studio 配置和历史保留时间。
- 当检测到新的 `person` 或已跟踪的 `person` 发生移动时，可触发截图分析。
- LM Studio 视觉分析结果会保存为历史记录，并带有过期清理、截图保存和重复事件合并逻辑。

## 技术栈

- 前端：React 19、Vite 6、TypeScript、Lucide React
- 后端：FastAPI、WebSocket、Pydantic、SQLite
- 推理：ONNX Runtime、YOLO ONNX、Pillow、NumPy
- 测试：pytest、FastAPI TestClient、Vite/TypeScript build

## 目录结构

```text
.
├── backend/
│   ├── app/
│   │   ├── detectors/
│   │   │   ├── base.py
│   │   │   ├── demo_detector.py
│   │   │   └── onnx_detector.py
│   │   ├── main.py
│   │   ├── model_registry.py
│   │   ├── schemas.py
│   │   ├── settings_store.py
│   │   └── vision_store.py
│   ├── models/
│   │   └── .gitkeep
│   ├── scripts/
│   │   └── benchmark.py
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── styles.css
│   │   └── types.ts
│   └── package.json
├── design.md
├── run-dev.sh
└── README.md
```

## 环境要求

推荐在本机直接运行，不把 Docker 作为默认开发路径。这样浏览器摄像头权限、Mac 上的 CoreML/MPS/ANE 能力和本地推理调试都会更直接。

需要准备：

- Python 3.10 或更新版本
- Node.js 18 或更新版本
- npm
- 支持摄像头权限的现代浏览器
- 可选：LM Studio，用于本地视觉模型分析
- 可选：Ultralytics CLI，用于导出 YOLO ONNX 模型

## 快速启动

在仓库根目录运行：

```bash
./run-dev.sh
```

脚本会自动：

- 创建或复用 `backend/.venv`
- 安装后端依赖
- 安装前端依赖
- 启动后端 `http://127.0.0.1:8010`
- 启动前端 `http://127.0.0.1:5173`

打开 `http://127.0.0.1:5173`，允许浏览器访问摄像头，然后选择摄像头和可用模型即可开始识别。按 `Ctrl+C` 会同时停止前后端服务。

如果没有准备真实 ONNX 模型，后端会自动选择 `demo-local-detector`，用于验证摄像头、WebSocket 和 UI 叠加流程。

`run-dev.sh` 支持通过环境变量覆盖默认地址：

```bash
BACKEND_HOST=127.0.0.1 BACKEND_PORT=8010 FRONTEND_HOST=127.0.0.1 FRONTEND_PORT=5173 ./run-dev.sh
```

## 手动启动后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

健康检查：

```bash
curl http://127.0.0.1:8010/health
```

## 手动启动前端

```bash
cd frontend
npm install
npm run dev
```

前端默认连接 `http://127.0.0.1:8010`。如需连接其它后端地址：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8010 npm run dev
```

## 模型准备

后端启动时会扫描 `backend/models/`，根据本地文件和运行时能力生成模型列表。真实 YOLO ONNX 模型不会提交到仓库，需要自行下载或导出。

当前模型 ID：

- `yolo-small-onnx-cpu`：推荐的高准确率默认模型，文件路径 `backend/models/yolo-small.onnx`
- `yolo11l-onnx-cpu`：YOLO11 Large ONNX CPU，文件路径 `backend/models/yolo11l.onnx`
- `yolo11x-onnx-cpu`：YOLO11 X ONNX CPU，文件路径 `backend/models/yolo11x.onnx`
- `yolo-small-onnx-coreml`、`yolo11l-onnx-coreml`、`yolo11x-onnx-coreml`：CoreML EP 可用时出现
- `demo-local-detector`：内置演示模型，总是可用

安装导出工具：

```bash
cd backend
source .venv/bin/activate
python -m pip install ultralytics
```

导出推荐模型：

```bash
mkdir -p models
yolo export model=yolov8s.pt format=onnx imgsz=640
mv yolov8s.onnx models/yolo-small.onnx
```

导出可选 YOLO11 模型：

```bash
yolo export model=yolo11l.pt format=onnx imgsz=640
mv yolo11l.onnx models/yolo11l.onnx

yolo export model=yolo11x.pt format=onnx imgsz=640
mv yolo11x.onnx models/yolo11x.onnx
```

`onnxruntime` 是推理引擎，不包含训练好的权重。只有当对应 `.onnx` 文件存在时，真实 YOLO 模型才会显示为可用。

## LM Studio 视觉事件

Seeya 可以在人物事件发生时截取当前画面，并请求 LM Studio 的 OpenAI 兼容接口进行图像描述。默认地址为：

```text
http://192.168.4.181:1234/v1
```

默认模型 ID 为：

```text
qwen/qwen3-v1-4b
```

使用方式：

1. 在 LM Studio 中启动支持图像输入的本地模型，并开启 OpenAI Compatible Server。
2. 在 Seeya 右侧设置面板中填写 LM Studio URL。
3. 点击检测模型，前端会通过后端请求 `{baseUrl}/models`。
4. 选择视觉模型并保存提示词。
5. 当新人物出现或人物移动超过阈值时，前端会向后端发送截图分析请求。

后端会调用 `{baseUrl}/chat/completions`。截图只会发送到你配置的 LM Studio 地址；同时，为了历史记录预览，截图会保存在本机数据目录中，并按保留时间自动清理。

## 数据存储

默认本地数据目录为：

```text
backend/data/
```

其中包含：

- `seeya.db`：SQLite 数据库，保存前端设置和视觉事件记录
- `screenshots/`：视觉事件截图

可以通过环境变量修改数据目录：

```bash
SEEYA_DATA_DIR=/path/to/seeya-data uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

视觉历史默认保留 1 天，可在 UI 中调整。后端也支持用环境变量设置默认值：

```bash
SEEYA_VISION_RETENTION_DAYS=7 uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

`backend/data/`、模型文件、虚拟环境和前端依赖都已加入 `.gitignore`，不会被提交。

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BACKEND_HOST` | `127.0.0.1` | `run-dev.sh` 启动后端时使用的监听地址 |
| `BACKEND_PORT` | `8010` | `run-dev.sh` 启动后端时使用的端口 |
| `FRONTEND_HOST` | `127.0.0.1` | `run-dev.sh` 启动 Vite 时使用的监听地址 |
| `FRONTEND_PORT` | `5173` | `run-dev.sh` 启动 Vite 时使用的端口 |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8010` | 前端访问后端 REST API 和 WebSocket 的基址 |
| `SEEYA_DATA_DIR` | `backend/data` | 后端 SQLite 数据库和截图保存目录 |
| `SEEYA_VISION_RETENTION_DAYS` | `1` | 视觉事件默认保留天数 |

## 后端 API

基础状态与模型：

- `GET /health`：返回服务状态、当前模型和运行时信息。
- `GET /models`：返回模型列表、可用性、推荐状态和当前选中模型。
- `POST /models/select`：切换当前模型，请求体为 `{ "modelId": "..." }`。

前端设置：

- `GET /settings`：读取前端设置。
- `PUT /settings`：保存前端设置。

LM Studio 与视觉事件：

- `POST /vision/models`：根据 `baseUrl` 读取 LM Studio 模型列表。
- `POST /vision/analyze`：发送截图和检测上下文，返回视觉模型描述。
- `GET /vision/events`：查询历史事件，支持 `startAt`、`endAt`、`keyword`、`limit`。
- `DELETE /vision/events`：清空历史事件和本地截图。
- `GET /vision/events/{event_id}`：读取单条历史事件。
- `GET /vision/events/{event_id}/screenshot`：读取历史事件截图。

实时检测：

- `WS /ws/detect`：接收前端帧数据，返回 `frameId`、`modelId`、`inferenceMs` 和检测结果。

WebSocket 请求示意：

```json
{
  "frameId": 1,
  "imageData": "data:image/jpeg;base64,...",
  "width": 1280,
  "height": 720,
  "modelId": "yolo-small-onnx-cpu",
  "confidenceThreshold": 0.55
}
```

WebSocket 响应示意：

```json
{
  "frameId": 1,
  "modelId": "yolo-small-onnx-cpu",
  "inferenceMs": 42.5,
  "detections": [
    {
      "label": "person",
      "confidence": 0.96,
      "box": {
        "x": 120,
        "y": 80,
        "width": 240,
        "height": 420
      }
    }
  ]
}
```

## 开发与验证

运行后端测试：

```bash
cd backend
source .venv/bin/activate
python -m pytest
```

运行前端构建：

```bash
cd frontend
npm run build
```

目前前端没有单独的测试脚本，`npm run build` 是主要的前端类型检查和构建验证方式。

运行模型 benchmark：

```bash
cd backend
source .venv/bin/activate
python scripts/benchmark.py
```

benchmark 会输出模型 ID、runtime、输入尺寸、平均推理耗时、P95 推理耗时和估算 FPS。

## 常见问题

摄像头无法打开：

- 确认浏览器允许 `http://127.0.0.1:5173` 使用摄像头。
- 确认没有其它应用独占摄像头。
- 优先使用本机运行方式，不建议用 Docker 跑第一阶段摄像头链路。

真实 YOLO 模型不可用：

- 确认 `.onnx` 文件放在 `backend/models/` 下的指定路径。
- 确认后端虚拟环境已安装 `onnxruntime`。
- 刷新页面或重启后端，让模型注册表重新扫描。

CoreML EP 不可用：

- 确认当前 `onnxruntime` 构建支持 `CoreMLExecutionProvider`。
- 即使 CoreML EP 不可用，ONNX CPU 和 `demo-local-detector` 仍可使用。

LM Studio 请求失败：

- 确认 LM Studio OpenAI Compatible Server 已启动。
- 确认 URL 以 `http://` 或 `https://` 开头，并包含 `/v1`。
- 确认选择的模型支持图像输入。

历史记录没有截图：

- 只有触发 LM Studio 视觉事件时才会保存截图。
- 过期记录和截图会按保留时间清理。
- 手动清空历史会删除所有事件和截图。

## 隐私说明

实时摄像头帧会发送到本机 FastAPI 后端进行推理，不会作为普通检测流程保存到磁盘。只有触发 LM Studio 视觉事件时，截图才会发送到配置的 LM Studio 地址，并在本地保存为历史记录截图，直到过期或被手动清理。

## 相关文档

- `design.md`：项目第一阶段设计文档，包含目标、架构、API 设计和测试计划。

## License

This project is intended to be licensed under the MIT License. Add a `LICENSE` file before distributing it as an open-source package.
