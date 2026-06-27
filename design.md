# Seeya Design 文档

## 1. 项目概述

Seeya 是一个本地运行的实时视觉识别 Web 应用。第一阶段目标是通过浏览器摄像头动态识别通用物体，并在屏幕画面上实时绘制检测框和标签。

第一阶段采用 Web 前端 + 本地后端架构：前端负责摄像头采集、画面展示、抽帧和标签叠加；后端负责加载物体检测模型并返回结构化检测结果。

## 2. 第一阶段目标

- 浏览器打开应用后可选择摄像头并启动识别。
- 系统能识别人、杯子、手机、书、键盘等通用物体。
- 检测结果以 bounding box + 标签 + 置信度形式叠加在实时画面上。
- 前端 UI 支持选择后端模型。
- UI 显示当前模型、连接状态、FPS、推理耗时、端到端延迟和检测对象列表。
- 默认稳定运行在 Mac 本机。

## 3. 技术架构

- 前端：`React + Vite + TypeScript`
- 后端：`FastAPI + WebSocket`
- 默认模型路径：`YOLO + ONNX Runtime CPU`
- 可选运行时：
  - `PyTorch + MPS`
  - `ONNX Runtime CoreML Execution Provider`
- 不纳入第一阶段默认目标：
  - MLX 推理后端
  - 自定义物体训练
  - 云部署
  - 视频存储
  - 用户登录和历史记录

## 4. 开发与运行环境

第一阶段默认使用本地开发环境，不把 Docker 作为主运行方式。

推荐环境：

- 后端使用本地 Python 虚拟环境：`backend/.venv`
- 前端使用本地 Node 依赖目录：`frontend/node_modules`
- 模型文件和推理缓存保存在本地目录，不提交到仓库。
- README 提供本地启动前端和后端的步骤。

选择本地虚拟环境作为默认方案的原因：

- 浏览器摄像头权限和本机后端联调更直接。
- Mac 上测试 `PyTorch + MPS`、`ONNX Runtime CoreML EP`、ONNX CPU 的性能更可靠。
- Docker Desktop 在 macOS 上访问摄像头、Apple GPU/ANE、Core ML 能力都不如本机环境自然。
- 第一阶段目标是验证实时识别链路和 Mac 本机性能，本地环境调试成本最低。

Docker 策略：

- Docker 不作为第一阶段实时摄像头和 Mac runtime benchmark 的主环境。
- 第二阶段可补充 Docker，用于 CI、后端 API 部署、团队环境复现或非 Mac CPU-only 部署。
- 如果后续添加 Docker，必须在 README 中明确说明：Docker 路径可能不支持摄像头直连、MPS、CoreML EP 或 Apple ANE 加速。

## 5. 前端设计

前端采用“实时识别工作台”布局。

顶部区域：

- 应用名：`Seeya`
- 摄像头选择
- 后端模型选择
- 后端连接状态
- 设置入口

主区域：

- 实时摄像头画面
- canvas 检测框叠加层
- 物体标签和置信度

右侧面板：

- 当前模型
- 模型加载状态
- 开始/停止识别
- 置信度阈值
- FPS
- 后端推理耗时
- 端到端延迟
- 检测对象列表

前端行为：

- 使用 `getUserMedia` 获取摄像头画面。
- 默认以 `5 FPS` 抽帧发送给后端。
- 使用 WebSocket 发送帧并接收检测结果。
- 只渲染最新 `frameId` 对应的检测结果，丢弃过期响应。
- 通过后端 `/settings` API 将前端参数设置持久化到 SQLite 数据库。

## 6. 前端模型选择

前端必须提供选择后端模型的能力，但模型列表由后端提供，前端不硬编码模型文件路径。

UI 行为：

- 顶部或右侧面板显示“后端模型”下拉选择。
- 默认选择后端推荐模型。
- 模型未下载或依赖不可用时显示禁用态和原因。
- 切换模型时显示“模型加载中”。
- 识别中不允许直接切换模型；前端应先停止识别，再切换模型，并由用户手动重新开始。
- 前端通过后端 `/settings` API 记住上次选择的 `modelId`。

模型元数据：

```json
{
  "id": "yolo-small-onnx-cpu",
  "name": "YOLO Small ONNX CPU",
  "runtime": "onnx-cpu",
  "size": "small",
  "inputSize": 640,
  "available": true,
  "recommended": true,
  "unavailableReason": null
}
```

## 7. 后端设计

后端负责模型管理、推理和 WebSocket 通信。

核心职责：

- 启动时发现本地可用模型。
- 提供模型列表接口。
- 支持按前端选择切换模型。
- 接收前端发送的视频帧。
- 执行物体检测。
- 返回 JSON 检测结果，不返回已绘制图片。

推荐模块：

- `app/main.py`：FastAPI 入口和路由。
- `app/model_registry.py`：模型发现和元数据管理。
- `app/detectors/base.py`：Detector 抽象接口。
- `app/detectors/onnx_detector.py`：ONNX Runtime 实现。
- `app/detectors/torch_detector.py`：PyTorch/MPS 可选实现。
- `app/schemas.py`：接口数据结构。
- `scripts/benchmark.py`：模型性能测试脚本。

## 8. API 设计

### `GET /health`

返回后端、当前模型和运行时状态。

### `GET /models`

返回可选模型列表。

响应示例：

```json
{
  "models": [
    {
      "id": "yolo-small-onnx-cpu",
      "name": "YOLO Small ONNX CPU",
      "runtime": "onnx-cpu",
      "size": "small",
      "inputSize": 640,
      "available": true,
      "recommended": true,
      "unavailableReason": null
    }
  ],
  "selectedModelId": "yolo-small-onnx-cpu"
}
```

### `POST /models/select`

切换当前模型。

请求：

```json
{
  "modelId": "yolo-small-onnx-cpu"
}
```

规则：

- 识别中不允许直接切换模型。
- 前端应先停止识别，再切换模型。
- 模型不可用时返回明确错误原因。

### `WS /ws/detect`

前端发送帧，后端返回检测结果。

返回结构：

```json
{
  "frameId": 123,
  "modelId": "yolo-small-onnx-cpu",
  "inferenceMs": 42,
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

## 9. 模型与性能策略

默认模型：

- `YOLO Small ONNX CPU`
- 输入尺寸优先使用 `640`
- 默认抽帧 `5 FPS`

选择 ONNX CPU 作为默认路径的原因：

- Mac 上兼容性最好。
- 安装和运行稳定。
- 适合第一阶段 MVP。
- 对通用物体识别和低帧率实时标签足够可用。

CoreML EP 支持策略：

- 可作为 ONNX Runtime 的 Apple-native 优化路径。
- 需要验证 YOLO 模型算子、shape、首次编译和检测结果一致性。
- 第一阶段作为可选优化，不作为默认路径。

MLX 策略：

- MLX 不作为第一阶段默认目标。
- 原因是 Ultralytics YOLO 没有官方 MLX 推理后端。
- 后续若采用第三方 YOLO-MLX 或自研移植，再单独评估。

benchmark 要求：

- 增加 `scripts/benchmark.py`。
- 对比 ONNX Runtime CPU 和可选 ONNX Runtime CoreML EP。
- 输出模型 ID、runtime、输入尺寸、平均推理耗时、P95 推理耗时和估算 FPS。

## 10. 参考项目

主要参考：

- [mohamedabubasith/realtime-object-detection](https://github.com/mohamedabubasith/realtime-object-detection)

借鉴点：

- `FastAPI + React + Vite + YOLO/ONNX` 项目组织。
- 性能参数设计。
- 模型状态面板。
- benchmark 和运行时配置思路。

辅助参考：

- [shriya7756/fastapi-object-detection](https://github.com/shriya7756/fastapi-object-detection)
- [sitammeur/YoloDetect](https://github.com/sitammeur/YoloDetect)
- [Dheeraj9811/DetectNet-Web](https://github.com/Dheeraj9811/DetectNet-Web)

不直接 fork。Seeya 应按自己的目标重建，避免继承上传视频、RTSP、MJPEG 服务端绘制等第一阶段不需要的复杂度。

## 11. 测试计划

后端测试：

- `/health` 返回当前模型状态。
- `/models` 返回模型列表和推荐模型。
- `/models/select` 能切换可用模型，并拒绝不可用模型。
- 单张测试图片可成功推理。
- WebSocket 输入帧后返回合法检测结果。

前端测试：

- 摄像头权限成功和失败状态。
- 摄像头选择。
- 模型列表加载。
- 模型不可用禁用态。
- 模型切换加载态。
- 开始/停止识别流程。
- canvas 检测框随视频尺寸正确缩放。
- 置信度阈值调整后检测结果变化。

集成验证：

- 本机启动前后端。
- 浏览器打开应用后可看到摄像头画面。
- 默认模型可开始识别。
- 常见物体能被检测并显示标签。
- 切换另一个可用模型后重新识别，结果携带新的 `modelId`。
- 断开后端后 UI 显示错误，并可恢复连接。

性能验证：

- 记录 ONNX CPU 默认配置下的 FPS、平均推理耗时、P95 推理耗时。
- 对比 PyTorch MPS 和 CoreML EP。
- 根据实测结果决定后续是否调整默认模型。

## 12. 第一阶段验收标准

- 用户可以在浏览器中启动摄像头。
- 用户可以选择后端模型。
- 默认模型可成功加载。
- 点击开始识别后，画面中常见物体出现检测框和标签。
- UI 能显示当前模型、FPS、延迟和检测列表。
- 断开后端或模型不可用时，有清晰错误提示。
- 摄像头视频不保存到磁盘。
- README 能指导本地启动前端和后端。
