import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, Camera, CircleStop, Play, Settings, Wifi, WifiOff } from "lucide-react";
import type { Detection, DetectResponse, ModelInfo } from "./types";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8010";
const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");
const MODEL_STORAGE_KEY = "seeya.selectedModelId";
const LANGUAGE_STORAGE_KEY = "seeya.language";
const APPEARANCE_STORAGE_KEY = "seeya.appearance";
const TARGET_FPS = 5;
const CAMERA_OFF_ID = "camera-off";

type Language = "en" | "zh";
type Appearance = "system" | "dark" | "light";

const copy = {
  en: {
    camera: "Camera",
    cameraOff: "Camera Off",
    noCameraFound: "No camera found",
    backendModel: "Backend Model",
    unavailable: "unavailable",
    connected: "Connected",
    disconnected: "Disconnected",
    idle: "Idle",
    liveCamera: "Live camera",
    currentModel: "Current model",
    noModelSelected: "No model selected",
    modelAvailable: "Model is available for local inference.",
    modelStatus: "Model status",
    startRecognition: "Start recognition",
    stopRecognition: "Stop recognition",
    confidenceThreshold: "Confidence threshold",
    inference: "Inference",
    latency: "Latency",
    detections: "Detections",
    objects: "objects",
    noObjects: "No objects above threshold.",
    label: "Label",
    confidence: "Confidence",
    settings: "Settings",
    language: "Language",
    english: "English",
    chinese: "中文",
    appearance: "Appearance",
    system: "System",
    dark: "Dark",
    light: "Light",
    selectCamera: "Select a camera and start recognition",
    selectCameraError: "Select a camera before starting recognition",
    cameraUnavailable: "getUserMedia is unavailable in this browser",
    unableStartCamera: "Unable to start camera",
    backendUnavailable: "Unable to load backend models",
    selectModelError: "Select an available model first",
    websocketFailed: "WebSocket connection failed",
    ready: "Ready",
    running: "Running",
    loading: "Loading",
    error: "Error",
    notLoaded: "Not loaded",
    backendDown: "Backend unavailable",
    cameraIdle: "Camera idle",
    cameraActive: "Camera active",
    cameraApiUnavailable: "Camera API unavailable",
    cameraPermissionNeeded: "Camera permission needed",
  },
  zh: {
    camera: "摄像头",
    cameraOff: "关闭摄像头",
    noCameraFound: "未找到摄像头",
    backendModel: "后端模型",
    unavailable: "不可用",
    connected: "已连接",
    disconnected: "已断开",
    idle: "空闲",
    liveCamera: "实时画面",
    currentModel: "当前模型",
    noModelSelected: "未选择模型",
    modelAvailable: "模型可用于本地推理。",
    modelStatus: "模型状态",
    startRecognition: "开始识别",
    stopRecognition: "停止识别",
    confidenceThreshold: "置信度阈值",
    inference: "推理耗时",
    latency: "延迟",
    detections: "检测对象",
    objects: "个对象",
    noObjects: "没有超过阈值的对象。",
    label: "标签",
    confidence: "置信度",
    settings: "设置",
    language: "语言",
    english: "English",
    chinese: "中文",
    appearance: "外观",
    system: "系统",
    dark: "深色",
    light: "浅色",
    selectCamera: "选择摄像头并开始识别",
    selectCameraError: "开始识别前请选择摄像头",
    cameraUnavailable: "当前浏览器不支持 getUserMedia",
    unableStartCamera: "无法启动摄像头",
    backendUnavailable: "无法加载后端模型",
    selectModelError: "请先选择可用模型",
    websocketFailed: "WebSocket 连接失败",
    ready: "就绪",
    running: "运行中",
    loading: "加载中",
    error: "错误",
    notLoaded: "未加载",
    backendDown: "后端不可用",
    cameraIdle: "摄像头空闲",
    cameraActive: "摄像头已开启",
    cameraApiUnavailable: "摄像头 API 不可用",
    cameraPermissionNeeded: "需要摄像头权限",
  },
} satisfies Record<Language, Record<string, string>>;

function readStoredLanguage(): Language {
  return window.localStorage.getItem(LANGUAGE_STORAGE_KEY) === "zh" ? "zh" : "en";
}

function readStoredAppearance(): Appearance {
  const value = window.localStorage.getItem(APPEARANCE_STORAGE_KEY);
  return value === "dark" || value === "light" || value === "system" ? value : "system";
}

function normalizeStatus(status: string): string {
  return status.toLowerCase().replace(/\s+/g, "-");
}

function App() {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const overlayRef = React.useRef<HTMLCanvasElement | null>(null);
  const captureRef = React.useRef<HTMLCanvasElement | null>(null);
  const socketRef = React.useRef<WebSocket | null>(null);
  const latestFrameRef = React.useRef(0);
  const sentAtRef = React.useRef(new Map<number, number>());

  const [models, setModels] = React.useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = React.useState("");
  const [cameras, setCameras] = React.useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = React.useState("");
  const [cameraState, setCameraState] = React.useState("Camera idle");
  const [connectionState, setConnectionState] = React.useState<"idle" | "connected" | "error">("idle");
  const [modelStatus, setModelStatus] = React.useState("Not loaded");
  const [isRunning, setIsRunning] = React.useState(false);
  const [threshold, setThreshold] = React.useState(0.45);
  const [detections, setDetections] = React.useState<Detection[]>([]);
  const [fps, setFps] = React.useState(0);
  const [inferenceMs, setInferenceMs] = React.useState(0);
  const [latencyMs, setLatencyMs] = React.useState(0);
  const [error, setError] = React.useState("");
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [language, setLanguage] = React.useState<Language>(() => readStoredLanguage());
  const [appearance, setAppearance] = React.useState<Appearance>(() => readStoredAppearance());
  const [systemPrefersDark, setSystemPrefersDark] = React.useState(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  const t = copy[language];
  const effectiveAppearance = appearance === "system" ? (systemPrefersDark ? "dark" : "light") : appearance;

  const selectedModel = React.useMemo(
    () => models.find((model) => model.id === selectedModelId),
    [models, selectedModelId],
  );

  const displayModelStatus = React.useMemo(() => {
    const statusKey = normalizeStatus(modelStatus);
    if (statusKey === "ready") return t.ready;
    if (statusKey === "running") return t.running;
    if (statusKey === "loading") return t.loading;
    if (statusKey === "error") return t.error;
    if (statusKey === "not-loaded") return t.notLoaded;
    if (statusKey === "backend-unavailable") return t.backendDown;
    return modelStatus;
  }, [modelStatus, t]);

  const displayCameraState = React.useMemo(() => {
    const stateKey = normalizeStatus(cameraState);
    if (stateKey === "camera-idle") return t.cameraIdle;
    if (stateKey === "camera-active") return t.cameraActive;
    if (stateKey === "camera-api-unavailable") return t.cameraApiUnavailable;
    if (stateKey === "camera-permission-needed") return t.cameraPermissionNeeded;
    return cameraState;
  }, [cameraState, t]);

  React.useEffect(() => {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  }, [language]);

  React.useEffect(() => {
    window.localStorage.setItem(APPEARANCE_STORAGE_KEY, appearance);
  }, [appearance]);

  React.useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => setSystemPrefersDark(media.matches);
    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  const loadModels = React.useCallback(async () => {
    const response = await fetch(`${API_BASE_URL}/models`);
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const data = (await response.json()) as { models: ModelInfo[]; selectedModelId: string };
    const savedModelId = window.localStorage.getItem(MODEL_STORAGE_KEY);
    const savedModel = data.models.find((model) => model.id === savedModelId && model.available);
    setModels(data.models);
    setSelectedModelId(savedModel?.id ?? data.selectedModelId);
    setModelStatus("Ready");
  }, []);

  const refreshCameras = React.useCallback(async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((device) => device.kind === "videoinput");
    setCameras(videoDevices);
    setSelectedCameraId((current) => {
      if (current === CAMERA_OFF_ID || videoDevices.some((camera) => camera.deviceId === current)) {
        return current;
      }
      return videoDevices[0]?.deviceId || CAMERA_OFF_ID;
    });
  }, []);

  React.useEffect(() => {
    loadModels().catch((err: unknown) => {
      setConnectionState("error");
      setModelStatus("Backend unavailable");
      setError(err instanceof Error ? err.message : t.backendUnavailable);
    });
  }, [loadModels, t.backendUnavailable]);

  React.useEffect(() => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setCameraState("Camera API unavailable");
      return;
    }
    refreshCameras().catch(() => setCameraState("Camera permission needed"));
  }, [refreshCameras]);

  React.useEffect(() => {
    drawOverlay(videoRef.current, overlayRef.current, detections);
  }, [detections]);

  const startCamera = React.useCallback(async () => {
    if (selectedCameraId === CAMERA_OFF_ID) {
      throw new Error(t.selectCameraError);
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(t.cameraUnavailable);
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: selectedCameraId ? { deviceId: { exact: selectedCameraId } } : true,
      audio: false,
    });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
    setCameraState("Camera active");
    await refreshCameras();
  }, [refreshCameras, selectedCameraId, t.cameraUnavailable, t.selectCameraError]);

  const stopCamera = React.useCallback(() => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState("Camera idle");
  }, []);

  const stopRecognition = React.useCallback(() => {
    setIsRunning(false);
    socketRef.current?.close();
    socketRef.current = null;
    setConnectionState("idle");
    setFps(0);
  }, []);

  const handleCameraChange = React.useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      const nextCameraId = event.target.value;
      setSelectedCameraId(nextCameraId);
      if (nextCameraId !== CAMERA_OFF_ID) {
        return;
      }

      stopRecognition();
      stopCamera();
      setDetections([]);
      setInferenceMs(0);
      setLatencyMs(0);
      setModelStatus((current) => (current === "Running" || current === "Loading" ? "Ready" : current));
      setError("");
    },
    [stopCamera, stopRecognition],
  );

  const handleStart = React.useCallback(async () => {
    if (!selectedModel?.available) {
      setError(selectedModel?.unavailableReason ?? t.selectModelError);
      return;
    }
    setError("");
    setModelStatus("Loading");
    try {
      await startCamera();
    } catch (err) {
      setModelStatus("Ready");
      setError(err instanceof Error ? err.message : t.unableStartCamera);
      return;
    }
    const socket = new WebSocket(`${WS_BASE_URL}/ws/detect`);
    socketRef.current = socket;

    socket.onopen = () => {
      setConnectionState("connected");
      setModelStatus("Running");
      setIsRunning(true);
    };
    socket.onerror = () => {
      setConnectionState("error");
      setError(t.websocketFailed);
      setIsRunning(false);
    };
    socket.onclose = () => {
      setConnectionState((current) => (current === "error" ? "error" : "idle"));
      setIsRunning(false);
      setModelStatus("Ready");
    };
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as DetectResponse | { error: string };
      if ("error" in data) {
        setError(data.error);
        return;
      }
      if (data.frameId < latestFrameRef.current) {
        return;
      }
      latestFrameRef.current = data.frameId;
      const sentAt = sentAtRef.current.get(data.frameId);
      if (sentAt) {
        setLatencyMs(Math.round(performance.now() - sentAt));
        sentAtRef.current.delete(data.frameId);
      }
      setInferenceMs(data.inferenceMs);
      setDetections(data.detections);
    };
  }, [selectedModel, startCamera, t.selectModelError, t.unableStartCamera, t.websocketFailed]);

  React.useEffect(() => {
    if (!isRunning) {
      return;
    }
    let frameId = latestFrameRef.current;
    let sentFrames = 0;
    const fpsStartedAt = performance.now();

    const interval = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = captureRef.current;
      const socket = socketRef.current;
      if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.videoWidth === 0) {
        return;
      }
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) {
        return;
      }
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      frameId += 1;
      const imageData = canvas.toDataURL("image/jpeg", 0.72);
      sentAtRef.current.set(frameId, performance.now());
      socket.send(
        JSON.stringify({
          frameId,
          imageData,
          width: canvas.width,
          height: canvas.height,
          modelId: selectedModelId,
          confidenceThreshold: threshold,
        }),
      );
      sentFrames += 1;
      const elapsedSeconds = (performance.now() - fpsStartedAt) / 1000;
      setFps(Math.round((sentFrames / Math.max(elapsedSeconds, 1)) * 10) / 10);
    }, 1000 / TARGET_FPS);

    return () => window.clearInterval(interval);
  }, [isRunning, selectedModelId, threshold]);

  const handleModelChange = async (event: React.ChangeEvent<HTMLSelectElement>) => {
    const nextModelId = event.target.value;
    const nextModel = models.find((model) => model.id === nextModelId);
    if (!nextModel?.available || isRunning) {
      return;
    }
    setModelStatus("Loading");
    const response = await fetch(`${API_BASE_URL}/models/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelId: nextModelId }),
    });
    if (!response.ok) {
      setError(await response.text());
      setModelStatus("Error");
      return;
    }
    window.localStorage.setItem(MODEL_STORAGE_KEY, nextModelId);
    setSelectedModelId(nextModelId);
    setModelStatus("Ready");
  };

  const connectionLabel = connectionState === "connected" ? t.connected : connectionState === "error" ? t.disconnected : t.idle;

  return (
    <main className={`app-shell theme-${effectiveAppearance}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Activity size={18} /></div>
          <span>Seeya</span>
        </div>
        <label className="field">
          <span>{t.camera}</span>
          <select value={selectedCameraId} onChange={handleCameraChange}>
            <option value={CAMERA_OFF_ID}>{t.cameraOff}</option>
            {cameras.length === 0 ? <option disabled>{t.noCameraFound}</option> : null}
            {cameras.map((camera, index) => (
              <option key={camera.deviceId || index} value={camera.deviceId}>
                {camera.label || `${t.camera} ${index + 1}`}
              </option>
            ))}
          </select>
        </label>
        <label className="field model-field">
          <span>{t.backendModel}</span>
          <select value={selectedModelId} onChange={handleModelChange} disabled={isRunning || models.length === 0}>
            {models.map((model) => (
              <option key={model.id} value={model.id} disabled={!model.available}>
                {model.name}{model.available ? "" : ` (${t.unavailable})`}
              </option>
            ))}
          </select>
        </label>
        <div className={`connection ${connectionState}`}>
          {connectionState === "connected" ? <Wifi size={16} /> : <WifiOff size={16} />}
          {connectionLabel}
        </div>
        <div className="settings-wrap">
          <button
            className="icon-button"
            aria-label={t.settings}
            aria-expanded={settingsOpen}
            onClick={() => setSettingsOpen((current) => !current)}
          >
            <Settings size={18} />
          </button>
          {settingsOpen ? (
            <div className="settings-popover" role="dialog" aria-label={t.settings}>
              <div className="settings-title">{t.settings}</div>
              <label className="settings-control">
                <span>{t.language}</span>
                <select value={language} onChange={(event) => setLanguage(event.target.value as Language)}>
                  <option value="en">{t.english}</option>
                  <option value="zh">{t.chinese}</option>
                </select>
              </label>
              <label className="settings-control">
                <span>{t.appearance}</span>
                <select value={appearance} onChange={(event) => setAppearance(event.target.value as Appearance)}>
                  <option value="system">{t.system}</option>
                  <option value="dark">{t.dark}</option>
                  <option value="light">{t.light}</option>
                </select>
              </label>
            </div>
          ) : null}
        </div>
      </header>

      <section className="workspace">
        <section className="viewer">
          <div className="viewer-toolbar">
            <div>
              <strong>{t.liveCamera}</strong>
              <span>{displayCameraState}</span>
            </div>
            <div className="runtime-chip">{selectedModel?.runtime ?? "runtime"}</div>
          </div>
          <div className="video-stage">
            <video ref={videoRef} muted playsInline />
            <canvas ref={overlayRef} className="overlay" />
            {!isRunning ? (
              <div className="empty-state">
                <Camera size={42} />
                <p>{t.selectCamera}</p>
              </div>
            ) : null}
          </div>
        </section>

        <aside className="side-panel">
          <div className="panel-section">
            <span className="panel-label">{t.currentModel}</span>
            <h2>{selectedModel?.name ?? t.noModelSelected}</h2>
            <p>{selectedModel?.available ? t.modelAvailable : selectedModel?.unavailableReason}</p>
          </div>

          <div className="model-state-row">
            <span>{t.modelStatus}</span>
            <strong>{displayModelStatus}</strong>
          </div>

          <button className="start-button" onClick={handleStart} disabled={isRunning}>
            <Play size={18} />
            {t.startRecognition}
          </button>

          <button className="stop-button" onClick={stopRecognition} disabled={!isRunning}>
            <CircleStop size={18} />
            {t.stopRecognition}
          </button>

          <label className="threshold">
            <span>{t.confidenceThreshold}</span>
            <strong>{threshold.toFixed(2)}</strong>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </label>

          <div className="status-grid">
            <Metric label="FPS" value={fps.toFixed(1)} />
            <Metric label={t.inference} value={`${inferenceMs.toFixed(1)} ms`} />
            <Metric label={t.latency} value={`${latencyMs} ms`} />
          </div>

          {error ? <div className="error-box">{error}</div> : null}

          <div className="detections">
            <div className="detections-header">
              <span>{t.detections}</span>
              <strong>{language === "zh" ? `${detections.length} ${t.objects}` : `${detections.length} ${t.objects}`}</strong>
            </div>
            {detections.length === 0 ? (
              <p className="muted">{t.noObjects}</p>
            ) : (
              <div className="detection-table">
                <div className="detection-table-head">
                  <span>#</span>
                  <span>{t.label}</span>
                  <span>{t.confidence}</span>
                  <span>BBox</span>
                </div>
                {detections.map((detection, index) => (
                  <div className="detection-row" key={`${detection.label}-${detection.box.x}`}>
                    <span>{index + 1}</span>
                    <span className="label-cell">{detection.label}</span>
                    <strong>{detection.confidence.toFixed(2)}</strong>
                    <span>
                      {Math.round(detection.box.x)}, {Math.round(detection.box.y)}, {Math.round(detection.box.width)},{" "}
                      {Math.round(detection.box.height)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </section>
      <canvas ref={captureRef} className="capture-canvas" />
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function drawOverlay(video: HTMLVideoElement | null, canvas: HTMLCanvasElement | null, detections: Detection[]) {
  if (!video || !canvas) {
    return;
  }
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || video.videoWidth === 0 || video.videoHeight === 0) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;

  detections.forEach((detection) => {
    const x = detection.box.x * scaleX;
    const y = detection.box.y * scaleY;
    const width = detection.box.width * scaleX;
    const height = detection.box.height * scaleY;
    const label = `${detection.label} ${Math.round(detection.confidence * 100)}%`;

    ctx.strokeStyle = "#18f6a4";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    ctx.font = "600 13px Inter, system-ui, sans-serif";
    const textWidth = ctx.measureText(label).width + 14;
    ctx.fillStyle = "rgba(4, 20, 18, 0.92)";
    ctx.fillRect(x, Math.max(0, y - 28), textWidth, 24);
    ctx.fillStyle = "#bfffe8";
    ctx.fillText(label, x + 7, Math.max(16, y - 11));
  });
}

createRoot(document.getElementById("root")!).render(<App />);
