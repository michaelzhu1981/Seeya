import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, Camera, CircleStop, FileText, Image, Play, RefreshCw, Search, Settings, Wifi, WifiOff, X } from "lucide-react";
import type {
  AppSettings,
  Detection,
  DetectResponse,
  ModelInfo,
  VisionAnalyzeResponse,
  VisionEventRecord,
  VisionEventsResponse,
  VisionModelInfo,
  VisionModelsResponse,
  VisionTriggerSettings,
} from "./types";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8010";
const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");
const TARGET_FPS = 5;
const CAMERA_OFF_ID = "camera-off";
const DEFAULT_CONFIDENCE_THRESHOLD = 0.55;
const FRAME_JPEG_QUALITY = 0.88;
const DEFAULT_LM_STUDIO_URL = "http://192.168.4.181:1234/v1";
const DEFAULT_LM_STUDIO_MODEL = "qwen/qwen3-v1-4b";
const DEFAULT_LM_STUDIO_PROMPT = "请用中文简洁描述截图中可见的人、动作、位置变化和明显风险。";
const DEFAULT_HISTORY_RETENTION_DAYS = 1;
const MAX_VISION_MESSAGES = 50;
const BOX_SMOOTHING_ALPHA = 0.65;

type Language = "en" | "zh";
type Appearance = "system" | "dark" | "light";
type VisionEventType = "new_person" | "person_moved";
type VisionMessageStatus = "pending" | "success" | "error";
type VisionMessage = {
  id: string;
  eventType: VisionEventType;
  status: VisionMessageStatus;
  timestamp: string;
  summary: string;
  message: string;
  frameId: number;
  modelId: string;
};
type TrackedDetection = Detection & {
  trackId: number;
  hits: number;
  lastSeenFrame: number;
  newPersonReported: boolean;
  eventAnchorBox: Detection["box"];
};
type DetectionTracker = {
  nextTrackId: number;
  tracks: TrackedDetection[];
};
type TrackedFrameResult = {
  detections: Detection[];
  events: VisionEvent[];
};
type VisionEvent = {
  eventType: VisionEventType;
  trackId: number;
  detection: Detection;
};

const DEFAULT_TRIGGER_SETTINGS: VisionTriggerSettings = {
  cooldownSeconds: 8,
  stableConfirmFrames: 2,
  missToleranceFrames: 2,
  trackIouThreshold: 0.35,
  movementDistancePercent: 8,
  movementIouThreshold: 0.55,
};

const DEFAULT_APP_SETTINGS: AppSettings = {
  selectedModelId: null,
  selectedCameraId: null,
  language: "en",
  appearance: "system",
  lmStudioUrl: DEFAULT_LM_STUDIO_URL,
  lmStudioModelId: DEFAULT_LM_STUDIO_MODEL,
  lmStudioPrompt: DEFAULT_LM_STUDIO_PROMPT,
  visionTriggerSettings: DEFAULT_TRIGGER_SETTINGS,
  historyRetentionDays: DEFAULT_HISTORY_RETENTION_DAYS,
  confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
};

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
    historyRetentionDays: "History data retention",
    historyRetentionDaysHelp: "How many days database records and saved screenshots are kept before automatic cleanup.",
    days: "days",
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
    lmStudioVision: "LM Studio vision",
    lmStudioUrl: "LM Studio URL",
    editLmStudioPrompt: "Edit prompt",
    promptDialogTitle: "LM Studio prompt",
    promptDialogHelp: "This prompt is saved in the backend database and sent with each LM Studio vision request.",
    savePrompt: "Save",
    cancelPrompt: "Cancel",
    checkModels: "Check models",
    checkingModels: "Checking",
    visionModel: "Vision model",
    noVisionModels: "No vision models loaded",
    globalCooldown: "Global cooldown",
    stableFrames: "Stable frames",
    missTolerance: "Miss tolerance",
    trackIou: "Track IoU",
    moveDistance: "Move distance",
    moveIou: "Move IoU",
    globalCooldownHelp: "Minimum time between screenshot requests. While cooling down, no person event sends an image to LM Studio.",
    stableFramesHelp: "How many consecutive frames a new person must appear before triggering a new-person event.",
    missToleranceHelp: "How many missed frames a track can survive before it is removed.",
    trackIouHelp: "Minimum bounding-box overlap needed to treat a detection as the same tracked person.",
    moveDistanceHelp: "How far a tracked person's box center must move, as a percent of the frame diagonal, to trigger movement.",
    moveIouHelp: "If overlap with the last event box falls below this value, trigger a movement event.",
    visionMessages: "Vision messages",
    noVisionMessages: "No vision messages yet.",
    messageDetail: "Message detail",
    pending: "Pending",
    success: "Success",
    newPerson: "New person",
    personMoved: "Person moved",
    selectVisionModel: "Select an LM Studio model first",
    historyRecords: "History records",
    historyStart: "Start",
    historyEnd: "End",
    historyKeyword: "Keyword",
    historyKeywordPlaceholder: "Message, model, or event",
    viewHistory: "View history",
    loadingHistory: "Loading history",
    historyDialogTitle: "Vision history",
    noHistoryRecords: "No history records in this range.",
    viewImage: "View image",
    noScreenshot: "No screenshot",
    lastSeen: "Last seen",
    repeated: "Repeated",
    close: "Close",
    imagePreview: "Image preview",
    imageLoadFailed: "Unable to load image",
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
    historyRetentionDays: "历史数据保存时间",
    historyRetentionDaysHelp: "数据库记录和已保存图片自动清理前保留的天数。",
    days: "天",
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
    lmStudioVision: "LM Studio 图像识别",
    lmStudioUrl: "LM Studio URL",
    editLmStudioPrompt: "修改提示词",
    promptDialogTitle: "LM Studio 提示词",
    promptDialogHelp: "提示词会保存在后端数据库，并随每次 LM Studio 图像识别请求发送。",
    savePrompt: "保存",
    cancelPrompt: "取消",
    checkModels: "检测模型",
    checkingModels: "检测中",
    visionModel: "图像模型",
    noVisionModels: "未加载图像模型",
    globalCooldown: "全局冷却",
    stableFrames: "新人稳定帧",
    missTolerance: "丢失容忍帧",
    trackIou: "Track IoU",
    moveDistance: "移动距离",
    moveIou: "移动 IoU",
    globalCooldownHelp: "两次截图识别请求之间的最短间隔。冷却期间，任何人员事件都不会发送截图到 LM Studio。",
    stableFramesHelp: "新的人需要连续出现多少帧，才触发“新的人”事件。",
    missToleranceHelp: "Track 丢失后还能保留多少帧，超过后会被移除。",
    trackIouHelp: "检测框重叠度达到多少，才认为是同一个被跟踪的人。",
    moveDistanceHelp: "人的检测框中心移动超过画面对角线的多少比例，才触发“人移动”事件。",
    moveIouHelp: "当前框与上次触发事件时的框重叠度低于该值时，触发“人移动”事件。",
    visionMessages: "图像识别消息",
    noVisionMessages: "暂无图像识别消息。",
    messageDetail: "消息详情",
    pending: "请求中",
    success: "成功",
    newPerson: "新的人",
    personMoved: "人移动",
    selectVisionModel: "请先选择 LM Studio 模型",
    historyRecords: "历史记录",
    historyStart: "开始时间",
    historyEnd: "结束时间",
    historyKeyword: "关键字",
    historyKeywordPlaceholder: "消息、模型或事件",
    viewHistory: "查看历史记录",
    loadingHistory: "加载历史中",
    historyDialogTitle: "图像识别历史",
    noHistoryRecords: "该时间范围内没有历史记录。",
    viewImage: "查看图片",
    noScreenshot: "无截图",
    lastSeen: "最后出现",
    repeated: "重复",
    close: "关闭",
    imagePreview: "图片预览",
    imageLoadFailed: "无法加载图片",
  },
} satisfies Record<Language, Record<string, string>>;

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, value));
}

function normalizeTriggerSettings(value: Partial<VisionTriggerSettings> | undefined): VisionTriggerSettings {
  return {
    cooldownSeconds: clampNumber(value?.cooldownSeconds, 1, 60, DEFAULT_TRIGGER_SETTINGS.cooldownSeconds),
    stableConfirmFrames: Math.round(clampNumber(value?.stableConfirmFrames, 1, 10, DEFAULT_TRIGGER_SETTINGS.stableConfirmFrames)),
    missToleranceFrames: Math.round(clampNumber(value?.missToleranceFrames, 0, 10, DEFAULT_TRIGGER_SETTINGS.missToleranceFrames)),
    trackIouThreshold: clampNumber(value?.trackIouThreshold, 0.1, 0.9, DEFAULT_TRIGGER_SETTINGS.trackIouThreshold),
    movementDistancePercent: clampNumber(value?.movementDistancePercent, 1, 50, DEFAULT_TRIGGER_SETTINGS.movementDistancePercent),
    movementIouThreshold: clampNumber(value?.movementIouThreshold, 0.1, 0.95, DEFAULT_TRIGGER_SETTINGS.movementIouThreshold),
  };
}

function normalizeAppSettings(value: Partial<AppSettings> | undefined): AppSettings {
  const language = value?.language === "zh" ? "zh" : "en";
  const appearance =
    value?.appearance === "dark" || value?.appearance === "light" || value?.appearance === "system" ? value.appearance : "system";
  return {
    selectedModelId: value?.selectedModelId || null,
    selectedCameraId: value?.selectedCameraId || null,
    language,
    appearance,
    lmStudioUrl: value?.lmStudioUrl?.trim() || DEFAULT_LM_STUDIO_URL,
    lmStudioModelId: value?.lmStudioModelId?.trim() || DEFAULT_LM_STUDIO_MODEL,
    lmStudioPrompt: value?.lmStudioPrompt?.trim() || DEFAULT_LM_STUDIO_PROMPT,
    visionTriggerSettings: normalizeTriggerSettings(value?.visionTriggerSettings),
    historyRetentionDays: Math.round(clampNumber(value?.historyRetentionDays, 1, 365, DEFAULT_HISTORY_RETENTION_DAYS)),
    confidenceThreshold: clampNumber(value?.confidenceThreshold, 0, 1, DEFAULT_CONFIDENCE_THRESHOLD),
  };
}

function normalizeStatus(status: string): string {
  return status.toLowerCase().replace(/\s+/g, "-");
}

function resolveInitialModelId(models: ModelInfo[], savedModelId: string | null, backendSelectedModelId: string): string {
  const recommendedModel = models.find((model) => model.available && model.recommended);
  const savedModel = models.find((model) => model.id === savedModelId && model.available);

  return savedModel?.id ?? recommendedModel?.id ?? backendSelectedModelId;
}

async function fetchAppSettings(): Promise<AppSettings> {
  const response = await fetch(`${API_BASE_URL}/settings`);
  if (!response.ok) {
    throw new Error(await readApiError(response));
  }
  return normalizeAppSettings((await response.json()) as Partial<AppSettings>);
}

async function saveAppSettings(settings: AppSettings): Promise<AppSettings> {
  const response = await fetch(`${API_BASE_URL}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!response.ok) {
    throw new Error(await readApiError(response));
  }
  return normalizeAppSettings((await response.json()) as Partial<AppSettings>);
}

function createSessionId(): string {
  return window.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function toDateTimeInputValue(date: Date): string {
  const offsetMs = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offsetMs).toISOString().slice(0, 16);
}

function dateTimeInputToIso(value: string): string | null {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

function dateTimeInputToDateInputValue(value: string): string {
  return value.slice(0, 10);
}

function dateInputToDateTimeInputValue(value: string, boundary: "start" | "end"): string {
  return value ? `${value}${boundary === "start" ? "T00:00" : "T23:59"}` : "";
}

function readDefaultHistoryRange(retentionDays = DEFAULT_HISTORY_RETENTION_DAYS): { start: string; end: string } {
  const end = new Date();
  const start = new Date(end.getTime() - retentionDays * 24 * 60 * 60 * 1000);
  return {
    start: toDateTimeInputValue(start),
    end: toDateTimeInputValue(end),
  };
}

function App() {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const overlayRef = React.useRef<HTMLCanvasElement | null>(null);
  const captureRef = React.useRef<HTMLCanvasElement | null>(null);
  const socketRef = React.useRef<WebSocket | null>(null);
  const latestFrameRef = React.useRef(0);
  const sentAtRef = React.useRef(new Map<number, number>());
  const detectionTrackerRef = React.useRef<DetectionTracker>({ nextTrackId: 1, tracks: [] });
  const lastVisionRequestAtRef = React.useRef(-Number.POSITIVE_INFINITY);
  const triggerSettingsRef = React.useRef<VisionTriggerSettings>(DEFAULT_TRIGGER_SETTINGS);
  const historyRetentionDaysRef = React.useRef(DEFAULT_HISTORY_RETENTION_DAYS);
  const lmStudioUrlRef = React.useRef(DEFAULT_LM_STUDIO_URL);
  const lmStudioPromptRef = React.useRef(DEFAULT_LM_STUDIO_PROMPT);
  const selectedVisionModelIdRef = React.useRef("");
  const settingsHydratedRef = React.useRef(false);
  const sessionIdRef = React.useRef(createSessionId());

  const [models, setModels] = React.useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = React.useState("");
  const [visionModels, setVisionModels] = React.useState<VisionModelInfo[]>([]);
  const [selectedVisionModelId, setSelectedVisionModelId] = React.useState(DEFAULT_LM_STUDIO_MODEL);
  const [lmStudioUrl, setLmStudioUrl] = React.useState(DEFAULT_LM_STUDIO_URL);
  const [lmStudioPrompt, setLmStudioPrompt] = React.useState(DEFAULT_LM_STUDIO_PROMPT);
  const [promptDraft, setPromptDraft] = React.useState("");
  const [promptDialogOpen, setPromptDialogOpen] = React.useState(false);
  const [isCheckingVisionModels, setIsCheckingVisionModels] = React.useState(false);
  const [triggerSettings, setTriggerSettings] = React.useState<VisionTriggerSettings>(DEFAULT_TRIGGER_SETTINGS);
  const [historyRetentionDays, setHistoryRetentionDays] = React.useState(DEFAULT_HISTORY_RETENTION_DAYS);
  const [visionMessages, setVisionMessages] = React.useState<VisionMessage[]>([]);
  const [selectedVisionMessageId, setSelectedVisionMessageId] = React.useState("");
  const [historyRange, setHistoryRange] = React.useState(() => readDefaultHistoryRange(DEFAULT_HISTORY_RETENTION_DAYS));
  const [historyKeyword, setHistoryKeyword] = React.useState("");
  const [historyDialogOpen, setHistoryDialogOpen] = React.useState(false);
  const [historyEvents, setHistoryEvents] = React.useState<VisionEventRecord[]>([]);
  const [selectedHistoryEventId, setSelectedHistoryEventId] = React.useState("");
  const [isLoadingHistory, setIsLoadingHistory] = React.useState(false);
  const [historyError, setHistoryError] = React.useState("");
  const [historyImageUrl, setHistoryImageUrl] = React.useState("");
  const [isLoadingHistoryImage, setIsLoadingHistoryImage] = React.useState(false);
  const [historyImageError, setHistoryImageError] = React.useState("");
  const [cameras, setCameras] = React.useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = React.useState("");
  const [cameraState, setCameraState] = React.useState("Camera idle");
  const [connectionState, setConnectionState] = React.useState<"idle" | "connected" | "error">("idle");
  const [modelStatus, setModelStatus] = React.useState("Not loaded");
  const [isRunning, setIsRunning] = React.useState(false);
  const [threshold, setThreshold] = React.useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [detections, setDetections] = React.useState<Detection[]>([]);
  const [fps, setFps] = React.useState(0);
  const [inferenceMs, setInferenceMs] = React.useState(0);
  const [latencyMs, setLatencyMs] = React.useState(0);
  const [error, setError] = React.useState("");
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [language, setLanguage] = React.useState<Language>(DEFAULT_APP_SETTINGS.language);
  const [appearance, setAppearance] = React.useState<Appearance>(DEFAULT_APP_SETTINGS.appearance);
  const [systemPrefersDark, setSystemPrefersDark] = React.useState(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  const t = copy[language];
  const effectiveAppearance = appearance === "system" ? (systemPrefersDark ? "dark" : "light") : appearance;

  const selectedModel = React.useMemo(
    () => models.find((model) => model.id === selectedModelId),
    [models, selectedModelId],
  );
  const selectedVisionMessage = React.useMemo(
    () => visionMessages.find((message) => message.id === selectedVisionMessageId) ?? visionMessages[0],
    [selectedVisionMessageId, visionMessages],
  );
  const selectedHistoryEvent = React.useMemo(
    () => historyEvents.find((event) => event.id === selectedHistoryEventId) ?? historyEvents[0],
    [historyEvents, selectedHistoryEventId],
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
    return () => {
      if (historyImageUrl) {
        URL.revokeObjectURL(historyImageUrl);
      }
    };
  }, [historyImageUrl]);

  React.useEffect(() => {
    lmStudioUrlRef.current = lmStudioUrl;
  }, [lmStudioUrl]);

  React.useEffect(() => {
    lmStudioPromptRef.current = lmStudioPrompt;
  }, [lmStudioPrompt]);

  React.useEffect(() => {
    selectedVisionModelIdRef.current = selectedVisionModelId;
  }, [selectedVisionModelId]);

  React.useEffect(() => {
    triggerSettingsRef.current = triggerSettings;
  }, [triggerSettings]);

  React.useEffect(() => {
    historyRetentionDaysRef.current = historyRetentionDays;
  }, [historyRetentionDays]);

  const currentAppSettings = React.useMemo(
    () =>
      normalizeAppSettings({
        selectedModelId: selectedModelId || null,
        selectedCameraId: selectedCameraId || null,
        language,
        appearance,
        lmStudioUrl,
        lmStudioModelId: selectedVisionModelId,
        lmStudioPrompt,
        visionTriggerSettings: triggerSettings,
        historyRetentionDays,
        confidenceThreshold: threshold,
      }),
    [
      appearance,
      historyRetentionDays,
      language,
      lmStudioPrompt,
      lmStudioUrl,
      selectedCameraId,
      selectedModelId,
      selectedVisionModelId,
      threshold,
      triggerSettings,
    ],
  );

  React.useEffect(() => {
    if (!settingsHydratedRef.current) {
      return;
    }
    const timeout = window.setTimeout(() => {
      saveAppSettings(currentAppSettings).catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Unable to save settings");
      });
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [currentAppSettings]);

  React.useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => setSystemPrefersDark(media.matches);
    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  const applyAppSettings = React.useCallback((settings: AppSettings) => {
    setSelectedModelId(settings.selectedModelId ?? "");
    setSelectedCameraId(settings.selectedCameraId ?? "");
    setLanguage(settings.language);
    setAppearance(settings.appearance);
    setLmStudioUrl(settings.lmStudioUrl);
    setSelectedVisionModelId(settings.lmStudioModelId);
    setLmStudioPrompt(settings.lmStudioPrompt);
    setTriggerSettings(settings.visionTriggerSettings);
    setHistoryRetentionDays(settings.historyRetentionDays);
    setHistoryRange(readDefaultHistoryRange(settings.historyRetentionDays));
    setThreshold(settings.confidenceThreshold);
  }, []);

  const loadModels = React.useCallback(async (savedModelId: string | null) => {
    const response = await fetch(`${API_BASE_URL}/models`);
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const data = (await response.json()) as { models: ModelInfo[]; selectedModelId: string };
    const initialModelId = resolveInitialModelId(data.models, savedModelId, data.selectedModelId);
    setModels(data.models);
    setSelectedModelId(initialModelId);
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

  const appendVisionMessage = React.useCallback((message: VisionMessage) => {
    setVisionMessages((current) => {
      const nextMessages = [message, ...current].slice(0, MAX_VISION_MESSAGES);
      setSelectedVisionMessageId(message.id);
      return nextMessages;
    });
  }, []);

  const updateVisionMessage = React.useCallback((messageId: string, updates: Partial<VisionMessage>) => {
    setVisionMessages((current) =>
      current.map((message) => (message.id === messageId ? { ...message, ...updates } : message)),
    );
  }, []);

  const handleCheckVisionModels = React.useCallback(async () => {
    setIsCheckingVisionModels(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/vision/models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ baseUrl: lmStudioUrl }),
      });
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const data = (await response.json()) as VisionModelsResponse;
      setVisionModels(data.models);
      setSelectedVisionModelId((current) => {
        const preferred = data.models.find((model) => model.id === DEFAULT_LM_STUDIO_MODEL);
        const currentAvailable = data.models.find((model) => model.id === current);
        return currentAvailable?.id ?? preferred?.id ?? data.models[0]?.id ?? "";
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load LM Studio models");
    } finally {
      setIsCheckingVisionModels(false);
    }
  }, [lmStudioUrl]);

  const openPromptDialog = React.useCallback(() => {
    setPromptDraft(lmStudioPrompt);
    setPromptDialogOpen(true);
  }, [lmStudioPrompt]);

  const savePromptDraft = React.useCallback(() => {
    setLmStudioPrompt(promptDraft.trim() || DEFAULT_LM_STUDIO_PROMPT);
    setPromptDialogOpen(false);
  }, [promptDraft]);

  const updateTriggerSetting = React.useCallback(
    <K extends keyof VisionTriggerSettings>(key: K, value: VisionTriggerSettings[K]) => {
      setTriggerSettings((current) => ({ ...current, [key]: value }));
    },
    [],
  );

  const updateHistoryRange = React.useCallback((key: "start" | "end", value: string) => {
    setHistoryRange((current) => ({ ...current, [key]: value }));
  }, []);

  const updateHistoryDateRange = React.useCallback((key: "start" | "end", value: string) => {
    setHistoryRange((current) => ({ ...current, [key]: dateInputToDateTimeInputValue(value, key) }));
  }, []);

  const fetchHistoryEvents = React.useCallback(async () => {
    setHistoryDialogOpen(true);
    setIsLoadingHistory(true);
    setHistoryError("");
    setHistoryImageError("");
    setHistoryImageUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return "";
    });
    try {
      const params = new URLSearchParams({ limit: "100" });
      const startAt = dateTimeInputToIso(historyRange.start);
      const endAt = dateTimeInputToIso(historyRange.end);
      if (startAt) {
        params.set("startAt", startAt);
      }
      if (endAt) {
        params.set("endAt", endAt);
      }
      const trimmedKeyword = historyKeyword.trim();
      if (trimmedKeyword) {
        params.set("keyword", trimmedKeyword);
      }
      const response = await fetch(`${API_BASE_URL}/vision/events?${params.toString()}`);
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const data = (await response.json()) as VisionEventsResponse;
      setHistoryEvents(data.events);
      setSelectedHistoryEventId(data.events[0]?.id ?? "");
    } catch (err) {
      setHistoryEvents([]);
      setSelectedHistoryEventId("");
      setHistoryError(err instanceof Error ? err.message : "Unable to load history");
    } finally {
      setIsLoadingHistory(false);
    }
  }, [historyKeyword, historyRange.end, historyRange.start]);

  const closeHistoryDialog = React.useCallback(() => {
    setHistoryDialogOpen(false);
    setHistoryImageError("");
    setHistoryImageUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return "";
    });
  }, []);

  const loadHistoryImage = React.useCallback(async () => {
    if (!selectedHistoryEvent?.hasScreenshot) {
      return;
    }
    setIsLoadingHistoryImage(true);
    setHistoryImageError("");
    setHistoryImageUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return "";
    });
    try {
      const response = await fetch(`${API_BASE_URL}/vision/events/${selectedHistoryEvent.id}/screenshot`);
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const blob = await response.blob();
      setHistoryImageUrl(URL.createObjectURL(blob));
    } catch (err) {
      setHistoryImageError(err instanceof Error ? err.message : t.imageLoadFailed);
    } finally {
      setIsLoadingHistoryImage(false);
    }
  }, [selectedHistoryEvent, t.imageLoadFailed]);

  const captureCurrentFrameImage = React.useCallback((): string => {
    const video = videoRef.current;
    const canvas = captureRef.current;
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
      return "";
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      return "";
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", FRAME_JPEG_QUALITY);
  }, []);

  const sendVisionEvent = React.useCallback(
    async (visionEvent: VisionEvent, frameId: number, frameDetections: Detection[]) => {
      const now = performance.now();
      const cooldownMs = triggerSettingsRef.current.cooldownSeconds * 1000;
      if (now - lastVisionRequestAtRef.current < cooldownMs) {
        return;
      }
      const modelId = selectedVisionModelIdRef.current.trim();
      if (!modelId) {
        const timestamp = new Date().toISOString();
        appendVisionMessage({
          id: `${frameId}-${now}`,
          eventType: visionEvent.eventType,
          status: "error",
          timestamp,
          summary: t.selectVisionModel,
          message: t.selectVisionModel,
          frameId,
          modelId: "",
        });
        lastVisionRequestAtRef.current = now;
        return;
      }
      const imageData = captureCurrentFrameImage();
      if (!imageData) {
        return;
      }

      lastVisionRequestAtRef.current = now;
      const timestamp = new Date().toISOString();
      const id = `${frameId}-${now}`;
      appendVisionMessage({
        id,
        eventType: visionEvent.eventType,
        status: "pending",
        timestamp,
        summary: t.pending,
        message: t.pending,
        frameId,
        modelId,
      });

      try {
        const response = await fetch(`${API_BASE_URL}/vision/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            baseUrl: lmStudioUrlRef.current,
            modelId,
            prompt: lmStudioPromptRef.current,
            imageData,
            eventType: visionEvent.eventType,
            frameId,
            retentionDays: historyRetentionDaysRef.current,
            sessionId: sessionIdRef.current,
            trackId: visionEvent.trackId,
            detections: frameDetections,
          }),
        });
        if (!response.ok) {
          throw new Error(await readApiError(response));
        }
        const data = (await response.json()) as VisionAnalyzeResponse;
        updateVisionMessage(id, {
          status: "success",
          timestamp: data.createdAt,
          summary: summarizeVisionMessage(data.message),
          message: data.message,
          modelId: data.modelId,
        });
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "LM Studio vision request failed";
        updateVisionMessage(id, {
          status: "error",
          summary: errorMessage,
          message: errorMessage,
        });
      }
    },
    [appendVisionMessage, captureCurrentFrameImage, t.pending, t.selectVisionModel, updateVisionMessage],
  );

  React.useEffect(() => {
    let cancelled = false;
    const loadInitialState = async () => {
      const settings = await fetchAppSettings();
      if (cancelled) {
        return;
      }
      applyAppSettings(settings);
      await loadModels(settings.selectedModelId);
      settingsHydratedRef.current = true;
    };
    loadInitialState().catch((err: unknown) => {
      if (cancelled) {
        return;
      }
      settingsHydratedRef.current = true;
      setConnectionState("error");
      setModelStatus("Backend unavailable");
      setError(err instanceof Error ? err.message : copy.en.backendUnavailable);
    });
    return () => {
      cancelled = true;
    };
  }, [applyAppSettings, loadModels]);

  React.useEffect(() => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setCameraState("Camera API unavailable");
      return;
    }
    refreshCameras().catch(() => setCameraState("Camera permission needed"));
  }, [refreshCameras]);

  React.useEffect(() => {
    setSelectedCameraId((current) => {
      if (current === CAMERA_OFF_ID || cameras.some((camera) => camera.deviceId === current)) {
        return current;
      }
      return cameras[0]?.deviceId || CAMERA_OFF_ID;
    });
  }, [cameras]);

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
    detectionTrackerRef.current = { nextTrackId: 1, tracks: [] };
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
      const tracked = trackDetections(
        data.detections,
        data.frameId,
        detectionTrackerRef.current,
        triggerSettingsRef.current,
        { width: videoRef.current?.videoWidth ?? 0, height: videoRef.current?.videoHeight ?? 0 },
      );
      setDetections(tracked.detections);
      if (tracked.events.length > 0) {
        void sendVisionEvent(tracked.events[0], data.frameId, tracked.detections);
      }
    };
  }, [selectedModel, sendVisionEvent, startCamera, t.selectModelError, t.unableStartCamera, t.websocketFailed]);

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
      const imageData = canvas.toDataURL("image/jpeg", FRAME_JPEG_QUALITY);
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
              <label className="settings-control" title={t.historyRetentionDaysHelp}>
                <span>{t.historyRetentionDays}</span>
                <div className="retention-control">
                  <input
                    type="number"
                    min={1}
                    max={365}
                    step={1}
                    value={historyRetentionDays}
                    onChange={(event) =>
                      setHistoryRetentionDays(Math.round(clampNumber(Number(event.target.value), 1, 365, DEFAULT_HISTORY_RETENTION_DAYS)))
                    }
                  />
                  <strong>{t.days}</strong>
                </div>
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

          <div className="vision-settings">
            <div className="section-header">
              <span>{t.lmStudioVision}</span>
            </div>
            <label className="stacked-control">
              <span>{t.lmStudioUrl}</span>
              <div className="lm-studio-url-row">
                <input value={lmStudioUrl} onChange={(event) => setLmStudioUrl(event.target.value)} />
                <button className="secondary-button prompt-button" type="button" onClick={openPromptDialog}>
                  <FileText size={16} />
                  {t.editLmStudioPrompt}
                </button>
              </div>
            </label>
            <button className="secondary-button" onClick={handleCheckVisionModels} disabled={isCheckingVisionModels}>
              <RefreshCw size={16} />
              {isCheckingVisionModels ? t.checkingModels : t.checkModels}
            </button>
            <label className="stacked-control">
              <span>{t.visionModel}</span>
              <select value={selectedVisionModelId} onChange={(event) => setSelectedVisionModelId(event.target.value)}>
                {visionModels.length === 0 ? <option value={selectedVisionModelId}>{selectedVisionModelId || t.noVisionModels}</option> : null}
                {visionModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.id}
                  </option>
                ))}
              </select>
            </label>
            <div className="trigger-grid">
              <NumberControl
                label={t.globalCooldown}
                tooltip={t.globalCooldownHelp}
                suffix="s"
                value={triggerSettings.cooldownSeconds}
                min={1}
                max={60}
                step={1}
                onChange={(value) => updateTriggerSetting("cooldownSeconds", value)}
              />
              <NumberControl
                label={t.stableFrames}
                tooltip={t.stableFramesHelp}
                value={triggerSettings.stableConfirmFrames}
                min={1}
                max={10}
                step={1}
                onChange={(value) => updateTriggerSetting("stableConfirmFrames", Math.round(value))}
              />
              <NumberControl
                label={t.missTolerance}
                tooltip={t.missToleranceHelp}
                value={triggerSettings.missToleranceFrames}
                min={0}
                max={10}
                step={1}
                onChange={(value) => updateTriggerSetting("missToleranceFrames", Math.round(value))}
              />
              <NumberControl
                label={t.trackIou}
                tooltip={t.trackIouHelp}
                value={triggerSettings.trackIouThreshold}
                min={0.1}
                max={0.9}
                step={0.05}
                precision={2}
                onChange={(value) => updateTriggerSetting("trackIouThreshold", value)}
              />
              <NumberControl
                label={t.moveDistance}
                tooltip={t.moveDistanceHelp}
                suffix="%"
                value={triggerSettings.movementDistancePercent}
                min={1}
                max={50}
                step={1}
                onChange={(value) => updateTriggerSetting("movementDistancePercent", value)}
              />
              <NumberControl
                label={t.moveIou}
                tooltip={t.moveIouHelp}
                value={triggerSettings.movementIouThreshold}
                min={0.1}
                max={0.95}
                step={0.05}
                precision={2}
                onChange={(value) => updateTriggerSetting("movementIouThreshold", value)}
              />
            </div>
          </div>

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

          <div className="history-section">
            <div className="section-header">
              <span>{t.historyRecords}</span>
            </div>
            <div className="history-controls">
              <label className="stacked-control">
                <span>{t.historyStart}</span>
                <input
                  type="date"
                  value={dateTimeInputToDateInputValue(historyRange.start)}
                  onChange={(event) => updateHistoryDateRange("start", event.target.value)}
                />
              </label>
              <label className="stacked-control">
                <span>{t.historyEnd}</span>
                <input
                  type="date"
                  value={dateTimeInputToDateInputValue(historyRange.end)}
                  onChange={(event) => updateHistoryDateRange("end", event.target.value)}
                />
              </label>
              <button className="secondary-button" type="button" onClick={fetchHistoryEvents} disabled={isLoadingHistory}>
                <Search size={16} />
                {isLoadingHistory ? t.loadingHistory : t.viewHistory}
              </button>
            </div>
            {historyError ? <div className="error-box compact">{historyError}</div> : null}
          </div>

          <div className="vision-messages">
            <div className="section-header">
              <span>{t.visionMessages}</span>
              <strong>{visionMessages.length}</strong>
            </div>
            {visionMessages.length === 0 ? (
              <p className="muted">{t.noVisionMessages}</p>
            ) : (
              <div className="vision-message-list">
                {visionMessages.map((message) => (
                  <button
                    className={`vision-message-item ${message.status} ${selectedVisionMessage?.id === message.id ? "selected" : ""}`}
                    key={message.id}
                    onClick={() => setSelectedVisionMessageId(message.id)}
                  >
                    <span>{formatTimestamp(message.timestamp)}</span>
                    <strong>{message.eventType === "new_person" ? t.newPerson : t.personMoved}</strong>
                    <em>{message.summary}</em>
                  </button>
                ))}
              </div>
            )}
            <label className="message-detail">
              <span>{t.messageDetail}</span>
              <textarea
                readOnly
                value={
                  selectedVisionMessage
                    ? `[${formatTimestamp(selectedVisionMessage.timestamp)}] ${
                        selectedVisionMessage.eventType === "new_person" ? t.newPerson : t.personMoved
                      }\n${selectedVisionMessage.message}`
                    : ""
                }
              />
            </label>
          </div>
        </aside>
      </section>
      <canvas ref={captureRef} className="capture-canvas" />
      {promptDialogOpen ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setPromptDialogOpen(false)}>
          <section className="prompt-dialog" role="dialog" aria-modal="true" aria-labelledby="prompt-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
            <div className="section-header">
              <span id="prompt-dialog-title">{t.promptDialogTitle}</span>
            </div>
            <p className="muted">{t.promptDialogHelp}</p>
            <textarea value={promptDraft} onChange={(event) => setPromptDraft(event.target.value)} autoFocus />
            <div className="dialog-actions">
              <button className="secondary-button" type="button" onClick={() => setPromptDialogOpen(false)}>
                {t.cancelPrompt}
              </button>
              <button className="secondary-button primary-action" type="button" onClick={savePromptDraft}>
                {t.savePrompt}
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {historyDialogOpen ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={closeHistoryDialog}>
          <section className="history-dialog" role="dialog" aria-modal="true" aria-labelledby="history-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
            <div className="history-dialog-header">
              <div>
                <span id="history-dialog-title">{t.historyDialogTitle}</span>
                <strong>{historyEvents.length}</strong>
              </div>
              <button className="icon-button" type="button" aria-label={t.close} onClick={closeHistoryDialog}>
                <X size={18} />
              </button>
            </div>
            <div className="history-dialog-filters">
              <label className="stacked-control">
                <span>{t.historyStart}</span>
                <input
                  type="datetime-local"
                  value={historyRange.start}
                  onChange={(event) => updateHistoryRange("start", event.target.value)}
                />
              </label>
              <label className="stacked-control">
                <span>{t.historyEnd}</span>
                <input
                  type="datetime-local"
                  value={historyRange.end}
                  onChange={(event) => updateHistoryRange("end", event.target.value)}
                />
              </label>
              <label className="stacked-control">
                <span>{t.historyKeyword}</span>
                <input
                  type="search"
                  value={historyKeyword}
                  placeholder={t.historyKeywordPlaceholder}
                  onChange={(event) => setHistoryKeyword(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      void fetchHistoryEvents();
                    }
                  }}
                />
              </label>
              <button className="secondary-button" type="button" onClick={fetchHistoryEvents} disabled={isLoadingHistory}>
                <Search size={16} />
                {isLoadingHistory ? t.loadingHistory : t.viewHistory}
              </button>
            </div>
            <div className="history-dialog-body">
              <div className="history-list">
                {isLoadingHistory ? <p className="muted">{t.loadingHistory}</p> : null}
                {!isLoadingHistory && historyEvents.length === 0 ? <p className="muted">{historyError || t.noHistoryRecords}</p> : null}
                {historyEvents.map((event) => (
                  <button
                    className={`history-item ${selectedHistoryEvent?.id === event.id ? "selected" : ""}`}
                    key={event.id}
                    type="button"
                    onClick={() => {
                      setSelectedHistoryEventId(event.id);
                      setHistoryImageError("");
                      setHistoryImageUrl((current) => {
                        if (current) {
                          URL.revokeObjectURL(current);
                        }
                        return "";
                      });
                    }}
                  >
                    <span>{formatTimestamp(event.createdAt)}</span>
                    <strong>{event.eventType === "new_person" ? t.newPerson : t.personMoved}</strong>
                    <em>{event.summary}</em>
                    {event.duplicateCount > 0 ? (
                      <small>
                        {t.repeated} {event.duplicateCount}
                      </small>
                    ) : null}
                  </button>
                ))}
              </div>
              <div className="history-detail">
                {selectedHistoryEvent ? (
                  <>
                    <div className="history-meta">
                      <span>{formatTimestamp(selectedHistoryEvent.createdAt)}</span>
                      <span>
                        {t.lastSeen}: {formatTimestamp(selectedHistoryEvent.lastSeenAt)}
                      </span>
                      <span>
                        {t.repeated}: {selectedHistoryEvent.duplicateCount}
                      </span>
                      <span>Track: {selectedHistoryEvent.trackId ?? "-"}</span>
                      <span>Frame: {selectedHistoryEvent.frameId}</span>
                      <span>{selectedHistoryEvent.modelId}</span>
                    </div>
                    <textarea readOnly value={selectedHistoryEvent.message} />
                    <div className="history-image-actions">
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={loadHistoryImage}
                        disabled={!selectedHistoryEvent.hasScreenshot || isLoadingHistoryImage}
                      >
                        <Image size={16} />
                        {selectedHistoryEvent.hasScreenshot ? t.viewImage : t.noScreenshot}
                      </button>
                      {historyImageError ? <span className="history-image-error">{historyImageError}</span> : null}
                    </div>
                    <div className="history-image-frame">
                      {isLoadingHistoryImage ? <p className="muted">{t.imagePreview}...</p> : null}
                      {historyImageUrl ? <img src={historyImageUrl} alt={t.imagePreview} /> : null}
                    </div>
                  </>
                ) : (
                  <p className="muted">{t.noHistoryRecords}</p>
                )}
              </div>
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}

function trackDetections(
  incomingDetections: Detection[],
  frameId: number,
  tracker: DetectionTracker,
  settings: VisionTriggerSettings,
  frameSize: { width: number; height: number },
): TrackedFrameResult {
  const matchedTrackIds = new Set<number>();
  const sortedDetections = [...incomingDetections].sort((left, right) => right.confidence - left.confidence);
  const events: VisionEvent[] = [];
  const frameDiagonal = Math.hypot(frameSize.width, frameSize.height) || 1;

  sortedDetections.forEach((detection) => {
    let bestTrackIndex = -1;
    let bestIou = 0;

    tracker.tracks.forEach((track, index) => {
      if (track.label !== detection.label || matchedTrackIds.has(track.trackId)) {
        return;
      }
      const overlap = boxIou(track.box, detection.box);
      if (overlap > bestIou) {
        bestIou = overlap;
        bestTrackIndex = index;
      }
    });

    if (bestTrackIndex >= 0 && bestIou >= settings.trackIouThreshold) {
      const bestTrack = tracker.tracks[bestTrackIndex];
      const previousAnchor = bestTrack.eventAnchorBox;
      bestTrack.confidence = detection.confidence;
      bestTrack.box = smoothBox(bestTrack.box, detection.box);
      bestTrack.hits += 1;
      bestTrack.lastSeenFrame = frameId;
      matchedTrackIds.add(bestTrack.trackId);
      collectPersonEvents(bestTrack, previousAnchor, settings, frameDiagonal, events);
      return;
    }

    const track: TrackedDetection = {
      ...detection,
      trackId: tracker.nextTrackId,
      hits: 1,
      lastSeenFrame: frameId,
      newPersonReported: false,
      eventAnchorBox: detection.box,
    };
    tracker.nextTrackId += 1;
    tracker.tracks.push(track);
    matchedTrackIds.add(track.trackId);
    collectPersonEvents(track, track.eventAnchorBox, settings, frameDiagonal, events);
  });

  tracker.tracks = tracker.tracks.filter((track) => frameId - track.lastSeenFrame <= settings.missToleranceFrames);
  const detections = tracker.tracks
    .filter((track) => isStableTrack(track, settings))
    .sort((left, right) => right.confidence - left.confidence)
    .map(stripTrackFields);
  return { detections, events };
}

function collectPersonEvents(
  track: TrackedDetection,
  previousAnchor: Detection["box"],
  settings: VisionTriggerSettings,
  frameDiagonal: number,
  events: VisionEvent[],
) {
  if (track.label !== "person" || !isStableTrack(track, settings)) {
    return;
  }
  if (!track.newPersonReported) {
    track.newPersonReported = true;
    track.eventAnchorBox = track.box;
    events.push({ eventType: "new_person", trackId: track.trackId, detection: stripTrackFields(track) });
    return;
  }

  const movementDistance = centerDistance(previousAnchor, track.box) / frameDiagonal;
  const movementIou = boxIou(previousAnchor, track.box);
  if (movementDistance >= settings.movementDistancePercent / 100 || movementIou < settings.movementIouThreshold) {
    track.eventAnchorBox = track.box;
    events.push({ eventType: "person_moved", trackId: track.trackId, detection: stripTrackFields(track) });
  }
}

function isStableTrack(track: TrackedDetection, settings: VisionTriggerSettings): boolean {
  return track.hits >= settings.stableConfirmFrames || track.confidence >= 0.75;
}

function stripTrackFields(track: TrackedDetection): Detection {
  return {
    label: track.label,
    confidence: track.confidence,
    box: track.box,
  };
}

function smoothBox(previous: Detection["box"], next: Detection["box"]): Detection["box"] {
  return {
    x: previous.x * (1 - BOX_SMOOTHING_ALPHA) + next.x * BOX_SMOOTHING_ALPHA,
    y: previous.y * (1 - BOX_SMOOTHING_ALPHA) + next.y * BOX_SMOOTHING_ALPHA,
    width: previous.width * (1 - BOX_SMOOTHING_ALPHA) + next.width * BOX_SMOOTHING_ALPHA,
    height: previous.height * (1 - BOX_SMOOTHING_ALPHA) + next.height * BOX_SMOOTHING_ALPHA,
  };
}

function boxIou(left: Detection["box"], right: Detection["box"]): number {
  const leftX2 = left.x + left.width;
  const leftY2 = left.y + left.height;
  const rightX2 = right.x + right.width;
  const rightY2 = right.y + right.height;

  const intersectionWidth = Math.max(0, Math.min(leftX2, rightX2) - Math.max(left.x, right.x));
  const intersectionHeight = Math.max(0, Math.min(leftY2, rightY2) - Math.max(left.y, right.y));
  const intersection = intersectionWidth * intersectionHeight;
  const union = left.width * left.height + right.width * right.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function centerDistance(left: Detection["box"], right: Detection["box"]): number {
  const leftCenterX = left.x + left.width / 2;
  const leftCenterY = left.y + left.height / 2;
  const rightCenterX = right.x + right.width / 2;
  const rightCenterY = right.y + right.height / 2;
  return Math.hypot(leftCenterX - rightCenterX, leftCenterY - rightCenterY);
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NumberControl({
  label,
  tooltip,
  suffix,
  value,
  min,
  max,
  step,
  precision = 0,
  onChange,
}: {
  label: string;
  tooltip: string;
  suffix?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  precision?: number;
  onChange: (value: number) => void;
}) {
  const displayValue = precision > 0 ? value.toFixed(precision) : String(value);
  return (
    <label className="number-control" data-tooltip={tooltip}>
      <span>{label}</span>
      <strong>
        {displayValue}
        {suffix ?? ""}
      </strong>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

async function readApiError(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: unknown };
    if (typeof data.detail === "string") {
      return data.detail;
    }
  } catch {
    return response.statusText;
  }
  return response.statusText;
}

function summarizeVisionMessage(message: string): string {
  const normalized = message.replace(/\s+/g, " ").trim();
  return normalized.length > 56 ? `${normalized.slice(0, 56)}...` : normalized;
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
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
