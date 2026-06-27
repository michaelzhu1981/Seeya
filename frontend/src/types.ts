export type ModelInfo = {
  id: string;
  name: string;
  runtime: string;
  size: string;
  inputSize: number;
  available: boolean;
  recommended: boolean;
  unavailableReason: string | null;
};

export type Detection = {
  label: string;
  confidence: number;
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

export type DetectResponse = {
  frameId: number;
  modelId: string;
  inferenceMs: number;
  detections: Detection[];
};

export type VisionModelInfo = {
  id: string;
  object: string | null;
};

export type VisionModelsResponse = {
  models: VisionModelInfo[];
};

export type VisionAnalyzeResponse = {
  message: string;
  createdAt: string;
  modelId: string;
  eventId: string | null;
  duplicateCount: number;
  deduplicated: boolean;
};

export type VisionEventRecord = {
  id: string;
  sessionId: string | null;
  trackId: number | null;
  eventType: "new_person" | "person_moved";
  modelId: string;
  frameId: number;
  message: string;
  summary: string;
  detections: Detection[];
  primaryBox: Detection["box"] | null;
  duplicateCount: number;
  firstSeenAt: string;
  lastSeenAt: string;
  createdAt: string;
  expiresAt: string;
  hasScreenshot: boolean;
  screenshotMimeType: string | null;
  screenshotSizeBytes: number;
  screenshotWidth: number;
  screenshotHeight: number;
};

export type VisionEventsResponse = {
  events: VisionEventRecord[];
};
