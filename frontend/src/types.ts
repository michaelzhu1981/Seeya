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
