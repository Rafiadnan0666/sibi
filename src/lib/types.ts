export interface HandLandmark {
  x: number;
  y: number;
  z?: number;
}

export interface SIBISign {
  id: number;
  character: string;
  filename: string;
  description?: string;
}

export interface RecognizedCharacter {
  character: string;
  confidence: number;
  timestamp: number;
  stabilized: boolean;
}

export interface WordPrediction {
  word: string;
  score: number;
}

export interface GestureControl {
  type: 'point' | 'pinch' | 'delete' | 'pause' | 'space' | 'confirm';
  detected: boolean;
}

export interface CameraState {
  active: boolean;
  error?: string;
  fps?: number;
}

export interface SIBIState {
  currentInput: string;
  recognizedWords: RecognizedCharacter[];
  predictions: WordPrediction[];
  isRecognizing: boolean;
  cameraState: CameraState;
  selectedPrediction?: number;
}