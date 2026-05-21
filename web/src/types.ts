export type LibraryItem = {
  id: string;
  label: string;
  kind: "song" | "preset";
  previewUrl: string;
  song?: string;
  numDrones?: number | null;
  createdAt?: string | null;
  createdLabel?: string | null;
};

export type LibraryResponse = {
  songs: LibraryItem[];
  presets: LibraryItem[];
};

export type LlmProvider = {
  id: "openai" | "ollama";
  label: string;
  models: string[];
  defaultModel: string | null;
};

export type LlmResponse = {
  providers: LlmProvider[];
  defaultProvider: "openai" | "ollama";
  defaultModel: string;
};

export type JobEvent = {
  id: number;
  type: string;
  createdAt: string;
  payload: Record<string, unknown>;
};

export type ChatMessage = {
  role: string;
  content: string;
};

export type Playback = {
  schemaVersion: number;
  audioUrl: string;
  song: string;
  numDrones: number;
  timestamps: number[];
  states: number[][][];
  fields: {
    pos: [number, number];
    quat: [number, number];
    vel: [number, number];
    angVel: [number, number];
  };
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  colors: number[][];
  sampleRate: number;
};
