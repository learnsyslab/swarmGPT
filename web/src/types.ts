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

// One drone-deck's compiled lighting cue list: step events under zero-order hold, baked at the
// same col_freq the hardware drains at. `times` is ascending seconds and always opens at 0;
// `rgb` is integer 0-255, one row per time.
export type LightingCues = {
  times: number[];
  rgb: number[][];
};

export type Playback = {
  schemaVersion: number;
  audioUrl: string;
  audioOffset: number;
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
  lighting: {
    top: LightingCues[];
    bot: LightingCues[];
  };
  sampleRate: number;
};
