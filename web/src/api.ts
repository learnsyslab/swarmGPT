import type { LibraryItem, LibraryResponse, LlmResponse, Playback } from "./types";

// "auto" asks a classifier whether the request needs a primitive that does not exist; "force"
// synthesizes unconditionally; "off" never does.
export type SynthesisMode = "auto" | "force" | "off";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    let detail: string | null = null;
    try {
      const payload = JSON.parse(text) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      }
    } catch {
      detail = null;
    }
    if (detail) {
      throw new Error(detail);
    }
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export function getLibrary(): Promise<LibraryResponse> {
  return request<LibraryResponse>("/api/library");
}

export function getLlm(): Promise<LlmResponse> {
  return request<LlmResponse>("/api/llm");
}

export function createJob(selection: string, provider: string, modelId: string) {
  return request<{ jobId: string; eventsUrl: string }>("/api/jobs", {
    method: "POST",
    body: JSON.stringify({ selection, provider, modelId })
  });
}

const PLAYBACK_SCHEMA_VERSION = 2;

export async function getPlayback(jobId: string): Promise<Playback> {
  const playback = await request<Playback>(`/api/jobs/${jobId}/playback`);
  // A stale bundle against a restarted server (web-dev, or a cached index.html on the SPA route)
  // would otherwise read a payload whose shape it does not understand. Fail into the error banner.
  if (playback.schemaVersion !== PLAYBACK_SCHEMA_VERSION) {
    throw new Error(
      `Playback schema ${playback.schemaVersion} is not supported (expected ${PLAYBACK_SCHEMA_VERSION}). Reload the page.`
    );
  }
  return playback;
}

export function refineJob(
  jobId: string,
  message: string,
  provider: string,
  modelId: string,
  synthesis: SynthesisMode
) {
  return request<{ jobId: string }>(`/api/jobs/${jobId}/refine`, {
    method: "POST",
    body: JSON.stringify({ message, provider, modelId, synthesis })
  });
}

export function deployJob(jobId: string) {
  return request<{ jobId: string }>(`/api/jobs/${jobId}/deploy`, { method: "POST" });
}

export function emergencyStopJob(jobId: string) {
  return request<{ jobId: string; emergencyStopped: boolean }>(
    `/api/jobs/${jobId}/emergency-stop`,
    { method: "POST" }
  );
}

export function savePreset(jobId: string) {
  return request<{ preset: LibraryItem }>(`/api/jobs/${jobId}/preset`, { method: "POST" });
}

export function deletePreset(presetId: string) {
  return request<{ deleted: string }>(`/api/presets/${encodeURIComponent(presetId)}`, {
    method: "DELETE"
  });
}

export function openJobEvents(jobId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${protocol}://${window.location.host}/api/jobs/${jobId}/events`);
}
