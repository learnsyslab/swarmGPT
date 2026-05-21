import type { LibraryItem, LibraryResponse, LlmResponse, Playback } from "./types";

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

export function getPlayback(jobId: string): Promise<Playback> {
  return request<Playback>(`/api/jobs/${jobId}/playback`);
}

export function refineJob(
  jobId: string,
  message: string,
  provider: string,
  modelId: string
) {
  return request<{ jobId: string }>(`/api/jobs/${jobId}/refine`, {
    method: "POST",
    body: JSON.stringify({ message, provider, modelId })
  });
}

export function deployJob(jobId: string) {
  return request<{ jobId: string }>(`/api/jobs/${jobId}/deploy`, { method: "POST" });
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
