import {
  Eye,
  ListMusic,
  Loader2,
  Music,
  Play,
  RefreshCw,
  Rocket,
  Save,
  Send,
  Wand2,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createJob,
  deletePreset,
  deployJob,
  getLibrary,
  getLlm,
  getPlayback,
  openJobEvents,
  refineJob,
  savePreset
} from "./api";
import { Player } from "./Player";
import type { ChatMessage, JobEvent, LibraryItem, LibraryResponse, LlmResponse, Playback } from "./types";

type Stage = "select" | "thinking" | "filtering" | "ready" | "playing" | "deploying" | "failed";

function eventLabel(type: string): string {
  return type.replaceAll("_", " ");
}

function isMessages(value: unknown): value is ChatMessage[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        item &&
        typeof item === "object" &&
        "role" in item &&
        "content" in item &&
        typeof item.role === "string" &&
        typeof item.content === "string"
    )
  );
}

export function App() {
  const [library, setLibrary] = useState<LibraryResponse>({ songs: [], presets: [] });
  const [llm, setLlm] = useState<LlmResponse | null>(null);
  const [provider, setProvider] = useState<"openai" | "ollama">("openai");
  const [modelId, setModelId] = useState("gpt-4o");
  const [selected, setSelected] = useState<LibraryItem | null>(null);
  const [previewing, setPreviewing] = useState<LibraryItem | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>("select");
  const [progress, setProgress] = useState(0);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [playback, setPlayback] = useState<Playback | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [refineOpen, setRefineOpen] = useState(false);
  const [refineText, setRefineText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [presetNotice, setPresetNotice] = useState<string | null>(null);
  const [savingPreset, setSavingPreset] = useState(false);
  const [deletingPreset, setDeletingPreset] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const providerInfo = llm?.providers.find((entry) => entry.id === provider);
  const modelOptions = providerInfo?.models ?? [];
  const busy = stage === "thinking" || stage === "filtering" || stage === "deploying";
  const visibleEvents = useMemo(
    () => events.filter((event) => event.type !== "safety_progress"),
    [events]
  );

  useEffect(() => {
    Promise.all([getLibrary(), getLlm()])
      .then(([libraryResponse, llmResponse]) => {
        setLibrary(libraryResponse);
        setLlm(llmResponse);
        setProvider(llmResponse.defaultProvider);
        setModelId(llmResponse.defaultModel);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const refreshLibrary = async () => {
    setLibrary(await getLibrary());
  };

  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  const connectEvents = (nextJobId: string) => {
    socketRef.current?.close();
    const socket = openJobEvents(nextJobId);
    socketRef.current = socket;
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data) as JobEvent;
      setEvents((current) => [...current, event]);

      if (event.type === "thinking_started") {
        setStage("thinking");
      }
      if (event.type === "conversation") {
        const messages = event.payload.messages;
        if (isMessages(messages)) {
          setConversation(messages);
        }
      }
      if (event.type === "safety_started") {
        setStage("filtering");
        setProgress(0);
      }
      if (event.type === "safety_progress") {
        const percent = Number(event.payload.percent ?? 0);
        setStage("filtering");
        setProgress(percent);
      }
      if (event.type === "ready") {
        setStage("ready");
        setProgress(1);
      }
      if (event.type === "deploy_started") {
        setStage("deploying");
      }
      if (event.type === "deploy_complete") {
        setStage("ready");
      }
      if (event.type === "failed") {
        setStage("failed");
        setError(String(event.payload.message ?? "Job failed"));
      }
    };
    socket.onerror = () => setError("Event stream disconnected.");
  };

  const start = async (item: LibraryItem) => {
    setSelected(item);
    setPreviewing(null);
    setJobId(null);
    setEvents([]);
    setConversation([]);
    setPlayback(null);
    setError(null);
    setPresetNotice(null);
    setProgress(0);
    setStage("thinking");
    setDetailsOpen(false);
    const job = await createJob(item.id, provider, modelId);
    setJobId(job.jobId);
    connectEvents(job.jobId);
  };

  const showPlayback = async () => {
    if (!jobId) {
      return;
    }
    const data = playback ?? (await getPlayback(jobId));
    setPlayback(data);
    setStage("playing");
  };

  const submitRefine = async () => {
    if (!jobId || !refineText.trim()) {
      return;
    }
    setPlayback(null);
    setError(null);
    setPresetNotice(null);
    setProgress(0);
    setStage("thinking");
    setDetailsOpen(false);
    await refineJob(jobId, refineText.trim(), provider, modelId);
    setRefineText("");
    setRefineOpen(false);
  };

  const deploy = async () => {
    if (!jobId) {
      return;
    }
    setError(null);
    setStage("deploying");
    await deployJob(jobId);
  };

  const saveSafePreset = async () => {
    if (!jobId) {
      return;
    }
    setSavingPreset(true);
    setError(null);
    setPresetNotice(null);
    try {
      const result = await savePreset(jobId);
      const name = result.preset.song ?? result.preset.label;
      setPresetNotice(`Saved safe preset for ${name}.`);
      await refreshLibrary();
    } finally {
      setSavingPreset(false);
    }
  };

  const deletePresetItem = async (item: LibraryItem) => {
    const name = item.song ?? item.label;
    const confirmed = window.confirm(
      `Are you sure you want to delete the preset for ${name}? This cannot be undone.`
    );
    if (!confirmed) {
      return;
    }
    setDeletingPreset(item.id);
    setError(null);
    setPresetNotice(null);
    try {
      await deletePreset(item.id);
      await refreshLibrary();
    } finally {
      setDeletingPreset(null);
    }
  };

  const reset = () => {
    socketRef.current?.close();
    setSelected(null);
    setPreviewing(null);
    setJobId(null);
    setStage("select");
    setProgress(0);
    setEvents([]);
    setConversation([]);
    setPlayback(null);
    setDetailsOpen(false);
    setRefineOpen(false);
    setRefineText("");
    setError(null);
    setPresetNotice(null);
  };

  if (stage === "playing" && playback) {
    return <Player playback={playback} onClose={() => setStage("ready")} />;
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">SwarmGPT</p>
            <h1>Drone choreography console</h1>
          </div>
          {jobId && (
            <button className="secondary-action" onClick={() => setDetailsOpen((open) => !open)}>
              <Eye size={18} />
              {detailsOpen ? "Hide details" : "Show details"}
            </button>
          )}
        </header>

        {error && <div className="error-banner">{error}</div>}

        {stage === "select" && (
          <>
            <section className="control-band">
              <label>
                LLM backend
                <select
                  value={provider}
                  onChange={(event) => {
                    const next = event.target.value as "openai" | "ollama";
                    setProvider(next);
                    const nextProvider = llm?.providers.find((entry) => entry.id === next);
                    setModelId(nextProvider?.defaultModel ?? nextProvider?.models[0] ?? "");
                  }}
                >
                  {llm?.providers.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Model
                <input
                  list="model-options"
                  value={modelId}
                  onChange={(event) => setModelId(event.target.value)}
                  placeholder="Model name"
                />
                <datalist id="model-options">
                  {modelOptions.map((model) => (
                    <option key={model} value={model} />
                  ))}
                </datalist>
              </label>
              <button
                className="secondary-action"
                onClick={() => getLlm().then(setLlm).catch((err: Error) => setError(err.message))}
              >
                <RefreshCw size={18} />
                Refresh models
              </button>
            </section>

            <section className="library">
              <div className="section-title">
                <ListMusic size={19} />
                <h2>Select a song</h2>
              </div>
              <div className="song-grid">
                {library.songs.map((item) => (
                  <article key={`${item.kind}-${item.id}`} className="song-card">
                    <div>
                      <span className="kind">{item.kind}</span>
                      <h3>{item.label}</h3>
                    </div>
                    <div className="song-actions">
                      <button className="secondary-action compact" onClick={() => setPreviewing(item)}>
                        <Music size={17} />
                        Preview
                      </button>
                      <button
                        className="primary-action compact"
                        disabled={!modelId.trim()}
                        onClick={() => start(item).catch((err: Error) => setError(err.message))}
                      >
                        <Wand2 size={17} />
                        Select
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="library preset-library">
              <div className="section-title">
                <Wand2 size={19} />
                <h2>Presets</h2>
              </div>
              <div className="song-grid preset-grid">
                {library.presets.map((item) => (
                  <article key={`${item.kind}-${item.id}`} className="song-card preset-card">
                    <button
                      className="preset-delete"
                      aria-label={`Delete preset ${item.song ?? item.label}`}
                      disabled={deletingPreset === item.id}
                      onClick={() => deletePresetItem(item).catch((err: Error) => setError(err.message))}
                      title="Delete preset"
                    >
                      <X size={14} />
                    </button>
                    <div>
                      <span className="kind preset-kind">preset</span>
                      <h3>{item.song ?? item.label}</h3>
                      {item.numDrones && <p className="preset-meta">{item.numDrones} Drones</p>}
                      {item.createdLabel && <p className="preset-time">{item.createdLabel}</p>}
                    </div>
                    <div className="song-actions">
                      <button className="secondary-action compact" onClick={() => setPreviewing(item)}>
                        <Music size={17} />
                        Preview song
                      </button>
                      <button
                        className="primary-action compact"
                        disabled={!modelId.trim()}
                        onClick={() => start(item).catch((err: Error) => setError(err.message))}
                      >
                        <Wand2 size={17} />
                        Load preset
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            {previewing && (
              <section className="preview-strip">
                <div>
                  <p className="eyebrow">Preview</p>
                  <strong>{previewing.label}</strong>
                </div>
                <audio src={previewing.previewUrl} controls autoPlay />
              </section>
            )}
          </>
        )}

        {stage !== "select" && selected && (
          <section className="job-panel">
            <div className="job-header">
              <div>
                <p className="eyebrow">{selected.kind}</p>
                <h2>{selected.label}</h2>
              </div>
              <span className={`status-pill ${stage}`}>
                {busy && <Loader2 size={16} className="spin" />}
                {stage}
              </span>
            </div>

            {(stage === "thinking" || stage === "filtering") && (
              <div className="progress-area">
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
                </div>
                <span>{stage === "thinking" ? "Thinking" : `${Math.round(progress * 100)}% safe`}</span>
              </div>
            )}

            {stage === "ready" && (
              <div className="ready-actions">
                <button className="primary-action" onClick={() => showPlayback().catch((err: Error) => setError(err.message))}>
                  <Play size={18} />
                  Play in browser
                </button>
                <button
                  className="secondary-action"
                  disabled={savingPreset}
                  onClick={() => saveSafePreset().catch((err: Error) => setError(err.message))}
                >
                  <Save size={18} />
                  {savingPreset ? "Saving" : "Save safe preset"}
                </button>
                <button className="secondary-action" onClick={() => setRefineOpen((open) => !open)}>
                  <Send size={18} />
                  Refine
                </button>
                <button className="secondary-action" onClick={() => deploy().catch((err: Error) => setError(err.message))}>
                  <Rocket size={18} />
                  Deploy
                </button>
                <button className="secondary-action" onClick={reset}>
                  <ListMusic size={18} />
                  Choose another
                </button>
              </div>
            )}

            {stage === "failed" && (
              <div className="ready-actions">
                <button className="secondary-action" onClick={reset}>
                  <ListMusic size={18} />
                  Choose another
                </button>
              </div>
            )}

            {presetNotice && <p className="status-message">{presetNotice}</p>}

            {refineOpen && (
              <div className="refine-box">
                <textarea
                  value={refineText}
                  onChange={(event) => setRefineText(event.target.value)}
                  placeholder="Describe the choreography change"
                />
                <button
                  className="primary-action"
                  disabled={!refineText.trim()}
                  onClick={() => submitRefine().catch((err: Error) => setError(err.message))}
                >
                  <Send size={18} />
                  Send refine
                </button>
              </div>
            )}
          </section>
        )}
      </section>

      {detailsOpen && (
        <aside className="details-panel">
          <div className="section-title">
            <Eye size={18} />
            <h2>Generation details</h2>
          </div>
          <p className="details-intro">
            Status changes and model messages for the current choreography. Safety filter progress is shown in the main bar.
          </p>
          <div className="event-list">
            {visibleEvents.map((event) => (
              <div key={event.id} className="event-row">
                <span>{eventLabel(event.type)}</span>
                <time>{new Date(event.createdAt).toLocaleTimeString()}</time>
              </div>
            ))}
          </div>
          <div className="conversation">
            {conversation.map((message, index) => (
              <article key={`${message.role}-${index}`} className="message">
                <span>
                  {message.role === "assistant"
                    ? "Generated choreography"
                    : message.role === "user"
                      ? "Choreography request"
                      : "Model instructions"}
                </span>
                <p>{message.content}</p>
              </article>
            ))}
          </div>
        </aside>
      )}
    </main>
  );
}
