import { test, expect } from "@playwright/test";

const silenceWav = Buffer.from(
  "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "base64"
);

function makePlayback() {
  const timestamps = [0, 0.5, 1.0, 1.5, 2.0];
  const positions = [
    [
      [-0.7, -0.4, 0.6],
      [0.0, 0.45, 0.8],
      [0.72, -0.25, 1.0]
    ],
    [
      [-0.45, -0.25, 0.75],
      [0.15, 0.35, 0.95],
      [0.62, -0.15, 1.08]
    ],
    [
      [-0.2, -0.1, 0.9],
      [0.3, 0.25, 1.05],
      [0.48, -0.02, 1.16]
    ],
    [
      [0.05, 0.06, 0.82],
      [0.48, 0.12, 0.9],
      [0.3, 0.16, 1.1]
    ],
    [
      [0.3, 0.2, 0.72],
      [0.62, -0.05, 0.82],
      [0.12, 0.34, 0.98]
    ]
  ];

  return {
    schemaVersion: 1,
    audioUrl: "/api/media/music/Harness",
    song: "Harness",
    numDrones: 3,
    timestamps,
    states: positions.map((frame) =>
      frame.map(([x, y, z]) => [x, y, z, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    ),
    fields: { pos: [0, 3], quat: [3, 7], vel: [7, 10], angVel: [10, 13] },
    bounds: { min: [-3, -3, 0.25], max: [3, 3, 1.75] },
    colors: [
      [0.95, 0.2, 0.18],
      [0.18, 0.72, 0.95],
      [0.96, 0.78, 0.22]
    ],
    sampleRate: 2
  };
}

async function configurePage(page) {
  await page.addInitScript(() => {
    const events = [
      { id: 1, type: "thinking_started", createdAt: new Date().toISOString(), payload: {} },
      {
        id: 2,
        type: "conversation",
        createdAt: new Date().toISOString(),
        payload: {
          messages: [
            { role: "user", content: "Create a compact test choreography." },
            { role: "assistant", content: "Generated a compact three-drone choreography." }
          ]
        }
      },
      { id: 3, type: "safety_started", createdAt: new Date().toISOString(), payload: {} },
      { id: 4, type: "safety_progress", createdAt: new Date().toISOString(), payload: { percent: 0.5 } },
      { id: 5, type: "ready", createdAt: new Date().toISOString(), payload: { duration: 2 } }
    ];

    class FakeWebSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;

      constructor(url) {
        this.url = url;
        this.readyState = FakeWebSocket.CONNECTING;
        this.listeners = new Map();
        setTimeout(() => {
          this.readyState = FakeWebSocket.OPEN;
          this.#emit("open", {});
          events.forEach((event, index) => {
            setTimeout(() => this.#emit("message", { data: JSON.stringify(event) }), 20 + index * 30);
          });
        }, 0);
      }

      addEventListener(type, handler) {
        const handlers = this.listeners.get(type) ?? [];
        handlers.push(handler);
        this.listeners.set(type, handlers);
      }

      removeEventListener(type, handler) {
        const handlers = this.listeners.get(type) ?? [];
        this.listeners.set(type, handlers.filter((entry) => entry !== handler));
      }

      close() {
        this.readyState = FakeWebSocket.CLOSED;
        this.#emit("close", {});
      }

      send() {}

      #emit(type, event) {
        const property = this[`on${type}`];
        if (typeof property === "function") {
          property.call(this, event);
        }
        for (const handler of this.listeners.get(type) ?? []) {
          handler.call(this, event);
        }
      }
    }

    window.WebSocket = FakeWebSocket;
    HTMLMediaElement.prototype.play = function play() {
      this.dispatchEvent(new Event("play"));
      return Promise.resolve();
    };
    HTMLMediaElement.prototype.pause = function pause() {
      this.dispatchEvent(new Event("pause"));
    };
  });

  await page.route("**/api/library", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        songs: [{ id: "Harness", label: "Harness", kind: "song", previewUrl: "/api/media/music/Harness" }],
        presets: [
          {
            id: "Harness | Compact preset",
            label: "Harness",
            kind: "preset",
            previewUrl: "/api/media/music/Harness",
            song: "Harness",
            numDrones: 3,
            createdAt: "2026-05-21T12:34:56",
            createdLabel: "2026-05-21 12:34"
          }
        ]
      })
    });
  });
  await page.route("**/api/llm", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        providers: [{ id: "openai", label: "OpenAI", models: ["gpt-4o"], defaultModel: "gpt-4o" }],
        defaultProvider: "openai",
        defaultModel: "gpt-4o"
      })
    });
  });
  await page.route("**/api/jobs", async (route) => {
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ jobId: "job", eventsUrl: "/api/jobs/job/events" })
    });
  });
  await page.route("**/api/jobs/job/playback", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(makePlayback()) });
  });
  await page.route("**/api/media/music/Harness", async (route) => {
    await route.fulfill({ contentType: "audio/wav", body: silenceWav });
  });
}

async function exercise(page) {
  const droneAssetStatuses = [];
  page.on("response", (response) => {
    if (response.url().includes("/api/assets/drone/")) {
      droneAssetStatuses.push(response.status());
    }
  });

  await configurePage(page);
  await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "Select a song" }).waitFor();
  await page.getByRole("heading", { name: "Presets" }).waitFor();
  await page.getByText("3 Drones").waitFor();
  await page.getByText("2026-05-21 12:34").waitFor();
  await page.locator(".song-card").filter({ hasText: "Harness" }).first().getByRole("button", { name: "Select" }).click();
  await page.getByRole("button", { name: "Play in browser" }).waitFor();
  await page.getByRole("button", { name: "Save safe preset" }).waitFor();
  await expect(page.locator(".details-panel")).toHaveCount(0);

  await page.getByRole("button", { name: "Show details" }).click();
  const detailsText = await page.locator(".details-panel").innerText();
  expect(detailsText).not.toMatch(/safety progress/i);
  expect(detailsText).toMatch(/Generated choreography/i);

  await page.getByRole("button", { name: "Play in browser" }).click();
  await page.locator("canvas").waitFor();
  await page.waitForTimeout(1800);

  const result = await page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) {
      return { ok: false, reason: "missing canvas" };
    }
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    if (!gl) {
      return { ok: false, reason: "missing webgl context" };
    }
    const width = canvas.width;
    const height = canvas.height;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let lit = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 50) {
        lit += 1;
      }
    }
    return {
      ok: lit > width * height * 0.02,
      width,
      height,
      lit,
      playButton: document.querySelector(".playback-controls button")?.textContent ?? ""
    };
  });
  expect(droneAssetStatuses.filter((status) => status === 200).length).toBeGreaterThanOrEqual(8);
  expect(result.ok, JSON.stringify(result)).toBeTruthy();
  expect(result.playButton).toMatch(/Pause/i);
  console.log(JSON.stringify({ ...result, droneAssets: droneAssetStatuses.length }));
}

const chromeExecutable = process.env.PLAYWRIGHT_CHROME_EXECUTABLE ?? "/usr/bin/google-chrome";

test.use({
  launchOptions: {
    executablePath: chromeExecutable,
    args: ["--no-sandbox", "--disable-dev-shm-usage"]
  }
});

test("desktop browser replay canvas", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await exercise(page);
});

test("mobile browser replay canvas", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await exercise(page);
});
