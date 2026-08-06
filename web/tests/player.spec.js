import { test, expect } from "@playwright/test";

const silenceWav = Buffer.from(
  "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "base64"
);

function makePlayback() {
  const timestamps = [0, 0.5, 1.0, 1.5, 2.0];
  // Drone 0 is the lighting probe (see the cue lists below), and it flies the same path the other
  // two do but at a fifth of the distance from the camera -- each of its rows is a fifth of the way
  // from the camera at (2.8, -3.2, 2.4) to where it would otherwise be. That keeps it on exactly the
  // same view ray, so the deck viewing angle the `magenta < cyan` comparison depends on is unchanged
  // and only its on-screen area grows, by 25x. At the original distance its underside covered a
  // single pixel in the mobile viewport and at half the distance six, which is no margin at all for
  // an assertion that has to tell the two decks apart on another GPU, driver or viewport.
  const positions = [
    [
      [2.1, -2.64, 2.04],
      [0.0, 0.45, 0.8],
      [0.72, -0.25, 1.0]
    ],
    [
      [2.15, -2.61, 2.07],
      [0.15, 0.35, 0.95],
      [0.62, -0.15, 1.08]
    ],
    [
      [2.2, -2.58, 2.1],
      [0.3, 0.25, 1.05],
      [0.48, -0.02, 1.16]
    ],
    [
      [2.25, -2.548, 2.084],
      [0.48, 0.12, 0.9],
      [0.3, 0.16, 1.1]
    ],
    [
      [2.3, -2.52, 2.064],
      [0.62, -0.05, 0.82],
      [0.12, 0.34, 0.98]
    ]
  ];

  // Compiled lighting cues: step events under zero-order hold, one list per drone per deck, ending
  // on the unconditional blackout. Drone 0 is the probe: at t = 1.0 its **top** deck turns CYAN and
  // its **bottom** deck turns MAGENTA, and nothing else in the scene is either colour. Everything
  // the lighting assertions in `exercise` key off comes from those two cues.
  //
  // Two cues on one drone, one per deck, because a single cue cannot say which mesh carries which
  // material: the two decks are otherwise interchangeable, and swapping which material lands on
  // z = +0.015 and z = -0.002 stayed invisible. With both decks lit differently the counts
  // themselves separate them: from a camera looking down, the upward-facing diffusor shows several
  // times the pixels the underside does, so cyan must outnumber magenta.
  //
  // Trails are one neutral grey and never follow the lighting (§9.2), so both counts are the
  // diffusors alone: no line contributes to either, and a colour count constrains the LEDs
  // directly. Cyan lost the trail line when that changed (605 -> 464 desktop, 342 -> 240 mobile);
  // magenta was never on it and did not move. The ~6x separation is the viewing angle, as intended.
  const blackout = 1.9;
  const lighting = {
    top: [
      {
        times: [0, 1.0, blackout],
        rgb: [
          [242, 51, 46],
          [0, 255, 255],
          [0, 0, 0]
        ]
      },
      {
        times: [0, 0.75, blackout],
        rgb: [
          [242, 140, 46],
          [160, 120, 30],
          [0, 0, 0]
        ]
      },
      {
        times: [0, blackout],
        rgb: [
          [245, 199, 56],
          [0, 0, 0]
        ]
      }
    ],
    bot: [
      {
        times: [0, 1.0, blackout],
        rgb: [
          [121, 26, 23],
          [255, 0, 255],
          [0, 0, 0]
        ]
      },
      {
        times: [0, 1.25, blackout],
        rgb: [
          [121, 26, 23],
          [160, 90, 40],
          [0, 0, 0]
        ]
      },
      {
        times: [0, blackout],
        rgb: [
          [122, 100, 28],
          [0, 0, 0]
        ]
      }
    ]
  };

  return {
    schemaVersion: 2,
    audioUrl: "/api/media/music/Harness",
    song: "Harness",
    numDrones: 3,
    timestamps,
    states: positions.map((frame) =>
      frame.map(([x, y, z]) => [x, y, z, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    ),
    fields: { pos: [0, 3], quat: [3, 7], vel: [7, 10], angVel: [10, 13] },
    // The ceiling clears the probe's new height; only min z (the floor plane) and min/max x, y (the
    // grid extent) are drawn, so this keeps the declared volume honest and changes nothing on screen.
    bounds: { min: [-3, -3, 0.25], max: [3, 3, 2.25] },
    lighting,
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

    const sockets = [];
    window.__emitJobEvent = (event) => {
      for (const socket of sockets) {
        socket.emitJson(event);
      }
    };

    class FakeWebSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;

      constructor(url) {
        this.url = url;
        this.readyState = FakeWebSocket.CONNECTING;
        this.listeners = new Map();
        sockets.push(this);
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
        const index = sockets.indexOf(this);
        if (index >= 0) {
          sockets.splice(index, 1);
        }
        this.#emit("close", {});
      }

      send() {}

      emitJson(event) {
        this.#emit("message", { data: JSON.stringify(event) });
      }

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

// The two probe colours, as channel indices: two channels above 90 and the third below 45. The
// tests are deliberately symmetric, so comparing their counts compares how much of each deck is
// visible and not how sensitive the two detectors are.
const MAGENTA = { hi: [0, 2], lo: 1 }; // drone 0's bottom deck from t = 1.0
const CYAN = { hi: [1, 2], lo: 0 }; // drone 0's top deck from t = 1.0

// Nothing else in the scene answers either test: the floor, grid, drone bodies and every trail are
// green and grey, and the other two drones' cues keep every channel out of range on both decks.
function countPixels(page, spec) {
  return page.evaluate(({ hi, lo }) => {
    const canvas = document.querySelector("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let count = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i + hi[0]] > 90 && pixels[i + hi[1]] > 90 && pixels[i + lo] < 45) {
        count += 1;
      }
    }
    return count;
  }, spec);
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

  // Lighting is looked up by playhead time every frame, not baked once at mesh construction: both
  // probe cues are unreachable at t = 0 and must appear after seeking past them.
  const magentaBefore = await countPixels(page, MAGENTA);
  const cyanBefore = await countPixels(page, CYAN);
  await page.locator(".timeline").fill("1.2");
  await page.waitForTimeout(300);
  const magentaAfter = await countPixels(page, MAGENTA);
  const cyanAfter = await countPixels(page, CYAN);
  expect(magentaBefore, `magenta at t=0 should be 0, was ${magentaBefore}`).toBe(0);
  expect(cyanBefore, `cyan at t=0 should be 0, was ${cyanBefore}`).toBe(0);
  expect(magentaAfter, `magenta after seeking should be > 0, was ${magentaAfter}`).toBeGreaterThan(0);
  expect(cyanAfter, `cyan after seeking should be > 0, was ${cyanAfter}`).toBeGreaterThan(0);

  // Which mesh carries which material. The camera looks down on the swarm, so the upward-facing
  // diffusor at z = +0.015 presents several times the pixels the underside at z = -0.002 does.
  // Swapping the two materials moves the large deck onto magenta and leaves cyan with the
  // underside, which inverts the inequality; nothing else in the suite notices the swap. A
  // comparison rather than a threshold, so it does not depend on the viewport or the renderer.
  expect(
    magentaAfter,
    `the bottom deck is mostly hidden from this camera, so magenta (${magentaAfter}) must stay under cyan (${cyanAfter})`
  ).toBeLessThan(cyanAfter);

  // Past the blackout, which is the *last* cue in every list. This is what `findCueIndex`'s
  // `length - 1` bound buys: clamped to `length - 2` like its `findSampleIndex` sibling, the
  // lookup can never reach the final cue, so every drone holds its previous colour through
  // landing and the show never goes dark. Seeking only as far as 1.2 leaves that bound free --
  // both probe cues are index 1 of 3, reachable under either bound.
  await page.locator(".timeline").fill("1.95");
  await page.waitForTimeout(300);
  const magentaAfterBlackout = await countPixels(page, MAGENTA);
  const cyanAfterBlackout = await countPixels(page, CYAN);
  expect(
    magentaAfterBlackout,
    `magenta after the blackout should be 0, was ${magentaAfterBlackout}`
  ).toBe(0);
  expect(
    cyanAfterBlackout,
    `cyan after the blackout should be 0, was ${cyanAfterBlackout}`
  ).toBe(0);
  console.log(
    JSON.stringify({ ...result, droneAssets: droneAssetStatuses.length, magentaAfter, cyanAfter })
  );
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

test("deploy failure returns to ready controls", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await configurePage(page);
  await page.route("**/api/jobs/job/deploy", async (route) => {
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ jobId: "job" })
    });
    await page.evaluate(() => {
      setTimeout(() => {
        window.__emitJobEvent({
          id: 6,
          type: "deploy_started",
          createdAt: new Date().toISOString(),
          payload: {}
        });
      }, 10);
      setTimeout(() => {
        window.__emitJobEvent({
          id: 7,
          type: "failed",
          createdAt: new Date().toISOString(),
          payload: { message: "Drone link lost during deploy" }
        });
      }, 30);
    });
  });

  await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "Select a song" }).waitFor();
  await page.locator(".song-card").filter({ hasText: "Harness" }).first().getByRole("button", { name: "Select" }).click();
  await page.getByRole("button", { name: "Deploy" }).waitFor();
  await page.getByRole("button", { name: "Deploy" }).click();

  await expect(page.getByText("Drone link lost during deploy")).toBeVisible();
  await expect(page.locator(".status-pill")).toHaveText("ready");
  await expect(page.getByRole("button", { name: "Play in browser" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Refine" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Deploy" })).toBeVisible();
});

test("deploying state shows e-stop button that sends emergency stop", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await configurePage(page);
  let emergencyStopRequested = false;
  await page.route("**/api/jobs/job/deploy", async (route) => {
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ jobId: "job" })
    });
    await page.evaluate(() => {
      setTimeout(() => {
        window.__emitJobEvent({
          id: 6,
          type: "deploy_started",
          createdAt: new Date().toISOString(),
          payload: {}
        });
      }, 10);
    });
  });
  await page.route("**/api/jobs/job/emergency-stop", async (route) => {
    emergencyStopRequested = true;
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ jobId: "job", emergencyStopped: true })
    });
    await page.evaluate(() => {
      window.__emitJobEvent({
        id: 7,
        type: "emergency_stop_sent",
        createdAt: new Date().toISOString(),
        payload: {}
      });
    });
  });

  await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "Select a song" }).waitFor();
  await page.locator(".song-card").filter({ hasText: "Harness" }).first().getByRole("button", { name: "Select" }).click();
  await page.getByRole("button", { name: "Deploy" }).waitFor();
  await page.getByRole("button", { name: "Deploy" }).click();
  await page.getByRole("button", { name: "E-stop" }).click();

  expect(emergencyStopRequested).toBeTruthy();
  await expect(page.getByText("Emergency stop sent.")).toBeVisible();
});
