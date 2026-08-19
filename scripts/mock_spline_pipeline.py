"""Mock end-to-end spline pipeline test — interactive 3-D viewer.

Builds a fake SongStructure, constructs a hand-written choreography string (as the LLM
would produce it), feeds it through Choreographer.response2trajectory, and opens an
interactive draggable 3-D plot with a time slider and playback button.

Run with:
    .pixi/envs/default/bin/python scripts/mock_spline_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")  # interactive backend — must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.utils.music_analyzer import Bar, Beat, Segment, SongStructure

# ---------------------------------------------------------------------------
# 1.  Fake song structure — 2 segments, 2 bars each, 4 beats per bar at 120 BPM.
# ---------------------------------------------------------------------------
BPM = 120
BEAT_S = 60.0 / BPM   # 0.5 s per beat
BAR_S = 4 * BEAT_S    # 2 s per bar


def _make_bar(bar_id: int, bar_start: float, n_beats: int = 4) -> Bar:
    beats = [
        Beat(id=i + 1, time_s=bar_start + i * BEAT_S, position_in_bar=i + 1)
        for i in range(n_beats)
    ]
    return Bar(id=bar_id, start_s=bar_start, beats=beats)


seg1_bars = [_make_bar(b, (b - 1) * BAR_S) for b in range(1, 3)]
seg2_start = 2 * BAR_S
seg2_bars = [_make_bar(b, seg2_start + (b - 1) * BAR_S) for b in range(1, 3)]

structure = SongStructure(
    schema_version=2,
    source_path="mock",
    song_sha256="0" * 64,
    analyzer="mock",
    bpm=BPM,
    segments=[
        Segment(id=1, label="intro", start_s=0.0, end_s=seg2_start, bars=seg1_bars),
        Segment(id=2, label="chorus", start_s=seg2_start, end_s=seg2_start + 2 * BAR_S, bars=seg2_bars),
    ],
)

# ---------------------------------------------------------------------------
# 2.  Hand-written choreography text.
#     Pattern: primitive  TRANSITION  primitive  (strict alternation required).
#     s1b1t1 = 0.0 s  |  s1b2t1 = 2.0 s (TRANSITION gap)  |  s2b1t1 = 4.0 s
# ---------------------------------------------------------------------------
CHOREOGRAPHY_TEXT = """\
song_mood: "energetic"
choreography_plan: "mock test: circle -> transition -> helix -> transition -> orbit"
choreography:
  s1b1t1: form_circle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 150, 100, 1.5)
  s1b2t1: TRANSITION
  s2b1t1: helix(4, 50, 120)
  s2b2t1: TRANSITION
  s2b2t3: orbit(180, 80)
  END
"""

# ---------------------------------------------------------------------------
# 3.  Build trajectories.
# ---------------------------------------------------------------------------
choreo = Choreographer(use_motion_primitives=True)
print(f"Drones: {choreo.num_drones}  limits: {choreo.lim_lower} .. {choreo.lim_upper}")

trajectories = choreo.response2trajectory(CHOREOGRAPHY_TEXT, structure)

for d, traj in trajectories.items():
    print(f"  Drone {d}: [{traj.t0:.2f}s, {traj.t1:.2f}s], {len(traj.segments)} segs")

# ---------------------------------------------------------------------------
# 4.  Pre-sample all trajectories at high resolution.
# ---------------------------------------------------------------------------
SAMPLE_HZ = 50
t0_global = min(traj.t0 for traj in trajectories.values())
t1_global = max(traj.t1 for traj in trajectories.values())
n_samples = int(round((t1_global - t0_global) * SAMPLE_HZ)) + 1
times = np.linspace(t0_global, t1_global, n_samples)  # shape (T,)

# all_pts[d] = (T, 3) in cm
all_pts: dict[int, np.ndarray] = {}
for d, traj in trajectories.items():
    all_pts[d] = traj.evaluate(times)

colors = plt.cm.tab10(np.linspace(0, 1, choreo.num_drones))

# ---------------------------------------------------------------------------
# 5.  Build the interactive figure.
#     Layout: large 3-D axes on top, time slider + play button below.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 9))
fig.suptitle("SwarmGPT — drag to rotate, slider to scrub", fontsize=11)

ax3d = fig.add_axes([0.05, 0.22, 0.90, 0.72], projection="3d")

# Full trajectory trails (static, thin)
trail_lines = {}
for d in trajectories:
    (ln,) = ax3d.plot(
        all_pts[d][:, 0], all_pts[d][:, 1], all_pts[d][:, 2],
        color=colors[d], lw=0.8, alpha=0.35,
    )
    trail_lines[d] = ln

# Current-position dots (updated by slider)
dot_x = [all_pts[d][0, 0] for d in trajectories]
dot_y = [all_pts[d][0, 1] for d in trajectories]
dot_z = [all_pts[d][0, 2] for d in trajectories]
dot_colors = [colors[d] for d in trajectories]
dots = ax3d.scatter(dot_x, dot_y, dot_z, c=dot_colors, s=60, zorder=5, depthshade=False)

# Past-trail highlight (head of trail up to current time)
head_lines = {}
for d in trajectories:
    (ln,) = ax3d.plot(
        [all_pts[d][0, 0]], [all_pts[d][0, 1]], [all_pts[d][0, 2]],
        color=colors[d], lw=1.8, alpha=0.9,
    )
    head_lines[d] = ln

# Axis labels and limits
all_xyz = np.concatenate(list(all_pts.values()), axis=0)
pad = 20
ax3d.set_xlim(all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad)
ax3d.set_ylim(all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad)
ax3d.set_zlim(all_xyz[:, 2].min() - pad, all_xyz[:, 2].max() + pad)
ax3d.set_xlabel("x (cm)")
ax3d.set_ylabel("y (cm)")
ax3d.set_zlabel("z (cm)")

# Beat markers — vertical lines at beat times projected onto the floor
z_floor = float(all_xyz[:, 2].min()) - pad
for seg in structure.segments:
    for bar in seg.bars:
        for beat in bar.beats:
            # thin vertical tick at beat time (we can't easily draw them in 3D without clutter,
            # so we add a small floor marker instead)
            pass  # kept intentionally sparse to avoid clutter

# Time label
time_text = ax3d.text2D(0.02, 0.97, "t = 0.00 s", transform=ax3d.transAxes, fontsize=10)

# ---------------------------------------------------------------------------
# 6.  Slider
# ---------------------------------------------------------------------------
ax_slider = fig.add_axes([0.12, 0.10, 0.70, 0.03])
slider = Slider(ax_slider, "Time (s)", t0_global, t1_global, valinit=t0_global, valstep=1.0 / SAMPLE_HZ)

ax_btn_play = fig.add_axes([0.85, 0.07, 0.08, 0.06])
btn_play = Button(ax_btn_play, "▶ Play")

ax_btn_reset = fig.add_axes([0.85, 0.01, 0.08, 0.05])
btn_reset = Button(ax_btn_reset, "Reset")


def _idx_at(t: float) -> int:
    return int(np.clip(round((t - t0_global) * SAMPLE_HZ), 0, n_samples - 1))


def update(t: float) -> None:
    idx = _idx_at(t)
    # Update dot positions
    new_x = [all_pts[d][idx, 0] for d in trajectories]
    new_y = [all_pts[d][idx, 1] for d in trajectories]
    new_z = [all_pts[d][idx, 2] for d in trajectories]
    dots._offsets3d = (new_x, new_y, new_z)  # type: ignore[attr-defined]
    # Update head trails
    for d in trajectories:
        head_lines[d].set_data(all_pts[d][: idx + 1, 0], all_pts[d][: idx + 1, 1])
        head_lines[d].set_3d_properties(all_pts[d][: idx + 1, 2])
    time_text.set_text(f"t = {t:.2f} s")
    fig.canvas.draw_idle()


slider.on_changed(update)

# Play animation
_playing = [False]
_timer = [None]


def _animate(t_start: float) -> None:
    import time as _time
    t = t_start
    dt = 1.0 / SAMPLE_HZ
    while _playing[0] and t <= t1_global:
        slider.set_val(t)
        fig.canvas.flush_events()
        _time.sleep(dt * 0.5)  # run at 2× real-time; adjust multiplier as desired
        t += dt
    _playing[0] = False
    btn_play.label.set_text("▶ Play")
    fig.canvas.draw_idle()


def on_play(event: object) -> None:
    if _playing[0]:
        _playing[0] = False
        btn_play.label.set_text("▶ Play")
    else:
        _playing[0] = True
        btn_play.label.set_text("⏹ Stop")
        import threading
        t = float(slider.val)
        if t >= t1_global:
            t = t0_global
        threading.Thread(target=_animate, args=(t,), daemon=True).start()


def on_reset(event: object) -> None:
    _playing[0] = False
    btn_play.label.set_text("▶ Play")
    slider.set_val(t0_global)


btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)

# Initial draw at t=0
update(t0_global)

plt.show()
