# Trajectory-era entries

These four were authored under the removed path, where the model wrote a whole
`(params, swarm_pos, tstart, tend, limits) -> (final_pos, waypoints)` primitive rather than the
equation of a shape. They no longer load: `PrimitiveManifest` parses shapes only, and this
directory sits outside `load_promoted`'s glob deliberately.

They are kept as evidence, not as library entries:

- `double_helix.json` — the first primitive to clear both gates (authored separation 1.265, flown
  1.235, 0 steps inside the envelope, 7 iterations).
- `upright_heart_outline.json` — a heart the choreographer had no way to express, y spread 0.000 m.
- `altitude_separated_double_helix.json` — collision-safe, and **two flat counter-rotating rings**,
  not a double helix. It is the evidence that the model certifies its own output: it passed 5/5 of
  the invariants it wrote for itself.
- `double_helix.png` — the render of the first of these.
