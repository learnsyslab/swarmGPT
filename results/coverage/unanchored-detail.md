# Unanchored elicitation probe — results

Source: `unanchored_20260817_154201.json`  |  model: `gpt-5.6-luna`

## What this measures

The model is asked for a choreography plan in plain language **with no primitive list in the prompt** (`blind`), and again with the real primitive list added and nothing else changed (`anchored`). A separate judge then decides, for each described moment, whether the real library can deliver it. The judge sees one intent and the primitive list, never the song, the surrounding plan, or which condition produced it.

The headline is the **difference** between conditions. An absolute figure would inherit whatever bias the judge has; a difference does not, as long as the judge cannot tell the conditions apart — which it cannot.

## Summary

| condition | intents | expressible | any shortfall | capability shortfall |
|---|---|---|---|---|
| blind | 88 | 33% | 67% (59) | 56% (49) |
| anchored | 88 | 76% | 24% (21) | 20% (18) |

**Anchoring delta, any shortfall: +43.2%**

**Anchoring delta, capability only: +35.2%** — the defensible figure, with colour-palette complaints excluded.

No intent in either condition was judged `not_expressible`. Every shortfall is `partial`, so the claim is that the library **degrades** what the model asks for, not that it cannot do it at all.

## Per song

| song | blind expressible | anchored expressible |
|---|---|---|
| Chopin - Nocturne Op. 9 No. 2 | 0/5 | 4/5 |
| Crazyflie Drones Theme | 4/9 | 8/9 |
| Crazyflie Drones Theme (short) | 1/5 | 3/5 |
| Dance of the Flying Machines | 3/5 | 4/5 |
| Debussy - Clair de Lune | 2/5 | 4/5 |
| Fearless2 | 2/5 | 5/5 |
| J. Strauss I - Radetzky March | 3/8 | 5/8 |
| J. Strauss II - The Blue Danube Op. 314 | 3/8 | 8/8 |
| Mortals | 4/6 | 6/6 |
| Mozart - Eine kleine Nachtmusik | 2/10 | 6/10 |
| On & On | 1/4 | 2/4 |
| Vivaldi - Summer | 3/8 | 5/8 |
| We Wish You a Merry Christmas | 1/10 | 7/10 |

## Blind-condition shortfalls (judge saw no condition label) — 59 of 88

**Chopin - Nocturne Op. 9 No. 2 · s1b1t1** — _capability + palette_

- wanted: Begin as a small, low cluster near the center, almost like a single glowing point. On the first beat, the ten drones slowly unfurl into a soft crescent, with the center drones slightly higher and the outer drones lower. The movement should feel weightless and organic, like a flower opening or a breath being drawn in, with no sudden acceleration.
- lighting: Use a very dim, warm ivory glow, concentrated at the center and fading toward the edges. The light should gently swell with the first phrase, creating the feeling of a private thought appearing in darkness rather than a theatrical entrance.
- judge says missing: The crescent and stepped vertical profile can be approximated with individual move actions over a long interval, but there is no explicit easing or organic acceleration control. The lighting palette has no ivory colour, and gradient changes colour rather than brightness, so it cannot create a true dim radial brightness fade from a glowing center to darker edges.

**Chopin - Nocturne Op. 9 No. 2 · s1b5t1** — _capability + palette_

- wanted: From the crescent, let the swarm drift into a broad, graceful arc spanning the space, as though the original thought is expanding into a memory. The drones should rise subtly while maintaining a slow, continuous curve, with a slight asymmetry that keeps the image human and hand-shaped rather than geometrically perfect.
- lighting: Deepen the ivory into a soft champagne-gold. Let the center brighten first, followed by a slow ripple of light toward both ends of the arc. The result should resemble candlelight traveling across a room—tender, fluid, and full of restrained longing.
- judge says missing: There is no ivory or champagne-gold palette colour, so the hue must be approximated with amber/orange/white. The ripple can approximate the center-outward candlelight effect only when the arc provides sufficiently different radii from the swarm centre.

**Chopin - Nocturne Op. 9 No. 2 · s2b1t1** — _capability + palette_

- wanted: After the transition into the second section, gather the arc inward and then release it into a larger, rising spiral. The drones should appear to orbit an invisible emotional center, climbing in a smooth helix while preserving clear spacing and an elegant sense of suspension. The spiral should feel like the music has found confidence without losing its intimacy.
- lighting: Shift gradually from champagne to moonlit blue-white, with a faint violet undertone. Light the drones in a traveling wave around the spiral, alternating between a soft shimmer and a calm sustained glow. The effect should be nocturnal and expansive, as if moonlight were being stirred through the air.
- judge says missing: No primitive produces a single continuously expanding, rising spiral/helix: spiral_speed provides radial growth while helix provides the climb, but combining them sequentially loses the unified smooth motion. The palette also has no champagne or violet, only approximate substitutes, and the lighting primitives do not provide a precise multi-hue champagne-to-blue-white transition with a persistent faint violet undertone in one look.

**Chopin - Nocturne Op. 9 No. 2 · s2b5t1** — _capability + palette_

- wanted: At this more developed point, open the spiral into a wide, sweeping ribbon that bends diagonally through the volume. Let the ten drones trace a single expressive gesture, rising at one end and dipping at the other, then reversing the emphasis in a slow, lyrical sway. It should feel like a dancer extending an arm through a long melodic line.
- lighting: Allow the blue-white light to bloom into silver with brief pearlescent flashes. The highlights should travel along the ribbon in the direction of its sweep, never strobing or becoming harsh. The desired impression is a sudden glint of brilliance inside an otherwise tender nocturne—emotion breaking briefly into radiance.
- judge says missing: There is no true ribbon or curved, coordinated multi-drone trajectory primitive, so the expressive rising-and-dipping gesture and reversal would need to be approximated with discrete repositioning. Silver and pearlescent colours are not in the palette; blue/white highlights can only approximate them.

**Chopin - Nocturne Op. 9 No. 2 · s3b1t1** — _capability + palette_

- wanted: For the final section, let the ribbon dissolve into a calm, symmetrical constellation: a shallow upward arc with the drones spaced like stars around an invisible horizon. Slowly reduce the height and scale of the formation until it feels suspended and almost motionless, leaving a small opening at the center as if the music has exhaled and departed.
- lighting: Fade from silver to a very soft pearl-white, then gradually cool it toward faint blue-gray. Keep the lights visible but delicate, with a final gentle pulse that passes through the constellation and disappears without a blackout. The ending should feel like moonlight lingering after the last thought, peaceful and unresolved rather than conclusively extinguished.
- judge says missing: There are no silver, pearl-white, or blue-gray palette colours, and no true gradual colour fade primitive. The motion primitives do not directly create a shallow upward arc with controlled star-like spacing and a specified central opening while continuously shrinking. A synchronized pulse does not spatially pass through the constellation; ripple_light may approximate it only when drones have varied radii. Thus the overall mood is achievable, but noticeable colour, motion-shape, and lighting-transition details are lost.

**Crazyflie Drones Theme · s1b9t1** — _palette only_

- wanted: The V folds inward and spirals into a rotating helix, with the drones passing through different heights before expanding into a large circular orbit. The effect should suggest a miniature galaxy or propeller turning in the air.
- lighting: Shift from blue to violet as the helix forms. Give the rotating circle a sequential chase around its perimeter, with one warm-white spark traveling through the ring and making the whole formation feel alive.
- judge says missing: The motion can be staged from an inverted cone/V-like formation through a rotating helix into a large circle, including height variation. A blue-to-violet shift is only approximate because violet is not an available palette colour. A perimeter chase is expressible, but a distinct warm-white spark travelling through the ring is not: only white is available, and chase controls brightness rather than assigning a separate moving colour.

**Crazyflie Drones Theme · s1b17t1** — _capability + palette_

- wanted: The butterfly collapses into a narrow vertical column, then bursts outward into a large five-pointed star with two drones emphasizing each major arm. The star should rotate slowly while maintaining a strong, unmistakable silhouette.
- lighting: Use a deep indigo background color on the drones, with vivid gold-white points at the ends of the star arms. A clean white flash marks the burst from the column, followed by a slow golden rotation around the shape.
- judge says missing: There is no primitive that forms a narrow vertical column with the required collapse, and form_cone cannot reliably produce one. The lighting palette has no gold colour, only approximate amber/yellow/white options. A rotating gold-only light pattern around the star is also not directly available; chase can move brightness but not a gold colour around the shape. Exact automatic assignment of two emphasizing drones to every star arm is not provided.

**Crazyflie Drones Theme · s1b25t1** — _capability + palette_

- wanted: The lightning bolt unfolds into a wide circular wave, with the ten drones spaced evenly around its rim. The ring breathes inward and outward twice, then tilts into a rising spiral that suggests the swarm is gathering momentum for a launch.
- lighting: Transition through saturated blue, purple, and fuchsia around the ring. Each expansion should brighten the drones, and the final rising spiral should carry a warm amber glow toward its upper end.
- judge says missing: There is no dedicated lightning-bolt formation or transition primitive, and the palette has no exact purple or fuchsia colours. The available primitives can approximate these with per-drone moves and indigo/magenta/pink, but an audience would notice the lost precision.

**Crazyflie Drones Theme · s1b29t1** — _capability + palette_

- wanted: The spiral compresses into a dense central point, pauses for dramatic tension, and then explodes into a broad radiant sunburst. The drones should hold the sunburst with alternating long and short rays, creating a powerful final image for the main section.
- lighting: Fade the colors toward darkness during the compression, leaving only a faint violet core. On the burst, ignite every drone in brilliant gold-white, with a saturated orange pulse expanding from the center and then settling into a steady celebratory glow.
- judge says missing: There is no true inward-compressing spiral trajectory, no violet palette colour, no direct brightness fade-to-dark control, and form_star does not inherently provide alternating long and short rays. These differences would be noticeable in the intended motion and lighting.

**Crazyflie Drones Theme · s2b1t1** — _capability_

- wanted: For the outro, the sunburst folds into a compact smiling crescent-like arc, then the drones drift outward and upward in a graceful farewell, ending as ten separated points across the upper half of the space. The final motion should feel weightless, friendly, and conclusive.
- lighting: Change the gold-white sunburst into warm festive amber with soft white highlights. The crescent receives one gentle synchronized pulse, then the drones gradually dim one by one until only a few warm sparks remain before fading out.
- judge says missing: There is no true per-drone, sequential gradual dimming/fade primitive. Chase can move illumination from drone to drone, and strobe_decay can fade each flash, but neither produces the described one-by-one gradual dimming of the drones themselves.

**Crazyflie Drones Theme (short) · s1b1t1** — _capability_

- wanted: Begin as a compact, low glowing cluster near the center, with the drones staggered vertically like a small floating spark. On the first beat, the cluster gives a subtle upward lift and opens just enough to become visibly alive, while retaining a clear, rounded silhouette.
- lighting: Use a dim cool-blue glow, with one or two drones briefly flashing white at the center like a spark catching. Keep the overall look restrained and anticipatory, as though the show is just waking up.
- judge says missing: No primitive can create a compact, rounded cluster with only a subtle vertical stagger: form_cone or helix produces a more pronounced non-rounded shape, while form_circle is flat. There is also no selector for the drones physically at the center, so the white flash cannot be reliably assigned to the visual center.

**Crazyflie Drones Theme (short) · s1b5t1** — _palette only_

- wanted: Spread the 10 drones into a crisp five-pointed star, with pairs of drones defining each arm and a slight height difference giving the shape depth. Make the star expand outward on the beat, then gently contract and rotate as a single object, creating a clean, playful reveal.
- lighting: Shift from blue to bright cyan and electric violet. Chase a white highlight around the star’s points in time with the rotation, making the shape feel energized without losing its recognizable outline.
- judge says missing: The exact electric-violet hue is unavailable; the palette only offers nearby indigo, magenta, or blends. The rest of the motion and the white neighbour chase can be approximated closely.

**Crazyflie Drones Theme (short) · s1b13t1** — _capability + palette_

- wanted: Gather rapidly into a compact central vortex, circling upward in a tight helix. At the beat, burst outward into a large, symmetrical flower-like formation with five rounded petals, then pulse the petals open and closed as if the shape is breathing.
- lighting: Start with deep purple and hot pink in the spinning vortex, then explode into brilliant gold and white at the flower reveal. Let each petal carry a different saturated color, with a warm white pulse traveling around the perimeter.
- judge says missing: There is no dedicated rounded five-petal flower or geometric breathing/open-close motion primitive; form_star is angular and pulse affects brightness rather than petal shape. The palette also lacks deep purple, hot pink, and gold, though indigo, pink, amber, yellow, and white can approximate them. A white perimeter chase and differently colored petal groups are achievable.

**Crazyflie Drones Theme (short) · s1b17t1** — _capability + palette_

- wanted: Finish the intro with a confident full-volume formation: expand into a wide, balanced arc with a raised center, like a smiling horizon above the audience. Give the arc one strong upward sweep and a synchronized settling motion, ending in a perfectly still, evenly spaced silhouette.
- lighting: Flood the formation in brilliant white with a warm golden edge, then send one final blue-to-gold shimmer from left to right. End on a sharp, unified white flash followed by a steady warm-white hold, making the swarm feel triumphant and ready for the next section.
- judge says missing: There is no true warm-white or gold colour; amber is only an approximation. A white-plus-amber two-deck treatment may suggest warmth but is visibly two-tone rather than a unified warm-white hold, so the exact lighting intent is not fully expressible.

**Dance of the Flying Machines · s1b5t1** — _capability + palette_

- wanted: Expand the two arcs outward across the space, then sweep them around each other in a broad horizontal spiral. The inner drones lead while the outer drones follow, making the swarm feel like a single rotating flying apparatus rather than ten separate objects.
- lighting: Shift from blue-white to metallic silver and pale cyan. Use a slow wave of brightness that follows the spiral, with the leading edge sparkling brightly and the trailing edge fading like an illuminated contrail.
- judge says missing: No primitive forms or preserves two outward-expanding arcs, nor does spiral motion provide controllable inner-versus-outer phase/lead-follow timing. The lighting palette has no metallic silver, and brightness effects cannot precisely create a spiral-following bright leading edge with a fading contrail trailing edge.

**Dance of the Flying Machines · s2b5t1** — _capability + palette_

- wanted: From the hovering line, let the drones drift apart into a high, airy constellation with varied heights, leaving generous space between them. They orbit gently around an invisible center for a final moment, then slow until the formation feels suspended and weightless, like flying machines becoming stars.
- lighting: Fade the electric blue into soft violet, then into quiet white points. Keep each drone subtly twinkling at a different time, with one final coordinated brightening across the constellation before the lights recede to a delicate, barely glowing afterimage.
- judge says missing: There is no precise orbit primitive with controllable deceleration or a final suspended hold. Violet is not an exact palette colour, and lighting has no independent per-drone phase control for truly asynchronous twinkles or a controllable low-level afterglow; strobe_decay is only an approximation. Colour changes and final brightness receding are also replacement/effect steps rather than explicit fades.

**Debussy - Clair de Lune · s1b1t1** — _capability + palette_

- wanted: Begin as a small, low, loosely scattered constellation near the center. Let the ten drones drift almost imperceptibly outward and upward, as if moonlight is slowly spreading across still water. Keep the movement soft, continuous, and slightly uneven so the formation feels alive rather than geometric.
- lighting: Use very dim, cool blue-white light with a silvery haze. The drones should appear like distant points of reflected moonlight, with no harsh edges and no strong focal point. The effect should be intimate, hushed, and just beginning to awaken.
- judge says missing: There is no direct dim-level or haze primitive, so the drones cannot remain consistently very dim with a silvery diffuse appearance. Pulse introduces visible brightness variation rather than a controlled low glow, and the motion primitives do not directly provide a purpose-built almost-imperceptible, organic drift.

**Debussy - Clair de Lune · s2b1t1** — _capability + palette_

- wanted: Open the swarm into a wide, ascending arc that reaches across the available space, with the center lifted above the outer drones. The ten drones should move as if drawn upward by a gentle current, then hover in a slow, collective inhale. The formation should feel larger and more spacious than before, while remaining fluid and weightless.
- lighting: Shift from pale blue to luminous silver-white, with a faint lavender undertone. Brighten the elevated center so the arc reads as a glowing horizon or rising moonlit swell. The effect should be a graceful expansion of emotional scale, not a sudden burst.
- judge says missing: The palette has no silver or lavender, so the luminous silver-white color with a faint lavender undertone cannot be represented exactly. A blue/white or white/magenta approximation would be visibly different.

**Debussy - Clair de Lune · s3b1t1** — _capability + palette_

- wanted: Gather the ten drones into a serene, elevated oval or incomplete ring, centered high enough to feel suspended but still comfortably within the space. Let the ring slowly tilt and dissolve at one point, creating a gentle opening toward the audience. End the moment in near-stillness, with only a barely perceptible drift, as if the entire sky has paused.
- lighting: Return to a refined moon-white with a soft lavender-blue halo. Make the upper portion of the oval subtly brighter while leaving the opening darker and inviting. The effect should be a final suspended image of clarity and longing—radiant, fragile, and calm rather than triumphant.
- judge says missing: The palette has no lavender, and there is no brightness primitive for a reliably subtle, static upper-half emphasis while keeping the opening gently darker. A true soft halo and barely perceptible continuous drift are also not directly available; the closest version would use white-to-blue gradients, selective darkening, and small discrete moves.

**Fearless2 · s1b5t1** — _capability_

- wanted: Unfold the cluster into a wide, low horizontal line, spreading decisively toward both sides while maintaining a clean, balanced shape. The movement should feel like the group is opening its stance and claiming space.
- lighting: Transition from cool blue into brighter white with a restrained gold tint. Let the brightness rise as the line expands, producing a clear sense of resolve without yet reaching full intensity.
- judge says missing: There is no controlled sub-full brightness level or smooth colour crossfade. Pulse reaches full intensity periodically, and lighting keys replace the look outright, so the restrained, gradual transition into brighter-but-not-full white cannot be reproduced exactly.

**Fearless2 · s2b1t1** — _unclear_

- wanted: At the chorus entrance, launch the line upward and outward into a broad, elevated V formation. The center should lead slightly, with the outer drones sweeping back and around so the whole swarm appears to break through an invisible barrier.
- lighting: Flash into brilliant white and saturated electric blue at the moment of expansion. Chase the brightness from the center toward the wings, then hold the full formation glowing strongly and evenly.
- judge says missing: The primitives can place drones into an elevated V and realise the white/blue chase and sustained glow, but they do not provide a coordinated curved or sweeping-back trajectory for the outer drones, nor a specific barrier-breaking motion. Individual moves would only approximate this with direct paths.

**Fearless2 · s2b5t1** — _capability + palette_

- wanted: Drive the swarm into a powerful forward surge, compressing briefly into a spearhead before opening into a large, symmetrical starburst. The motion should feel fast, fearless, and celebratory, with the drones occupying the full available width and rising through the vertical space.
- lighting: Use intense white cores with vivid gold and cyan accents that streak along the direction of travel. Finish the starburst with a strong synchronized flare, making the entire swarm feel like an explosion of confidence and momentum.
- judge says missing: There is no group translation or directional forward-surge primitive, so the swarm cannot convincingly surge forward as a unified formation. Gold is also unavailable as an exact colour, though amber or yellow is a close approximation.

**J. Strauss I - Radetzky March · s1b5t1** — _capability + palette_

- wanted: Re-form into two precise marching ranks and advance them in a short, strongly synchronized surge, preserving rigid military order while the whole formation rises slightly.
- lighting: Use bold gold lighting across both ranks, with alternating white highlights that travel from the front rank to the rear like marching accents.
- judge says missing: There is no gold colour in the palette, only approximations such as amber or yellow. The motion and travelling white accents are otherwise achievable.

**J. Strauss I - Radetzky March · s2b5t1** — _palette only_

- wanted: Spread into a wide, shallow wave that rolls from one side to the other, then level out into a broad ceremonial line facing the audience.
- lighting: Let a bright gold crest travel across the wave, followed by a clean white shimmer along the completed line.
- judge says missing: wave is a standing-wave surface rather than a crest that travels from one side to the other, and the palette has no exact gold colour; amber or yellow is only an approximation

**J. Strauss I - Radetzky March · s3b1t1** — _capability + palette_

- wanted: Break the line into a strong forward-pointing arrowhead and drive it upward and slightly forward, making the swarm feel like a triumphant procession surging ahead.
- lighting: Light the arrowhead in intense brass gold, with the tip burning white and the rest of the formation pulsing behind it in stepped accents.
- judge says missing: There is no brass-gold colour; the available palette can only approximate it, so the intended intense brass-gold appearance would be noticeably different.

**J. Strauss I - Radetzky March · s3b5t1** — _capability + palette_

- wanted: Strike outward from the arrowhead into a huge celebratory burst, then pull rapidly back into a tight, perfectly centered cluster, creating a dramatic cannon-blast-and-recoil effect.
- lighting: Use a blinding white-gold explosion on the outward strike, briefly scatter red-gold sparks at the edges, then return to a concentrated hot-gold glow.
- judge says missing: The exact white-gold, red-gold, and hot-gold hues are not available—only approximations such as white, amber, yellow, or orange. The primitives also do not directly create a true outward blast from an arrowhead or specifically target edge drones as scattered sparks, so the result would read as a large formation burst and recoil with approximate lighting.

**J. Strauss I - Radetzky March · s3b9t1** — _capability + palette_

- wanted: Launch the final formation into a grand rotating star and expand it to its widest, highest shape, holding the silhouette proudly before the drones make one final synchronized upward lift.
- lighting: Build to full golden brilliance with sparkling white tips and brief red accents, then finish in a powerful all-white flash that settles into a steady celebratory gold.
- judge says missing: There is no exact gold colour, and the primitives do not provide a direct selector for the star's geometric tips, so white sparkling tips cannot be guaranteed to appear specifically at the tips.

**J. Strauss II - The Blue Danube Op. 314 · s1b1t1** — _capability_

- wanted: Begin with the ten drones suspended in a loose, asymmetrical constellation, spread widely across the volume at different heights. Give the whole group a barely perceptible shared sway, as though it is being stirred awake, while two or three points drift gently toward the center without forming a rigid pattern.
- lighting: Keep the scene mostly dark. Use very dim, cool blue-white pinpricks, with a soft glimmer that passes slowly from one drone to another. The effect should be mysterious, delicate, and expectant.
- judge says missing: There is no controllable dim-level primitive for a consistently very dim, soft pinprick glow. The available chase/brightness effects are effectively full brightness when active, so the glimmer can be slow and sparse but not reliably delicate or dim as specified.

**J. Strauss II - The Blue Danube Op. 314 · s2b5t1** — _unclear_

- wanted: Expand the arc into a broad, flowing spiral. The outer drones sweep outward while the inner drones curve around them, creating the impression of a current curling into a waltz. Let the spiral breathe vertically, with a gentle wave traveling from one side of the formation to the other.
- lighting: Deepen the blue and add restrained turquoise along the moving edges. The brightest points should travel around the spiral rather than remain fixed, producing a liquid, reflective sensation.
- judge says missing: The primitives can create an expanding spiral, vertical variation, blue/turquoise colour, and a travelling brightness chase. However, wave is explicitly a 2D standing-wave surface rather than a directional wave travelling from one side to the other, and the motion primitives do not independently make inner drones curve around outward-sweeping outer drones with that precise layered flow.

**J. Strauss II - The Blue Danube Op. 314 · s3b5t1** — _capability_

- wanted: Break the ring into two interweaving curved lines that pass around and through one another without collision, then reform as a tilted double spiral. The exchange should feel like a flirtatious waltz between two currents, with a pronounced upward sweep followed by a soft settling motion.
- lighting: Let one curve glow blue and the other glow blue-green, exchanging brightness as they cross. Add brief pearl-white flashes at the moments of closest visual connection, evoking sparkling water and polished ballroom movement.
- judge says missing: There is no direct primitive for two interweaving curved lines with controlled non-colliding passage, nor for a tilted double spiral. A single spiral primitive cannot create two independently interweaving spiral currents or specify tilt, and brightness effects are not inherently tied to the moments when the curves visually cross. The result can approximate the intent but would lose noticeable motion and lighting synchronization details.

**J. Strauss II - The Blue Danube Op. 314 · s3b9t1** — _palette only_

- wanted: Gather the double spiral into a compact, ascending vortex, then release it into a wide fan that opens toward the audience. The opening should feel like a sudden discovery of a vast river or sky: expansive, confident, and still graceful rather than explosive.
- lighting: Build from saturated cobalt at the center to bright azure and silver at the widening edges. As the fan opens, let the drones brighten together, creating a clear swell of radiance while preserving the cool, aquatic character.
- judge says missing: The primitives can create a tightening/ascending spiral or helix and then a broad cone-like release, but they do not provide a true double-spiral-to-fan transition or an audience-oriented planar fan. Lighting can approximate the cool center-to-edge progression with blue/azure/white gradients and synchronized pulsing, but the exact cobalt and silver hues are unavailable.

**J. Strauss II - The Blue Danube Op. 314 · s4b1t1** — _palette only_

- wanted: Compress the fan into a sweeping diagonal ribbon that travels across the full volume, with each drone following a slightly different height and curvature. The ribbon should roll over itself once, like a cresting wave, then level into a poised forward-moving line.
- lighting: Shift toward a more celebratory palette: vivid royal blue with warm champagne-gold highlights. Gold should appear first at the crest of the wave and then flow along the entire ribbon, making the music feel triumphant without losing its elegance.
- judge says missing: The palette has no champagne-gold colour, so amber or yellow is only an approximation. The motion primitives also lack a native 3D ribbon/cresting-wave path; individual moves and helix/twister can approximate the roll but not reproduce it exactly.

**Mortals · s1b1t1** — _capability_

- wanted: The drones appear close together near the center as a small, suspended cluster. They gently breathe outward and inward, like a single organism waking up, with subtle differences in height that make the group feel alive rather than perfectly rigid.
- lighting: Use a very dim, cool-white glow with a faint blue tint. The light should slowly rise and fall with the breathing motion, creating the feeling of a distant ember or consciousness beginning to form.
- judge says missing: There is no continuous radial-breathing primitive, so expansion and contraction must be approximated by successive formation changes. The lighting cannot maintain a genuinely very dim baseline: pulse fades toward off and rises to full brightness, with no low-intensity control for a constant ember-like glow.

**Mortals · s2b13t1** — _capability + palette_

- wanted: The separated groups collapse toward one another and reform as a dense, angular spear pointing forward through the volume. The shape should surge with contained force, then hold in a poised, compressed state—as if everything is gathering before release.
- lighting: Drive the colors toward brilliant white at the leading point, with crimson and violet trailing behind. The swarm should appear intensely charged, almost overexposed, with a hard-edged flash at the moment the formation locks together.
- judge says missing: The primitives do not clearly support a spear pointing horizontally/forward or an explicitly forceful collapse from separated groups. Lighting cannot reliably place brilliant white specifically at the leading point while independently preserving crimson and violet trails; violet is not an available exact color, and spatial selectors do not provide precise leading/trailing placement.

**Mozart - Eine kleine Nachtmusik · s1b1t1** — _capability + palette_

- wanted: Begin as ten widely spaced points suspended at different heights, then draw them inward in a slow, graceful spiral until they form a compact diamond-shaped constellation.
- lighting: Start nearly dark with faint warm-white glimmers. As the drones converge, let each point brighten in sequence, creating a delicate champagne-gold arpeggio that resolves into a unified soft-white glow.
- judge says missing: There is no primitive that produces a compact diamond-shaped constellation after the spiral, and spiral direction/inward convergence is not explicitly controllable. The palette also lacks warm-white or champagne-gold, and the brightness primitives cannot provide independently faint, sequential glimmers as precisely described.

**Mozart - Eine kleine Nachtmusik · s1b5t1** — _palette only_

- wanted: Unfurl the diamond into a perfectly balanced fan, with the center drones slightly forward and the outer drones opening symmetrically like a formal curtain.
- lighting: Use luminous ivory and pale gold. The center should shine first, followed by a smooth outward wave, giving the impression of a polished musical phrase opening into space.
- judge says missing: Exact luminous ivory and pale-gold colours are unavailable; white and amber/yellow can only approximate them. The fan geometry and center-outward timing are otherwise achievable.

**Mozart - Eine kleine Nachtmusik · s1b9t1** — _capability + palette_

- wanted: Collapse the fan into a rotating ring, keeping the formation poised and circular while the drones rise and fall in a subtle alternating ripple.
- lighting: Shift toward brighter white with touches of silver-blue. Make the ring sparkle in alternating pairs, like elegant conversational phrases passing around the circle.
- judge says missing: The motion primitives cannot produce a sustained alternating vertical ripple that travels around the ring; even/odd move_z only creates coarse two-group offsets. There is also no silver colour, only approximate blue/azure tones.

**Mozart - Eine kleine Nachtmusik · s2b5t1** — _capability + palette_

- wanted: Send the ten drones through a fast, buoyant exchange: two interlaced spirals weave around one another and then settle into a tall, symmetrical lyre shape.
- lighting: Use sparkling white with flashes of warm gold at every crossing. The brightness should travel rapidly through the weave, producing the feeling of intricate orchestral dialogue.
- judge says missing: There is no primitive for two independently interlaced spirals or a lyre formation, and lighting cannot be triggered specifically at each geometric crossing. Amber can approximate warm gold, and the brightness travel can be staged rather than crossing-synchronised.

**Mozart - Eine kleine Nachtmusik · s2b9t1** — _capability + palette_

- wanted: Stretch the lyre vertically, then release it into a broad ascending wave: the drones rise in staggered groups from one side to the other, forming a sweeping diagonal crest before leveling into a broad horizon.
- lighting: Build from cool pearl at the low end to brilliant gold at the crest. The wave should brighten progressively, culminating in a full, triumphant white-gold flash across the formation.
- judge says missing: There is no direct lyre-stretch or traveling ascending-wave primitive: move_z only translates groups, and wave is a 2D standing wave rather than a staggered crest traveling across the formation. The palette also lacks exact pearl and gold; white/azure and amber/yellow are only approximations, and a true simultaneous white-gold flash is unavailable on one LED deck.

**Mozart - Eine kleine Nachtmusik · s2b13t1** — _capability + palette_

- wanted: Gather the broad horizon into a rotating ten-point star, then make the star gently precess so its points seem to turn through the air while the whole figure remains centered and regal.
- lighting: Give the star a crisp silver-white outline with a warm golden heart. Let individual points twinkle independently, then synchronize them into a strong, ceremonial pulse.
- judge says missing: The motion can form and rotate a centered star, but there is no explicit control for a ten-point topology or true precession beyond repeated rotation. Lighting can approximate silver-white and warm-golden tones with white and amber/yellow, but cannot create a crisp white outline with a distinct golden heart; gradient(radius) is blended. Independent point twinkling is only approximable through manually partitioned or patterned brightness effects, not a true independent twinkle.

**Mozart - Eine kleine Nachtmusik · s2b17t1** — _capability + palette_

- wanted: Burst the star outward into ten distinct points, each following a graceful curved path, then draw every point back into a tight, elevated crown-shaped formation.
- lighting: Make the outward burst vivid and jewel-like, alternating gold, ivory, and faint rose-white. During the return, intensify the lights until the crown becomes dazzling and unified.
- judge says missing: No primitive explicitly provides graceful curved trajectories for each star point or a crown formation. Individual move actions can place drones in a custom arrangement but do not specify curved paths. Gold, ivory, and rose-white are unavailable as exact palette colours, and the available brightness primitives do not provide a guaranteed continuous intensification synchronized specifically with the return.

**Mozart - Eine kleine Nachtmusik · s3b5t1** — _capability + palette_

- wanted: Finish by lifting the ellipse into a brilliant symmetrical crown above the center of the volume, with the outer drones framing the inner group. Hold the final shape absolutely still after one last slight upward sweep.
- lighting: Rise from soft pearl to radiant champagne-gold, with all ten drones reaching full brightness together. End in a clean, sustained warm-white chord: majestic, joyful, and completely resolved.
- judge says missing: The exact lighting palette is unavailable: there is no pearl, champagne-gold, or warm-white colour. White plus amber/yellow can approximate the progression, but the final warm-white chord and champagne-gold appearance would be noticeably different.

**On & On · s2b1t1** — _capability_

- wanted: The swarm rises and expands into a broad, balanced arc, with the drones moving outward in smooth counterbalanced paths rather than simply spreading apart. The arc gently contracts and expands in time with the beat, like a single breathing organism, while maintaining a strong shared center.
- lighting: Shift from blue into bright cyan and clean white. Each outward expansion sends a luminous wave from the center toward the ends of the arc; each contraction gathers the light back inward. Keep the rhythm clear and confident, as if the chorus has suddenly opened into a larger space.
- judge says missing: There is no inward-travelling lighting primitive to make each contraction visibly gather light back toward the center. The arc must also be manually approximated with coordinated move actions rather than formed directly.

**On & On · s2b5t1** — _capability + palette_

- wanted: Transform the arc into a traveling wave across the full formation. One side lifts while the other dips, and the motion passes through every drone in sequence, creating a smooth, visible ripple that repeats with increasing amplitude. The swarm should feel like a flag, a tide, or energy traveling forever from one side to the other.
- lighting: Use a saturated turquoise-to-violet gradient that travels with the wave. As each drone reaches the crest, it blooms to brilliant white, leaving a brief colored afterglow behind. The wave should look continuous rather than like separate flashes, with the whole formation visibly carrying the pulse onward.
- judge says missing: wave is a 2D standing wave rather than a genuinely traveling wave, and the other motion primitives do not directly provide a smooth sequential ripple with continually increasing amplitude. Lighting can approximate a moving pulse, but cannot continuously bind a brilliant-white crest and brief colored afterglow to each drone's exact wave phase. The palette also has no violet color, only approximate indigo/magenta substitutes.

**On & On · s2b9t1** — _capability_

- wanted: The ten drones spiral inward from the traveling wave into a large rotating ring, then open through its center into a powerful forward-facing loop or endless-infinity shape. The formation keeps turning smoothly, with alternating drones tracing the upper and lower curves so the figure appears to flow without a beginning or end. Finish this moment with the shape expanding to fill the available space while remaining perfectly unified.
- lighting: Build from violet and cyan into intense white-gold at the crossings and front edge of the shape. The light should chase continuously around the ring and infinity curves, with a bright flare whenever paths visually intersect. End on a sustained radiant outline with a subtle pulsing interior, conveying an unstoppable, luminous motion that truly goes on and on.
- judge says missing: The primitives can create a traveling wave, rotating ring, and an approximate loop/infinity formation using individual moves and rotation, with chasing and flashing lights. However, there is no native continuous loop/infinity trajectory or spatially triggered crossing flare, and the two LED rings cannot create a distinct pulsing interior within a sustained radiant outline. The result would be an approximation with noticeable losses.

**Vivaldi - Summer · s1b1t1** — _capability + palette_

- wanted: The ten drones hold a low, compact formation slightly below center, like a dark cloud gathering before the storm. They drift almost imperceptibly inward and downward, with a subtle irregular tremor rather than a perfectly still hover.
- lighting: Very dim, deep blue-violet light with only faint red embers appearing intermittently between the drones. The formation should feel heavy, tense, and expectant.
- judge says missing: There is no true deep blue-violet palette colour, only approximations such as blue or indigo. Brightness primitives cannot hold a uniformly very dim level; they mainly pulse or switch lights on and off. Motion primitives cannot produce a subtle irregular tremor or a controlled, nearly imperceptible inward-and-downward drift of a compact cloud; zig_zag or other multi-step motions would be visibly patterned rather than irregular and delicate.

**Vivaldi - Summer · s1b9t1** — _capability_

- wanted: The line folds into a broken spiral, with the drones circling around an empty center. The movement should be fluid but increasingly agitated, as though a hot wind is twisting the swarm into a vortex.
- lighting: Use stronger orange and red light in the moving inner arc, with cooler blue light on the outer drones. The colors should chase around the spiral and make the center feel dangerously charged.
- judge says missing: The motion primitives can create a spiral/vortex-like formation with increasingly agitated twisting when sequenced, and lighting can provide radial colour variation plus a travelling chase. However, they cannot precisely assign a distinct red-and-orange inner arc while keeping the outer drones blue: gradient supports only two endpoint colours, and the available selectors do not directly select an inner radial subset. The closest version would lose that deliberate three-zone colour treatment.

**Vivaldi - Summer · s2b1t1** — _capability + palette_

- wanted: The drones regroup into a tall, narrow column rising through the center of the volume. The column bends and sways as one body, then briefly buckles near its middle, like a flame being battered by wind.
- lighting: Begin with a concentrated white-hot core at the center, fading through gold and orange toward the edges. Let the lower portion glow red, giving the column a furnace-like intensity.
- judge says missing: There is no primitive for a cohesive column that bends and sways as one body while selectively buckling near its middle. The motion can be approximated with twisting, helical, or zig-zag movement, but the localized flame-like buckle would be visibly absent. The lighting can approximate the white-hot center, warm edges, and red lower section, though a precise multi-stop white-to-gold-to-orange gradient is not directly available.

**Vivaldi - Summer · s3b1t1** — _capability_

- wanted: The crown collapses inward and the swarm plunges into a low, jagged wave spanning the width of the space. The wave travels from one side to the other with sharp rises and drops, like a storm front striking in successive blows.
- lighting: Alternate icy blue-white flashes with dark red pulses as the wave travels. Each crest should flare brightly for an instant, while the troughs fall almost black, creating a stark visual rhythm.
- judge says missing: The wave primitive is explicitly a 2D standing wave, not a wave that travels across the width. The available motion primitives do not directly provide a propagating, successive-blow storm front, so the exact travelling-wave motion and its crest-synchronised lighting cannot be fully realised.

**Vivaldi - Summer · s3b5t1** — _capability_

- wanted: The drones shoot upward and outward into a fractured constellation, then freeze in an asymmetric suspended pattern with large empty gaps. A few drones make tiny, delayed adjustments, as if the storm has exhausted itself but has not truly disappeared.
- lighting: End with cold white points edged in pale blue, while one or two isolated drones retain a dim red glow. Fade the constellation slowly into darkness, leaving a final ominous ember rather than a warm resolution.
- judge says missing: There is no true dim-static brightness control or explicit fade-to-dark primitive. Pulse and strobe_decay continue cycling or flashing rather than reliably holding isolated drones at a dim red ember, and the exact pale-blue edge is not available as a named colour.

**We Wish You a Merry Christmas · s1b5t1** — _capability + palette_

- wanted: Gather into a large, slow-turning Christmas bell made from the ten drones. Let the bell swing slightly from side to side, with a graceful arc that gives it a sense of ringing without becoming frantic.
- lighting: Shift to warm gold and amber, with a brighter golden outline and a softer white glow inside the bell. Pulse the brightness gently on the implied ringing motion.
- judge says missing: There is no true bell-shaped formation or smooth curved swing trajectory. The side-to-side motion would be approximated by piecewise moves or zig-zag motion rather than a continuously graceful arc, and exact gold is unavailable (only amber/yellow/orange approximations).

**We Wish You a Merry Christmas · s1b9t1** — _capability + palette_

- wanted: Break the bell apart into a sweeping spiral, then send the drones outward into a circular ornament-like formation. The circle should rotate smoothly, as though a glass bauble is turning in the air.
- lighting: Use rich red, emerald green, and warm white distributed around the circle. Make the colors travel slowly around the rotating ornament, with occasional pinpoint sparkles like reflected lights.
- judge says missing: The palette provides white but not warm white, and rainbow introduces additional hues rather than smoothly travelling only the specified red, emerald green, and warm-white colours. The available colour primitives cannot continuously move a constrained three-colour pattern around the circle; lighting-key changes would be discrete rather than smooth.

**We Wish You a Merry Christmas · s1b13t1** — _capability + palette_

- wanted: Reassemble into a clearly recognizable five-pointed star, broad and balanced in the center of the volume. Hold the star proudly, then give it a slow, elegant tilt and return it to face the audience.
- lighting: Make the star brilliant warm white with a soft golden halo. Brighten the points one after another, followed by a full-star shimmer that feels like a magical reveal.
- judge says missing: The star formation, holding, tilt/return, white illumination, and shimmer are supported. However, there is no warm-white or dedicated golden-halo effect; white/amber gradients or separate LED decks only approximate it. A chase can brighten drones sequentially, but it cannot reliably target the five star points as distinct groups, so the point-by-point reveal may not read as intended.

**We Wish You a Merry Christmas · s1b17t1** — _capability + palette_

- wanted: Transform the star into the upper half of a Christmas tree, with the drones cascading downward into layered branches. Add a gentle vertical sway, as if the tree is alive in a quiet winter breeze.
- lighting: Use deep green as the main color, with warm white and red points appearing as ornaments. The top drone should glow gold, creating a clear tree topper while the lower branches twinkle in alternating colors.
- judge says missing: A cone or stacked circles can approximate the tree silhouette and layered branches, and repeated move_z actions can suggest vertical motion. However, the primitives do not create distinct cascading branch layers with a natural gentle sway as part of the formation. Lighting can approximate green with white and red ornaments, but there is no exact gold colour; amber or yellow would be an approximation for the topper.

**We Wish You a Merry Christmas · s1b21t1** — _capability + palette_

- wanted: Let the tree dissolve into falling snow: the drones travel in staggered downward paths, then sweep back upward in a soft looping pattern. Finish by gathering into a loose, joyful arc that suggests snowfall settling around the scene.
- lighting: Return to icy white, pale blue, and silver. Keep the lights sparkling asynchronously, with brief bright flakes and a gradual warm glow beginning to emerge near the center.
- judge says missing: There is no silver colour primitive, so the silver portion must be approximated with white and pale blue. The arc and asynchronous sparkle can be approximated through individual positioning and running-light effects rather than dedicated primitives.

**We Wish You a Merry Christmas · s2b1t1** — _palette only_

- wanted: Use the arc as the rim of a large wreath, closing the formation into a full circle with a clear opening in the middle. Rotate the wreath slowly while the drones bob with a buoyant, celebratory rhythm.
- lighting: Light the wreath in evergreen green with evenly spaced red and gold accents. Add a warm white inner glow so the center remains inviting and unmistakable.
- judge says missing: The exact lighting intent is not fully expressible: the palette has no gold or warm-white colour, only amber/yellow and white. Evenly spaced red-and-gold accent placement is also not guaranteed by the available spatial colour primitives.

**We Wish You a Merry Christmas · s2b5t1** — _capability + palette_

- wanted: Open the wreath and send the drones into a sweeping figure-eight around one another, like ribbons of festive light dancing through the air. The crossing should be smooth and symmetrical, with the swarm briefly forming two interlocking loops.
- lighting: Use alternating candy-cane red and warm white trails of light, enriched with small gold flashes at each crossing. Increase brightness through the center of the figure-eight for a feeling of joyful lift.
- judge says missing: The primitives can create rotating, spiralling, or twisted formations and approximate a symmetrical interweaving, but they do not define a smooth figure-eight path with two interlocking loops or crossing-specific motion. Red and white lighting can be alternated, and flashes can be added, but there is no warm-white or gold colour (amber/yellow are only approximations), no persistent light trails, and no direct brightness control that specifically increases at each figure-eight crossing.

**We Wish You a Merry Christmas · s2b9t1** — _capability_

- wanted: Collapse the figure-eight into ten rising candles or glowing points arranged in a gentle ascending fan. Have the fan breathe outward and inward, then lift together as though making a shared wish.
- lighting: Give each drone a warm candle-gold core with a soft amber halo. Flicker subtly and independently at first, then synchronize into one bright, steady golden swell.
- judge says missing: There is no true fan formation with individually arranged rising points, no direct collapse-from-figure-eight transition, and no formation-scaling primitive for smooth outward/inward breathing. The two LED rings provide directional two-tone light rather than a genuine candle core with a soft radial amber halo, and independent subtle flicker is only approximable rather than directly supported.

**We Wish You a Merry Christmas · s2b13t1** — _palette only_

- wanted: From the rising fan, form a large radiant star high in the available space, with the remaining shape reading as a small wreath or cradle beneath it. Hold the complete Christmas tableau for the ending, adding one slow upward float and a final calm settling motion.
- lighting: Use brilliant warm white and gold for the star, surrounded by a restrained halo of red and green from the lower wreath. Finish with all ten lights glowing together, then soften slightly while preserving the star as the last clear image.
- judge says missing: The primitives do not provide a true rising fan formation, and form_star applies to the whole swarm, so creating a separate lower wreath requires moving a subset afterward and may leave the star incomplete. Gold is also only approximated by amber or yellow.


## Anchored-condition shortfalls, for comparison — 21 of 88

**Chopin - Nocturne Op. 9 No. 2 · s1b1t1** — _unclear_

- wanted: The full swarm opens as one calm, perfectly spaced circle, floating at about 0.90 m with an 0.85 m radius. The formation should feel suspended and almost weightless, like the first sustained phrase of the nocturne. Use form_circle on drone_ids '1-10' with radius 85 cm, z 90 cm, and a 1.5 s arrival.
- lighting: Give every drone a deep indigo-to-cyan gradient radiating from the centre across both LED decks. The whole circle should breathe very slowly together over 4 beats, creating a cool, glassy pulse rather than a hard flash.
- judge says missing: The circle places every drone at the same radius, so a radius-based gradient collapses to one colour rather than producing a visible indigo-to-cyan gradient radiating from the centre. The synchronized four-beat breathing across both decks is expressible.

**Crazyflie Drones Theme · s1b25t1** — _capability_

- wanted: Split the swarm into two distinct stacked rings: drones 1-5 form a lower amber ring at about 0.75 m, while drones 6-10 form an upper cyan ring at about 1.45 m. The two rings should rotate visually as a layered double halo with a clear vertical gap.
- lighting: Give drones 1-5 a warm amber colour and drones 6-10 a cool cyan colour. Make the two groups blink in opposite phases over 2 beats, creating a crisp alternating lower-ring/upper-ring exchange.
- judge says missing: The two arbitrary ID groups can be coloured separately, but no brightness primitive provides a phase offset between exactly drones 1-5 and 6-10. alternate_blink only splits a selected group by parity or stage side, so it cannot create the required lower-ring/upper-ring antiphase exchange.

**Crazyflie Drones Theme (short) · s1b13t1** — _capability_

- wanted: Split the swarm into two coordinated five-drone rings: one ring held low and one ring held substantially higher, with both rings centered on the same vertical axis. Have the rings rotate in opposing directions and maintain a clear vertical gap, creating a suspended double-orbit effect that fills the stage from roughly 0.7 m to 1.4 m.
- lighting: Give every drone a deliberate two-tone appearance: red on its upper LED ring and blue on its lower LED ring. Make the lower and upper rings alternate in brightness against one another, so the two orbital layers appear to trade energy while the red-over-blue colour separation remains unmistakable.
- judge says missing: rotate applies to all drones in one direction, so the two rings cannot rotate in opposing directions. Brightness primitives do not directly alternate the upper and lower drone groups in antiphase by height, though timed lighting replacements could approximate it.

**Crazyflie Drones Theme (short) · s1b17t1** — _capability_

- wanted: Collapse the two rings toward the center in a fast inward spiral, with the upper and lower layers winding together rather than simply dropping straight down. Finish in a compact, elevated whirl ready to launch into the following section; the final approach should feel like the whole stage is being pulled into a single point.
- lighting: Shift to a hot magenta-and-orange vortex. Let the colour blend radiate from the center outward while brightness ripples travel inward toward the collapsing swarm. Finish with a short, vivid orange-white flare at the center of the whirl, leaving the drones visibly energized for the next phrase.
- judge says missing: The primitives do not provide an inward spiral-collapse trajectory: spiral_speed only supports a positive radius increase, while center does not preserve the winding inward approach. ripple_light travels outward rather than inward, and no lighting primitive provides a localized, center-only orange-white flare that radiates or converges as described.

**Dance of the Flying Machines · s1b1t1** — _palette only_

- wanted: Form the whole swarm into a broad circle with radius 140 cm at z=90 cm, arriving over 2.5 seconds. The effect is a calm but imposing orbital machine, with all ten drones evenly distributed around the ring.
- lighting: Use a cyan-to-blue gradient across the whole swarm by radius on both LED decks, with the entire ring breathing together in a 4-beat pulse. The center should feel cool, deep, and slowly powered up.
- judge says missing: The motion and synchronized 4-beat breathing are expressible. However, every drone on a circle has the same radius, so a radius-based gradient collapses to one uniform hue rather than producing a visible cyan-to-blue gradient across the ring.

**Debussy - Clair de Lune · s3b1t1** — _capability_

- wanted: Open the cone into a broad, undulating standing wave. The swarm should rise and dip in coordinated layers, producing a visible rolling surface rather than a simple bounce, with the final shape filling the available space while remaining within the flight volume. Use wave(3, 70) for drones 1-10.
- lighting: Turn the swarm nearly white at the crest of the wave, with indigo carried underneath for depth. Alternate the brightness of the two sides in antiphase so the wave appears to travel across the formation, while a soft pulse keeps the whole image alive. Use light_color(all, white, top), light_color(all, indigo, bot), alternate_blink(all, 2, side, top), and pulse(all, 4, bot).
- judge says missing: wave has no drone selector, so it cannot be restricted to drones 1-10 when the swarm contains additional drones. The standing-wave motion and lighting effects are otherwise supported.

**J. Strauss I - Radetzky March · s1b9t1** — _capability_

- wanted: Split the formation into two perfectly separated horizontal rings: one low and one high, with a dramatic vertical gap between them. The lower ring should hold steady while the upper ring turns in the opposite direction, creating a stately, interlocking mechanical movement.
- lighting: Make the upper ring bright white with a cool cyan edge, and the lower ring deep red with an amber edge. The two layers should pulse in alternating opposition—when the upper ring swells, the lower ring recedes—so the vertical separation reads clearly as two independent bands.
- judge says missing: There is no rotation primitive targeting only one ring, so the upper ring cannot turn opposite to a stationary lower ring. The lighting can approximate the two-tone edge using the two LED decks and can schedule alternating looks, but it cannot provide a true smoothly phase-opposed pulse for the upper and lower groups with independent phase control.

**J. Strauss I - Radetzky March · s3b5t1** — _unclear_

- wanted: Release the helix into a broad, undulating wave across the stage. The drones should crest and dip in staggered timing, producing a visible three-dimensional ripple that travels from one side to the other and then rebounds, with the formation remaining spread across both width and height.
- lighting: Use a cool cyan, teal, and blue palette with a moving rainbow undertone. A luminous ripple should travel outward through the formation in time with the beat, while the two sides blink in alternation so the wave appears to pass through the light as well as the bodies.
- judge says missing: The motion primitives can create helix height, a broad 2D standing wave, and vertical undulation, but they do not provide a true traveling wave with staggered crests that propagates from one side to the other and then rebounds. The lighting effects are expressible, but the motion's directional traveling-and-rebound behavior would be visibly missing.

**J. Strauss I - Radetzky March · s3b9t1** — _capability + palette_

- wanted: Drive the wave inward at once, then explode the ten drones into a huge, symmetric starburst filling the available width and height without touching the boundaries. After the burst, draw the points inward just enough to create a powerful final salute: a stable, evenly spaced star suspended high and clearly facing the audience.
- lighting: Finish with brilliant white across the star, edged in yellow and amber, with red accents at alternating points. Every beat receives a hard white flash followed by a warm golden decay; the final hold should remain fully illuminated, majestic, and unmistakably triumphant.
- judge says missing: There is no inward-travelling wave primitive: wave is a 2D standing-wave oscillation. The starburst and final stable star are achievable with form_star and vertical positioning. Lighting can provide white, yellow/amber gradients, alternating red accents, beat-locked flashes and decay, and a fully illuminated final hold, but strobe_decay keeps one colour during its decay; it cannot make each beat transition from a white flash into a warm golden decay. Exact white-edged yellow/amber treatment with simultaneous alternating red point accents is also only approximable.

**Mozart - Eine kleine Nachtmusik · s1b1t1** — _capability_

- wanted: All ten drones gather into a small, precise horizontal circle, like the first glint of a constellation appearing in the night. The arrival is deliberate and graceful rather than abrupt.
- lighting: A deep nocturnal palette: the drones carry a slow amber-to-indigo color gradient around the circle. Their brightness gently breathes together over four beats, creating a quiet, elegant entrance.
- judge says missing: A true amber-to-indigo gradient arranged around the circle is not directly expressible: gradient supports index, x, y, z, or radius, but not angular/neighbour ordering. On a flat circle, radius is identical for every drone, and index order is not guaranteed to match spatial order.

**Mozart - Eine kleine Nachtmusik · s1b5t1** — _capability_

- wanted: The circle begins a compact three-step spiral, turning inward and then opening slightly, as if the constellation is waking and rotating in space. The movement should feel fluid and poised.
- lighting: The warm amber and indigo evolve into a neighboring rainbow that travels around the formation. The whole ring pulses slowly, with no hard flashes.
- judge says missing: No primitive supports a single fluid three-step spiral whose radius first decreases and then increases slightly. spiral_speed only provides a positive radius increase, while successive form_circle actions would look like discrete formation changes rather than a continuous inward-then-outward spiral.

**Mozart - Eine kleine Nachtmusik · s2b9t1** — _capability_

- wanted: Collapse the helix into a broad upright cone, with its point lifted and its base spread toward the audience. The transformation should feel like the swarm is being drawn into a single dramatic spear of light.
- lighting: Use a white center fading through yellow and orange toward the wider lower portion. Each strong beat produces a crisp strobe-decay accent, emphasizing the cone's pointed silhouette.
- judge says missing: The motion and strobe-decay are expressible, but the lighting primitives cannot provide a clean multi-stop white-to-yellow-to-orange gradient specifically from the cone’s centre toward its wider lower portion. A two-colour gradient or coarse subset assignments would lose a noticeable part of that colour transition.

**Mozart - Eine kleine Nachtmusik · s3b5t1** — _capability_

- wanted: The star slowly contracts toward the center and lowers into a compact, calm constellation. The final motion should feel like the music settling back into the night, with all ten drones ending together and visibly balanced.
- lighting: Transition to a cool blue-white look: blue at the lower-facing light and white above it. The brightness fades in a slow four-beat breath, leaving a soft luminous afterglow rather than a sudden blackout.
- judge says missing: There is no one-shot slow fade that ends in a persistent soft afterglow. pulse repeats rather than leaving a terminal afterglow, while strobe_decay provides a decay but not the requested four-beat breath.

**On & On · s2b5t1** — _capability_

- wanted: Break the star into two clearly separated counterbalanced rings: drones 1-5 form a lower ring at 75 cm and drones 6-10 form an upper ring at 145 cm, with enough vertical separation to make the layers unmistakable. Have the two rings rotate as a single opposing structure: form_circle(1-5, 105, 75, 1.0); form_circle(6-10, 105, 145, 1.0); rotate(all, 180, z).
- lighting: Color the lower ring deep blue and the upper ring cyan, with each layer breathing in antiphase so the stacked structure alternates between a cool lower wave and a luminous upper wave. Use light_color(1-5, blue, both); light_color(6-10, cyan, both); alternate_blink(all, 2, by, both).
- judge says missing: The two vertically separated rings and their rotation are expressible, as are the blue and cyan layer colours. However, alternate_blink cannot split drones by upper versus lower layer: it only alternates by parity or stage side. Pulse has no phase-offset control, so continuous antiphase breathing between the two rings cannot be produced directly.

**On & On · s2b9t1** — _capability_

- wanted: Collapse the two-ring architecture into a tall, inverted cone that rises toward the audience like a beam opening upward, then twist it visibly through the remaining phrase. Use form_cone(70, 55, 0, 1.0) followed by twister(3, 12, 15).
- lighting: Finish with a hot white-and-magenta climax: sweep magenta through the vertical cone while white beat flashes punch through the color. Use gradient(all, magenta, white, z, both); sweep(all, 1, z, both); strobe_decay(all, 1, both).
- judge says missing: The cone and visible twist are expressible, and the lighting can provide a static magenta-to-white vertical gradient, a vertical brightness sweep, and beat-locked decay flashes. However, sweep changes brightness rather than moving magenta colour through the cone, and strobe_decay cannot turn the magenta regions white on each beat. The white flashes therefore cannot fully punch through the magenta as described.

**Vivaldi - Summer · s1b5t1** — _capability_

- wanted: The circle opens and begins a measured upward spiral, with the swarm rotating as one body. The rise should be graceful rather than explosive, creating the impression of hot air lifting the drones into the sky.
- lighting: A flowing orange-to-yellow gradient travels around the formation. The lights pulse slowly together, brightening at the top of each musical breath without flashing harshly.
- judge says missing: There is no primitive for a two-colour orange-to-yellow gradient that travels around the formation over time. rainbow provides motion of colour but cycles through the full spectrum, while gradient is static, so the closest version loses the intended restricted orange-to-yellow flowing gradient.

**Vivaldi - Summer · s2b1t1** — _capability_

- wanted: At the start of the second section, the swarm snaps into two clearly separated rotating rings, one high and one low. The rings counter-rotate with a strong vertical gap, creating a compact double-orbit that feels mechanically precise and suddenly more powerful.
- lighting: The lower ring carries red and orange; the upper ring carries yellow and white. Both rings pulse together, while their two light decks use contrasting colors so every drone reads as a hot two-tone spark.
- judge says missing: The rotate primitive applies to all drones and cannot rotate the two disjoint rings in opposite directions. Thus the stacked rings and lighting are achievable, but the mechanically precise counter-rotation is not.

**Vivaldi - Summer · s3b1t1** — _capability_

- wanted: The drones spread into a wide, shallow wave across the stage, rising and falling in a coordinated standing pattern. The wave should travel laterally through the swarm, like a shimmering wall of heat seen above scorched ground.
- lighting: A cyan-to-azure gradient sweeps across the wave, with white accents on the highest crests. The brightness travels from one side to the other in a smooth chase, then briefly swells across the entire wave on each major beat.
- judge says missing: wave is explicitly a 2D standing-wave surface, not a laterally travelling wave, so it cannot reproduce the shimmering wall travelling from side to side. White accents cannot dynamically track whichever crests are highest throughout the wave motion.

**We Wish You a Merry Christmas · s2b1t1** — _capability_

- wanted: After the first section, draw all ten drones inward and reform them as a clean, wide circle at mid-height. Rotate the circle through a full quarter turn during the interval, making the transition feel like a reset before the next verse.
- lighting: Switch to cool winter colour: cyan blending to azure by neighbour position. Add a smooth four-beat pulse, giving the circle the calm, glowing quality of moonlit snow.
- judge says missing: The gradient primitive cannot blend cyan to azure by neighbour position; its spatial/index modes are limited to index, x, y, z, or radius. The motion and smooth pulse are expressible, but the specified neighbour-position colour arrangement is not exact.

**We Wish You a Merry Christmas · s2b5t1** — _capability_

- wanted: Transform the circle into a broad two-dimensional wave across the stage. The wave should travel from audience-left to audience-right in three visible undulations, with the drones rising and falling in a coordinated rolling surface.
- lighting: Use white on the wave peaks and blue in the troughs, blended by height. Add a directional sweep along the x-axis on a two-beat period so the illumination travels with the wave.
- judge says missing: wave is a standing wave rather than a travelling left-to-right wave, and it does not specify or guarantee three visible undulations. The static height gradient also does not dynamically track moving peaks and troughs throughout the wave.

**We Wish You a Merry Christmas · s2b13t1** — _capability_

- wanted: Finish with the strongest image: expand all ten drones into a large upright star, hold it for recognition, then make a fast two-step spiral flourish that returns to the star on the final beat. The closing silhouette should be bold, balanced, and unmistakably festive.
- lighting: Set the final star to brilliant white with red on the upper LED ring and amber on the lower ring, creating a two-tone Christmas glow. Add a one-beat strobe-decay for the final musical hits, with the star fully radiant between accents rather than fading into darkness.
- judge says missing: The two LED rings can be red and amber, but the same final look cannot also retain brilliant white on those rings. A strobe_decay naturally fades after each flash; light_on can restore full brightness between accents only via separate replacement keys, not simultaneously with the decay.


## Full plans

### BLIND — Chopin - Nocturne Op. 9 No. 2

A delicate, intimate opening that gradually gathers emotional weight, then blooms into a luminous, expressive phrase before resolving into a quiet, suspended final image. The ten drones should behave like one breathing musical organism, with individual separation appearing only when the music calls for it.

- `s1b1t1` **Begin as a small, low cluster near the center, almost like a single glowing point. On the first beat, the ten drones slowly unfurl into a soft crescent, with the center drones slightly higher and the outer drones lower. The movement should feel weightless and organic, like a flower opening or a breath being drawn in, with no sudden acceleration.** — Use a very dim, warm ivory glow, concentrated at the center and fading toward the edges. The light should gently swell with the first phrase, creating the feeling of a private thought appearing in darkness rather than a theatrical entrance.
- `s1b5t1` **From the crescent, let the swarm drift into a broad, graceful arc spanning the space, as though the original thought is expanding into a memory. The drones should rise subtly while maintaining a slow, continuous curve, with a slight asymmetry that keeps the image human and hand-shaped rather than geometrically perfect.** — Deepen the ivory into a soft champagne-gold. Let the center brighten first, followed by a slow ripple of light toward both ends of the arc. The result should resemble candlelight traveling across a room—tender, fluid, and full of restrained longing.
- `s2b1t1` **After the transition into the second section, gather the arc inward and then release it into a larger, rising spiral. The drones should appear to orbit an invisible emotional center, climbing in a smooth helix while preserving clear spacing and an elegant sense of suspension. The spiral should feel like the music has found confidence without losing its intimacy.** — Shift gradually from champagne to moonlit blue-white, with a faint violet undertone. Light the drones in a traveling wave around the spiral, alternating between a soft shimmer and a calm sustained glow. The effect should be nocturnal and expansive, as if moonlight were being stirred through the air.
- `s2b5t1` **At this more developed point, open the spiral into a wide, sweeping ribbon that bends diagonally through the volume. Let the ten drones trace a single expressive gesture, rising at one end and dipping at the other, then reversing the emphasis in a slow, lyrical sway. It should feel like a dancer extending an arm through a long melodic line.** — Allow the blue-white light to bloom into silver with brief pearlescent flashes. The highlights should travel along the ribbon in the direction of its sweep, never strobing or becoming harsh. The desired impression is a sudden glint of brilliance inside an otherwise tender nocturne—emotion breaking briefly into radiance.
- `s3b1t1` **For the final section, let the ribbon dissolve into a calm, symmetrical constellation: a shallow upward arc with the drones spaced like stars around an invisible horizon. Slowly reduce the height and scale of the formation until it feels suspended and almost motionless, leaving a small opening at the center as if the music has exhaled and departed.** — Fade from silver to a very soft pearl-white, then gradually cool it toward faint blue-gray. Keep the lights visible but delicate, with a final gentle pulse that passes through the constellation and disappears without a blackout. The ending should feel like moonlight lingering after the last thought, peaceful and unresolved rather than conclusively extinguished.

### ANCHORED — Chopin - Nocturne Op. 9 No. 2

A restrained nocturne that gradually transforms from a cool, breathing circle into a warm star, then a rising sculptural cone, two separated harmonic rings, and finally a luminous rotating spectrum. All positions remain within the 4 m by 4 m horizontal volume and the 0.25–1.70 m height range.

- `s1b1t1` **The full swarm opens as one calm, perfectly spaced circle, floating at about 0.90 m with an 0.85 m radius. The formation should feel suspended and almost weightless, like the first sustained phrase of the nocturne. Use form_circle on drone_ids '1-10' with radius 85 cm, z 90 cm, and a 1.5 s arrival.** — Give every drone a deep indigo-to-cyan gradient radiating from the centre across both LED decks. The whole circle should breathe very slowly together over 4 beats, creating a cool, glassy pulse rather than a hard flash.
- `s1b5t1` **The circle contracts inward and resolves into a delicate five-point star, with the points clearly legible but still soft and elegant. Keep the star centred near 1.00 m, with generous spacing so it reads as a precise musical ornament. Use form_star for the full swarm with height 100 cm, spacing 55 cm, delta_radius 80 cm, and a 1.2 s arrival.** — Replace the cool opening look with warm amber on both decks, accented by a subtle white emphasis at the outer tips through a radial amber-to-white gradient. A neighbour chase should travel around the star once per beat, with three drones illuminated at a time, like a graceful arpeggio rather than a strobe.
- `s2b1t1` **The star stretches vertically into a tall, elegant cone, lifting the upper points while keeping the base broad enough to remain readable from the audience. The movement should feel like the melody opening upward into a more impassioned phrase. Use a non-inverted cone for the full swarm with a 65 cm height change, 55 cm spacing, and a 1.5 s arrival.** — Use a vertical gradient from indigo at the lower drones through magenta toward the top, on both decks. A slow two-beat sweep along the z axis should travel upward through the cone, making the structure appear to sing from its base to its apex.
- `s2b5t1` **Break the cone into two distinct, vertically separated circles: drones '1-5' form a lower ring at 75 cm, while drones '6-10' form an upper ring at 145 cm. Keep both rings centred and aligned so they read as one floating, two-level wreath, with a clearly visible 70 cm vertical separation. Both rings should arrive over roughly 1.5 s, creating a broad harmonic expansion.** — Treat the two levels as two voices. On the lower ring, use amber on the top deck and red on the bottom deck; on the upper ring, use azure on the top deck and indigo on the bottom deck. Across all drones, alternate the two stage sides in a two-beat antiphase blink, so the warm and cool harmonies trade places without ever becoming frantic.
- `s3b1t1` **For the closing phrase, the separated rings unwind into a slow three-step spiral with a strong vertical sweep, rising and turning as one coherent figure. The motion should feel like the entire nocturne dissolving upward into the air rather than snapping to a final pose. Use a 3-step spiral for the full swarm with a 110 cm height span.** — Flood both decks with a neighbour-ordered rainbow that completes a full colour cycle over 4 beats, allowing the spectrum to travel around the spiral. Layer a one-beat strobe-decay accent over it so each principal beat briefly catches the whole rising shape, then lets the colour fade into the next turn.

### BLIND — Crazyflie Drones Theme

A playful, futuristic flight through the intro: the ten drones begin as a quiet spark, repeatedly transform between precise geometric symbols and flowing swarms, then build toward a bright, celebratory finale in the outro. All formations stay within the 4 m × 4 m horizontal space and 0.25–1.70 m altitude range.

- `s1b1t1` **The ten drones appear as a compact glowing cluster near the center, then gently separate into a small, balanced constellation with two staggered rings: five slightly higher and five slightly lower. The movement should feel like a system waking up and discovering its own shape.** — Begin in near darkness with soft cool-white pinpricks. As the constellation opens, each drone gains a pale electric-blue glow, with a subtle pulse spreading through the group like a startup signal.
- `s1b5t1` **The constellation contracts into a crisp arrowhead pointing forward, then sweeps sideways as one coordinated flock, leaving the impression of a single agile craft changing course. Finish in a broad, level V formation.** — Use bright cyan-white lights at the leading point and progressively deeper blue toward the trailing drones. The arrowhead should flash once as it locks into shape, then settle into a smooth traveling shimmer.
- `s1b9t1` **The V folds inward and spirals into a rotating helix, with the drones passing through different heights before expanding into a large circular orbit. The effect should suggest a miniature galaxy or propeller turning in the air.** — Shift from blue to violet as the helix forms. Give the rotating circle a sequential chase around its perimeter, with one warm-white spark traveling through the ring and making the whole formation feel alive.
- `s1b13t1` **The circle breaks into two interleaving arcs that sweep past one another, cross, and reform as a symmetrical butterfly or pair of wings. The wings should gently beat once through a coordinated rise and fall before holding open.** — Color the two arcs in contrasting electric blue and magenta. During the crossing, brighten the center overlap to brilliant white, then let the wing tips pulse outward in alternating colors.
- `s1b17t1` **The butterfly collapses into a narrow vertical column, then bursts outward into a large five-pointed star with two drones emphasizing each major arm. The star should rotate slowly while maintaining a strong, unmistakable silhouette.** — Use a deep indigo background color on the drones, with vivid gold-white points at the ends of the star arms. A clean white flash marks the burst from the column, followed by a slow golden rotation around the shape.
- `s1b21t1` **The star dissolves into a fast, playful swarm that darts in small arcs and corkscrews, then snaps into a stylized lightning bolt cutting diagonally across the volume. The movement should feel mischievous, energetic, and highly controlled rather than chaotic.** — Animate hot pink and electric cyan streak-like pulses through the swarm. When the lightning bolt locks in, illuminate it in brilliant white-blue with a sharp flicker along its zigzag path.
- `s1b25t1` **The lightning bolt unfolds into a wide circular wave, with the ten drones spaced evenly around its rim. The ring breathes inward and outward twice, then tilts into a rising spiral that suggests the swarm is gathering momentum for a launch.** — Transition through saturated blue, purple, and fuchsia around the ring. Each expansion should brighten the drones, and the final rising spiral should carry a warm amber glow toward its upper end.
- `s1b29t1` **The spiral compresses into a dense central point, pauses for dramatic tension, and then explodes into a broad radiant sunburst. The drones should hold the sunburst with alternating long and short rays, creating a powerful final image for the main section.** — Fade the colors toward darkness during the compression, leaving only a faint violet core. On the burst, ignite every drone in brilliant gold-white, with a saturated orange pulse expanding from the center and then settling into a steady celebratory glow.
- `s2b1t1` **For the outro, the sunburst folds into a compact smiling crescent-like arc, then the drones drift outward and upward in a graceful farewell, ending as ten separated points across the upper half of the space. The final motion should feel weightless, friendly, and conclusive.** — Change the gold-white sunburst into warm festive amber with soft white highlights. The crescent receives one gentle synchronized pulse, then the drones gradually dim one by one until only a few warm sparks remain before fading out.

### ANCHORED — Crazyflie Drones Theme

A playful, high-energy 130 BPM show that grows from a glowing circular formation into expanding geometric shapes, vertical motion, layered rings, and a final unified white flash. The swarm stays within the 4 m × 4 m × 1.45 m flight volume, with strong beat-locked lighting changes at each musical landmark.

- `s1b1t1` **The ten drones snap into a broad, level circle at about 0.90 m altitude, creating a clean opening emblem that feels confident and immediately visible. The circle arrives sharply in roughly one second.** — Use a full-spectrum rainbow distributed around the circle, with the colours travelling from drone to drone over 4 beats. Add a synchronized 2-beat pulse so the opening breathes with the music.
- `s1b5t1` **The circle rises into a lively three-step helix, with the drones winding upward and creating a clearly three-dimensional corkscrew rather than remaining flat. The motion should feel like the theme is taking off.** — Keep the moving rainbow, now cycling over 2 beats around neighbouring drones. Add a 1-beat strobe-decay accent so every musical beat produces a bright travelling flare.
- `s1b9t1` **The helix resolves into a dramatic upright cone, with a broad lower edge and a sharply elevated point. Hold the cone long enough for the audience to read the silhouette, making it feel like an animated rocket or party hat.** — Colour the cone from amber at its lower, wider region to pink at its upper point. Run a slow vertical sweep over 4 beats, making the light climb the cone from base to tip.
- `s1b13t1` **The cone bursts outward into a large five-pointed star, with the ten drones tracing its points and inner structure at about 1.10 m altitude. The arrival should be crisp and theatrical, like a visual cymbal hit.** — Use alternating cyan and magenta across the star. Apply a short 1-beat strobe-decay to make the star appear to punch into existence, then let the colour remain vivid between accents.
- `s1b17t1` **The star melts into a broad undulating wave across the stage. The drones should rise and fall in a coordinated standing-wave pattern, producing a visibly rippling surface that travels across the formation.** — Use a cyan-to-blue gradient across the wave from one side to the other. Add a 2-beat pulse and a directional sweep along the y-axis so the illumination appears to travel with the physical wave.
- `s1b21t1` **The wave contracts and spins into a three-step outward spiral, with the formation opening as it turns. The effect should feel like the whole swarm is drawing a bright vortex in the air while retaining enough radius to stay legible.** — Run a neighbour-based rainbow around the spiral over 1 beat, combined with an outward ripple of light from the centre. The brightness should repeatedly surge from the core toward the outer drones.
- `s1b25t1` **Split the swarm into two distinct stacked rings: drones 1-5 form a lower amber ring at about 0.75 m, while drones 6-10 form an upper cyan ring at about 1.45 m. The two rings should rotate visually as a layered double halo with a clear vertical gap.** — Give drones 1-5 a warm amber colour and drones 6-10 a cool cyan colour. Make the two groups blink in opposite phases over 2 beats, creating a crisp alternating lower-ring/upper-ring exchange.
- `s1b29t1` **The stacked rings collapse into a fast three-step twister, winding the swarm around its vertical axis with a pronounced corkscrew profile. The motion should be exuberant and slightly chaotic, like the final instrumental lift before the outro.** — Use a neighbour rainbow cycling every beat, with a z-axis sweep so the colours climb and descend through the twisting shape. Add a full-beat strobe-decay for sharp musical flashes.
- `s2b1t1` **For the outro, the twister resolves instantly into one compact, symmetrical circle centred at roughly 0.95 m altitude. The drones settle together as a clean final emblem and hold still for the ending hit.** — Turn every drone brilliant white on both LED rings and use a single strong 1-beat strobe-decay flash, leaving the final image as a unified white burst rather than a fading multicolour effect.

### BLIND — Crazyflie Drones Theme (short)

A playful, high-energy five-part intro in which 10 drones evolve from a quiet spark into a bright, tightly synchronized finish. The choreography should feel musical and expressive: clean silhouettes on the downbeats, elastic expansion and contraction, and vivid color changes that make the swarm read as one animated character.

- `s1b1t1` **Begin as a compact, low glowing cluster near the center, with the drones staggered vertically like a small floating spark. On the first beat, the cluster gives a subtle upward lift and opens just enough to become visibly alive, while retaining a clear, rounded silhouette.** — Use a dim cool-blue glow, with one or two drones briefly flashing white at the center like a spark catching. Keep the overall look restrained and anticipatory, as though the show is just waking up.
- `s1b5t1` **Spread the 10 drones into a crisp five-pointed star, with pairs of drones defining each arm and a slight height difference giving the shape depth. Make the star expand outward on the beat, then gently contract and rotate as a single object, creating a clean, playful reveal.** — Shift from blue to bright cyan and electric violet. Chase a white highlight around the star’s points in time with the rotation, making the shape feel energized without losing its recognizable outline.
- `s1b9t1` **Break the star into a broad, flowing wave across the space: the drones rise and fall in sequence from one side to the other, then sweep through a loose spiral. The motion should feel elastic and exuberant, like a ribbon being flicked through the air, while keeping the full swarm coordinated.** — Use a vivid rainbow gradient that travels along the wave, with saturated magenta, yellow, cyan, and blue passing from drone to drone. Add a brief bright-white crest as the wave reaches its highest point.
- `s1b13t1` **Gather rapidly into a compact central vortex, circling upward in a tight helix. At the beat, burst outward into a large, symmetrical flower-like formation with five rounded petals, then pulse the petals open and closed as if the shape is breathing.** — Start with deep purple and hot pink in the spinning vortex, then explode into brilliant gold and white at the flower reveal. Let each petal carry a different saturated color, with a warm white pulse traveling around the perimeter.
- `s1b17t1` **Finish the intro with a confident full-volume formation: expand into a wide, balanced arc with a raised center, like a smiling horizon above the audience. Give the arc one strong upward sweep and a synchronized settling motion, ending in a perfectly still, evenly spaced silhouette.** — Flood the formation in brilliant white with a warm golden edge, then send one final blue-to-gold shimmer from left to right. End on a sharp, unified white flash followed by a steady warm-white hold, making the swarm feel triumphant and ready for the next section.

### ANCHORED — Crazyflie Drones Theme (short)

A five-part opening arc: the ten drones begin as a restrained rising cone, bloom into a rotating helix, snap into a sharp starburst, separate into two elevated counter-rotating rings, then draw back together in a fast inward spiral for the next musical section. The lighting evolves from a low amber pulse to a travelling spectrum, a brilliant star-like flash, a dramatic red-and-blue split, and finally a vivid magenta-orange vortex.

- `s1b1t1` **Begin in a compact, low formation around the center, then rise and open into a clean upright cone. The formation should feel like a spark gathering strength, with the point clearly defined and the body of the cone filling the central airspace without spreading beyond the middle of the volume.** — Use a restrained amber glow across the swarm with a slow, unified breath. Add a brief warm-white beat accent at the musical hits so the cone appears to ignite from within rather than flash randomly.
- `s1b5t1` **Bloom the cone into a broad, graceful helix. The drones should climb and orbit around the central axis in a visibly twisting column, with enough vertical separation to make the motion read as a three-dimensional corkscrew. The movement should feel like the opening suddenly gaining lift and momentum.** — Turn the swarm into a travelling spectrum that runs around the helix, shifting continuously through cyan, blue, indigo, magenta, red, orange, and back toward yellow. Let the entire colour field breathe slowly while a brighter wave travels through the formation in the same direction as the twist.
- `s1b9t1` **Snap outward from the helix into a crisp, symmetrical ten-point starburst centered on the audience. Hold the star long enough to read as a strong graphic symbol, with the points evenly distributed and the drones making a decisive outward hit rather than a soft drift.** — Make the star predominantly brilliant white with a cool azure edge. Use a fast travelling brightness chase around the star so its points appear to flare in sequence, then let the whole shape pulse once as a single hard musical accent.
- `s1b13t1` **Split the swarm into two coordinated five-drone rings: one ring held low and one ring held substantially higher, with both rings centered on the same vertical axis. Have the rings rotate in opposing directions and maintain a clear vertical gap, creating a suspended double-orbit effect that fills the stage from roughly 0.7 m to 1.4 m.** — Give every drone a deliberate two-tone appearance: red on its upper LED ring and blue on its lower LED ring. Make the lower and upper rings alternate in brightness against one another, so the two orbital layers appear to trade energy while the red-over-blue colour separation remains unmistakable.
- `s1b17t1` **Collapse the two rings toward the center in a fast inward spiral, with the upper and lower layers winding together rather than simply dropping straight down. Finish in a compact, elevated whirl ready to launch into the following section; the final approach should feel like the whole stage is being pulled into a single point.** — Shift to a hot magenta-and-orange vortex. Let the colour blend radiate from the center outward while brightness ripples travel inward toward the collapsing swarm. Finish with a short, vivid orange-white flare at the center of the whirl, leaving the drones visibly energized for the next phrase.

### BLIND — Dance of the Flying Machines

A ten-drone piece that begins as a restrained awakening, grows into a sweeping mechanical dance, breaks into a bright airborne chorus, and ends with the machines dissolving into a calm constellation.

- `s1b1t1` **The ten drones appear as a tight, low cluster near the center, then slowly separate into two interlocking arcs that tilt in opposite directions, like the first graceful movements of a machine learning to fly. Keep the formation compact and poised, with no sudden jumps.** — Begin with a very dim, cool blue-white glow. Let the drones brighten one after another in a traveling pulse through the cluster, creating the feeling of systems powering up and becoming conscious.
- `s1b5t1` **Expand the two arcs outward across the space, then sweep them around each other in a broad horizontal spiral. The inner drones lead while the outer drones follow, making the swarm feel like a single rotating flying apparatus rather than ten separate objects.** — Shift from blue-white to metallic silver and pale cyan. Use a slow wave of brightness that follows the spiral, with the leading edge sparkling brightly and the trailing edge fading like an illuminated contrail.
- `s1b9t1` **Break the spiral into a dramatic rising fan: the drones lift from a low, compressed arrangement into staggered heights, opening toward the audience while maintaining a strong symmetrical shape. At the peak, fold the fan inward and sweep it forward as though a flock of flying machines is launching together.** — Introduce vivid amber and warm white highlights over the cool cyan base. The fan should flare brighter as it opens, then produce a single strong golden flash at the forward sweep, followed by brief shimmering afterglows that suggest propellers and sparks in motion.
- `s2b1t1` **Re-form the swarm into a wide, undulating wave spanning the performance volume. The wave travels from one side to the other, with each drone rising and dipping in sequence, then all ten crest together in a powerful synchronized lift before settling into a hovering line.** — Move into saturated electric blue with sharp white accents. Chase a bright pulse along the wave, then illuminate all drones at the shared crest so the entire formation appears to breathe and surge as one machine.
- `s2b5t1` **From the hovering line, let the drones drift apart into a high, airy constellation with varied heights, leaving generous space between them. They orbit gently around an invisible center for a final moment, then slow until the formation feels suspended and weightless, like flying machines becoming stars.** — Fade the electric blue into soft violet, then into quiet white points. Keep each drone subtly twinkling at a different time, with one final coordinated brightening across the constellation before the lights recede to a delicate, barely glowing afterimage.

### ANCHORED — Dance of the Flying Machines

A progressive mechanical ballet: the swarm begins as a breathing blue orbital system, expands into a rotating star, splits into two contrasting altitude rings, then recombines as a vertical helix before collapsing into a bright ascending cone. The lighting evolves from cool atmospheric motion to saturated machine-like pulses, two-tone vertical color, and a final white impact.

- `s1b1t1` **Form the whole swarm into a broad circle with radius 140 cm at z=90 cm, arriving over 2.5 seconds. The effect is a calm but imposing orbital machine, with all ten drones evenly distributed around the ring.** — Use a cyan-to-blue gradient across the whole swarm by radius on both LED decks, with the entire ring breathing together in a 4-beat pulse. The center should feel cool, deep, and slowly powered up.
- `s1b5t1` **Transform the whole swarm into a tall star using height 110 cm, spacing 55 cm, and delta_radius 80 cm, arriving over 1.5 seconds. The star should feel like the flying machines have unfolded mechanical wings and are reaching sharply toward the audience.** — Run a neighbor-ordered rainbow across both decks of the whole swarm with a 2-beat cycle, combined with a 2-beat pulse. The color should travel around the star while the entire form inhales and exhales.
- `s1b9t1` **Split the swarm into two disjoint five-drone rings: form_circle(1-5, radius 80 cm, z=70 cm, time 1.0 s) and form_circle(6-10, radius 80 cm, z=140 cm, time 1.0 s). The two rings should hover as a deliberately separated double rotor, turning the swarm into a layered flying-machine assembly.** — Make drones 1-5 amber on both decks and drones 6-10 magenta on both decks. Drive a 1-beat strobe_decay across the whole swarm so the lower and upper rotors flash together like synchronized engines.
- `s2b1t1` **Recombine the whole swarm into a rising helix with 3 steps, delta_height 70 cm, and height 90 cm. The helix should visibly wind upward through the volume, creating a strong vertical corkscrew rather than a flat formation.** — Give every drone a two-tone machine glow: red on the top deck and blue on the bottom deck. Add a side-based alternate_blink with a 1-beat period on both decks, so left and right halves trade flashes while the helix climbs.
- `s2b5t1` **Collapse the whole swarm into an upright cone with delta_height 70 cm, spacing 55 cm, non-inverted orientation, arriving over 1.2 seconds. The final image should be a forceful ascending spear or launch plume, with the drones stacked clearly through the height of the volume.** — Turn both decks of the whole swarm white and drive a 1-beat strobe_decay. The cone should hit as a brilliant mechanical flare, with each beat producing a crisp launch-like flash.

### BLIND — Debussy - Clair de Lune

A quiet, moonlit opening grows into a more expansive, breathing landscape, then gathers into a luminous and emotionally suspended final image. The ten drones should feel like one living constellation rather than separate objects.

- `s1b1t1` **Begin as a small, low, loosely scattered constellation near the center. Let the ten drones drift almost imperceptibly outward and upward, as if moonlight is slowly spreading across still water. Keep the movement soft, continuous, and slightly uneven so the formation feels alive rather than geometric.** — Use very dim, cool blue-white light with a silvery haze. The drones should appear like distant points of reflected moonlight, with no harsh edges and no strong focal point. The effect should be intimate, hushed, and just beginning to awaken.
- `s1b5t1` **Transform the constellation into a slow, open crescent, with the drones tracing a broad curve across the middle of the space. The curve should gently roll and breathe: one side rises while the other settles, creating the sensation of a wave or a drifting cloud bank. Preserve generous spacing and an unhurried, floating quality.** — Increase the light slightly toward pearly white while retaining a cool blue tint. Let the crescent have a subtle inner glow, brighter near its center and softer toward its ends. The effect should suggest moonlight becoming visible in the landscape without losing the music’s delicacy.
- `s2b1t1` **Open the swarm into a wide, ascending arc that reaches across the available space, with the center lifted above the outer drones. The ten drones should move as if drawn upward by a gentle current, then hover in a slow, collective inhale. The formation should feel larger and more spacious than before, while remaining fluid and weightless.** — Shift from pale blue to luminous silver-white, with a faint lavender undertone. Brighten the elevated center so the arc reads as a glowing horizon or rising moonlit swell. The effect should be a graceful expansion of emotional scale, not a sudden burst.
- `s2b5t1` **Release the arc into a slow spiral-like drift, with the drones circling around an invisible center while gradually changing height. The motion should feel like falling leaves caught in a mild updraft—layered, graceful, and never mechanical. Bring the swarm briefly closer together, then let it expand again, as though the music is breathing through the entire volume.** — Use richer blue-violet shadows around a warm silver core. Allow the lights to softly swell and recede in intensity as the formation contracts and opens. The effect should be dreamlike and immersive: a luminous current that surrounds the audience rather than merely drawing a symbol in the sky.
- `s3b1t1` **Gather the ten drones into a serene, elevated oval or incomplete ring, centered high enough to feel suspended but still comfortably within the space. Let the ring slowly tilt and dissolve at one point, creating a gentle opening toward the audience. End the moment in near-stillness, with only a barely perceptible drift, as if the entire sky has paused.** — Return to a refined moon-white with a soft lavender-blue halo. Make the upper portion of the oval subtly brighter while leaving the opening darker and inviting. The effect should be a final suspended image of clarity and longing—radiant, fragile, and calm rather than triumphant.

### ANCHORED — Debussy - Clair de Lune

A restrained moonlit opening that gradually expands from a breathing spiral into a radiant star, separates into two vertical rings, twists upward into a sculptural cone, and ends in a luminous wave. The lighting moves from indigo stillness through silver-blue accents into layered cyan, amber, and white highlights.

- `s1b1t1` **The entire swarm gathers into a slow, elegant three-step spiral, rising through about 55 cm as if a moonlit current is lifting the formation. Use spiral(3, 55) for drones 1-10.** — Give all drones a deep indigo-blue moonlit colour on both decks, with a slow four-beat pulse so the whole spiral gently breathes rather than flashes. Use light_color(all, indigo, both) and pulse(all, 4, both).
- `s1b5t1` **The spiral resolves into a clear, expansive five-point star. The points should open decisively but gracefully, with a tall central rise and broad spacing that makes the star read as a single emblem in the sky. Use form_star(110, 60, 80, 1.5) for drones 1-10.** — Shift from indigo toward a pale lunar gradient, with azure at the centre and white at the outer reaches. Add a beat-locked strobe-decay accent so the star blooms on each musical pulse and fades like reflected light. Use gradient(all, azure, white, radius, both) and strobe_decay(all, 1, both).
- `s2b1t1` **Break the star into two distinct, vertically separated rings: drones 1-5 form a lower ring at 80 cm, while drones 6-10 form an upper ring at 145 cm. They should arrive together and hold as two calm orbiting halos with a strong vertical gap. Use form_circle(1-5, 45, 80, 1.5) and form_circle(6-10, 45, 145, 1.5).** — Make every drone two-tone: amber on its upper LED ring and blue on its lower LED ring, creating warm moon-gold above cool night-blue. Let the upper layer breathe on a two-beat cycle while the lower layer breathes more slowly over four beats. Use light_color(all, amber, top), light_color(all, blue, bot), pulse(all, 2, top), and pulse(all, 4, bot).
- `s2b5t1` **The two halos collapse into a rising, rotating cone. The formation should feel like the music gathering upward into a luminous funnel, with enough vertical spread for the upper drones to peel visibly away from the lower ones. Use form_cone(65, 60, 0, 1.5) for drones 1-10, followed by twister(3, 12, 18) for a measured three-step corkscrew motion.** — Run a cool teal-to-indigo colour gradient across the cone from one side to the other. A neighbour chase should travel around the sculptural form in small groups, like moonlight winding up the funnel. Use gradient(all, teal, indigo, x, both) and chase(all, 2, 3, 1, neighbour, both).
- `s3b1t1` **Open the cone into a broad, undulating standing wave. The swarm should rise and dip in coordinated layers, producing a visible rolling surface rather than a simple bounce, with the final shape filling the available space while remaining within the flight volume. Use wave(3, 70) for drones 1-10.** — Turn the swarm nearly white at the crest of the wave, with indigo carried underneath for depth. Alternate the brightness of the two sides in antiphase so the wave appears to travel across the formation, while a soft pulse keeps the whole image alive. Use light_color(all, white, top), light_color(all, indigo, bot), alternate_blink(all, 2, side, top), and pulse(all, 4, bot).

### BLIND — Fearless2

A bold, escalating journey from restrained tension into expansive, fearless energy. The ten-d drone swarm begins contained and watchful, breaks open at the first chorus, surges forward with confidence, and ends in a sharp, triumphant burst.

- `s1b1t1` **Hold the ten drones in a compact, slightly uneven cluster at mid-height, with a subtle inward-and-outward breathing motion. The formation should feel alert and contained, as if gathering courage before moving.** — Use a dim, cool blue-white glow with soft pulses synchronized to the breath of the formation. Keep the edges subdued and the center slightly brighter, creating a feeling of tension and anticipation.
- `s1b5t1` **Unfold the cluster into a wide, low horizontal line, spreading decisively toward both sides while maintaining a clean, balanced shape. The movement should feel like the group is opening its stance and claiming space.** — Transition from cool blue into brighter white with a restrained gold tint. Let the brightness rise as the line expands, producing a clear sense of resolve without yet reaching full intensity.
- `s2b1t1` **At the chorus entrance, launch the line upward and outward into a broad, elevated V formation. The center should lead slightly, with the outer drones sweeping back and around so the whole swarm appears to break through an invisible barrier.** — Flash into brilliant white and saturated electric blue at the moment of expansion. Chase the brightness from the center toward the wings, then hold the full formation glowing strongly and evenly.
- `s2b5t1` **Drive the swarm into a powerful forward surge, compressing briefly into a spearhead before opening into a large, symmetrical starburst. The motion should feel fast, fearless, and celebratory, with the drones occupying the full available width and rising through the vertical space.** — Use intense white cores with vivid gold and cyan accents that streak along the direction of travel. Finish the starburst with a strong synchronized flare, making the entire swarm feel like an explosion of confidence and momentum.
- `s3b1t1` **Snap the drones into a crisp, unified crown-like arc at high-mid height, with a slight upward lift on the final beat. The ending should be immediate and decisive rather than fading, as if the swarm has reached a victorious summit.** — Give the arc a brilliant warm-white glow edged with rich gold. Hit one clean, blinding final accent across all ten drones, then hold the victorious image for the brief closing moment.

### ANCHORED — Fearless2

A dark-to-radiant arc: the swarm begins as a rotating, breathing color field, tightens into a rising cone, splits into powerful two-level chorus rings, then twists vertically before exploding into a final starburst.

- `s1b1t1` **The full swarm slowly unfurls into a broad, rotating spiral, creating the feeling of fear opening into motion. Use spiral(all, 3, 80) with a sustained, graceful reveal.** — Give the whole swarm a continuous spectral flow with rainbow(all, 4, neighbour, both), softened by pulse(all, 4, both). The colors should circulate while the entire shape breathes with the verse.
- `s1b5t1` **Pull the swarm upward into a sharply readable cone, like a rising point of resolve. Use form_cone(all, 60, 55, 0, 1.0), with the cone arriving decisively on the beat.** — Shift from the diffuse spectrum into a warm-to-cool vertical build: gradient(all, amber, cyan, z, both), with pulse(all, 2, both) for a tense, accelerating glow.
- `s2b1t1` **Break the cone open into two clearly separated chorus rings: a lower ring carrying drones 1 through 5 and a high ring carrying drones 6 through 10. Use form_circle(1-5, 65, 75, 1.0) and form_circle(6-10, 65, 145, 1.0). The two levels should feel like a large, suspended architectural emblem.** — Make every drone two-tone: light_color(all, red, top) and light_color(all, blue, bot). Add strobe_decay(all, 1, both) so the rings hit with a bright, musical flash on every beat.
- `s2b5t1` **Drive the two-level emblem into a vigorous vertical twist, with the swarm winding through the full available height and visibly changing which drones read high and low. Use helix(all, 3, 70, 80), followed by twister(all, 3, 12, 20) as the motion develops.** — Let the color climb through the twisting height with rainbow(all, 2, z, both). Add alternate_blink(all, 1, side, both) so opposing sides of the twisting swarm answer each other in sharp antiphase.
- `s3b1t1` **End with an instantaneous, expansive starburst: all ten drones shoot from the twist into a large, unmistakable star that fills the permitted volume. Use form_star(150, 70, 100, 0.8).** — Finish in brilliant white with a hot chromatic edge: light_color(all, white, top), gradient(all, magenta, azure, radius, bot), and strobe_decay(all, 1, both). The final hit should read as a blinding, celebratory release rather than a fade.

### BLIND — J. Strauss I - Radetzky March

A bright, theatrical march that grows from disciplined military precision into expansive, celebratory motion. The ten-drone swarm repeatedly snaps into clear formations on the downbeats, with brass-gold light, crisp white accents, and increasingly energetic bursts toward the final phrase.

- `s1b1t1` **Begin as a compact, low diamond-like cluster, then snap outward into a proud symmetrical fan, as if the march is announcing itself and taking the stage.** — Start dim and warm, then ignite in brilliant brass gold at the outward snap, with a clean white flash at the center of the formation.
- `s1b5t1` **Re-form into two precise marching ranks and advance them in a short, strongly synchronized surge, preserving rigid military order while the whole formation rises slightly.** — Use bold gold lighting across both ranks, with alternating white highlights that travel from the front rank to the rear like marching accents.
- `s1b9t1` **Collapse briefly toward the center, then burst into a broad ten-point star before rotating the star a quarter turn with crisp, unified timing.** — Flash hot white at the collapse, explode into saturated golden light at the starburst, and leave each outer point glowing warmly as it turns.
- `s2b1t1` **Gather into a compact vertical column and unwind into a rising spiral, giving the impression that the march has gained lift, confidence, and momentum.** — Shift from deep amber at the base to vivid gold and white toward the top, with a traveling sparkle that climbs through the spiral.
- `s2b5t1` **Spread into a wide, shallow wave that rolls from one side to the other, then level out into a broad ceremonial line facing the audience.** — Let a bright gold crest travel across the wave, followed by a clean white shimmer along the completed line.
- `s3b1t1` **Break the line into a strong forward-pointing arrowhead and drive it upward and slightly forward, making the swarm feel like a triumphant procession surging ahead.** — Light the arrowhead in intense brass gold, with the tip burning white and the rest of the formation pulsing behind it in stepped accents.
- `s3b5t1` **Strike outward from the arrowhead into a huge celebratory burst, then pull rapidly back into a tight, perfectly centered cluster, creating a dramatic cannon-blast-and-recoil effect.** — Use a blinding white-gold explosion on the outward strike, briefly scatter red-gold sparks at the edges, then return to a concentrated hot-gold glow.
- `s3b9t1` **Launch the final formation into a grand rotating star and expand it to its widest, highest shape, holding the silhouette proudly before the drones make one final synchronized upward lift.** — Build to full golden brilliance with sparkling white tips and brief red accents, then finish in a powerful all-white flash that settles into a steady celebratory gold.

### ANCHORED — J. Strauss I - Radetzky March

A ceremonial opening grows into a bold, highly dimensional marching spectacle: the ten drones begin as a disciplined amber pulse, then divide into rings, stars, cones, and a rising helix before ending in a brilliant, audience-facing burst. Every change is designed to land on the first beat of a four-bar phrase and remain legible across the full flying volume.

- `s1b1t1` **The ten drones begin as a compact, low formation near the center, then rise and spread into a clean, imposing circle at medium height. The movement should feel like a regiment snapping into parade order on the first downbeat—precise, confident, and immediately visible.** — Use a warm amber-and-orange military glow. The whole formation breathes together once per bar, with a crisp white flash on each first beat before returning to amber. Keep the light unified and ceremonial rather than colorful or playful.
- `s1b5t1` **The circle sweeps around its own center as though the entire formation is turning to face the audience, while the drones subtly change altitude in a coordinated rolling rise and fall. The effect is a disciplined rotating wheel with no loss of spacing or control.** — Shift into a saturated red and orange look. A bright running chase travels around the circle once per beat, creating the impression of drum cadence moving through the ranks. The chase should remain strong enough that individual drones visibly hand the light from one to the next.
- `s1b9t1` **Split the formation into two perfectly separated horizontal rings: one low and one high, with a dramatic vertical gap between them. The lower ring should hold steady while the upper ring turns in the opposite direction, creating a stately, interlocking mechanical movement.** — Make the upper ring bright white with a cool cyan edge, and the lower ring deep red with an amber edge. The two layers should pulse in alternating opposition—when the upper ring swells, the lower ring recedes—so the vertical separation reads clearly as two independent bands.
- `s2b1t1` **Collapse the two rings inward and reform the swarm as a large, symmetrical five-point star, broad enough to command the full stage without approaching the boundaries. The star should arrive sharply, like a brass fanfare expanding from the center.** — Use a bold white core with yellow and amber radiating through the star’s arms. Add a one-beat strobe-decay accent on each downbeat, making every point of the star flare like polished brass catching sunlight.
- `s2b5t1` **The star folds upward into a tall, inverted cone, with its wide base hovering low and its apex rising high. The cone should slowly corkscrew around its vertical axis, giving the formation a proud, ascending march rather than a static pyramid.** — Color the cone in a vertical gradient from red at its lower body through orange to yellow at the apex. A bright sweep travels upward along the cone once per bar, emphasizing the climb and making the height of the formation unmistakable.
- `s3b1t1` **Unfurl the cone into a full-height helix that winds from the lower stage to near the top of the flying volume. The ten drones should travel as a tightly spaced rising spiral, like a regiment becoming a twisting column of brass and motion, then hold the helix briefly at its peak.** — Use a rainbow that travels spatially along the helix, with neighboring drones carrying clearly separated hues. Let the entire spectrum breathe once per bar, while a white flash runs upward through the column on each first beat.
- `s3b5t1` **Release the helix into a broad, undulating wave across the stage. The drones should crest and dip in staggered timing, producing a visible three-dimensional ripple that travels from one side to the other and then rebounds, with the formation remaining spread across both width and height.** — Use a cool cyan, teal, and blue palette with a moving rainbow undertone. A luminous ripple should travel outward through the formation in time with the beat, while the two sides blink in alternation so the wave appears to pass through the light as well as the bodies.
- `s3b9t1` **Drive the wave inward at once, then explode the ten drones into a huge, symmetric starburst filling the available width and height without touching the boundaries. After the burst, draw the points inward just enough to create a powerful final salute: a stable, evenly spaced star suspended high and clearly facing the audience.** — Finish with brilliant white across the star, edged in yellow and amber, with red accents at alternating points. Every beat receives a hard white flash followed by a warm golden decay; the final hold should remain fully illuminated, majestic, and unmistakably triumphant.

### BLIND — J. Strauss II - The Blue Danube Op. 314

A graceful awakening that grows from a quiet, suspended shimmer into a confident blue-and-gold waltz. The ten drones begin as scattered points in a dark, spacious sky, then gather, open into circular dancing formations, and finally sweep through a broad, celebratory finale. Motion should feel buoyant and musical rather than mechanical, with visible rises, dips, turns, and flowing changes of direction.

- `s1b1t1` **Begin with the ten drones suspended in a loose, asymmetrical constellation, spread widely across the volume at different heights. Give the whole group a barely perceptible shared sway, as though it is being stirred awake, while two or three points drift gently toward the center without forming a rigid pattern.** — Keep the scene mostly dark. Use very dim, cool blue-white pinpricks, with a soft glimmer that passes slowly from one drone to another. The effect should be mysterious, delicate, and expectant.
- `s2b1t1` **At the arrival of the first clear waltz impulse, let the constellation draw together into a graceful three-lobed arc, like the beginning of a turning dance. The drones should rise slightly as they gather, with the formation rotating slowly around its center and retaining an airy, uneven elegance.** — Introduce a richer river-blue glow across the group, brightening on the beat. A few drones should catch pale silver highlights as they turn, suggesting moonlight appearing on moving water.
- `s2b5t1` **Expand the arc into a broad, flowing spiral. The outer drones sweep outward while the inner drones curve around them, creating the impression of a current curling into a waltz. Let the spiral breathe vertically, with a gentle wave traveling from one side of the formation to the other.** — Deepen the blue and add restrained turquoise along the moving edges. The brightest points should travel around the spiral rather than remain fixed, producing a liquid, reflective sensation.
- `s3b1t1` **Open the spiral into a large, elegant rotating ring with a clear empty center. The ten drones should orbit at varied heights, rising and falling in a slow three-part rhythm so the ring feels buoyant and alive. Keep the rotation smooth and stately, like dancers sweeping across a ballroom.** — Use luminous sapphire blue as the dominant color, with alternating silver-white accents that pulse in a graceful one-two-three pattern. The center should remain darker, creating depth and a strong sense of space.
- `s3b5t1` **Break the ring into two interweaving curved lines that pass around and through one another without collision, then reform as a tilted double spiral. The exchange should feel like a flirtatious waltz between two currents, with a pronounced upward sweep followed by a soft settling motion.** — Let one curve glow blue and the other glow blue-green, exchanging brightness as they cross. Add brief pearl-white flashes at the moments of closest visual connection, evoking sparkling water and polished ballroom movement.
- `s3b9t1` **Gather the double spiral into a compact, ascending vortex, then release it into a wide fan that opens toward the audience. The opening should feel like a sudden discovery of a vast river or sky: expansive, confident, and still graceful rather than explosive.** — Build from saturated cobalt at the center to bright azure and silver at the widening edges. As the fan opens, let the drones brighten together, creating a clear swell of radiance while preserving the cool, aquatic character.
- `s4b1t1` **Compress the fan into a sweeping diagonal ribbon that travels across the full volume, with each drone following a slightly different height and curvature. The ribbon should roll over itself once, like a cresting wave, then level into a poised forward-moving line.** — Shift toward a more celebratory palette: vivid royal blue with warm champagne-gold highlights. Gold should appear first at the crest of the wave and then flow along the entire ribbon, making the music feel triumphant without losing its elegance.
- `s4b5t1` **Finish with the ten drones arranged in a grand, open circular wreath surrounding a clear center. Give the wreath one final slow, dignified turn, with alternating drones lifting and dipping to create a living three-beat pulse. End in a balanced, spacious tableau that feels both resolved and still in motion.** — Bring the full group to its brightest state: radiant blue, clear white sparkle, and warm gold accents distributed around the wreath. Let the gold and white shimmer travel once around the circle, then settle into a sustained luminous blue-and-gold finish.

### ANCHORED — J. Strauss II - The Blue Danube Op. 314

A graceful, continuously transforming waltz: the ten drones begin as a calm blue orbital jewel, gather into increasingly expressive star, spiral, cone, and layered-ring forms, then resolve into a bright, celebratory finale. Motion stays within the 4 m by 4 m volume and 0.25–1.70 m height limits, with lighting evolving from cool blue motion to warm gold and white triumph.

- `s1b1t1` **form_circle("1-10", 100, 75, 1.2); rotate(45, "z"). Effect: establish a poised, moonlit waltz with the whole swarm turning as one elegant blue ring.** — light_color(all, blue, both); pulse(all, 4, both). The ring breathes slowly and evenly, like light reflected on the Danube.
- `s2b1t1` **form_star(95, 55, 80, 1.3); rotate(-35, "z"). Effect: the circular calm opens into a crisp, flowering star, giving the music its first clear lift and definition.** — gradient(all, azure, indigo, radius, both); pulse(all, 2, both). Cool blue tones deepen toward the star's outer points while the whole shape swells with the phrase.
- `s2b5t1` **spiral(3, 55); move_z("1-10", 35); rotate(90, "z"). Effect: the star unwinds into a rising corkscrew, creating the sensation of dancers climbing through the waltz rather than simply changing formation.** — rainbow(all, 3, neighbour, both); chase(all, 1, 2, 1, neighbour, both). A concentrated band of color runs around the ascending spiral, making the rotation visibly travel through the swarm.
- `s3b1t1` **form_cone(60, 55, 0, 1.4); rotate(-70, "z"). Effect: the spiral gathers into a proud upright cone, with the swarm appearing to rise toward a single musical summit.** — gradient(all, teal, cyan, z, both); sweep(all, 2, z, both). Light climbs from the cone's base to its tip in a cool vertical sweep, emphasizing the new height.
- `s3b5t1` **form_circle("1-5", 65, 75, 1.0); form_circle("6-10", 65, 145, 1.0); rotate(120, "z"). Effect: split the swarm into two clearly separated, counterpoised waltz rings—one low and one high—so the music feels spatially layered and architectural.** — light_color(all, red, top); light_color(all, blue, bot); pulse(all, 1, both). Every drone carries a warm red upper glow and a deep blue lower glow, making the stacked rings read as a two-tone floating orchestra.
- `s3b9t1` **helix(3, 55, 85); spiral_speed(3, 75, 240, 2); rotate(-160, "z"). Effect: the two levels reconnect into a broad twisting helix, with the swarm visibly threading upward and around itself in a grand sustained sweep.** — rainbow(all, 2, radius, both); ripple_light(all, 2, both). Color radiates outward from the helix's center while a traveling pulse follows its changing radius, producing a luminous whirlpool.
- `s4b1t1` **form_star(110, 60, 90, 1.2); move_z("1-10", -30); rotate(180, "z"). Effect: after the expansive helix, snap into a large, high-energy star that feels like the waltz has opened into a full ballroom.** — gradient(all, amber, yellow, radius, both); strobe_decay(all, 1, both). Each beat strikes the star with a golden flash and a graceful decay, giving the phrase a strong orchestral accent.
- `s4b5t1` **form_circle("1-10", 105, 120, 1.5); rotate(360, "z"); move_z("1-10", 20). Effect: finish in a radiant, elevated circular finale—an immaculate rotating crown that holds together while the music resolves.** — light_color(all, white, top); light_color(all, amber, bot); pulse(all, 1, both). The upper ring is brilliant white while the lower ring glows celebratory gold, breathing on every beat for a luminous final waltz.

### BLIND — Mortals

A ten-drone performance that begins as a fragile spark, grows into a living and increasingly tense presence through the verse, then breaks open into a bright, liberated solo. The swarm should feel organic and intentional—sometimes unified, sometimes pulled apart—while remaining within the flying volume.

- `s1b1t1` **The drones appear close together near the center as a small, suspended cluster. They gently breathe outward and inward, like a single organism waking up, with subtle differences in height that make the group feel alive rather than perfectly rigid.** — Use a very dim, cool-white glow with a faint blue tint. The light should slowly rise and fall with the breathing motion, creating the feeling of a distant ember or consciousness beginning to form.
- `s2b1t1` **The cluster opens into a broad, shallow arc across the space, as though the presence has suddenly become aware of its surroundings. The drones drift with calm confidence, maintaining a clear shared shape while leaving generous space between one another.** — Shift to a stronger icy white with restrained blue-violet undertones. Light the arc from the center outward so the swarm feels like a wave of awareness spreading through the darkness.
- `s2b5t1` **The arc folds and twists into a slow, circulating spiral. The formation should feel pulled by an invisible current, with the drones orbiting one another while the overall shape gradually travels through the volume.** — Introduce saturated violet and deep electric blue, with individual drones brightening in sequence around the spiral. The result should feel hypnotic and slightly dangerous, as if the swarm has developed its own will.
- `s2b9t1` **The spiral suddenly breaks into two opposing groups that stretch away from one another, creating a tense divide across the space. After reaching maximum separation, both sides lean back toward the center without fully resolving the conflict.** — Contrast cold blue light on one side with burning magenta-red on the other. Keep the center comparatively dark so the separation reads as an emotional fracture, not merely a change of color.
- `s2b13t1` **The separated groups collapse toward one another and reform as a dense, angular spear pointing forward through the volume. The shape should surge with contained force, then hold in a poised, compressed state—as if everything is gathering before release.** — Drive the colors toward brilliant white at the leading point, with crimson and violet trailing behind. The swarm should appear intensely charged, almost overexposed, with a hard-edged flash at the moment the formation locks together.
- `s3b1t1` **The compressed spear bursts apart into a wide, three-dimensional constellation. The drones rise and spread across the full available space, moving in sweeping, liberated arcs before settling into a large, balanced emblem that feels expansive and triumphant.** — Open into radiant gold and clean white, with brief rainbow-like color glints moving through the constellation. The final impression should be release and transcendence: ten separate lights becoming a vast, luminous sky.

### ANCHORED — Mortals

A dark-to-bright arc: the swarm begins as a compact ignition, opens into a breathing ring, fractures into a rising spectrum, separates into two contrasting altitude bands, twists into a living helix, and finishes as a sharp inverted cone with a full-swarm white impact. All formations stay within the 4 m × 4 m × 1.45 m flying volume.

- `s1b1t1` **The ten drones snap into a tall five-pointed star, like a single spark suddenly becoming a symbol. Hold the star with a subtle rotational presence rather than drifting or wandering. Action: form_star(height_cm=110, spacing=60, delta_radius=80, time_to_finish_s=1.0); rotate(angle_deg=18, axis=z).** — A cold azure-to-white gradient runs from the lower points toward the upper points while the entire star slowly breathes: gradient(all, azure, white, by=z, both); pulse(all, period_beats=4, both).
- `s2b1t1` **The star melts outward into one broad, perfectly legible ring at mid-height. It should feel like the song has opened from a private ignition into a shared orbit. Action: form_circle(all, radius_cm=125, z_coord_cm=95, time_to_finish_s=1.2).** — The ring carries a complete spectrum that travels around the formation, with a slow four-beat breath underneath it: rainbow(all, period_beats=4, spread=neighbour, both); pulse(all, period_beats=4, both).
- `s2b5t1` **The ring rises and falls into a continuously turning helix, creating visible vertical depth and the sensation that the swarm is climbing through the music. Action: helix(steps=3, delta_height_cm=55, height_cm=85).** — A saturated spectrum runs along the helix while a bright musical chase circles through its spatial order: rainbow(all, period_beats=2, spread=neighbour, both); chase(all, period_beats=1, length=3, group_size=1, spread=neighbour, both).
- `s2b9t1` **The helix resolves into two separate, vertically stacked rings: a lower ring at 80 cm and an upper ring at 145 cm. The bands should read as two coordinated worlds, close enough to belong together but with a strong physical gap between them. Actions: form_circle(1-5, radius_cm=90, z_coord_cm=80, time_to_finish_s=1.0); form_circle(6-10, radius_cm=90, z_coord_cm=145, time_to_finish_s=1.0).** — The upper ring glows hot amber and the lower ring glows deep indigo, with each drone visibly split into two tones. The upper band flickers in opposing halves while the lower band breathes: light_color(all, amber, top); light_color(all, indigo, bot); alternate_blink(upper, period_beats=1, by=side, both); pulse(lower, period_beats=2, both).
- `s2b13t1` **Both rings collapse into a single aggressive corkscrew. The swarm should look braided and unstable, with drones winding around one another in a controlled vertical twist rather than merely rotating as a flat shape. Action: twister(steps=3, omega_times_ten=14, z_spacing_cm=20).** — A fast rainbow chases the braid while a radial ripple flashes outward from its core, making the twist appear to pulse from inside: rainbow(all, period_beats=1, spread=neighbour, both); ripple_light(all, period_beats=2, both).
- `s3b1t1` **For the solo entrance, the corkscrew compresses and strikes into a dramatic upright inverted cone, a pointed beacon with a broad crown and a sharp lower tip. It should arrive decisively, as if the entire swarm has been pulled into one final gesture. Action: form_cone(delta_height=70, spacing=60, is_inverted=1, time_to_finish_s=1.0).** — The whole cone turns brilliant white for the impact, with a clean one-beat flash and decaying afterglow that exposes the cone's silhouette: light_color(all, white, both); strobe_decay(all, period_beats=1, both).

### BLIND — Mozart - Eine kleine Nachtmusik

A precise, courtly-to-exuberant light sculpture for ten drones: the swarm begins as a quiet constellation, gathers into elegant geometric motifs, breaks into sparkling counterpoint, and closes in a radiant, perfectly resolved formation. The movement should feel musical, buoyant, and unmistakably classical rather than mechanical.

- `s1b1t1` **Begin as ten widely spaced points suspended at different heights, then draw them inward in a slow, graceful spiral until they form a compact diamond-shaped constellation.** — Start nearly dark with faint warm-white glimmers. As the drones converge, let each point brighten in sequence, creating a delicate champagne-gold arpeggio that resolves into a unified soft-white glow.
- `s1b5t1` **Unfurl the diamond into a perfectly balanced fan, with the center drones slightly forward and the outer drones opening symmetrically like a formal curtain.** — Use luminous ivory and pale gold. The center should shine first, followed by a smooth outward wave, giving the impression of a polished musical phrase opening into space.
- `s1b9t1` **Collapse the fan into a rotating ring, keeping the formation poised and circular while the drones rise and fall in a subtle alternating ripple.** — Shift toward brighter white with touches of silver-blue. Make the ring sparkle in alternating pairs, like elegant conversational phrases passing around the circle.
- `s2b1t1` **Break the ring into two opposing arcs that sweep apart dramatically, then curve back toward one another without touching, creating a suspended oval with a clear central opening.** — Introduce a richer golden-white brilliance. Light the two arcs in call-and-response, with a brief dark breath at the center before both sides flare together.
- `s2b5t1` **Send the ten drones through a fast, buoyant exchange: two interlaced spirals weave around one another and then settle into a tall, symmetrical lyre shape.** — Use sparkling white with flashes of warm gold at every crossing. The brightness should travel rapidly through the weave, producing the feeling of intricate orchestral dialogue.
- `s2b9t1` **Stretch the lyre vertically, then release it into a broad ascending wave: the drones rise in staggered groups from one side to the other, forming a sweeping diagonal crest before leveling into a broad horizon.** — Build from cool pearl at the low end to brilliant gold at the crest. The wave should brighten progressively, culminating in a full, triumphant white-gold flash across the formation.
- `s2b13t1` **Gather the broad horizon into a rotating ten-point star, then make the star gently precess so its points seem to turn through the air while the whole figure remains centered and regal.** — Give the star a crisp silver-white outline with a warm golden heart. Let individual points twinkle independently, then synchronize them into a strong, ceremonial pulse.
- `s2b17t1` **Burst the star outward into ten distinct points, each following a graceful curved path, then draw every point back into a tight, elevated crown-shaped formation.** — Make the outward burst vivid and jewel-like, alternating gold, ivory, and faint rose-white. During the return, intensify the lights until the crown becomes dazzling and unified.
- `s3b1t1` **From the crown, descend into a compact revolving sphere-like cluster, then expand smoothly into a low, wide ellipse that feels weightless and intimate.** — Soften to moonlit white and pale blue, with a subtle warm glow at the center. The light should breathe gently, creating a moment of graceful suspension after the larger climax.
- `s3b5t1` **Finish by lifting the ellipse into a brilliant symmetrical crown above the center of the volume, with the outer drones framing the inner group. Hold the final shape absolutely still after one last slight upward sweep.** — Rise from soft pearl to radiant champagne-gold, with all ten drones reaching full brightness together. End in a clean, sustained warm-white chord: majestic, joyful, and completely resolved.

### ANCHORED — Mozart - Eine kleine Nachtmusik

A 10-drone arc that grows from a restrained opening circle into increasingly complex vertical, radial, and traveling formations, reaching a bright kinetic climax before resolving into a calm, unified ending. The lighting evolves from warm nocturnal colors through cool spectral motion, sharp musical accents, and finally a soft blue-white afterglow.

- `s1b1t1` **All ten drones gather into a small, precise horizontal circle, like the first glint of a constellation appearing in the night. The arrival is deliberate and graceful rather than abrupt.** — A deep nocturnal palette: the drones carry a slow amber-to-indigo color gradient around the circle. Their brightness gently breathes together over four beats, creating a quiet, elegant entrance.
- `s1b5t1` **The circle begins a compact three-step spiral, turning inward and then opening slightly, as if the constellation is waking and rotating in space. The movement should feel fluid and poised.** — The warm amber and indigo evolve into a neighboring rainbow that travels around the formation. The whole ring pulses slowly, with no hard flashes.
- `s1b9t1` **The swarm snaps outward into a tall five-pointed star, with a clear vertical lift and strong radial definition. It should feel like the music suddenly finds its shape and announces itself.** — Use bright gold and white across the star, with the color strongest toward its outer radius. A short strobe-decay accent marks the beat, then the star settles into full luminous presence.
- `s2b1t1` **Divide the swarm into two disjoint five-drone circles: one ring low and one ring high, separated clearly in altitude. The two rings should rotate into a balanced celestial instrument, with the upper ring visibly floating above the lower one.** — Make the lower ring rich blue and the upper ring amber-gold. The two levels blink in alternating antiphase, so the stage seems to exchange energy between the low and high rings.
- `s2b5t1` **Unwind both rings into one vertically extended helix. The drones should travel continuously through height, producing a clear ascending-and-descending ribbon rather than a flat rotation.** — A cyan-to-indigo gradient follows the height of the helix. The color breathes slowly while a gentle upward brightness sweep travels through the vertical structure.
- `s2b9t1` **Collapse the helix into a broad upright cone, with its point lifted and its base spread toward the audience. The transformation should feel like the swarm is being drawn into a single dramatic spear of light.** — Use a white center fading through yellow and orange toward the wider lower portion. Each strong beat produces a crisp strobe-decay accent, emphasizing the cone's pointed silhouette.
- `s2b13t1` **The cone twists into a vigorous three-step corkscrew, with the drones climbing and descending around a shared vertical axis. The movement should feel energetic, tightly wound, and unmistakably three-dimensional.** — Run a vivid rainbow outward by radius, so the colors ripple from the core to the outer drones. Add a traveling radial light ripple that repeatedly expands through the corkscrew.
- `s2b17t1` **Break the corkscrew into a large, sharply articulated zigzag that travels laterally across the volume while alternating height. The swarm should strike each corner cleanly, like musical phrases being drawn in the air.** — Use a saturated orange-to-teal gradient across the formation. A fast neighboring chase runs along the zigzag, with bright clusters stepping from one turn to the next.
- `s3b1t1` **Gather the zigzag into a large, open star, wider and more commanding than the earlier star. The drones rise into a unified celebratory emblem, holding the shape with confident symmetry.** — Flood the star with white and amber, with the outer points carrying the warmest gold. The brightness swells over two beats and then gives one strong full-beat flash at the arrival.
- `s3b5t1` **The star slowly contracts toward the center and lowers into a compact, calm constellation. The final motion should feel like the music settling back into the night, with all ten drones ending together and visibly balanced.** — Transition to a cool blue-white look: blue at the lower-facing light and white above it. The brightness fades in a slow four-beat breath, leaving a soft luminous afterglow rather than a sudden blackout.

### BLIND — On & On

A ten-drone chorus journey that begins as a quiet ignition, grows into a synchronized breathing pattern, breaks into a sweeping wave, and culminates in a radiant, continuously turning emblem of motion. The swarm should feel alive, unified, and always carrying the music forward.

- `s1b1t1` **All ten drones appear tightly grouped at low altitude as a compact, slightly irregular ember. On the beat, the cluster gives a small upward surge and opens just enough to suggest something coming to life, then settles into a slow, hovering pulse.** — Start nearly dark, with a faint cool blue glow hidden inside the cluster. At the beat, flash a crisp white-blue core, followed by a soft electric-blue halo that fades but never disappears. The effect should feel like the first spark of an endless cycle.
- `s2b1t1` **The swarm rises and expands into a broad, balanced arc, with the drones moving outward in smooth counterbalanced paths rather than simply spreading apart. The arc gently contracts and expands in time with the beat, like a single breathing organism, while maintaining a strong shared center.** — Shift from blue into bright cyan and clean white. Each outward expansion sends a luminous wave from the center toward the ends of the arc; each contraction gathers the light back inward. Keep the rhythm clear and confident, as if the chorus has suddenly opened into a larger space.
- `s2b5t1` **Transform the arc into a traveling wave across the full formation. One side lifts while the other dips, and the motion passes through every drone in sequence, creating a smooth, visible ripple that repeats with increasing amplitude. The swarm should feel like a flag, a tide, or energy traveling forever from one side to the other.** — Use a saturated turquoise-to-violet gradient that travels with the wave. As each drone reaches the crest, it blooms to brilliant white, leaving a brief colored afterglow behind. The wave should look continuous rather than like separate flashes, with the whole formation visibly carrying the pulse onward.
- `s2b9t1` **The ten drones spiral inward from the traveling wave into a large rotating ring, then open through its center into a powerful forward-facing loop or endless-infinity shape. The formation keeps turning smoothly, with alternating drones tracing the upper and lower curves so the figure appears to flow without a beginning or end. Finish this moment with the shape expanding to fill the available space while remaining perfectly unified.** — Build from violet and cyan into intense white-gold at the crossings and front edge of the shape. The light should chase continuously around the ring and infinity curves, with a bright flare whenever paths visually intersect. End on a sustained radiant outline with a subtle pulsing interior, conveying an unstoppable, luminous motion that truly goes on and on.

### ANCHORED — On & On

A four-hit chorus arc: begin with a restrained amber spark, open into a full-spectrum starburst, separate into two opposing aerial rings, then resolve into a rising, rotating cone with a bright beat-locked finish. The swarm stays within the 4 m × 4 m × 1.45 m flight volume.

- `s1b1t1` **Start as a compact, low circular formation and send the swarm through a fast, graceful spiral that rises slightly, like a spark gathering momentum. Use form_circle(all, 100, 55, 0.6) followed by spiral_speed(3, 25, 180, 2).** — Give the whole swarm a warm amber-to-orange gradient by radius. Add a soft pulse over 2 beats so the opening feels like a controlled glow rather than a hard flash: gradient(all, amber, orange, radius, both); pulse(all, 2, both).
- `s2b1t1` **Hit the next chorus entrance with an expansive five-point starburst, making the swarm suddenly read as a large, symmetrical emblem. Use form_star(110, 55, 90, 1.0), then rotate(all, 180, z) so the star turns through the following interval.** — Make the starburst vivid and celebratory: run a rainbow around the formation while every drone flashes with a beat-locked decay. Use rainbow(all, 4, neighbour, both) and strobe_decay(all, 1, both).
- `s2b5t1` **Break the star into two clearly separated counterbalanced rings: drones 1-5 form a lower ring at 75 cm and drones 6-10 form an upper ring at 145 cm, with enough vertical separation to make the layers unmistakable. Have the two rings rotate as a single opposing structure: form_circle(1-5, 105, 75, 1.0); form_circle(6-10, 105, 145, 1.0); rotate(all, 180, z).** — Color the lower ring deep blue and the upper ring cyan, with each layer breathing in antiphase so the stacked structure alternates between a cool lower wave and a luminous upper wave. Use light_color(1-5, blue, both); light_color(6-10, cyan, both); alternate_blink(all, 2, by, both).
- `s2b9t1` **Collapse the two-ring architecture into a tall, inverted cone that rises toward the audience like a beam opening upward, then twist it visibly through the remaining phrase. Use form_cone(70, 55, 0, 1.0) followed by twister(3, 12, 15).** — Finish with a hot white-and-magenta climax: sweep magenta through the vertical cone while white beat flashes punch through the color. Use gradient(all, magenta, white, z, both); sweep(all, 1, z, both); strobe_decay(all, 1, both).

### BLIND — Vivaldi - Summer

A ten-drone arc that begins as a restrained, ominous ember, gathers into heat and instability, then breaks into violent lightning-like motion before ending in a stark, suspended threat. The swarm should feel like one living force that repeatedly forms, fractures, and surges through the full volume.

- `s1b1t1` **The ten drones hold a low, compact formation slightly below center, like a dark cloud gathering before the storm. They drift almost imperceptibly inward and downward, with a subtle irregular tremor rather than a perfectly still hover.** — Very dim, deep blue-violet light with only faint red embers appearing intermittently between the drones. The formation should feel heavy, tense, and expectant.
- `s1b5t1` **The compact cloud slowly stretches sideways into a long, uneven horizon. The outer drones lag behind and then catch up, creating a breathing, wave-like expansion across the space.** — The blue-violet deepens while narrow amber glows begin at the center and travel outward along the formation. The light should suggest heat starting to spread through the air.
- `s1b9t1` **The line folds into a broken spiral, with the drones circling around an empty center. The movement should be fluid but increasingly agitated, as though a hot wind is twisting the swarm into a vortex.** — Use stronger orange and red light in the moving inner arc, with cooler blue light on the outer drones. The colors should chase around the spiral and make the center feel dangerously charged.
- `s1b13t1` **The spiral suddenly tears open into two opposing streams. The drones pull apart toward opposite sides, then recoil toward one another without fully colliding, creating the sensation of a violent gust splitting and compressing the cloud.** — Flash the swarm with brief, hard white accents at each separation and recoil, surrounded by saturated red-orange. The flashes should feel like heat lightning inside a dark storm.
- `s2b1t1` **The drones regroup into a tall, narrow column rising through the center of the volume. The column bends and sways as one body, then briefly buckles near its middle, like a flame being battered by wind.** — Begin with a concentrated white-hot core at the center, fading through gold and orange toward the edges. Let the lower portion glow red, giving the column a furnace-like intensity.
- `s2b5t1` **The column explodes outward into a wide radial burst, then the drones race along curved paths that form a rapidly rotating crown. The crown should feel grand, unstable, and close to breaking apart.** — Use brilliant gold and searing white on the outward burst, followed by fast-moving red and orange bands around the crown. Add brief blue-violet shadows between bursts so the brightness feels violent rather than uniformly bright.
- `s3b1t1` **The crown collapses inward and the swarm plunges into a low, jagged wave spanning the width of the space. The wave travels from one side to the other with sharp rises and drops, like a storm front striking in successive blows.** — Alternate icy blue-white flashes with dark red pulses as the wave travels. Each crest should flare brightly for an instant, while the troughs fall almost black, creating a stark visual rhythm.
- `s3b5t1` **The drones shoot upward and outward into a fractured constellation, then freeze in an asymmetric suspended pattern with large empty gaps. A few drones make tiny, delayed adjustments, as if the storm has exhausted itself but has not truly disappeared.** — End with cold white points edged in pale blue, while one or two isolated drones retain a dim red glow. Fade the constellation slowly into darkness, leaving a final ominous ember rather than a warm resolution.

### ANCHORED — Vivaldi - Summer

A tense, heat-building swarm that begins as a dim ember, gradually opens into orbiting geometric forms, then erupts into a bright, restless climax. The ten drones remain within the 4 m by 4 m stage and 0.25–1.70 m altitude envelope.

- `s1b1t1` **The ten drones begin as a compact, low hovering cluster, then slowly breathe outward into a small, orderly circle. The movement should feel restrained and expectant, as though the swarm is gathering heat before the first attack of summer.** — A subdued ember palette: deep red and orange across the swarm, with a slow synchronized breathing glow. The lower lights remain red while the upper lights gently warm from orange toward amber.
- `s1b5t1` **The circle opens and begins a measured upward spiral, with the swarm rotating as one body. The rise should be graceful rather than explosive, creating the impression of hot air lifting the drones into the sky.** — A flowing orange-to-yellow gradient travels around the formation. The lights pulse slowly together, brightening at the top of each musical breath without flashing harshly.
- `s1b9t1` **The drones expand into a broad starburst, each point reaching outward from the center, then begin a slow rotation around the star. The shape should feel sharp and sun-like, with clear radial tension.** — Bright amber and yellow define the star points, with white highlights at the center-facing portions. A travelling chase runs around the star so the points ignite in sequence, like sparks racing through a hot metal shape.
- `s1b13t1` **The star folds into a tall, twisting column. The drones climb and descend through different heights while circling, producing a visible corkscrew that fills the vertical space without becoming chaotic.** — The upper lights burn yellow-white while the lower lights remain orange and red, making the column look molten. A vertical sweep moves through the formation from bottom to top, followed by a brief full-bright accent on the beat.
- `s2b1t1` **At the start of the second section, the swarm snaps into two clearly separated rotating rings, one high and one low. The rings counter-rotate with a strong vertical gap, creating a compact double-orbit that feels mechanically precise and suddenly more powerful.** — The lower ring carries red and orange; the upper ring carries yellow and white. Both rings pulse together, while their two light decks use contrasting colors so every drone reads as a hot two-tone spark.
- `s2b5t1` **The two rings collapse into a single violently twisting helix. The helix surges upward, tightens, then releases outward, giving the impression of a gust or vortex tearing through the swarm.** — Use a fast rainbow moving around the helix, with a strong beat-locked strobe decay layered over it. The flashes should land like lightning strikes while the underlying color continues to race through the twisting form.
- `s3b1t1` **The drones spread into a wide, shallow wave across the stage, rising and falling in a coordinated standing pattern. The wave should travel laterally through the swarm, like a shimmering wall of heat seen above scorched ground.** — A cyan-to-azure gradient sweeps across the wave, with white accents on the highest crests. The brightness travels from one side to the other in a smooth chase, then briefly swells across the entire wave on each major beat.
- `s3b5t1` **For the final surge, the wave gathers inward and launches into a tall, rotating cone that opens rapidly into a brilliant starburst. Hold the expanded star at full height and breadth as the last image: a violent, radiant sun rather than a gentle fade.** — Begin with indigo and magenta at the base, transition through pink and orange, and finish with blazing yellow and white. The lights race outward from the center, strobe with a short, sharp beat accent, then hold fully bright in a white-hot final tableau.

### BLIND — We Wish You a Merry Christmas

A warm, storybook Christmas journey in which ten drones begin as scattered winter lights, discover bells and ornaments, build a glowing tree, dissolve into snowfall, and finish as a bright star above a festive wreath. The movement should feel musical, buoyant, and graceful, with clear picture changes at each four-bar phrase while always remaining inside the flying volume.

- `s1b1t1` **Begin as ten gentle points of light spread loosely across the space at different heights, drifting inward and upward as if snowflakes are being carried by a soft breeze. The swarm should feel exploratory and magical rather than static.** — Use dim, cool white and pale blue lights with subtle shimmering variation. A few points should glint briefly like ice crystals while the overall image remains delicate and spacious.
- `s1b5t1` **Gather into a large, slow-turning Christmas bell made from the ten drones. Let the bell swing slightly from side to side, with a graceful arc that gives it a sense of ringing without becoming frantic.** — Shift to warm gold and amber, with a brighter golden outline and a softer white glow inside the bell. Pulse the brightness gently on the implied ringing motion.
- `s1b9t1` **Break the bell apart into a sweeping spiral, then send the drones outward into a circular ornament-like formation. The circle should rotate smoothly, as though a glass bauble is turning in the air.** — Use rich red, emerald green, and warm white distributed around the circle. Make the colors travel slowly around the rotating ornament, with occasional pinpoint sparkles like reflected lights.
- `s1b13t1` **Reassemble into a clearly recognizable five-pointed star, broad and balanced in the center of the volume. Hold the star proudly, then give it a slow, elegant tilt and return it to face the audience.** — Make the star brilliant warm white with a soft golden halo. Brighten the points one after another, followed by a full-star shimmer that feels like a magical reveal.
- `s1b17t1` **Transform the star into the upper half of a Christmas tree, with the drones cascading downward into layered branches. Add a gentle vertical sway, as if the tree is alive in a quiet winter breeze.** — Use deep green as the main color, with warm white and red points appearing as ornaments. The top drone should glow gold, creating a clear tree topper while the lower branches twinkle in alternating colors.
- `s1b21t1` **Let the tree dissolve into falling snow: the drones travel in staggered downward paths, then sweep back upward in a soft looping pattern. Finish by gathering into a loose, joyful arc that suggests snowfall settling around the scene.** — Return to icy white, pale blue, and silver. Keep the lights sparkling asynchronously, with brief bright flakes and a gradual warm glow beginning to emerge near the center.
- `s2b1t1` **Use the arc as the rim of a large wreath, closing the formation into a full circle with a clear opening in the middle. Rotate the wreath slowly while the drones bob with a buoyant, celebratory rhythm.** — Light the wreath in evergreen green with evenly spaced red and gold accents. Add a warm white inner glow so the center remains inviting and unmistakable.
- `s2b5t1` **Open the wreath and send the drones into a sweeping figure-eight around one another, like ribbons of festive light dancing through the air. The crossing should be smooth and symmetrical, with the swarm briefly forming two interlocking loops.** — Use alternating candy-cane red and warm white trails of light, enriched with small gold flashes at each crossing. Increase brightness through the center of the figure-eight for a feeling of joyful lift.
- `s2b9t1` **Collapse the figure-eight into ten rising candles or glowing points arranged in a gentle ascending fan. Have the fan breathe outward and inward, then lift together as though making a shared wish.** — Give each drone a warm candle-gold core with a soft amber halo. Flicker subtly and independently at first, then synchronize into one bright, steady golden swell.
- `s2b13t1` **From the rising fan, form a large radiant star high in the available space, with the remaining shape reading as a small wreath or cradle beneath it. Hold the complete Christmas tableau for the ending, adding one slow upward float and a final calm settling motion.** — Use brilliant warm white and gold for the star, surrounded by a restrained halo of red and green from the lower wreath. Finish with all ten lights glowing together, then soften slightly while preserving the star as the last clear image.

### ANCHORED — We Wish You a Merry Christmas

A bright, theatrical Christmas progression for ten drones: begin with a gathering spiral, expand into stars and cones, split into elevated red-and-gold rings, then build through a twisting, rainbow finale. Every formation stays within x/y ±2.0 m and z 0.25–1.70 m.

- `s1b1t1` **All ten drones rise together in a tight spiral, then open into a broad, confident circular sweep. The movement should feel like the show is coming alive from a single point and filling the sky.** — Use a slow rainbow travelling around the formation with a gentle four-beat pulse. The colours should move continuously rather than flash, creating a welcoming, magical opening.
- `s1b5t1` **Drive all ten drones sharply into a large five-point star, with a fast one-second arrival and a brief held silhouette. The star should read clearly and symmetrically from the audience viewpoint.** — Make the star warm Christmas gold: amber fading to yellow by radius, with a one-beat strobe-decay accent on every beat so each musical pulse hits like a sparkling ornament.
- `s1b9t1` **Collapse the star into an inverted cone, spreading the drones vertically so the formation changes from a flat emblem into a dimensional tree-like shape. The cone should arrive decisively, then breathe through a slow three-step vertical motion.** — Colour the cone from green at its broad lower edge through teal toward its upper point. Run a slow vertical sweep with a three-beat pulse, making the light appear to climb the tree.
- `s1b13t1` **Split the swarm into two distinct, disjoint rings: drones 1-5 form a lower circle at about 0.80 m, while drones 6-10 form a smaller upper circle at about 1.45 m. The two rings should rotate into alignment as a floating two-tier Christmas bauble.** — Give drones 1-5 a red lower ring and drones 6-10 a white upper ring. Add an alternating side blink across the whole swarm on a two-beat period, so the two tiers answer one another without going dark between accents.
- `s1b17t1` **Unstack the rings into a tall helix. The drones should wind upward and around one another, visibly occupying the full vertical volume, then make two measured turns rather than snapping directly to the next shape.** — Run a neighbour rainbow around the helix with a six-beat period, combined with a slow two-beat pulse. The colour should travel upward and around the twisting form like lights wrapped on a tree.
- `s1b21t1` **Resolve the helix into a large upright star with a pronounced vertical reach, then make the entire star perform a crisp two-step outward-and-inward flourish. This is the first major arrival and should feel celebratory rather than delicate.** — Use a saturated red-and-amber radius gradient, with a one-beat strobe-decay on every drone. The result should be a fiery Christmas star that flashes strongly on each beat and leaves a warm afterglow.
- `s2b1t1` **After the first section, draw all ten drones inward and reform them as a clean, wide circle at mid-height. Rotate the circle through a full quarter turn during the interval, making the transition feel like a reset before the next verse.** — Switch to cool winter colour: cyan blending to azure by neighbour position. Add a smooth four-beat pulse, giving the circle the calm, glowing quality of moonlit snow.
- `s2b5t1` **Transform the circle into a broad two-dimensional wave across the stage. The wave should travel from audience-left to audience-right in three visible undulations, with the drones rising and falling in a coordinated rolling surface.** — Use white on the wave peaks and blue in the troughs, blended by height. Add a directional sweep along the x-axis on a two-beat period so the illumination travels with the wave.
- `s2b9t1` **Wind the wave into a vigorous vertical twister. The ten drones should orbit at different heights and spacing, producing a compact spinning column that climbs and descends visibly rather than remaining a flat rotation.** — Use a neighbour rainbow with a four-beat period and a one-beat pulse. The colour should chase around the column while the whole twister breathes in time with the music.
- `s2b13t1` **Finish with the strongest image: expand all ten drones into a large upright star, hold it for recognition, then make a fast two-step spiral flourish that returns to the star on the final beat. The closing silhouette should be bold, balanced, and unmistakably festive.** — Set the final star to brilliant white with red on the upper LED ring and amber on the lower ring, creating a two-tone Christmas glow. Add a one-beat strobe-decay for the final musical hits, with the star fully radiant between accents rather than fading into darkness.
