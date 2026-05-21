import { Pause, Play, RotateCcw, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { Playback } from "./types";

type PlayerProps = {
  playback: Playback;
  onClose: () => void;
};

type DroneScene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  drones: THREE.Group[];
  trails: THREE.Line[];
  animationId: number | null;
  lastUiUpdate: number;
};

const TRAIL_SECONDS = 2.4;
const TRAIL_SAMPLES = 48;
const STL_SCALE = 0.001;
const geometryCache = new Map<string, Promise<THREE.BufferGeometry>>();

type MeshPart = {
  file: string;
  color: number;
  position?: [number, number, number];
  rotationZ?: number;
  rotationX?: number;
};

const CF21B_STATIC_PARTS: MeshPart[] = [
  { file: "cf21B/cf21B_pcb.stl", color: 0x4d4d4d },
  { file: "cf21B/cf21B_motors.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_prop-guards.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_connectors.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_connector-pins.stl", color: 0xf7e099 },
  { file: "cf21B/cf21B_battery.stl", color: 0xb3b3b3 },
  { file: "cf21B/cf21B_battery-holder.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_PropL.stl", color: 0x85e625, position: [0.03536, -0.03536, 0.012], rotationZ: 45 },
  { file: "cf21B/cf21B_PropR.stl", color: 0x85e625, position: [-0.03536, -0.03536, 0.012], rotationZ: 135 },
  { file: "cf21B/cf21B_PropL.stl", color: 0x85e625, position: [-0.03536, 0.03536, 0.012], rotationZ: 225 },
  { file: "cf21B/cf21B_PropR.stl", color: 0x85e625, position: [0.03536, 0.03536, 0.012], rotationZ: 315 }
];

function findSampleIndex(timestamps: number[], time: number): number {
  if (time <= timestamps[0]) {
    return 0;
  }
  let lo = 0;
  let hi = timestamps.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (timestamps[mid] <= time) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return Math.max(0, Math.min(timestamps.length - 2, hi));
}

function sampleDroneState(
  playback: Playback,
  time: number,
  droneIndex: number,
  pos: THREE.Vector3,
  quat: THREE.Quaternion
): void {
  const timestamps = playback.timestamps;
  if (timestamps.length === 1) {
    const only = playback.states[0][droneIndex];
    pos.set(only[0], only[1], only[2]);
    quat.set(only[3], only[4], only[5], only[6]).normalize();
    return;
  }

  const clampedTime = Math.max(timestamps[0], Math.min(time, timestamps[timestamps.length - 1]));
  const index = findSampleIndex(timestamps, clampedTime);
  const t0 = timestamps[index];
  const t1 = timestamps[index + 1];
  const alpha = t1 > t0 ? (clampedTime - t0) / (t1 - t0) : 0;
  const a = playback.states[index][droneIndex];
  const b = playback.states[index + 1][droneIndex];

  pos.set(
    THREE.MathUtils.lerp(a[0], b[0], alpha),
    THREE.MathUtils.lerp(a[1], b[1], alpha),
    THREE.MathUtils.lerp(a[2], b[2], alpha)
  );

  const qa = new THREE.Quaternion(a[3], a[4], a[5], a[6]).normalize();
  const qb = new THREE.Quaternion(b[3], b[4], b[5], b[6]).normalize();
  quat.copy(qa).slerp(qb, alpha);
}

function loadGeometry(file: string): Promise<THREE.BufferGeometry> {
  if (!geometryCache.has(file)) {
    const loader = new STLLoader();
    geometryCache.set(
      file,
      new Promise((resolve, reject) => {
        loader.load(`/api/assets/drone/${file}`, (geometry) => {
          geometry.computeVertexNormals();
          resolve(geometry);
        }, undefined, reject);
      })
    );
  }
  return geometryCache.get(file)!;
}

function addMeshPart(parent: THREE.Group, part: MeshPart, material: THREE.Material): void {
  const partGroup = new THREE.Group();
  if (part.position) {
    partGroup.position.set(part.position[0], part.position[1], part.position[2]);
  }
  if (part.rotationX) {
    partGroup.rotation.x = THREE.MathUtils.degToRad(part.rotationX);
  }
  if (part.rotationZ) {
    partGroup.rotation.z = THREE.MathUtils.degToRad(part.rotationZ);
  }
  parent.add(partGroup);

  loadGeometry(part.file).then((geometry) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.setScalar(STL_SCALE);
    partGroup.add(mesh);
  }).catch(() => {
    // Keep the player usable if a mesh asset is missing.
  });
}

function makeDrone(color: THREE.Color): THREE.Group {
  const group = new THREE.Group();
  for (const part of CF21B_STATIC_PARTS) {
    addMeshPart(
      group,
      part,
      new THREE.MeshStandardMaterial({ color: part.color, roughness: 0.5, metalness: 0.05 })
    );
  }

  const deckMaterial = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.45,
    roughness: 0.35,
    transparent: true,
    opacity: 0.92
  });
  addMeshPart(group, { file: "cf21B/cf_led-diffusor.stl", color: 0xffffff, position: [0, 0, 0.015], rotationX: 180 }, deckMaterial);
  addMeshPart(group, { file: "cf21B/cf_led-diffusor.stl", color: 0xffffff, position: [0, 0, -0.002] }, deckMaterial);
  return group;
}

function makeFlightArea(playback: Playback): THREE.Group {
  const group = new THREE.Group();
  const [minX, minY, minZ] = playback.bounds.min;
  const [maxX, maxY] = playback.bounds.max;
  const width = maxX - minX;
  const depth = maxY - minY;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth),
    new THREE.MeshStandardMaterial({
      color: 0x16231f,
      roughness: 0.8,
      transparent: true,
      opacity: 0.48,
      side: THREE.DoubleSide
    })
  );
  floor.position.set(centerX, centerY, minZ);
  group.add(floor);

  const lineVertices: number[] = [];
  const addLine = (a: [number, number, number], b: [number, number, number]) => {
    lineVertices.push(...a, ...b);
  };
  addLine([minX, minY, minZ + 0.002], [maxX, minY, minZ + 0.002]);
  addLine([maxX, minY, minZ + 0.002], [maxX, maxY, minZ + 0.002]);
  addLine([maxX, maxY, minZ + 0.002], [minX, maxY, minZ + 0.002]);
  addLine([minX, maxY, minZ + 0.002], [minX, minY, minZ + 0.002]);

  const spacing = 0.5;
  for (let x = Math.ceil(minX / spacing) * spacing; x <= maxX; x += spacing) {
    addLine([x, minY, minZ + 0.001], [x, maxY, minZ + 0.001]);
  }
  for (let y = Math.ceil(minY / spacing) * spacing; y <= maxY; y += spacing) {
    addLine([minX, y, minZ + 0.001], [maxX, y, minZ + 0.001]);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(lineVertices, 3));
  group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x7fae98, transparent: true, opacity: 0.7 })));
  return group;
}

export function Player({ playback, onClose }: PlayerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const sceneRef = useRef<DroneScene | null>(null);
  const playheadRef = useRef(0);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const duration = useMemo(
    () => playback.timestamps[playback.timestamps.length - 1] ?? 0,
    [playback.timestamps]
  );

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08100e);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(2.8, -3.2, 2.4);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.9);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xdbeee7, 0x23322d, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(2.5, -2, 4);
    scene.add(key);

    scene.add(makeFlightArea(playback));

    const drones: THREE.Group[] = [];
    const trails: THREE.Line[] = [];
    playback.colors.forEach((rgb, index) => {
      const color = new THREE.Color(rgb[0], rgb[1], rgb[2]);
      const drone = makeDrone(color);
      drones.push(drone);
      scene.add(drone);

      const trailGeometry = new THREE.BufferGeometry();
      trailGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(TRAIL_SAMPLES * 3), 3)
      );
      const trail = new THREE.Line(
        trailGeometry,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.65 })
      );
      trail.name = `trail-${index}`;
      trails.push(trail);
      scene.add(trail);
    });

    const onResize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener("resize", onResize);

    sceneRef.current = {
      renderer,
      scene,
      camera,
      controls,
      drones,
      trails,
      animationId: null,
      lastUiUpdate: 0
    };

    return () => {
      window.removeEventListener("resize", onResize);
      const active = sceneRef.current;
      if (active && active.animationId !== null) {
        cancelAnimationFrame(active.animationId);
      }
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, [playback]);

  useEffect(() => {
    const tempPos = new THREE.Vector3();
    const tempQuat = new THREE.Quaternion();
    const trailPos = new THREE.Vector3();
    const trailQuat = new THREE.Quaternion();

    const renderAt = (time: number) => {
      const active = sceneRef.current;
      if (!active) {
        return;
      }

      playback.colors.forEach((_, droneIndex) => {
        sampleDroneState(playback, time, droneIndex, tempPos, tempQuat);
        active.drones[droneIndex].position.copy(tempPos);
        active.drones[droneIndex].quaternion.copy(tempQuat);

        const attr = active.trails[droneIndex].geometry.getAttribute("position") as THREE.BufferAttribute;
        for (let i = 0; i < TRAIL_SAMPLES; i += 1) {
          const offset = ((TRAIL_SAMPLES - 1 - i) / (TRAIL_SAMPLES - 1)) * TRAIL_SECONDS;
          sampleDroneState(playback, Math.max(0, time - offset), droneIndex, trailPos, trailQuat);
          attr.setXYZ(i, trailPos.x, trailPos.y, trailPos.z);
        }
        attr.needsUpdate = true;
      });

      active.controls.update();
      active.renderer.render(active.scene, active.camera);
    };

    const animate = () => {
      const active = sceneRef.current;
      const audio = audioRef.current;
      if (!active) {
        return;
      }
      const time = audio && !audio.paused ? Math.min(audio.currentTime, duration) : playheadRef.current;
      playheadRef.current = time;
      renderAt(time);
      const now = performance.now();
      if (now - active.lastUiUpdate > 80) {
        active.lastUiUpdate = now;
        setPlayhead(time);
      }
      active.animationId = requestAnimationFrame(animate);
    };

    const activeScene = sceneRef.current;
    if (activeScene) {
      activeScene.animationId = requestAnimationFrame(animate);
    }
    return () => {
      const active = sceneRef.current;
      if (active && active.animationId !== null) {
        cancelAnimationFrame(active.animationId);
        active.animationId = null;
      }
    };
  }, [duration, playback]);

  const setTime = (time: number) => {
    const nextTime = Math.max(0, Math.min(time, duration));
    playheadRef.current = nextTime;
    setPlayhead(nextTime);
    if (audioRef.current) {
      audioRef.current.currentTime = nextTime;
    }
  };

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (audio.paused) {
      audio.currentTime = playheadRef.current;
      await audio.play();
      setPlaying(true);
    } else {
      audio.pause();
      setPlaying(false);
    }
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.currentTime = 0;
    audio.play().then(() => {
      setPlaying(true);
    }).catch(() => {
      setPlaying(false);
    });
  }, [playback.audioUrl]);

  const restart = () => {
    setTime(0);
    audioRef.current?.pause();
    setPlaying(false);
  };

  return (
    <section className="player-shell">
      <div className="player-toolbar">
        <div>
          <p className="eyebrow">Browser playback</p>
          <h2>{playback.song}</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close player">
          <X size={18} />
        </button>
      </div>
      <div className="player-canvas" ref={mountRef} />
      <div className="playback-controls">
        <button className="primary-action compact" onClick={togglePlay}>
          {playing ? <Pause size={18} /> : <Play size={18} />}
          {playing ? "Pause" : "Play"}
        </button>
        <button className="secondary-action compact" onClick={restart}>
          <RotateCcw size={18} />
          Restart
        </button>
        <input
          className="timeline"
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={playhead}
          onChange={(event) => setTime(Number(event.target.value))}
        />
        <span className="timecode">
          {playhead.toFixed(1)} / {duration.toFixed(1)}s
        </span>
      </div>
      <audio
        ref={audioRef}
        src={playback.audioUrl}
        preload="auto"
        onEnded={() => setPlaying(false)}
      />
    </section>
  );
}
