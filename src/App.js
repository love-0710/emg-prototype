// src/App.js

import React, { useState, useRef, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import heartGif from "./heart.gif"; // Place heart.gif in src/

// --- Signal Chart ---
function SignalChart({ data, label }) {
  return (
    <div style={{ padding: 10, background: "#ddd", borderRadius: 8, minWidth: 350 }}>
      <h3 style={{ color: "#444" }}>{label.toUpperCase()}</h3>
      <LineChart width={340} height={160} data={data}>
        <CartesianGrid stroke="#bbb" strokeDasharray="3 3" />
        <XAxis dataKey="t" stroke="#666" />
        <YAxis domain={[-2, 2]} stroke="#666" />
        <Tooltip />
        <Line type="monotone" dataKey="value" stroke="#666" dot={false} />
      </LineChart>
    </div>
  );
}

// --- Heart on Chest (Animated GIF as Html overlay) ---
function Heart({ heartRate }) {
  // Color bar: blue (low), green (normal), red (high)
  let color = "#2ecc40"; // green
  if (heartRate < 60) color = "#3498db"; // blue
  else if (heartRate > 100) color = "#e74c3c"; // red

  // Bar height
  const barHeight = Math.max(1.2, Math.min(1.2, (heartRate - 40) / 140 + 0.2));

  return (
    <group>
      {/* Heart GIF as HTML overlay, sized and centered on the body */}
      <Html position={[0, 1.0, 0.26]} center>
        <img
          src={heartGif}
          alt="heart"
          style={{
            width: "150px",
            height: "200px",
            objectFit: "cover",
            pointerEvents: "none",
            userSelect: "none",
            borderRadius: "18px",
            boxShadow: "0 0 12px #0006",
            background: "transparent"
          }}
        />
      </Html>
      {/* Heart rate bar (further right, outside body) */}
      <mesh position={[1.25, 1.5, 0.28]}>
        <boxGeometry args={[0.09, barHeight, 0.05]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {/* Heart rate text */}
      <Html position={[1.25, 1.5 + barHeight / 2 + 0.14, 0.28]} center>
        <div style={{
          color,
          fontWeight: "bold",
          fontSize: 18,
          background: "rgba(255,255,255,0.7)",
          borderRadius: 6,
          padding: "1px 4px"
        }}>
          {Math.round(heartRate)} bpm
        </div>
      </Html>
    </group>
  );
}

// --- Improved HumanModel ---
function HumanModel({ bodyAction, eyeAction, autoRotate, heartRate }) {
  const group = useRef();

  // Animated state for smooth transitions
  const [leftArm, setLeftArm] = useState(0);
  const [rightArm, setRightArm] = useState(0);
  const [leftLeg, setLeftLeg] = useState(0);
  const [rightLeg, setRightLeg] = useState(0);
  const [head, setHead] = useState(0);

  // Target state
  const target = {
    leftArm: 0, rightArm: 0, leftLeg: 0, rightLeg: 0, head: 0
  };
  switch (bodyAction) {
    case "left":
      target.leftArm = Math.PI / 4;
      target.rightArm = 0;
      target.leftLeg = Math.PI / 8;
      target.rightLeg = 0;
      target.head = Math.PI / 8;
      break;
    case "right":
      target.rightArm = -Math.PI / 4;
      target.leftArm = 0;
      target.rightLeg = Math.PI / 8;
      target.leftLeg = 0;
      target.head = -Math.PI / 8;
      break;
    case "up":
      target.leftArm = -Math.PI / 4;
      target.rightArm = -Math.PI / 4;
      target.leftLeg = 0;
      target.rightLeg = 0;
      target.head = 0;
      break;
    case "down":
      target.leftArm = Math.PI / 4;
      target.rightArm = Math.PI / 4;
      target.leftLeg = 0;
      target.rightLeg = 0;
      target.head = 0;
      break;
    case "clinch":
      target.leftArm = Math.PI / 2;
      target.rightArm = -Math.PI / 2;
      break;
    case "palam":
      target.leftArm = -Math.PI / 2;
      target.rightArm = Math.PI / 2;
      break;
    default:
      break;
  }

  // Animate body parts smoothly and slowly
  useEffect(() => {
    const id = setInterval(() => {
      setLeftArm((v) => v + (target.leftArm - v) * 0.05);
      setRightArm((v) => v + (target.rightArm - v) * 0.05);
      setLeftLeg((v) => v + (target.leftLeg - v) * 0.05);
      setRightLeg((v) => v + (target.rightLeg - v) * 0.05);
      setHead((v) => v + (target.head - v) * 0.05);
    }, 30);
    return () => clearInterval(id);
  }, [bodyAction]);

  // Eye state
  const [eye, setEye] = useState({ x: 0, y: 0, blink: false });
  let eyeTarget = { x: 0, y: 0, blink: false };
  switch (eyeAction) {
    case "w": eyeTarget = { x: 0, y: 0.22, blink: false }; break;
    case "s": eyeTarget = { x: 0, y: -0.22, blink: false }; break;
    case "a": eyeTarget = { x: -0.22, y: 0, blink: false }; break;
    case "d": eyeTarget = { x: 0.22, y: 0, blink: false }; break;
    case "b": eyeTarget = { x: 0, y: 0, blink: true }; break;
    default: break;
  }
  useEffect(() => {
    const id = setInterval(() => {
      setEye((e) => ({
        x: e.x + (eyeTarget.x - e.x) * 0.05,
        y: e.y + (eyeTarget.y - e.y) * 0.05,
        blink: eyeTarget.blink
      }));
    }, 30);
    return () => clearInterval(id);
  }, [eyeAction]);

  useEffect(() => {
    if (!eye.blink) return;
    const timeout = setTimeout(() => setEye((e) => ({ ...e, blink: false })), 400);
    return () => clearTimeout(timeout);
  }, [eye.blink]);

  useFrame(() => {
    if (group.current && autoRotate) group.current.rotation.y += 0.002;
  });

  return (
    <group ref={group} position={[0, -1, 0]}>
      {/* Torso */}
      <mesh position={[0, 1, 0]}>
        <boxGeometry args={[1, 1.5, 0.5]} />
        <meshStandardMaterial color="#777" />
      </mesh>
      {/* Heart and heart rate bar */}
      <Heart heartRate={heartRate} />
      {/* Head */}
      <group position={[0, 2.25, 0]} rotation-y={head}>
        <mesh>
          <sphereGeometry args={[0.5, 32, 32]} />
          <meshStandardMaterial color="#f0c080" />
        </mesh>
        {/* Eyes and eyebrows */}
        {[ -0.18, 0.18 ].map((x, i) => (
          <group key={i} position={[x, 0.13, 0.45]}>
            {/* Eyebrow */}
            <mesh position={[0, 0.11, 0]}>
              <boxGeometry args={[0.18, 0.03, 0.03]} />
              <meshStandardMaterial color="#222" />
            </mesh>
            {/* Eyeball */}
            <mesh scale-y={eye.blink ? 0.1 : 1} scale-x={eye.blink ? 1.5 : 1}>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial color="#fff" />
            </mesh>
            {/* Pupil (kept inside eyeball) */}
            <mesh
              position={[
                Math.max(-0.09, Math.min(0.09, eye.x)),
                Math.max(-0.09, Math.min(0.09, eye.y)),
                0.03
              ]}
              scale-y={eye.blink ? 0.1 : 1}
              scale-x={eye.blink ? 1.5 : 1}
            >
              <sphereGeometry args={[0.05, 16, 16]} />
              <meshStandardMaterial color="#000" />
            </mesh>
          </group>
        ))}
      </group>
      {/* Left Arm */}
      <group position={[-0.75, 1.25, 0]} rotation-z={leftArm} rotation-x={leftArm / 2}>
        <mesh position={[0, -0.5, 0]}>
          <boxGeometry args={[0.3, 1, 0.3]} />
          <meshStandardMaterial color="#888" />
        </mesh>
      </group>
      {/* Right Arm */}
      <group position={[0.75, 1.25, 0]} rotation-z={rightArm} rotation-x={rightArm / 2}>
        <mesh position={[0, -0.5, 0]}>
          <boxGeometry args={[0.3, 1, 0.3]} />
          <meshStandardMaterial color="#888" />
        </mesh>
      </group>
      {/* Left Leg */}
      <group position={[-0.3, 0, 0]} rotation-x={leftLeg}>
        <mesh position={[0, -0.75, 0]}>
          <boxGeometry args={[0.4, 1.5, 0.4]} />
          <meshStandardMaterial color="#555" />
        </mesh>
      </group>
      {/* Right Leg */}
      <group position={[0.3, 0, 0]} rotation-x={rightLeg}>
        <mesh position={[0, -0.75, 0]}>
          <boxGeometry args={[0.4, 1.5, 0.4]} />
          <meshStandardMaterial color="#555" />
        </mesh>
      </group>
    </group>
  );
}

// Generate synthetic EMG/EOG-like signals for given label
function generateSignal(label, t) {
  const freqMap = {
    left: 1.5,
    right: 1.6,
    up: 1.2,
    down: 1.3,
    clinch: 2,
    palam: 1.8,
    w: 1.1,
    s: 1.1,
    a: 1.1,
    d: 1.1,
    b: 3,
    idle: 0.7,
  };
  const freq = freqMap[label] || 1;
  return Array.from({ length: 60 }, (_, i) => ({
    t: (i * 0.1).toFixed(1),
    value:
      Math.sin((freq * (i + t * 10)) / 10) +
      (label === "idle" ? 0 : Math.random() * 0.2 - 0.1),
  }));
}

// --- Smooth ECG Signal Generator ---
function generateECGSignal(heartRate, t) {
  const bpm = Math.max(40, Math.min(180, heartRate));
  // Slower scroll: 0.2s per point, 60 points = 12 seconds window
  return Array.from({ length: 60 }, (_, i) => ({
    t: (t + i * 0.2).toFixed(1),
    value: bpm
  }));
}



// ...imports and SignalChart/Heart/HumanModel unchanged...

export default function App() {
  // --- State ---
  const [bodyAction, setBodyAction] = useState(null);
  const [eyeAction, setEyeAction] = useState(null);
  const [emgData, setEmgData] = useState(generateSignal("idle", 0));
  const [eogData, setEogData] = useState(generateSignal("idle", 0));
  const [combinedData, setCombinedData] = useState(generateSignal("idle", 0));
  const [ecgData, setEcgData] = useState(generateECGSignal(72, 0));
  const [emgLabel, setEmgLabel] = useState("idle");
  const [eogLabel, setEogLabel] = useState("idle");
  const [combinedLabel, setCombinedLabel] = useState("idle");
  const [autoRotate, setAutoRotate] = useState(true);
  const [emgLastActive, setEmgLastActive] = useState(Date.now());
  const [eogLastActive, setEogLastActive] = useState(Date.now());
  const [combineLastActive, setCombineLastActive] = useState(Date.now());
  const [heartRate, setHeartRate] = useState(72);

  // --- Refs for latest state in interval ---
  const emgLabelRef = useRef(emgLabel);
  const eogLabelRef = useRef(eogLabel);
  const combinedLabelRef = useRef(combinedLabel);
  const bodyActionRef = useRef(bodyAction);
  const eyeActionRef = useRef(eyeAction);
  const emgLastActiveRef = useRef(emgLastActive);
  const eogLastActiveRef = useRef(eogLastActive);
  const combineLastActiveRef = useRef(combineLastActive);
  const heartRateRef = useRef(heartRate);
  const heartRateTarget = useRef(72);
  const bpmOsc = useRef(0);

  // Keep refs in sync
  useEffect(() => { emgLabelRef.current = emgLabel; }, [emgLabel]);
  useEffect(() => { eogLabelRef.current = eogLabel; }, [eogLabel]);
  useEffect(() => { combinedLabelRef.current = combinedLabel; }, [combinedLabel]);
  useEffect(() => { bodyActionRef.current = bodyAction; }, [bodyAction]);
  useEffect(() => { eyeActionRef.current = eyeAction; }, [eyeAction]);
  useEffect(() => { emgLastActiveRef.current = emgLastActive; }, [emgLastActive]);
  useEffect(() => { eogLastActiveRef.current = eogLastActive; }, [eogLastActive]);
  useEffect(() => { combineLastActiveRef.current = combineLastActive; }, [combineLastActive]);
  useEffect(() => { heartRateRef.current = heartRate; }, [heartRate]);

  // --- Heart rate keyboard control (independent) ---
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "l") {
        heartRateTarget.current = Math.max(40, heartRateTarget.current - 2);
      }
      if (e.key === "h") {
        heartRateTarget.current = Math.min(180, heartRateTarget.current + 2);
      }
    }
    function handleKeyUp(e) {
      if (e.key === "l" || e.key === "h") {
        heartRateTarget.current = 72;
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // --- Key handling for EMG/EOG/Combined (unchanged) ---
  useEffect(() => {
    function handleKeyDown(e) {
      let ba = null, ea = null;
      switch (e.key) {
        case "ArrowLeft": ba = "left"; break;
        case "ArrowRight": ba = "right"; break;
        case "ArrowUp": ba = "up"; break;
        case "ArrowDown": ba = "down"; break;
        case "c": ba = "clinch"; break;
        case "p": ba = "palam"; break;
        case "w": ea = "w"; break;
        case "s": ea = "s"; break;
        case "a": ea = "a"; break;
        case "d": ea = "d"; break;
        case "b": ea = "b"; break;
        default: break;
      }
      if (ba && ba !== bodyActionRef.current) {
        setBodyAction(ba);
        setEmgLabel(ba);
        setEmgLastActive(Date.now());
      }
      if (ea && ea !== eyeActionRef.current) {
        setEyeAction(ea);
        setEogLabel(
          ea === "w"
            ? "up"
            : ea === "a"
            ? "left"
            : ea === "s"
            ? "down"
            : ea === "d"
            ? "right"
            : ea === "b"
            ? "b"
            : "idle"
        );
        setEogLastActive(Date.now());
      }
    }
    function handleKeyUp(e) {
      if (
        ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "c", "p"].includes(
          e.key
        )
      ) {
        setBodyAction(null);
        setEmgLastActive(Date.now());
      }
      if (["w", "a", "s", "d", "b"].includes(e.key)) {
        setEyeAction(null);
        setEogLastActive(Date.now());
      }
      setCombineLastActive(Date.now());
    }
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // --- Main interval for all signals and heart rate ---
  useEffect(() => {
    let start = Date.now();
    const interval = setInterval(() => {
      let t = ((Date.now() - start) / 1000);

      // Heart rate: idle fluctuation or animate toward target
      setHeartRate((hr) => {
        if (heartRateTarget.current === 72) {
          bpmOsc.current += 0.05;
          const base = 73.5 + Math.sin(bpmOsc.current) * 1.5; // 72-75
          return hr + (base - hr) * 0.08;
        }
        return hr + (heartRateTarget.current - hr) * 0.08;
      });

      // EMG
      setEmgData(generateSignal(emgLabelRef.current, t));
      if (
        emgLabelRef.current !== "idle" &&
        !bodyActionRef.current &&
        Date.now() - emgLastActiveRef.current > 6000
      ) {
        setEmgLabel("idle");
      }

      // EOG
      setEogData(generateSignal(eogLabelRef.current, t));
      if (
        eogLabelRef.current !== "idle" &&
        !eyeActionRef.current &&
        Date.now() - eogLastActiveRef.current > 6000
      ) {
        setEogLabel("idle");
      }

      // Combined
      let emgDir = ["left", "right", "up", "down"].includes(emgLabelRef.current)
        ? emgLabelRef.current
        : null;
      let eogDir = ["left", "right", "up", "down"].includes(eogLabelRef.current)
        ? eogLabelRef.current
        : null;
      let combined = "idle";
      if (emgDir && eogDir && emgDir === eogDir) {
        combined = emgDir;
        setCombineLastActive(Date.now());
      }
      setCombinedLabel(combined);
      setCombinedData(generateSignal(combined, t));
      if (
        combinedLabelRef.current !== "idle" &&
        combined === "idle" &&
        Date.now() - combineLastActiveRef.current > 6000
      ) {
        setCombinedLabel("idle");
      }

      // ECG: always reflect current heartRate
      setEcgData(generateECGSignal(heartRateRef.current, t));
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // --- UI Layout (unchanged) ---
  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={{ flex: "0 0 600px", background: "#222", display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1 }}>
          <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
            <ambientLight intensity={0.6} />
            <directionalLight position={[0, 5, 5]} intensity={0.8} />
            <HumanModel
              bodyAction={bodyAction}
              eyeAction={eyeAction}
              autoRotate={autoRotate}
              heartRate={heartRate}
            />
            <OrbitControls enablePan={false} enableZoom={false} enableRotate={true} autoRotate={autoRotate} autoRotateSpeed={0.5} />
          </Canvas>
        </div>
        <button
          style={{
            margin: 16,
            padding: "8px 18px",
            background: autoRotate ? "#bbb" : "#888",
            color: "#222",
            border: "none",
            borderRadius: 6,
            fontWeight: "bold",
            cursor: "pointer"
          }}
          onClick={() => setAutoRotate((v) => !v)}
        >
          {autoRotate ? "Disable" : "Enable"} Model Auto-Rotate
        </button>
      </div>
      <div style={{ flex: "1", padding: 20, background: "#eee", display: "flex", flexDirection: "column" }}>
        {/* Top row: EMG, EOG, ECG */}
        <div style={{ display: "flex", gap: 20, justifyContent: "center" }}>
          <SignalChart data={emgData} label={`EMG: ${emgLabel}`} />
          <SignalChart data={eogData} label={`EOG: ${eogLabel}`} />
          <SignalChart
            data={ecgData}
            label={`ECG: ${Math.round(heartRate)} bpm`}
            yDomain={[40, 180]} // pass this prop only for ECG
          />
        </div>
        {/* Bottom row: Combined */}
        <div style={{ display: "flex", justifyContent: "center", marginTop: 30 }}>
          <div>
            <SignalChart data={combinedData} label={`Combined: ${combinedLabel}`} />
            {["left", "right", "up", "down"].includes(combinedLabel) && (
              <div style={{ textAlign: "center", fontWeight: "bold", fontSize: 22, marginTop: 8 }}>
                {`Direction: ${combinedLabel.toUpperCase()}`}
              </div>
            )}
          </div>
        </div>
        {/* Controls */}
        <div style={{ marginTop: 20 }}>
          <h3>Controls</h3>
          <p>
            <b>Body Movement (EMG):</b> Arrow keys for directions, <code>c</code> for clinch, <code>p</code> for palam
          </p>
          <p>
            <b>Eye Movement (EOG):</b> <code>w</code> (up), <code>s</code> (down), <code>a</code> (left), <code>d</code> (right), <code>b</code> (blink)
          </p>
          <p>
            <b>Heart Rate (ECG):</b> <code>l</code> (lower), <code>h</code> (higher). Hold to change, release to return to normal.
          </p>
          <p>
            <b>Combined:</b> Only shows when both body and eye are the same direction (left/right/up/down).
          </p>
          <p>
            Hold a key to see continuous movement and EMG/EOG signal waveform. Release or wait 6 seconds to reset.
          </p>
        </div>
      </div>
    </div>
  );
}
