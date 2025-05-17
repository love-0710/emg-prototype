// src/App.js

import React, { useState, useRef, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

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

// --- Improved HumanModel ---
function HumanModel({ bodyAction, eyeAction, autoRotate }) {
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

  // Animate body parts smoothly
  useEffect(() => {
    const id = setInterval(() => {
      setLeftArm((v) => v + (target.leftArm - v) * 0.15);
      setRightArm((v) => v + (target.rightArm - v) * 0.15);
      setLeftLeg((v) => v + (target.leftLeg - v) * 0.15);
      setRightLeg((v) => v + (target.rightLeg - v) * 0.15);
      setHead((v) => v + (target.head - v) * 0.15);
    }, 30);
    return () => clearInterval(id);
  }, [bodyAction]);

  // Eye state
  const [eye, setEye] = useState({ x: 0, y: 0, blink: false });
  // Target for eye
  let eyeTarget = { x: 0, y: 0, blink: false };
  switch (eyeAction) {
    case "w": eyeTarget = { x: 0, y: 0.22, blink: false }; break;
    case "s": eyeTarget = { x: 0, y: -0.22, blink: false }; break;
    case "a": eyeTarget = { x: -0.22, y: 0, blink: false }; break;
    case "d": eyeTarget = { x: 0.22, y: 0, blink: false }; break;
    case "b": eyeTarget = { x: 0, y: 0, blink: true }; break;
    default: break;
  }
  // Animate eye smoothly
  useEffect(() => {
    const id = setInterval(() => {
      setEye((e) => ({
        x: e.x + (eyeTarget.x - e.x) * 0.2,
        y: e.y + (eyeTarget.y - e.y) * 0.2,
        blink: eyeTarget.blink
      }));
    }, 30);
    return () => clearInterval(id);
  }, [eyeAction]);

  // Blink timer reset
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

// --- Main App ---
export default function App() {
  const [bodyAction, setBodyAction] = useState(null);
  const [eyeAction, setEyeAction] = useState(null);
  const [emgData, setEmgData] = useState(generateSignal("idle", 0));
  const [eogData, setEogData] = useState(generateSignal("idle", 0));
  const [combinedData, setCombinedData] = useState(generateSignal("idle", 0));
  const [emgLabel, setEmgLabel] = useState("idle");
  const [eogLabel, setEogLabel] = useState("idle");
  const [combinedLabel, setCombinedLabel] = useState("idle");
  const [autoRotate, setAutoRotate] = useState(true);
  const [emgLastActive, setEmgLastActive] = useState(Date.now());
  const [eogLastActive, setEogLastActive] = useState(Date.now());
  const [combineLastActive, setCombineLastActive] = useState(Date.now());
  const intervalRef = useRef();

  // Key handling
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
      if (ba && ba !== bodyAction) {
        setBodyAction(ba);
        setEmgLabel(ba);
        setEmgLastActive(Date.now());
      }
      if (ea && ea !== eyeAction) {
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
  }, [bodyAction, eyeAction]);

  // Real-time signal update and auto-reset after 6s of inactivity for each panel
  useEffect(() => {
    intervalRef.current && clearInterval(intervalRef.current);
    let start = Date.now();
    intervalRef.current = setInterval(() => {
      let t = ((Date.now() - start) / 1000);

      // EMG
      setEmgData(generateSignal(emgLabel, t));
      if (emgLabel !== "idle" && !bodyAction && Date.now() - emgLastActive > 6000) {
        setEmgLabel("idle");
      }

      // EOG
      setEogData(generateSignal(eogLabel, t));
      if (eogLabel !== "idle" && !eyeAction && Date.now() - eogLastActive > 6000) {
        setEogLabel("idle");
      }

      // Combined: only when both are set and same direction (left/right/up/down)
      let emgDir = ["left", "right", "up", "down"].includes(emgLabel) ? emgLabel : null;
      let eogDir = ["left", "right", "up", "down"].includes(eogLabel) ? eogLabel : null;
      let combined = "idle";
      if (emgDir && eogDir && emgDir === eogDir) {
        combined = emgDir;
        setCombineLastActive(Date.now());
      }
      setCombinedLabel(combined);
      setCombinedData(generateSignal(combined, t));
      if (
        combinedLabel !== "idle" &&
        combined === "idle" &&
        Date.now() - combineLastActive > 6000
      ) {
        setCombinedLabel("idle");
      }
    }, 100);
    return () => clearInterval(intervalRef.current);
  }, [emgLabel, eogLabel, bodyAction, eyeAction, emgLastActive, eogLastActive, combineLastActive, combinedLabel]);

  // --- UI Layout ---
  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={{ flex: "0 0 600px", background: "#222", display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1 }}>
          <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
            <ambientLight intensity={0.6} />
            <directionalLight position={[0, 5, 5]} intensity={0.8} />
            <HumanModel bodyAction={bodyAction} eyeAction={eyeAction} autoRotate={autoRotate} />
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
        {/* Top row: EMG and EOG */}
        <div style={{ display: "flex", gap: 20, justifyContent: "center" }}>
          <SignalChart data={emgData} label={`EMG: ${emgLabel}`} />
          <SignalChart data={eogData} label={`EOG: ${eogLabel}`} />
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
            <b>Combined:</b> Only shows when both body and eye are the same direction (left/right/up/down).
          </p>
          <p>
            Hold a key to see continuous movement and EMG/EOG signal waveform. Release or wait 10 seconds to reset.
          </p>
        </div>
      </div>
    </div>
  );
}

