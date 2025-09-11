// import React, { useEffect, useRef, useState } from "react";
// import { io } from "socket.io-client";

// const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || "http://127.0.0.1:5001";

// // minimal scene presets so /scene works
// const SCENES = {
//   calm:   { stroke:"#00e5ff", fill:"rgba(0,229,255,0.18)", lw:3, glow:6,  blend:"screen" },
//   forest: { stroke:"#7bd389", fill:"rgba(123,211,137,0.22)", lw:4, glow:10, blend:"multiply" },
//   neon:   { stroke:"#ff006e", fill:"rgba(255,0,110,0.18)", lw:5, glow:20, blend:"lighter" },
//   flame:  { stroke:"#ffd166", fill:"rgba(255,209,102,0.20)", lw:6, glow:24, blend:"screen" },
//   aurora: { stroke:"#b692ff", fill:"rgba(182,146,255,0.20)", lw:4, glow:16, blend:"screen" },
// };

// export default function OverlayPolys({ zIndex = 2147483647, opacity = 1, visible = true }) {
//   const canvasRef = useRef(null);
//   const [polys, setPolys] = useState([]);
//   const [style, setStyle] = useState(SCENES.calm);

//   // connect to backend
//   useEffect(() => {
//     const socket = io(SOCKET_URL, { transports: ["websocket"] });
//     socket.on("connect", () => console.log("[Overlay] connected:", SOCKET_URL));
//     socket.on("mask",  (msg) => setPolys(msg?.polys || []));
//     socket.on("style", (msg) => setStyle((s) => ({ ...s, ...msg })));
//     socket.on("scene", (msg) => { if (msg?.name && SCENES[msg.name]) setStyle(SCENES[msg.name]); });
//     return () => socket.disconnect();
//   }, []);

//   // size to device pixels
//   useEffect(() => {
//     const c = canvasRef.current;
//     const dpr = window.devicePixelRatio || 1;
//     const resize = () => {
//       const w = window.innerWidth, h = window.innerHeight;
//       c.style.width = "100vw"; c.style.height = "100vh";
//       c.width = Math.floor(w * dpr); c.height = Math.floor(h * dpr);
//     };
//     resize(); window.addEventListener("resize", resize);
//     return () => window.removeEventListener("resize", resize);
//   }, []);

//   // draw
//   useEffect(() => {
//     const c = canvasRef.current;
//     if (!c) return;
//     const ctx = c.getContext("2d");
//     const { width, height } = c;
//     const dpr = window.devicePixelRatio || 1;

//     ctx.clearRect(0, 0, width, height);
//     ctx.globalCompositeOperation = style.blend || "source-over";
//     ctx.lineWidth   = (style.lw ?? 3) * dpr;
//     ctx.strokeStyle = style.stroke || "#00ff00";
//     ctx.fillStyle   = style.fill   || "rgba(0,255,0,0.18)";
//     ctx.shadowColor = style.glow ? (style.stroke || "#00ff00") : "transparent";
//     ctx.shadowBlur  = style.glow || 0;

//     for (const poly of polys) {
//       if (!poly || poly.length < 3) continue;
//       ctx.beginPath();
//       for (let i = 0; i < poly.length; i++) {
//         const x = poly[i][0] * width;
//         const y = poly[i][1] * height;
//         if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
//       }
//       ctx.closePath();
//       ctx.fill();
//       ctx.stroke();
//     }
//   }, [polys, style]);

//   if (!visible) return null;
//   return (
//     <canvas
//       ref={canvasRef}
//       style={{
//         position: "fixed",
//         bottom: 0,
//         // width: "800px",
//         // height: "500px",
//         pointerEvents: "none", 
//         zIndex,
//         opacity
//       }}
//     />
//   );
// }

// src/components/OverlayPolys.jsx
import React, { useEffect, useRef, useState, useMemo } from "react";
import { io } from "socket.io-client";

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || "http://127.0.0.1:5001";

const SCENES = {
  calm:   { stroke:"#00e5ff", fill:"rgba(0,229,255,0.18)", lw:3, glow:6,  blend:"screen" },
  forest: { stroke:"#7bd389", fill:"rgba(123,211,137,0.22)", lw:4, glow:10, blend:"multiply" },
  neon:   { stroke:"#ff006e", fill:"rgba(255,0,110,0.18)", lw:5, glow:20, blend:"lighter" },
  flame:  { stroke:"#ffd166", fill:"rgba(255,209,102,0.20)", lw:6, glow:24, blend:"screen" },
  aurora: { stroke:"#b692ff", fill:"rgba(182,146,255,0.20)", lw:4, glow:16, blend:"screen" },
  white: { stroke:"#ffffff", fill:"rgba(255,255,255,0.20)", lw:4, glow:16, blend:"screen" },
};

// identity homography (no warp)
const IDENTITY = [1,0,0, 0,1,0, 0,0,1];

function warpPoint(u, v, H) {
  const X = H[0]*u + H[1]*v + H[2];
  const Y = H[3]*u + H[4]*v + H[5];
  const W = H[6]*u + H[7]*v + H[8];
  return [X / W, Y / W];
}

export default function OverlayPolys({ zIndex = 2147483647, opacity = 1, visible = true }) {
  const canvasRef = useRef(null);
  const [polys, setPolys] = useState([]);
  const [style, setStyle] = useState(SCENES.neon);

  // --- Hard-code or load calibration homography here ---
  const H = useMemo(() => {
    // Replace this with your calibrated 3x3 if you have one
    return IDENTITY;
  }, []);

  

  // socket connection
  useEffect(() => {
    const socket = io(SOCKET_URL, { transports: ["websocket"] });
    socket.on("mask",  (msg) => setPolys(msg?.polys || []));
    socket.on("style", (msg) => setStyle((s) => ({ ...s, ...msg })));
    socket.on("scene", (msg) => { if (msg?.name && SCENES[msg.name]) setStyle(SCENES[msg.name]); });
    return () => socket.disconnect();
  }, []);

  // resize
  useEffect(() => {
    const c = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    const resize = () => {
      const w = window.innerWidth, h = window.innerHeight;
      c.style.width = "100vw"; c.style.height = "100vh";
      c.width = Math.floor(w * dpr); c.height = Math.floor(h * dpr);
    };
    resize(); window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  // draw
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const { width, height } = c;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, width, height);
    ctx.globalCompositeOperation = style.blend || "source-over";
    ctx.lineWidth   = (style.lw ?? 3) * dpr;
    ctx.strokeStyle = style.stroke || "#00ff00";
    ctx.fillStyle   = style.fill   || "rgba(0,255,0,0.18)";
    ctx.shadowColor = style.glow ? (style.stroke || "#00ff00") : "transparent";
    ctx.shadowBlur  = style.glow || 0;

    for (const poly of polys) {
      if (!poly || poly.length < 3) continue;
      ctx.beginPath();
      for (let i = 0; i < poly.length; i++) {
        const scale = 1.5; 
        const offsetX = 0.12, offsetY = -0.15; 
const [uw, vw] = warpPoint(
  poly[i][0] * scale + offsetX,
  poly[i][1] * scale + offsetY,
  H
);

        const x = uw * width;
        const y = vw * height;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }, [polys, style, H]);

  if (!visible) return null;

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        pointerEvents: "none",
        zIndex,
        opacity,
      }}
    />
  );
}
