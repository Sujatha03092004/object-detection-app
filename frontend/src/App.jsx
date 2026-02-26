import { useRef, useState, useEffect, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const fpsTimer = useRef(null);
  const frameCounter = useRef(0);
  const isProcessing = useRef(false);

  const [isRunning, setIsRunning] = useState(false);
  const [detections, setDetections] = useState([]);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      videoRef.current.srcObject = stream;
      setIsRunning(true);
    } catch {
      alert("Camera access denied. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    clearInterval(intervalRef.current);
    clearInterval(fpsTimer.current);
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    isProcessing.current = false;
    setIsRunning(false);
    setDetections([]);
    setFrameCount(0);
    setFps(0);
    frameCounter.current = 0;
  };

  const detectFrame = useCallback(async () => {
    // Skip if previous request hasn't finished
    if (isProcessing.current) return;
    isProcessing.current = true;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      isProcessing.current = false;
      return;
    }

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) {
      isProcessing.current = false;
      return;
    }

    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw;
      canvas.height = vh;
    }

    const ctx = canvas.getContext("2d");

    // Capture frame to temp canvas for API — video renders itself
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = vw;
    tempCanvas.height = vh;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(video, 0, 0, vw, vh);

    tempCanvas.toBlob(async (blob) => {
      if (!blob) {
        isProcessing.current = false;
        return;
      }

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        const response = await axios.post(`${API_URL}/detect`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        const { detections } = response.data;
        setDetections(detections);
        frameCounter.current += 1;
        setFrameCount((c) => c + 1);

        // Clear old boxes and draw new ones
        ctx.clearRect(0, 0, vw, vh);

        detections.forEach(({ label, confidence, box }) => {
          const { xmin, ymin, xmax, ymax } = box;
          const w = xmax - xmin;
          const h = ymax - ymin;

          // Main box
          ctx.strokeStyle = "#c8f135";
          ctx.lineWidth = 2;
          ctx.strokeRect(xmin, ymin, w, h);

          // Corner accents
          const cs = 12;
          ctx.strokeStyle = "#c8f135";
          ctx.lineWidth = 3;
          // top-left
          ctx.beginPath();
          ctx.moveTo(xmin, ymin + cs);
          ctx.lineTo(xmin, ymin);
          ctx.lineTo(xmin + cs, ymin);
          ctx.stroke();
          // top-right
          ctx.beginPath();
          ctx.moveTo(xmax - cs, ymin);
          ctx.lineTo(xmax, ymin);
          ctx.lineTo(xmax, ymin + cs);
          ctx.stroke();
          // bottom-left
          ctx.beginPath();
          ctx.moveTo(xmin, ymax - cs);
          ctx.lineTo(xmin, ymax);
          ctx.lineTo(xmin + cs, ymax);
          ctx.stroke();
          // bottom-right
          ctx.beginPath();
          ctx.moveTo(xmax - cs, ymax);
          ctx.lineTo(xmax, ymax);
          ctx.lineTo(xmax - cs, ymax);
          ctx.stroke();

          // Label
          const text = `${label}  ${(confidence * 100).toFixed(0)}%`;
          ctx.font = "bold 13px monospace";
          const tw = ctx.measureText(text).width;
          ctx.fillStyle = "#c8f135";
          ctx.fillRect(xmin, ymin - 24, tw + 12, 24);
          ctx.fillStyle = "#0a0a0a";
          ctx.fillText(text, xmin + 6, ymin - 7);
        });
      } catch (err) {
        console.error("Detection error:", err);
      } finally {
        // Always release lock — whether success or error
        isProcessing.current = false;
      }
    }, "image/jpeg", 0.8);
  }, []);

  useEffect(() => {
    if (isRunning) {
      fpsTimer.current = setInterval(() => {
        setFps(frameCounter.current);
        frameCounter.current = 0;
      }, 1000);
      intervalRef.current = setInterval(detectFrame, 500);
    }
    return () => {
      clearInterval(intervalRef.current);
      clearInterval(fpsTimer.current);
    };
  }, [isRunning, detectFrame]);

  return (
    <div className="app">
      <div className="header">
        <div className="header-left">
          <h1>Object Detection</h1>
          <p>SSD MobileNet V2 · COCO · FastAPI · React</p>
        </div>
        <div className={`status-badge ${isRunning ? "active" : ""}`}>
          <div className="pulse-dot" />
          {isRunning ? "Live" : "Offline"}
        </div>
      </div>

      <div className="camera-section">
        <div className="camera-wrapper">
          <div className="camera-inner">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} />
            {!isRunning && (
              <div className="idle-overlay">
                <div className="icon">⬡</div>
                <p>Awaiting feed</p>
              </div>
            )}
          </div>
        </div>

        <div className="controls">
          {!isRunning ? (
            <button className="btn-start" onClick={startCamera}>
              ▶ Start Detection
            </button>
          ) : (
            <button className="btn-stop" onClick={stopCamera}>
              ■ Stop
            </button>
          )}
        </div>
      </div>

      <div className="sidebar">
        <div className="sidebar-section">
          <h2>Session Stats</h2>
          <div className="stats-grid">
            <div className="stat-box">
              <div className="stat-value">{detections.length}</div>
              <div className="stat-label">Objects</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{fps}</div>
              <div className="stat-label">FPS</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{frameCount}</div>
              <div className="stat-label">Frames</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">
                {detections.length > 0
                  ? `${(Math.max(...detections.map((d) => d.confidence)) * 100).toFixed(0)}%`
                  : "—"}
              </div>
              <div className="stat-label">Top Conf</div>
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <h2>Detections</h2>
          {detections.length === 0 ? (
            <p className="empty-state">No objects detected</p>
          ) : (
            detections.map((d, i) => (
              <div key={i} className="detection-item">
                <div className="det-bar-wrap">
                  <div className="det-label">{d.label}</div>
                  <div className="det-bar-bg">
                    <div
                      className="det-bar-fill"
                      style={{ width: `${d.confidence * 100}%` }}
                    />
                  </div>
                </div>
                <div className="det-pct">
                  {(d.confidence * 100).toFixed(0)}%
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}