import { useState, useRef, useCallback, useEffect } from "react";
import "./App.css";

// ── Types ────────────────────────────────────────────────────────────────────

interface Detection {
  bbox: [number, number, number, number];
  conf: number;
  cls: number;
}

interface DetectResponse {
  uid: string;
  upload_url: string;
  result_url: string;
  original_size: { w: number; h: number };
  inf_time_ms: number;
  nms_time_ms: number;
  detections: Detection[];
  viz_urls: Record<string, string> | null;
}

const API = "";

const VIZ_LABELS: Record<string, string> = {
  letterboxed: "Letterboxed input",
  det_logits_heatmap: "Detection logits",
  da_logits_heatmap: "Drivable area logits",
  ll_logits_heatmap: "Lane line logits",
  da_mask: "Drivable area mask",
  ll_mask: "Lane line mask",
  final_result: "Final result",
};

// ── App ──────────────────────────────────────────────────────────────────────

function App() {
  // state
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vizOpen, setVizOpen] = useState(false);
  const [dragging, setDragging] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  // cleanup preview URL on unmount
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const loadFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setVizOpen(false);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) loadFile(f);
    },
    [loadFile],
  );

  const detect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        filename: file.name,
        lanes_only: "true",
        visualize: "true",
      });
      console.log("[detect] params sent to backend →", {
        filename: file.name,
        lanes_only: true,
        visualize: true,
        url: `${API}/api/detect?${params}`,
      });
      const res = await fetch(`${API}/api/detect?${params}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: await file.arrayBuffer(),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      const data: DetectResponse = await res.json();
      setResult(data);
      setVizOpen(false);
      setTimeout(
        () =>
          resultRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
          }),
        100,
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unexpected error.");
    } finally {
      setLoading(false);
    }
  };

  const vizEntries = result?.viz_urls ? Object.entries(result.viz_urls) : [];

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  d="M3 17 C6 14 9 11 12 11 C15 11 18 14 21 17"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                />
                <path
                  d="M3 13 C6 9  10 6  12 6  C14 6  18 9  21 13"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeOpacity="0.5"
                />
                <rect
                  x="1"
                  y="1"
                  width="22"
                  height="22"
                  rx="5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeOpacity="0.2"
                />
              </svg>
            </span>
            <span className="logo-text">
              Lane<span className="logo-accent">Vision</span>
            </span>
          </div>
          <div className="header-badge">YOLOPv2</div>
        </div>
      </header>

      {/* ── Main grid ──────────────────────────────────────────────── */}
      <main className="main-grid">
        {/* ── LEFT: Upload panel ──────────────────────────────────── */}
        <section className="panel panel-left">
          <div className="panel-title">
            <span className="panel-number">01</span>
            <h2>Select Image</h2>
          </div>

          {/* Drop zone */}
          <div
            className={`dropzone${dragging ? " dragging" : ""}${preview ? " has-preview" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !preview && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Upload image"
            onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          >
            {preview ? (
              <>
                <img src={preview} alt="Preview" className="preview-img" />
                <div className="preview-overlay">
                  <button
                    className="overlay-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      inputRef.current?.click();
                    }}
                  >
                    Change image
                  </button>
                </div>
              </>
            ) : (
              <div className="dropzone-empty">
                <div className="drop-icon">
                  <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
                    <rect
                      x="4"
                      y="10"
                      width="40"
                      height="32"
                      rx="6"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <circle
                      cx="17"
                      cy="22"
                      r="4"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <path
                      d="M4 34 L15 24 L22 31 L30 23 L44 34"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M24 4 L24 16 M20 8 L24 4 L28 8"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
                <p className="drop-primary">Drop image here</p>
                <p className="drop-sub">
                  or <span className="drop-link">browse files</span>
                </p>
                <p className="drop-hint">JPG, PNG, WEBP · up to 20 MB</p>
              </div>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) loadFile(f);
            }}
          />

          {/* File info */}
          {file && (
            <div className="file-chip">
              <span className="file-chip-icon">
                <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <rect
                    x="2"
                    y="1"
                    width="10"
                    height="14"
                    rx="2"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                  <path
                    d="M5 5h6M5 8h6M5 11h4"
                    stroke="currentColor"
                    strokeWidth="1.2"
                    strokeLinecap="round"
                  />
                </svg>
              </span>
              <span className="file-chip-name">{file.name}</span>
              <span className="file-chip-size">
                {(file.size / 1024).toFixed(0)} KB
              </span>
              <button
                className="file-chip-remove"
                aria-label="Remove"
                onClick={() => {
                  setFile(null);
                  setPreview(null);
                  setResult(null);
                }}
              >
                ×
              </button>
            </div>
          )}

          {/* CTA */}
          <button
            className={`detect-btn${loading ? " loading" : ""}`}
            disabled={!file || loading}
            onClick={detect}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Detecting…
              </>
            ) : (
              <>
                <svg
                  viewBox="0 0 20 20"
                  fill="none"
                  aria-hidden="true"
                  className="btn-icon"
                >
                  <circle
                    cx="10"
                    cy="10"
                    r="8"
                    stroke="currentColor"
                    strokeWidth="1.8"
                  />
                  <path
                    d="M6 10 C7.5 8 8.5 7 10 7 C11.5 7 12.5 8 14 10"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                  />
                  <circle cx="10" cy="10" r="1.5" fill="currentColor" />
                </svg>
                Detect Lanes
              </>
            )}
          </button>

          {error && (
            <div className="error-box" role="alert">
              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <circle
                  cx="8"
                  cy="8"
                  r="7"
                  stroke="currentColor"
                  strokeWidth="1.4"
                />
                <path
                  d="M8 5v3.5M8 10.5v.5"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                />
              </svg>
              {error}
            </div>
          )}
        </section>

        {/* ── RIGHT: Result panel ─────────────────────────────────── */}
        <section className="panel panel-right" ref={resultRef}>
          <div className="panel-title">
            <span className="panel-number">02</span>
            <h2>Detection Result</h2>
            {result && (
              <div className="result-badges">
                <span className="badge badge-time">
                  <svg viewBox="0 0 12 12" fill="none" aria-hidden="true">
                    <circle
                      cx="6"
                      cy="6"
                      r="5"
                      stroke="currentColor"
                      strokeWidth="1.2"
                    />
                    <path
                      d="M6 3.5v3l1.5 1"
                      stroke="currentColor"
                      strokeWidth="1.2"
                      strokeLinecap="round"
                    />
                  </svg>
                  {(result.inf_time_ms + result.nms_time_ms).toFixed(0)} ms
                </span>
                <span className="badge badge-det">
                  {result.detections.length} detection
                  {result.detections.length !== 1 ? "s" : ""}
                </span>
              </div>
            )}
          </div>

          {result ? (
            <div className="result-area">
              <div className="result-img-wrap">
                <img
                  src={`${API}${result.result_url}`}
                  alt="Lane detection result"
                  className="result-img"
                />
                <div className="result-img-badge">
                  {result.original_size.w} × {result.original_size.h}
                </div>
              </div>

              {/* Timing breakdown */}
              <div className="timing-row">
                <div className="timing-item">
                  <span className="timing-label">Inference</span>
                  <span className="timing-val">
                    {result.inf_time_ms.toFixed(1)} ms
                  </span>
                </div>
                <div className="timing-divider" />
                <div className="timing-item">
                  <span className="timing-label">NMS</span>
                  <span className="timing-val">
                    {result.nms_time_ms.toFixed(1)} ms
                  </span>
                </div>
                <div className="timing-divider" />
                <div className="timing-item">
                  <span className="timing-label">Total</span>
                  <span className="timing-val">
                    {(result.inf_time_ms + result.nms_time_ms).toFixed(1)} ms
                  </span>
                </div>
              </div>

              {/* Viz expand */}
              {vizEntries.length > 0 && (
                <div className="viz-section">
                  <button
                    className={`viz-toggle${vizOpen ? " open" : ""}`}
                    onClick={() => setVizOpen((v) => !v)}
                    aria-expanded={vizOpen}
                  >
                    <svg
                      className="viz-toggle-chevron"
                      viewBox="0 0 16 16"
                      fill="none"
                      aria-hidden="true"
                    >
                      <path
                        d="M5 7l3 3 3-3"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    Visualize pipeline steps
                    <span className="viz-count">{vizEntries.length} steps</span>
                  </button>

                  {vizOpen && (
                    <div className="viz-grid">
                      {vizEntries.map(([key, url]) => (
                        <div className="viz-card" key={key}>
                          <img src={`${API}${url}`} alt={key} loading="lazy" />
                          <span className="viz-label">
                            {VIZ_LABELS[key] ?? key.replace(/_/g, " ")}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="result-empty">
              <div className="empty-illustration" aria-hidden="true">
                <svg
                  viewBox="0 0 180 120"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect
                    x="10"
                    y="10"
                    width="160"
                    height="100"
                    rx="12"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeOpacity="0.15"
                  />
                  <path
                    d="M10 80 C35 60 55 50 90 50 C125 50 145 60 170 80"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeOpacity="0.25"
                  />
                  <path
                    d="M10 65 C40 40 60 30 90 30 C120 30 140 40 170 65"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeOpacity="0.12"
                  />
                  <circle
                    cx="90"
                    cy="60"
                    r="16"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeOpacity="0.2"
                  />
                  <circle
                    cx="90"
                    cy="60"
                    r="5"
                    fill="currentColor"
                    fillOpacity="0.15"
                  />
                </svg>
              </div>
              <p className="empty-title">No result yet</p>
              <p className="empty-sub">
                Upload an image and click <strong>Detect Lanes</strong>
              </p>
            </div>
          )}
        </section>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────── */}
      <footer className="footer">
        YOLOPv2 Lane Detection · powered by FastAPI + React
      </footer>
    </div>
  );
}

export default App;
