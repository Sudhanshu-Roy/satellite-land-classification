import { useEffect, useRef, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

const SAMPLES = [
  { label: "AnnualCrop", name: "Annual crop" },
  { label: "Forest", name: "Forest" },
  { label: "HerbaceousVegetation", name: "Herbaceous vegetation" },
  { label: "Highway", name: "Highway" },
  { label: "Industrial", name: "Industrial" },
  { label: "Pasture", name: "Pasture" },
  { label: "PermanentCrop", name: "Permanent crop" },
  { label: "Residential", name: "Residential" },
  { label: "River", name: "River" },
  { label: "SeaLake", name: "Sea or lake" },
];

const CLASS_COLOR = {
  AnnualCrop: "#8a6a2e",
  Forest: "#1f5c3a",
  HerbaceousVegetation: "#4e6a32",
  Highway: "#5c5348",
  Industrial: "#4a433c",
  Pasture: "#3f6b45",
  PermanentCrop: "#6e4e24",
  Residential: "#8d4e3a",
  River: "#2c5870",
  SeaLake: "#234a66",
};

export default function App() {
  const inputRef = useRef(null);
  const requestRef = useRef(null);
  const previewRef = useRef("");
  const generation = useRef(0);
  const [ready, setReady] = useState(false);
  const [patient, setPatient] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState("");
  const [fileName, setFileName] = useState("");
  const [phase, setPhase] = useState("idle");
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [barsReady, setBarsReady] = useState(false);
  const [sampleCursor, setSampleCursor] = useState(0);
  const [sampleName, setSampleName] = useState("");

  useEffect(() => {
    try {
      localStorage.removeItem("landcover.api");
    } catch {
      // Storage can be blocked. The address is no longer read from it.
    }

    let cancelled = false;
    let retryTimer;
    const stop = new AbortController();

    async function ping() {
      const attempt = new AbortController();
      const timeout = setTimeout(() => attempt.abort(), 8000);
      const cancelAttempt = () => attempt.abort();
      stop.signal.addEventListener("abort", cancelAttempt);
      try {
        const response = await fetch(`${API}/health`, { signal: attempt.signal });
        if (!cancelled && response.ok) {
          setReady(true);
          return;
        }
      } catch {
        // The service is still starting. Keep the loading screen up.
      } finally {
        clearTimeout(timeout);
        stop.signal.removeEventListener("abort", cancelAttempt);
      }
      if (!cancelled) retryTimer = setTimeout(ping, 3000);
    }

    ping();
    const patientTimer = setTimeout(() => {
      if (!cancelled) setPatient(true);
    }, 8000);

    return () => {
      cancelled = true;
      stop.abort();
      clearTimeout(retryTimer);
      clearTimeout(patientTimer);
    };
  }, []);

  useEffect(() => {
    if (!result) return undefined;
    setBarsReady(false);
    const frame = requestAnimationFrame(() => setBarsReady(true));
    return () => cancelAnimationFrame(frame);
  }, [result]);

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  async function classify(file, groundTruth = "") {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setSampleName("");
      setError("Use an image file, such as a PNG or JPEG tile.");
      setPhase("error");
      return;
    }
    setSampleName(groundTruth);

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const id = ++generation.current;

    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const url = URL.createObjectURL(file);
    previewRef.current = url;
    setPreview(url);
    setFileName(file.name);
    setResult(null);
    setError("");
    setSlow(false);
    setPhase("waiting");

    const slowTimer = setTimeout(() => {
      if (generation.current === id) setSlow(true);
    }, 8000);
    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        body,
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = payload && payload.detail;
        throw new Error(typeof detail === "string" ? detail : "The API rejected this image.");
      }
      if (generation.current !== id) return;
      setResult(payload);
      setPhase("result");
    } catch (err) {
      if (err.name === "AbortError" || generation.current !== id) return;
      setError(
        err.message === "Failed to fetch"
          ? "The classifier did not respond. Try the image again in a moment."
          : err.message
      );
      setPhase("error");
    } finally {
      clearTimeout(slowTimer);
      if (generation.current === id) setSlow(false);
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragOver(false);
    classify(event.dataTransfer.files?.[0]);
  }

  async function useSample() {
    const sample = SAMPLES[sampleCursor % SAMPLES.length];
    setSampleCursor((current) => current + 1);
    try {
      const response = await fetch(`/samples/${sample.label}.jpg`);
      if (!response.ok) throw new Error("The sample tile could not be loaded.");
      const blob = await response.blob();
      const file = new File([blob], `${sample.label}.jpg`, { type: blob.type || "image/jpeg" });
      await classify(file, sample.name);
    } catch (err) {
      setSampleName("");
      setError(err.message || "The sample tile could not be loaded.");
      setPhase("error");
    }
  }

  const accent = result ? CLASS_COLOR[result.label] || "#1f5c3a" : "#1f5c3a";

  if (!ready) {
    return (
      <div className="boot" role="status" aria-live="polite">
        <h1>Satellite Land Cover Classifier</h1>
        <p>Loading application</p>
        <div className="boot-line" aria-hidden="true">
          <span />
        </div>
        {patient && <p className="boot-note">The model is starting. The first open can take a minute.</p>}
      </div>
    );
  }

  return (
    <div className="sheet">
      <header className="mast">
        <h1>Satellite Land Cover Classifier</h1>
        <p className="lede">
        Deep learning inference engine for Land Use and Land Cover (LULC) mapping.
        </p>
        <p className="brief">
        Utilizes a fine-tuned ResNet18 backbone evaluated on the EuroSAT (Sentinel-2) benchmark (27,000 multispectral tiles, 10 classes) with 96.8% top-1 validation accuracy.
        </p>
      </header>

      <div className="work">
        <section
          className={`plate ${dragOver ? "over" : ""} ${preview ? "filled" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          {preview ? (
            <img src={preview} alt={fileName ? `Uploaded satellite image ${fileName}` : "Uploaded satellite image"} />
          ) : (
            <div className="empty-plate">
              <p>Drop a tile here</p>
              <span>PNG or JPEG, up to 8 MB</span>
            </div>
          )}
          <div className="plate-actions">
            <button type="button" onClick={() => inputRef.current?.click()} disabled={phase === "waiting"}>
              {preview ? "Choose another image" : "Choose an image"}
            </button>
            <button type="button" className="quiet" onClick={useSample} disabled={phase === "waiting"}>
              {sampleName ? "Try another sample" : "Try a sample"}
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              onChange={(event) => {
                classify(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
            {sampleName && <p className="sample-note">Sample from {sampleName}</p>}
          </div>
        </section>

        <section className="reading" aria-live="polite">
          {phase === "waiting" && (
            <div className="status-block">
              <p className="reading-title">Reading the tile</p>
              <p>
                {slow
                  ? "Still working. The first classification after a pause can take a minute."
                  : "The model is scoring this image against the ten land classes."}
              </p>
            </div>
          )}

          {phase === "error" && (
            <div className="status-block">
              <p className="reading-title">Could not classify</p>
              <p>{error}</p>
            </div>
          )}

          {phase === "idle" && (
            <div className="status-block">
              <p className="reading-title">Awaiting Input Tile</p>
              <p>
                Upload a satellite patch or choose a sample to run inference across 10 recognizable surface classes:
              </p>
              <ul className="class-groups">
                <li>
                  <strong>Vegetation &amp; Agriculture:</strong>{" "}
                  <em>Annual Crop, Permanent Crop, Pasture, Herbaceous, Forest</em>
                </li>
                <li>
                  <strong>Built Environment:</strong> <em>Residential, Industrial, Highway</em>
                </li>
                <li>
                  <strong>Water Bodies:</strong> <em>River, Sea &amp; Lake</em>
                </li>
              </ul>
            </div>
          )}

          {phase === "result" && result && (
            <div className="verdict">
              <h2 style={{ textDecorationColor: accent }}>{result.name}</h2>
              <p className="confidence">
                <span>{(result.confidence * 100).toFixed(1)}%</span> confidence
              </p>
              <ol className="scores">
                {result.scores.map((score) => (
                  <li key={score.label}>
                    <span className="score-name">{score.name}</span>
                    <span className="track" aria-hidden="true">
                      <span
                        className={`fill ${barsReady ? "shown" : ""}`}
                        style={{
                          "--p": score.confidence,
                          background: CLASS_COLOR[score.label] || accent,
                        }}
                      />
                    </span>
                    <span className="pct">{(score.confidence * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
