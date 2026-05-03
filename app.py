from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import torch
import uvicorn

from model_custom import CrossModalModel
from utils import align, process_audio, process_video

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "model_final.pth"
SAMPLE_VIDEO_PATH = BASE_DIR / "sample.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CrossModalModel().to(device)
state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict, strict=True)
model.eval()

LABELS = {0: "REAL", 1: "FAKE"}

HOME_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VerifyAI | Deepfake Detection</title>
  <style>
    :root {
      --bg-0: #06101c;
      --bg-1: #0a1a31;
      --panel: rgba(9, 15, 28, 0.82);
      --panel-border: rgba(126, 225, 255, 0.16);
      --text: #eef5ff;
      --muted: #9fb0c8;
      --accent: #61efd0;
      --accent-2: #6f83ff;
      --danger: #ff6b6b;
      --success: #3fe08f;
    }

    * { box-sizing: border-box; }

    html, body { min-height: 100%; }

    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(97, 239, 208, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(111, 131, 255, 0.18), transparent 30%),
        linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 52%, #050812 100%);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: radial-gradient(circle at center, black 16%, transparent 85%);
    }

    .shell {
      position: relative;
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 26px;
    }

    .brand {
      font-size: 1.05rem;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }

    .pill {
      border: 1px solid rgba(97, 239, 208, 0.28);
      background: rgba(97, 239, 208, 0.08);
      color: var(--accent);
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 0.85rem;
      white-space: nowrap;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.08fr 0.92fr;
      gap: 24px;
      align-items: stretch;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(20px);
      border-radius: 28px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }

    .hero-copy {
      padding: 34px;
    }

    .kicker {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 0.76rem;
      font-weight: 700;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 16px var(--accent);
    }

    h1 {
      margin: 18px 0 14px;
      font-size: clamp(2.6rem, 5vw, 4.8rem);
      line-height: 0.94;
      letter-spacing: -0.05em;
    }

    h1 .accent {
      background: linear-gradient(90deg, var(--accent), #c2f9f2);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .lead {
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.7;
      max-width: 60ch;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 24px 0 0;
    }

    .chip {
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #dbe7ff;
      font-size: 0.88rem;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 28px;
    }

    .stat {
      padding: 18px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stat strong {
      display: block;
      font-size: 1.5rem;
      margin-bottom: 6px;
    }

    .stat span {
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.45;
    }

    .action-card {
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .preview {
      border-radius: 22px;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(97, 239, 208, 0.12), rgba(111, 131, 255, 0.08));
      border: 1px dashed rgba(255, 255, 255, 0.14);
      min-height: 250px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: none;
    }

    .placeholder {
      text-align: center;
      color: var(--muted);
      padding: 28px;
    }

    .placeholder strong {
      display: block;
      color: var(--text);
      margin-bottom: 10px;
      font-size: 1.05rem;
    }

    .controls {
      display: grid;
      gap: 12px;
    }

    input[type="file"] {
      width: 100%;
      color: var(--muted);
    }

    .row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    button {
      appearance: none;
      border: none;
      cursor: pointer;
      border-radius: 14px;
      padding: 13px 16px;
      font-weight: 700;
      transition: transform 0.2s ease, opacity 0.2s ease;
    }

    button:hover { transform: translateY(-1px); }

    .primary {
      background: linear-gradient(90deg, var(--accent), #83f5df);
      color: #03111b;
    }

    .secondary {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .result {
      padding: 20px 24px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.06);
      min-height: 120px;
    }

    .result h2 {
      margin: 0 0 10px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
    }

    .result .state {
      font-size: 1.9rem;
      font-weight: 800;
      margin-bottom: 8px;
    }

    .result .meta {
      color: var(--muted);
      line-height: 1.6;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      margin-top: 12px;
      font-size: 0.85rem;
      font-weight: 700;
    }

    .badge.real {
      background: rgba(63, 224, 143, 0.12);
      color: var(--success);
      border: 1px solid rgba(63, 224, 143, 0.3);
    }

    .badge.fake {
      background: rgba(255, 107, 107, 0.12);
      color: var(--danger);
      border: 1px solid rgba(255, 107, 107, 0.3);
    }

    .bar {
      height: 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      overflow: hidden;
      margin-top: 12px;
    }

    .fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent-2), var(--accent));
      border-radius: 999px;
      transition: width 0.4s ease;
    }

    .footer {
      margin-top: 20px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      color: var(--muted);
      font-size: 0.9rem;
      flex-wrap: wrap;
    }

    a {
      color: var(--accent);
      text-decoration: none;
    }

    .loading { opacity: 0.72; }

    @media (max-width: 940px) {
      .hero { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr; }
      .hero-copy { padding: 24px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="brand">VerifyAI</div>
      <div class="pill">FastAPI deepfake detection</div>
    </div>

    <section class="hero">
      <div class="panel hero-copy">
        <div class="kicker"><span class="dot"></span> Cross-modal analysis</div>
        <h1>Detect deepfakes with <span class="accent">audio and video</span> together</h1>
        <p class="lead">
          Upload a video or use the bundled sample clip. The model extracts frames,
          computes a log-mel audio representation, and returns a REAL or FAKE
          prediction with confidence.
        </p>

        <div class="chips">
          <span class="chip">16-frame video sampling</span>
          <span class="chip">64-bin log-mel audio</span>
          <span class="chip">PyTorch checkpoint inference</span>
          <span class="chip">FastAPI upload endpoint</span>
        </div>

        <div class="stats">
          <div class="stat">
            <strong>1</strong>
            <span>Upload one `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm` file</span>
          </div>
          <div class="stat">
            <strong>2</strong>
            <span>Click Analyze or use the sample clip</span>
          </div>
          <div class="stat">
            <strong>3</strong>
            <span>Get a prediction and confidence score</span>
          </div>
        </div>
      </div>

      <div class="panel action-card">
        <div class="preview" id="previewWrap">
          <video id="preview" controls></video>
          <div class="placeholder" id="placeholder">
            <strong>Preview area</strong>
            Choose a file or analyze the bundled sample.
          </div>
        </div>

        <form id="uploadForm" class="controls">
          <input type="file" id="videoFile" name="file" accept="video/*" />
          <div class="row">
            <button type="submit" class="primary">Analyze uploaded video</button>
            <button type="button" class="secondary" id="sampleBtn">Use sample video</button>
          </div>
        </form>

        <div class="result" id="resultBox">
          <h2>Result</h2>
          <div class="state" id="resultState">Waiting for a file</div>
          <div class="meta" id="resultMeta">
            Open /docs for the API schema or use the buttons above to test inference.
          </div>
          <div class="bar"><div class="fill" id="confidenceBar"></div></div>
          <div id="badgeWrap"></div>
        </div>

        <div class="footer">
          <span>Health check: <a href="/health">/health</a></span>
          <span>API docs: <a href="/docs">/docs</a></span>
        </div>
      </div>
    </section>
  </div>

  <script>
    const form = document.getElementById("uploadForm");
    const input = document.getElementById("videoFile");
    const sampleBtn = document.getElementById("sampleBtn");
    const preview = document.getElementById("preview");
    const placeholder = document.getElementById("placeholder");
    const resultState = document.getElementById("resultState");
    const resultMeta = document.getElementById("resultMeta");
    const confidenceBar = document.getElementById("confidenceBar");
    const badgeWrap = document.getElementById("badgeWrap");
    const resultBox = document.getElementById("resultBox");

    function setBusy(message) {
      resultState.textContent = message;
      resultMeta.textContent = "Please wait while the model processes the clip.";
      confidenceBar.style.width = "18%";
      badgeWrap.innerHTML = "";
      resultBox.classList.add("loading");
    }

    function showError(message) {
      resultState.textContent = "Error";
      resultMeta.textContent = message;
      confidenceBar.style.width = "0%";
      badgeWrap.innerHTML = "";
      resultBox.classList.remove("loading");
    }

    function showResult(data) {
      const prediction = String(data.prediction || "").toUpperCase();
      const confidence = Number(data.confidence || 0);
      resultState.textContent = prediction || "UNKNOWN";
      resultMeta.textContent = "Confidence: " + confidence.toFixed(2) + "%";
      confidenceBar.style.width = Math.max(0, Math.min(confidence, 100)) + "%";
      badgeWrap.innerHTML =
        '<span class="badge ' + (prediction === "FAKE" ? "fake" : "real") + '">' +
        (prediction === "FAKE" ? "Likely synthetic" : "Likely authentic") +
        "</span>";
      resultBox.classList.remove("loading");
    }

    function setPreviewFromFile(file) {
      if (!file) return;
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.style.display = "block";
      placeholder.style.display = "none";
    }

    async function predictFile(file, label) {
      if (!file) {
        throw new Error("Choose a video file first.");
      }
      setBusy(label || ("Analyzing " + file.name + "..."));
      const formData = new FormData();
      formData.append("file", file, file.name);
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || "Prediction failed.");
      }
      showResult(payload);
    }

    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (file) {
        setPreviewFromFile(file);
        resultState.textContent = "Ready to analyze";
        resultMeta.textContent = file.name;
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        const file = input.files && input.files[0];
        await predictFile(file);
      } catch (error) {
        showError(error.message || "Prediction failed.");
      }
    });

    sampleBtn.addEventListener("click", async () => {
      try {
        setBusy("Loading bundled sample video...");
        const response = await fetch("/sample.mp4");
        if (!response.ok) {
          throw new Error("Bundled sample video is unavailable.");
        }
        const blob = await response.blob();
        const file = new File([blob], "sample.mp4", {
          type: blob.type || "video/mp4",
        });
        setPreviewFromFile(file);
        await predictFile(file, "Analyzing bundled sample...");
      } catch (error) {
        showError(error.message || "Prediction failed.");
      }
    });
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "Deepfake Detection API is running"}


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(HOME_PAGE_HTML)


@app.get("/sample.mp4")
def sample_video() -> FileResponse:
    if not SAMPLE_VIDEO_PATH.is_file():
        raise HTTPException(status_code=404, detail="Bundled sample video not found.")
    return FileResponse(
        SAMPLE_VIDEO_PATH,
        media_type="video/mp4",
        filename="sample.mp4",
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    suffix = Path(file.filename or "temp_video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        shutil.copyfileobj(file.file, temp_file)

    try:
        frames = process_video(temp_path)
        mel = process_audio(temp_path)
        frames, mel = align(frames, mel)

        frames = frames.unsqueeze(0).float().to(device)
        mel = mel.unsqueeze(0).float().to(device)

        with torch.no_grad():
            output, _ = model(frames, mel)
            probs = torch.softmax(output, dim=1).cpu()

        pred = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred].item() * 100.0)

        return {
            "prediction": LABELS[pred],
            "confidence": round(confidence, 2),
        }
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
