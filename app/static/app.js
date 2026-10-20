const form = document.querySelector("#job-form");
const submitButton = document.querySelector("#submit-button");
const submitLabel = document.querySelector("#submit-label");
const scriptArea = document.querySelector("#script");
const transcriptArea = document.querySelector("#transcript");
const scriptPanel = document.querySelector("#script-panel");
const audioPanel = document.querySelector("#audio-panel");
const audioInput = document.querySelector("#audio-file");
const uploadZone = document.querySelector("#upload-zone");
const uploadTitle = document.querySelector("#upload-title");
const uploadHint = document.querySelector("#upload-hint");
const stateBadge = document.querySelector("#result-state");
const resultTitle = document.querySelector("#result-title");
const emptyState = document.querySelector("#empty-state");
const jobState = document.querySelector("#job-state");
const progressText = document.querySelector("#progress-message");
const progressPercent = document.querySelector("#progress-percent");
const progressBar = document.querySelector("#progress-bar");
const progressTrack = document.querySelector(".progress-track");
const friendlyMessage = document.querySelector("#friendly-message");
const videoPreview = document.querySelector("#video-preview");
const downloadsBox = document.querySelector("#downloads");
const charCount = document.querySelector("#char-count");

let activeJobId = null;

function selectedMode() {
  return document.querySelector('input[name="input-mode"]:checked').value;
}

function setMode(mode) {
  const isScript = mode === "script";
  scriptPanel.classList.toggle("hidden", !isScript);
  audioPanel.classList.toggle("hidden", isScript);
  scriptArea.required = isScript;
  audioInput.required = !isScript;
  document.querySelectorAll("[data-mode-card]").forEach((card) => {
    const selected = card.dataset.modeCard === mode;
    card.classList.toggle("selected", selected);
  });
}

function setProgress(value) {
  const bounded = Math.max(0, Math.min(100, Number(value) || 0));
  progressBar.style.width = `${bounded}%`;
  progressPercent.textContent = `${bounded}%`;
  progressTrack.setAttribute("aria-valuenow", String(bounded));
}

function setState(state, label) {
  stateBadge.className = `state-badge ${state}`;
  stateBadge.textContent = label;
}

function addDownload(labelText, filename, url) {
  const link = document.createElement("a");
  link.className = "download-link";
  link.href = url;
  link.setAttribute("download", filename);
  const label = document.createElement("span");
  label.textContent = labelText;
  const arrow = document.createElement("span");
  arrow.textContent = "↓";
  arrow.setAttribute("aria-hidden", "true");
  link.append(label, arrow);
  downloadsBox.append(link);
}

async function pollJob(jobId) {
  let delay = 1000;
  while (activeJobId === jobId) {
    const response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error("Could not read job status.");
    }
    setProgress(payload.progress);
    if (payload.status === "queued") {
      progressText.textContent = "Waiting for an available renderer…";
      setState("running", "QUEUED");
    } else if (payload.status === "running") {
      progressText.textContent = payload.progress < 20
        ? "Preparing narration and scenes…"
        : payload.progress < 45
          ? "Creating your scene visuals…"
          : "Rendering and captioning your video…";
      setState("running", "RENDERING");
    } else if (payload.status === "succeeded") {
      setProgress(100);
      progressText.textContent = "Your video is ready.";
      resultTitle.textContent = "Your story is ready";
      setState("success", "COMPLETE");
      friendlyMessage.classList.add("hidden");
      downloadsBox.replaceChildren();
      const files = payload.result?.files || {};
      const videos = Object.entries(files).filter(([name]) => name.toLowerCase().endsWith(".mp4"));
      const captions = Object.entries(files).find(([name]) => name.toLowerCase() === "captions.srt");
      if (videos.length) {
        videoPreview.src = videos[0][1];
        videoPreview.classList.remove("hidden");
        for (const [filename, url] of videos) {
          const ratio = filename.match(/_(16_9|9_16|1_1)\.mp4$/i)?.[1]?.replaceAll("_", ":");
          addDownload(ratio ? `Download video · ${ratio}` : "Download video", filename, url);
        }
      }
      if (captions) addDownload("Download captions", captions[0], captions[1]);
      downloadsBox.classList.remove("hidden");
      submitButton.disabled = false;
      submitLabel.textContent = "Create another video";
      return;
    } else if (payload.status === "failed") {
      progressText.textContent = "We couldn’t finish this video just now.";
      resultTitle.textContent = "Let’s give it another try";
      setState("failed", "TRY AGAIN");
      friendlyMessage.textContent = "Your story is safe. Please try again in a moment.";
      friendlyMessage.classList.remove("hidden");
      submitButton.disabled = false;
      submitLabel.textContent = "Try another render";
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, delay));
    delay = Math.min(Math.round(delay * 1.4), 5000);
  }
}

document.querySelectorAll('input[name="input-mode"]').forEach((radio) => {
  radio.addEventListener("change", () => setMode(selectedMode()));
});

scriptArea.addEventListener("input", () => {
  charCount.textContent = `${scriptArea.value.length.toLocaleString()} / 30,000`;
});

audioInput.addEventListener("change", () => {
  const file = audioInput.files?.[0];
  uploadTitle.textContent = file ? file.name : "Choose an audio file";
  uploadHint.textContent = file
    ? `${(file.size / (1024 * 1024)).toFixed(1)} MB · Ready to upload`
    : "MP3, WAV or M4A · 50 MB max";
});

for (const eventName of ["dragenter", "dragover"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragover");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragover");
  });
}
uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!["mp3", "wav", "m4a"].includes(extension)) {
    uploadHint.textContent = "Please choose an MP3, WAV or M4A file.";
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  audioInput.files = transfer.files;
  audioInput.dispatchEvent(new Event("change", { bubbles: true }));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  friendlyMessage.classList.add("hidden");
  videoPreview.pause();
  videoPreview.removeAttribute("src");
  videoPreview.load();
  videoPreview.classList.add("hidden");
  downloadsBox.classList.add("hidden");
  downloadsBox.replaceChildren();
  jobState.classList.remove("hidden");
  emptyState.classList.add("hidden");
  resultTitle.textContent = "Preparing your story";
  progressText.textContent = "Sending your project…";
  setProgress(0);
  setState("running", "SUBMITTING");
  submitButton.disabled = true;
  submitLabel.textContent = "Starting your render…";
  activeJobId = null;

  const body = new FormData();
  const mode = selectedMode();
  const script = mode === "script" ? scriptArea.value.trim() : transcriptArea.value.trim();
  if (script) body.append("script", script);
  const language = document.querySelector("#language").value;
  if (language) body.append("language", language);
  if (mode === "audio") body.append("audio_file", audioInput.files[0]);
  const ratios = [...document.querySelectorAll('input[name="aspect_ratios"]:checked')].map((item) => item.value);
  body.append("aspect_ratios", JSON.stringify(ratios));
  body.append("style", document.querySelector("#style").value.trim());
  body.append("avoid", JSON.stringify(
    document.querySelector("#avoid").value.split(",").map((word) => word.trim()).filter(Boolean),
  ));
  body.append("captions", JSON.stringify({
    burn_in: document.querySelector("#burn-in").checked,
    srt: true,
  }));

  try {
    const response = await fetch("/api/v1/broll/jobs", {
      method: "POST",
      headers: { "Idempotency-Key": crypto.randomUUID() },
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error("Request could not be started.");
    }
    activeJobId = payload.job_id;
    await pollJob(payload.job_id);
  } catch {
    resultTitle.textContent = "Let’s give it another try";
    progressText.textContent = "We couldn’t start this video just now.";
    setState("failed", "TRY AGAIN");
    friendlyMessage.textContent = "Please check your story and try again in a moment.";
    friendlyMessage.classList.remove("hidden");
    submitButton.disabled = false;
    submitLabel.textContent = "Try again";
  }
});
