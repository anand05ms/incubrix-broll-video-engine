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
const jobIdText = document.querySelector("#job-id");
const errorBox = document.querySelector("#error-box");
const warningsBox = document.querySelector("#warnings");
const creditsBox = document.querySelector("#credits");
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

function safeErrorMessage(payload, fallback) {
  return payload?.error?.message || fallback;
}

function addDownload(name, url) {
  const link = document.createElement("a");
  link.className = "download-link";
  link.href = url;
  link.setAttribute("download", name);
  const label = document.createElement("span");
  label.textContent = name;
  const arrow = document.createElement("span");
  arrow.textContent = "↓";
  arrow.setAttribute("aria-hidden", "true");
  link.append(label, arrow);
  downloadsBox.append(link);
}

function showWarnings(items) {
  warningsBox.replaceChildren();
  if (!items?.length) {
    warningsBox.classList.add("hidden");
    return;
  }
  const heading = document.createElement("p");
  heading.textContent = "A note about this result:";
  warningsBox.append(heading);
  for (const warning of items) {
    const line = document.createElement("p");
    line.textContent = warning;
    warningsBox.append(line);
  }
  warningsBox.classList.remove("hidden");
}

function addExternalLink(parent, label, href) {
  let url;
  try {
    url = new URL(href);
  } catch {
    return false;
  }
  if (url.protocol !== "https:") return false;
  const link = document.createElement("a");
  link.href = url.href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = label;
  parent.append(link);
  return true;
}

function showCredits(timeline) {
  creditsBox.replaceChildren();
  const stockScenes = (timeline?.scenes || []).filter((scene) => scene.clip?.source === "Pexels");
  if (!stockScenes.length) {
    creditsBox.classList.add("hidden");
    return;
  }

  const heading = document.createElement("h3");
  heading.textContent = "Footage credits";
  creditsBox.append(heading);
  const providerLine = document.createElement("p");
  providerLine.append(document.createTextNode("Stock footage provided by "));
  if (!addExternalLink(providerLine, "Pexels", "https://www.pexels.com/")) {
    providerLine.append(document.createTextNode("Pexels"));
  }
  providerLine.append(document.createTextNode("."));
  providerLine.append(document.createTextNode(" License: "));
  addExternalLink(providerLine, "Pexels License", "https://www.pexels.com/license/");
  creditsBox.append(providerLine);

  for (const scene of stockScenes) {
    const line = document.createElement("p");
    line.append(document.createTextNode(`${scene.scene_id}: Video by `));
    if (!addExternalLink(
      line,
      scene.clip.photographer_name || "Pexels contributor",
      scene.clip.photographer_url || "",
    )) {
      line.append(document.createTextNode(scene.clip.photographer_name || "Pexels contributor"));
    }
    line.append(document.createTextNode(" on Pexels · "));
    if (!addExternalLink(line, "clip page", scene.clip.source_url || "")) {
      line.append(document.createTextNode("clip page unavailable"));
    }
    creditsBox.append(line);
  }
  creditsBox.classList.remove("hidden");
}

async function pollJob(jobId) {
  let delay = 1000;
  while (activeJobId === jobId) {
    const response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(safeErrorMessage(payload, "Could not read job status."));
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
      showWarnings(payload.result?.warnings || []);
      creditsBox.classList.add("hidden");
      const timelineUrl = payload.result?.files?.["timeline.json"];
      if (timelineUrl) {
        try {
          const timelineResponse = await fetch(timelineUrl);
          if (!timelineResponse.ok) throw new Error(`Timeline request failed (${timelineResponse.status}).`);
          showCredits(await timelineResponse.json());
        } catch (error) {
          const message = error instanceof Error ? error.message : "Unexpected timeline retrieval error.";
          showWarnings([...(payload.result?.warnings || []), `Could not load footage credits: ${message}`]);
        }
      }
      downloadsBox.replaceChildren();
      for (const [name, url] of Object.entries(payload.result?.files || {})) {
        addDownload(name, url);
      }
      downloadsBox.classList.remove("hidden");
      submitButton.disabled = false;
      submitLabel.textContent = "Create another video";
      return;
    } else if (payload.status === "failed") {
      progressText.textContent = "This render could not be completed.";
      setState("failed", "FAILED");
      errorBox.textContent = `${payload.error?.code || "JOB_FAILED"}: ${payload.error?.message || "The job failed."}`;
      errorBox.classList.remove("hidden");
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
  errorBox.classList.add("hidden");
  warningsBox.classList.add("hidden");
  creditsBox.classList.add("hidden");
  creditsBox.replaceChildren();
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
      throw new Error(safeErrorMessage(payload, `Request failed (${response.status}).`));
    }
    activeJobId = payload.job_id;
    jobIdText.textContent = `Job ${payload.job_id}`;
    await pollJob(payload.job_id);
  } catch (error) {
    resultTitle.textContent = "We couldn’t start your render";
    progressText.textContent = "Check your input and try again.";
    setState("failed", "ERROR");
    errorBox.textContent = error instanceof Error ? error.message : "Unexpected request error.";
    errorBox.classList.remove("hidden");
    submitButton.disabled = false;
    submitLabel.textContent = "Try again";
  }
});
