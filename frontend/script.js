const statusEl = document.getElementById("status");
const linkEl = document.getElementById("midi-link");
const fileInput = document.getElementById("audio-input");
const engineSelect = document.getElementById("engine-select");
const uploadBtn = document.getElementById("upload-btn");
const extractBtn = document.getElementById("extract-btn");
const midiBtn = document.getElementById("midi-btn");
const resetBtn = document.getElementById("reset-btn");

let projectId = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#f87171" : "#e2e8f0";
}

async function handleUpload() {
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Please choose a WAV file first.", true);
    return;
  }
  const body = new FormData();
  body.append("file", file);
  setStatus("Uploading audio...");
  linkEl.hidden = true;

  try {
    const res = await fetch("/upload_audio", {
      method: "POST",
      body,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Upload failed.");
    }
    projectId = data.project_id;
    setStatus("Audio uploaded. Ready to extract.");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function callJsonEndpoint(path, extra = {}) {
  if (!projectId) {
    throw new Error("Upload audio before continuing.");
  }
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: projectId, ...extra }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

async function handleExtract() {
  setStatus("Extracting melody...");
  linkEl.hidden = true;
  try {
    const selectedEngine = engineSelect.value;
    const data = await callJsonEndpoint("/extract_melody", { engine: selectedEngine });
    setStatus(`Melody extracted with ${data.engine} (${data.frames} frames). Ready for MIDI.`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function handleMidi() {
  setStatus("Building MIDI file...");
  linkEl.hidden = true;
  try {
    const data = await callJsonEndpoint("/make_midi");
    linkEl.href = data.midi_url;
    linkEl.hidden = false;
    setStatus("MIDI ready. Download below.");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function handleReset() {
  linkEl.hidden = true;
  fileInput.value = "";
  if (!projectId) {
    setStatus("Ready for a new project.");
    return;
  }
  try {
    await callJsonEndpoint("/cleanup_project");
    setStatus("Project cleaned up. Upload a new WAV to continue.");
  } catch (err) {
    setStatus(err.message, true);
    return;
  }
  projectId = null;
}

uploadBtn.addEventListener("click", handleUpload);
extractBtn.addEventListener("click", handleExtract);
midiBtn.addEventListener("click", handleMidi);
resetBtn.addEventListener("click", handleReset);
