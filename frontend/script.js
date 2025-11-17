const state = {
  projectId: null,
  downloadPath: null,
};

const audioInput = document.getElementById('audio-input');
const uploadBtn = document.getElementById('upload-btn');
const convertBtn = document.getElementById('convert-btn');
const makeMidiBtn = document.getElementById('make-midi-btn');
const uploadStatus = document.getElementById('upload-status');
const convertStatus = document.getElementById('convert-status');
const downloadLink = document.getElementById('download-link');
const chatLog = document.getElementById('chat-log');
const chatText = document.getElementById('chat-text');
const chatSend = document.getElementById('chat-send');

const BASE_URL = '';

function appendChatLine(text, role = 'bot') {
  const line = document.createElement('div');
  line.className = `chat-line ${role}`;
  line.textContent = text;
  chatLog.appendChild(line);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function uploadAudio() {
  if (!audioInput.files.length) {
    uploadStatus.textContent = 'Choose a WAV or MP3 file first.';
    return;
  }
  const formData = new FormData();
  formData.append('file', audioInput.files[0]);

  uploadStatus.textContent = 'Uploading...';
  const response = await fetch(`${BASE_URL}/upload_audio`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    uploadStatus.textContent = 'Upload failed.';
    return;
  }

  const result = await response.json();
  state.projectId = result.project_id;
  uploadStatus.textContent = `Project created: ${result.project_id}`;
  convertBtn.disabled = false;
}

async function extractMelody() {
  if (!state.projectId) return;
  convertStatus.textContent = 'Running CREPE...';
  const response = await fetch(`${BASE_URL}/extract_midi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project_id: state.projectId }),
  });

  if (!response.ok) {
    convertStatus.textContent = 'Melody extraction failed.';
    return;
  }

  const result = await response.json();
  convertStatus.textContent = `Detected ${result.frames} frames.`;
  makeMidiBtn.disabled = false;
}

async function createMidi() {
  if (!state.projectId) return;
  convertStatus.textContent = 'Building MIDI...';
  const response = await fetch(`${BASE_URL}/make_midi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project_id: state.projectId }),
  });

  if (!response.ok) {
    convertStatus.textContent = 'MIDI generation failed.';
    return;
  }

  const result = await response.json();
  state.downloadPath = result.midi_path;
  downloadLink.href = `${BASE_URL}/${result.midi_path}`;
  downloadLink.hidden = false;
  convertStatus.textContent = 'MIDI ready!';
}

async function sendChat() {
  const message = chatText.value.trim();
  if (!message) return;
  appendChatLine(message, 'user');
  chatText.value = '';

  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    appendChatLine('Chat service unavailable.');
    return;
  }
  const result = await response.json();
  appendChatLine(result.response || '...');
}

uploadBtn.addEventListener('click', uploadAudio);
convertBtn.addEventListener('click', extractMelody);
makeMidiBtn.addEventListener('click', createMidi);
chatSend.addEventListener('click', sendChat);
chatText.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    sendChat();
  }
});

appendChatLine('Welcome! Upload audio to begin.');
