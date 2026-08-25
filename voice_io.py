"""
Voice input (microphone -> text via faster-whisper) and
voice output (text -> speech).

TTS defaults to pyttsx3 (free, local, no API key or restrictions) since
ElevenLabs' free tier doesn't allow API access to cloned or library voices.
Set TTS_PROVIDER=elevenlabs in .env to switch back once you have a working
ElevenLabs voice/plan.
"""

import os
import io
import tempfile
import requests
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel
import pygame
import pyttsx3

SAMPLE_RATE = 16000
RECORD_SECONDS = 5  # length of each command recording after the wake word fires
WAKE_CHUNK = 1280  # openWakeWord expects 80ms chunks at 16kHz (1280 samples)
WAKE_THRESHOLD = 0.5

_whisper_model = None
_wakeword_model = None
_pyttsx3_engine = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        # "base.en" is a good speed/accuracy tradeoff for English on CPU.
        # Use "small.en" or "medium.en" for better accuracy if you have a GPU.
        _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _whisper_model


def _get_wakeword_model():
    global _wakeword_model
    if _wakeword_model is None:
        # "hey_jarvis" is one of openWakeWord's built-in pretrained models.
        # inference_framework="onnx" avoids needing tflite-runtime, which
        # doesn't have reliable Windows wheels - onnxruntime is already
        # installed as a faster-whisper dependency.
        _wakeword_model = WakeWordModel(
            wakeword_models=["hey_jarvis"], inference_framework="onnx"
        )
    return _wakeword_model


def wait_for_wake_word():
    """Block until the wake word ('Hey Jarvis') is detected from the mic."""
    print("Waiting for wake word ('Hey Jarvis')...")
    model = _get_wakeword_model()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                         blocksize=WAKE_CHUNK) as stream:
        while True:
            audio_chunk, _ = stream.read(WAKE_CHUNK)
            audio_chunk = audio_chunk.flatten()
            predictions = model.predict(audio_chunk)
            score = predictions.get("hey_jarvis", 0.0)
            if score > WAKE_THRESHOLD:
                model.reset()
                print("Wake word detected!")
                return


def listen() -> str:
    """Record a few seconds of audio from the mic and transcribe it."""
    print("Listening...")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write_wav(f.name, SAMPLE_RATE, audio)
        tmp_path = f.name

    model = _get_whisper_model()
    segments, _ = model.transcribe(tmp_path)
    text = " ".join(seg.text for seg in segments).strip()
    os.remove(tmp_path)

    print(f"You said: {text}")
    return text


def _get_pyttsx3_engine():
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        _pyttsx3_engine = pyttsx3.init()
        # Try to pick a female-sounding voice if one is installed, since V is
        # meant to sound female. Windows ships at least one female voice
        # (usually "Zira") by default alongside the male default.
        for voice in _pyttsx3_engine.getProperty("voices"):
            if "zira" in voice.name.lower() or "female" in voice.name.lower():
                _pyttsx3_engine.setProperty("voice", voice.id)
                break
        _pyttsx3_engine.setProperty("rate", 180)  # slightly faster than default
    return _pyttsx3_engine


def _speak_pyttsx3(text: str):
    engine = _get_pyttsx3_engine()
    engine.say(text)
    engine.runAndWait()


def _speak_elevenlabs(text: str):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    if not api_key or not voice_id:
        print(f"[No ElevenLabs config found, falling back to local TTS] V: {text}")
        _speak_pyttsx3(text)
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        print(f"[ElevenLabs TTS error {resp.status_code}, falling back to local TTS] {resp.text}")
        _speak_pyttsx3(text)
        return

    audio_bytes = io.BytesIO(resp.content)
    pygame.mixer.init()
    pygame.mixer.music.load(audio_bytes, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def speak(text: str):
    """Convert text to speech and play it aloud. Defaults to free local
    pyttsx3. Set TTS_PROVIDER=elevenlabs in .env to use ElevenLabs instead
    (requires a plan that allows API access to your chosen voice)."""
    print(f"V: {text}")
    provider = os.getenv("TTS_PROVIDER", "pyttsx3").lower()
    if provider == "elevenlabs":
        _speak_elevenlabs(text)
    else:
        _speak_pyttsx3(text)
