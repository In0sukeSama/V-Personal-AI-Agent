# JARVIS

A personal voice assistant powered by the Claude API, inspired by Iron Man's JARVIS.

## Setup

1. **Install Python 3.10+** if you don't have it already.

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set up your API keys:**
   - Copy `.env.example` to `.env`
   - Fill in your `ANTHROPIC_API_KEY` (from platform.claude.com)
   - Fill in your `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`
   - (Optional) Add an `OPENWEATHER_API_KEY` from openweathermap.org (free tier) to enable the weather tool

4. **Test it in text mode first** (no mic/voice needed, fastest way to check the brain + tools work):
   ```
   python main.py --text
   ```

5. **Run full voice mode:**
   ```
   python main.py
   ```
   It records ~5 seconds after printing "Listening...", transcribes what you said, sends it to
   Claude, and speaks the reply back using your ElevenLabs voice.

## Project structure

- `main.py` - entry point, the listen -> think -> speak loop
- `brain.py` - talks to the Claude API, holds conversation history, handles tool-calling
- `tools.py` - the actions JARVIS can take (open apps, open websites, check weather)
- `voice_io.py` - microphone recording + Whisper transcription, and ElevenLabs text-to-speech
- `.env` - your API keys (never commit this file)

## Adding new tools

1. Write a function in `tools.py` that performs the action and returns a string result.
2. Add a schema for it to `TOOL_SCHEMAS` describing its name, purpose, and parameters.
3. Add a branch for it in `execute_tool()`.

Claude will automatically start using new tools when they're relevant - no changes needed
in `brain.py`.

## Notes

- Voice mode now waits silently for the wake word "Hey Jarvis" (via `openWakeWord`'s built-in
  pretrained model) before it starts recording your command. It replies "Yes, sir?" then
  records a 5-second command clip. To change the wake phrase to a fully custom one later,
  you'd train a custom openWakeWord model - the pretrained "hey_jarvis" is used for now.
- Whisper runs locally on CPU using the `base.en` model. Swap to `small.en` or `medium.en` in
  `voice_io.py` for better accuracy if you have a decent CPU or a GPU.
- `open_application` uses Windows' `os.startfile`, so this is Windows-specific as built.
