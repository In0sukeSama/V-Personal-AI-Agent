"""
JARVIS - main loop.

Two modes:
  python main.py           -> voice mode (mic in, ElevenLabs voice out)
  python main.py --text     -> text mode (type in terminal, text out) - good for
                                testing the brain + tools without mic/voice setup
"""

import sys
from dotenv import load_dotenv
from brain import JarvisBrain

load_dotenv()  # reads .env file into environment variables


def text_mode():
    print("V text mode. Type 'quit' to exit.\n")
    v = JarvisBrain()
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("quit", "exit"):
            print("V: Signing off. Stay sharp out there.")
            break
        if not user_text:
            continue
        reply = v.ask(user_text)
        print(f"V: {reply}\n")


def voice_mode():
    import time
    from voice_io import listen, speak, wait_for_wake_word

    LISTEN_WINDOW_SECONDS = 30  # stay active this long after each reply before re-requiring the wake word

    print("V voice mode. Say 'Hey Jarvis' to activate (placeholder trigger for now - V answers after). Ctrl+C to force stop.\n")
    v = JarvisBrain()
    while True:
        wait_for_wake_word()
        speak("Yeah, I'm here.")

        # Stay in active listening mode for a while after each exchange, so
        # the user doesn't have to repeat the wake word for follow-up commands.
        last_activity = time.time()
        while time.time() - last_activity < LISTEN_WINDOW_SECONDS:
            user_text = listen()
            if not user_text:
                continue  # silence - keep waiting within the window, don't reset the timer
            last_activity = time.time()
            if any(w in user_text.lower() for w in ("quit", "goodbye", "exit")):
                speak("Signing off. Stay sharp out there.")
                return
            reply = v.ask(user_text)
            print(f"V: {reply}\n")
            speak(reply)
        print("Listening window closed - say 'Hey Jarvis' to wake me again.\n")


if __name__ == "__main__":
    if "--text" in sys.argv:
        text_mode()
    else:
        voice_mode()
