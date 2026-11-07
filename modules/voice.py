import os
import tempfile
from groq import Groq

def transcribe_audio(audio_file):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Configure it before using voice input.")
    client = Groq(api_key=key)
    suffix = ".wav"
    if getattr(audio_file, "name", None):
        _, ext = os.path.splitext(audio_file.name)
        if ext:
            suffix = ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getvalue())
        path = tmp.name
    try:
        with open(path, "rb") as audio:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(path), audio.read()),
                model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3"),
                response_format="text",
            )
        return str(result).strip()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
