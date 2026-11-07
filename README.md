# NyayaMitra – AI Legal Assistant for India

A simple Streamlit conversational assistant with two automatic modes:
- **General conversation:** answers ordinary questions using Groq.
- **Legal assistance:** retrieves relevant text from a user-provided legal knowledge base using local Sentence Transformers + FAISS, then asks Groq to explain it in plain language.

It supports typed messages and voice transcription through Groq Whisper.

## Features
- ChatGPT-style single-page chat interface
- English/Hindi conversation (model follows the user's language)
- Automatic general/legal routing
- RAG over PDFs and TXT files
- Source labels for retrieved uploaded documents
- Voice input and transcription
- No OpenAI API, paid embeddings, or paid vector database

## 1. Requirements
- Python 3.10 or newer (Python 3.11 recommended)
- A Groq API key: https://console.groq.com/keys
- Internet access for Groq API calls and first-time embedding model download

## 2. Run locally

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your key:
```env
GROQ_API_KEY=your_actual_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_WHISPER_MODEL=whisper-large-v3
```

Run:
```bash
streamlit run app.py
```

## 3. Add legal documents
The repository intentionally does not include invented legal content. Add current official PDFs/TXT files to `knowledge_base/`, or upload documents through the sidebar during a session.

For a persistent local FAISS index:
```bash
python ingest.py
```

The ingestion script writes `data/legal.faiss` and `data/chunks.json`. The current app supports sidebar uploads for the active session; persistent-index loading can be added if desired.

Use official sources and verify current amendments, commencement, jurisdiction and applicability. India Code: https://www.indiacode.nic.in/

## 4. Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Create an app at https://share.streamlit.io/
3. Select the repository and `app.py`.
4. In Streamlit app Settings → Secrets, add:

```toml
GROQ_API_KEY = "your_actual_key"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3"
```

5. Deploy.

Do not commit `.env` or API keys to GitHub.

### Deployment notes
- Free hosting resources are limited. `sentence-transformers` and `faiss-cpu` increase installation size and startup time.
- The embedding model downloads on first use and needs outbound internet access.
- Streamlit Cloud may restart the app and clear session-uploaded documents.
- The microphone recorder requires a supported browser and HTTPS.
- Groq free-plan limits and available models can change. Check your Groq console.
- If the selected model is unavailable, set `GROQ_MODEL` to a currently supported chat model in your account.

## 5. Privacy and legal safety
- Do not upload confidential legal documents or unnecessary personal information to third-party AI services.
- Voice recordings are sent to Groq for transcription.
- Legal answers are general information, not professional legal representation.
- The assistant must not invent statutes, sections, court decisions or legal deadlines.
- The quality of legal answers depends on the accuracy and currency of documents you provide.
- If no legal knowledge base is loaded, the assistant should say that it cannot verify the legal position rather than pretending to have retrieved law.

## 6. Tests
Test scenarios are in `tests/test_cases.json`. They describe expected behavior; they are not claimed to be executed model outputs. Run the app manually with both general and legal questions and record actual outputs for your submission.
