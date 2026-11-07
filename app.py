import streamlit as st
import streamlit.components.v1 as components
import json
import os
from dotenv import load_dotenv

from modules.assistant import answer_question
from modules.rag import (
    build_index_from_uploaded_files,
    load_index,
    retrieve_debug
)
from modules.voice import transcribe_audio
from modules.case_analyzer import analyze_case
from modules.document_generator import generate_legal_document

load_dotenv()

# Load Streamlit Community Cloud secrets into environment variables.
# Local development continues to use the project's .env file.
try:
    for secret_name in ("GROQ_API_KEY", "GROQ_MODEL"):
        if secret_name in st.secrets and not os.getenv(secret_name):
            os.environ[secret_name] = str(st.secrets[secret_name])
except Exception:
    # st.secrets may be unavailable during local development.
    pass

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NyayaMitra | AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

if "generated_document" not in st.session_state:
    st.session_state.generated_document = None

if "active_feature" not in st.session_state:
    st.session_state.active_feature = None

if "legal_position" not in st.session_state:
    st.session_state.legal_position = None

if "case_analysis" not in st.session_state:
    st.session_state.case_analysis = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = True

if "legal_index" not in st.session_state:
    try:
        index, chunks = load_index()
        st.session_state.legal_index = index
        st.session_state.legal_chunks = chunks

    except Exception as exc:
        st.session_state.legal_index = None
        st.session_state.legal_chunks = []
        st.error(f"Could not load legal knowledge base: {exc}")

if "legal_chunks" not in st.session_state:
    st.session_state.legal_chunks = []

# --------------------------------------------------
# CUSTOM CSS  (fits one browser window, no page scroll)
# --------------------------------------------------

st.markdown("""
<style>
    :root {
        --ink: #12263f;
        --teal: #167d8d;
        --teal-soft: rgba(22,125,141,.10);
        --brass: #b8893b;
        --line: rgba(128,128,128,.22);
        --surface: rgba(128,128,128,.06);
        --muted: #7a8594;
    }

    /* ---------- Frame: one window, no page scroll ---------- */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
    }
    /* Safety net: if a window is ever too short, the page scrolls
       instead of hiding the message bar. */
    [data-testid="stMain"] {
        overflow-y: auto !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        height: 2.2rem;
    }
    [data-testid="stToolbar"], [data-testid="stDecoration"], footer {
        display: none !important;
    }
    .block-container {
        max-width: 1180px;
        padding: 2.2rem 1.4rem .4rem 1.4rem;
    }
    div[data-testid="stVerticalBlock"] { gap: .55rem; }

    /* ---------- Hide Streamlit sidebar completely ---------- */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* ---------- Top bar ---------- */
    .st-key-topbar {
        background: linear-gradient(110deg, #0f2238 0%, #174b70 65%, #167d8d 100%);
        border-radius: 16px;
        padding: .55rem .9rem;
        box-shadow: 0 8px 24px rgba(16,36,62,.14);
    }
    .brand { line-height: 1.15; }
    .brand b {
        color: #fff;
        font-size: 1.28rem;
        font-weight: 760;
        letter-spacing: -.5px;
    }
    .brand span {
        display: block;
        color: #cfe6ef;
        font-size: .76rem;
        margin-top: 1px;
    }
    .st-key-topbar button {
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.28);
        color: #fff;
        border-radius: 10px;
        min-height: 40px;
        font-weight: 600;
    }
    .st-key-topbar button:hover {
        background: rgba(255,255,255,.22);
        border-color: #fff;
        color: #fff;
    }
    .st-key-topbar button p { color: #fff; }

    /* ---------- Navigation pills ---------- */
    .st-key-navbar button {
        border-radius: 999px;
        min-height: 38px;
        font-weight: 600;
        font-size: .86rem;
        border: 1px solid var(--line);
        background: transparent;
        transition: border-color .15s ease, background .15s ease;
    }
    .st-key-navbar button:hover {
        border-color: var(--teal);
        color: var(--teal);
    }
    .st-key-navbar button[kind="primary"] {
        background: var(--ink);
        border-color: var(--ink);
        color: #fff;
    }
    .st-key-navbar button[kind="primary"] p { color: #fff; }

    /* ---------- Scroll regions ---------- */
    .st-key-chat_area {
        height: calc(100vh - 292px) !important;
        max-height: calc(100vh - 292px) !important;
        min-height: 240px;
        flex: 0 0 auto !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: .2rem .6rem .2rem 0;
    }
    .st-key-workspace_area {
        height: calc(100vh - 210px) !important;
        max-height: calc(100vh - 210px) !important;
        min-height: 300px;
        flex: 0 0 auto !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: .2rem .6rem .2rem 0;
    }
    /* Children must keep their natural height and scroll, not squash */
    .st-key-chat_area > *,
    .st-key-workspace_area > * { flex-shrink: 0 !important; }

    /* Message bar always pinned at the bottom of the window */
    .st-key-input_row {
        position: sticky;
        bottom: 0;
        z-index: 50;
        padding-bottom: .15rem;
        background: var(--background-color, inherit);
    }

    /* Always-visible, slim scrollbars */
    .st-key-chat_area, .st-key-workspace_area {
        scrollbar-width: thin;
        scrollbar-color: rgba(22,125,141,.55) transparent;
    }
    .st-key-chat_area::-webkit-scrollbar,
    .st-key-workspace_area::-webkit-scrollbar { width: 8px; }
    .st-key-chat_area::-webkit-scrollbar-thumb,
    .st-key-workspace_area::-webkit-scrollbar-thumb {
        background: rgba(22,125,141,.55);
        border-radius: 8px;
    }
    .st-key-chat_area::-webkit-scrollbar-track,
    .st-key-workspace_area::-webkit-scrollbar-track { background: transparent; }

    /* ---------- Chat messages ---------- */
    div[data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 4px;
        background: transparent;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--teal-soft);
        border-color: rgba(22,125,141,.28);
    }

    /* ---------- Welcome / empty state ---------- */
    .welcome { padding: 1.1rem .2rem .4rem .2rem; }
    .welcome h2 {
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: -.6px;
        margin: 0 0 .25rem 0;
        padding: 0;
    }
    .welcome p { color: var(--muted); font-size: .92rem; margin: 0; }
    .st-key-suggestions button {
        text-align: left;
        justify-content: flex-start;
        min-height: 62px;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: var(--surface);
        font-size: .85rem;
        padding: 10px 14px;
        transition: border-color .15s ease, background .15s ease;
    }
    .st-key-suggestions button:hover {
        border-color: var(--teal);
        background: var(--teal-soft);
        color: inherit;
    }

    /* ---------- Message bar ---------- */
    .st-key-input_row div[data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: .25rem .4rem .25rem 1.1rem;
        background: var(--surface);
    }
    .st-key-input_row div[data-testid="stForm"]:focus-within {
        border-color: var(--teal);
        box-shadow: 0 0 0 3px var(--teal-soft);
    }
    .st-key-input_row div[data-baseweb="input"],
    .st-key-input_row div[data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
    }
    .st-key-input_row input { font-size: .98rem; }
    .st-key-input_row div[data-testid="stForm"] button {
        border-radius: 50%;
        width: 42px;
        height: 42px;
        min-height: 42px;
        padding: 0;
        background: var(--teal);
        border: none;
        color: #fff;
        font-size: 1.05rem;
    }
    .st-key-input_row div[data-testid="stForm"] button:hover { background: var(--ink); }
    .st-key-input_row div[data-testid="stPopover"] button {
        border-radius: 50%;
        width: 50px;
        height: 50px;
        min-height: 50px;
        padding: 0;
        border: 1px solid var(--line);
        background: var(--surface);
        font-size: 1.15rem;
    }
    .st-key-input_row div[data-testid="stPopover"] button:hover {
        border-color: var(--brass);
        background: rgba(184,137,59,.12);
    }
    .fine-print {
        color: var(--muted);
        text-align: center;
        font-size: .72rem;
        margin-top: -.2rem;
    }

    /* ---------- Misc surfaces ---------- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    div.stButton > button { border-radius: 10px; font-weight: 600; }
    div[data-testid="stTextArea"] textarea { border-radius: 10px; line-height: 1.5; }
    div[data-baseweb="tab-list"] { gap: .3rem; }
    h3 { font-size: 1.15rem !important; padding-top: .2rem !important; }

    /* hidden helper iframe (auto-scroll script) */
    div[data-testid="stElementContainer"]:has(iframe[height="0"]) {
        position: absolute;
        height: 0;
        overflow: hidden;
    }

    @media (max-width: 700px) {
        .block-container { padding-left: .7rem; padding-right: .7rem; }
        .brand span { display: none; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SOURCE DISPLAY FUNCTION
# --------------------------------------------------

def display_sources(sources):

    if not sources:
        return

    with st.expander("Sources used"):

        for src in sources:

            metadata = src.get("metadata", src)

            title = metadata.get("title", "Legal document")
            page = metadata.get("page", "N/A")
            source = metadata.get("source", "")

            st.markdown(
                f"- **{title}** — Page {page}"
                + (f" · {source}" if source and source != title else "")
            )


# --------------------------------------------------
# RETRIEVAL DEBUGGER DISPLAY FUNCTION
# --------------------------------------------------

def display_retrieval_debug(results):

    if not results:
        return

    with st.expander(
        f"Retrieval Debugger — Top {len(results)} Passages"
    ):

        st.caption(
            "Similarity scores are cosine similarity values, "
            "not confidence probabilities."
        )

        for item in results:

            metadata = item.get("metadata", {})

            rank = item.get("rank", "-")
            score = item.get("score", 0.0)
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "N/A")
            chunk = metadata.get("chunk", "N/A")

            st.markdown(
                f"**Rank #{rank} | Similarity: {score:.4f}**"
            )

            st.caption(
                f"Source: {source} | Page: {page} | Chunk: {chunk}"
            )

            st.text(item.get("text", ""))

            st.divider()


# --------------------------------------------------
# TOP BAR: Brand · Language · Settings
# --------------------------------------------------

with st.container(key="topbar"):
    brand_col, lang_col, set_col = st.columns(
        [5.5, 1.5, .65],
        vertical_alignment="center"
    )

    with brand_col:
        st.markdown(
            '<div class="brand"><b>⚖️ NyayaMitra</b>'
            '<span>Understand your rights. Organize your case. '
            'Prepare your next step.</span></div>',
            unsafe_allow_html=True
        )

    with lang_col:
        language = st.selectbox(
            "Response language",
            ["Auto-detect", "English", "हिंदी"],
            key="response_language",
            label_visibility="collapsed"
        )

    with set_col:
        with st.popover("⚙️", help="Settings", use_container_width=True):
            st.checkbox(
                "Enable Retrieval Debugger",
                key="debug_mode",
                help="Show the top five retrieved passages for legal questions."
            )
            st.divider()
            st.markdown("**Legal knowledge base**")
            total_chunks = len(st.session_state.legal_chunks)
            if st.session_state.legal_index is not None:
                st.success(f"Knowledge base ready: {total_chunks} sections")
            else:
                st.warning("No knowledge base loaded yet.")

            uploaded_docs = st.file_uploader(
                "Upload additional PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                key="additional_legal_docs",
                help="Uploaded documents are used only in this session."
            )
            if st.button("Use uploaded documents", key="index_uploaded_docs", use_container_width=True):
                if not uploaded_docs:
                    st.warning("Please upload at least one document.")
                else:
                    with st.spinner("Reading and indexing documents..."):
                        try:
                            index, chunks = build_index_from_uploaded_files(uploaded_docs)
                            st.session_state.legal_index = index
                            st.session_state.legal_chunks = chunks
                            st.success(f"Successfully indexed {len(chunks)} sections.")
                        except Exception as exc:
                            st.error(f"Could not index documents: {exc}")
            st.caption("Permanent documents: add them to knowledge_base/ and run ingest.py.")

# --------------------------------------------------
# NAVIGATION PILLS
# --------------------------------------------------

nav_items = [
    (None, "💬 Chat"),
    ("Build My Case", "🗂️ Build My Case"),
    ("Understand My Legal Position", "⚖️ Legal Position"),
    ("Prepare Legal Documents", "📄 Legal Documents"),
]

with st.container(key="navbar"):
    nav_cols = st.columns(len(nav_items), gap="small")
    for i, (col, (feature_name, label)) in enumerate(zip(nav_cols, nav_items)):
        with col:
            is_active = st.session_state.active_feature == feature_name
            if st.button(
                label,
                key=f"nav_{i}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                if not is_active:
                    st.session_state.active_feature = feature_name
                    st.rerun()

# ==================================================
# WORKSPACES (Build Case / Legal Position / Documents)
# ==================================================

if st.session_state.active_feature:

    with st.container(key="workspace_area"):

        # --------------------------------------------------
        # BUILD MY CASE
        # --------------------------------------------------

        if st.session_state.active_feature == "Build My Case":

            input_col, output_col = st.columns([1, 1.5], gap="large")

            with input_col:

                st.markdown("### Organize your legal situation")

                st.write(
                    "Describe what happened in your own words. "
                    "NyayaMitra will organize your story into facts, "
                    "a timeline and an evidence checklist."
                )

                case_narrative = st.text_area(
                    "Describe your situation",
                    placeholder=(
                        "Explain what happened, when it happened, "
                        "who was involved and what proof you have..."
                    ),
                    height=200,
                    key="case_narrative"
                )

                if st.button("Analyze My Case", use_container_width=True):

                    if not case_narrative.strip():

                        st.warning("Please describe your situation first.")

                    else:

                        try:

                            with st.spinner("Organizing your case..."):

                                st.session_state.case_analysis = analyze_case(
                                    case_narrative,
                                    language=language
                                )
                                st.session_state.legal_position = None
                                st.session_state.generated_document = None

                        except Exception as exc:

                            st.error(f"Case analysis failed: {exc}")

            # DISPLAY CASE ANALYSIS

            with output_col:

                if st.session_state.get("case_analysis"):

                    result = st.session_state.case_analysis

                    tab_summary, tab_timeline, tab_facts, tab_evidence, tab_gaps = st.tabs(
                        ["Summary", "Timeline", "Facts", "Evidence", "Gaps"]
                    )

                    with tab_summary:

                        st.markdown("### Case Summary")
                        st.write(result.get("case_summary", ""))

                        st.markdown("### People Involved")

                        for item in result.get("parties_involved", []):
                            st.markdown(f"- {item}")

                    with tab_timeline:

                        st.markdown("### Timeline")

                        for event in result.get("timeline", []):

                            st.markdown(
                                f"**{event.get('date', 'Not specified')}**"
                            )
                            st.write(event.get("event", ""))

                    with tab_facts:

                        st.markdown("### Key Facts")

                        for item in result.get("key_facts", []):
                            st.markdown(f"- {item}")

                        st.markdown("### Disputed Facts")

                        for item in result.get("disputed_facts", []):
                            st.markdown(f"- {item}")

                    with tab_evidence:

                        st.markdown("### Evidence You Have")

                        for item in result.get("evidence_available", []):
                            st.markdown(f"- {item}")

                        st.markdown("### Evidence You Should Collect")

                        for item in result.get("evidence_to_collect", []):
                            st.markdown(f"- {item}")

                    with tab_gaps:

                        st.markdown("### Missing Information")

                        for item in result.get("missing_information", []):
                            st.markdown(f"- {item}")

                    st.caption(
                        "This is an AI-generated organization of your story. "
                        "Verify all facts before using it in legal proceedings."
                    )

                else:

                    st.info(
                        "Your organized case summary will appear here "
                        "after you click Analyze My Case."
                    )


        # --------------------------------------------------
        # EXPLAIN MY LEGAL POSITION
        # --------------------------------------------------

        if st.session_state.active_feature == "Understand My Legal Position":

            st.markdown("### Understand your rights and options")

            st.write(
                "Use your analyzed case to understand the relevant "
                "legal position, possible remedies and practical next steps."
            )

            if not st.session_state.get("case_analysis"):

                st.info(
                    "First use Build My Case to organize your situation."
                )

            else:

                if st.button(
                    "Explain My Legal Position",
                    use_container_width=True
                ):

                    case = st.session_state.case_analysis

                    # Convert structured case analysis into context
                    case_context = f"""
Analyze the following user's legal situation under Indian law.

CASE SUMMARY:
{case.get('case_summary', '')}

PARTIES INVOLVED:
{json.dumps(case.get('parties_involved', []), ensure_ascii=False)}

TIMELINE:
{json.dumps(case.get('timeline', []), ensure_ascii=False)}

KEY FACTS:
{json.dumps(case.get('key_facts', []), ensure_ascii=False)}

DISPUTED FACTS:
{json.dumps(case.get('disputed_facts', []), ensure_ascii=False)}

EVIDENCE AVAILABLE:
{json.dumps(case.get('evidence_available', []), ensure_ascii=False)}

MISSING INFORMATION:
{json.dumps(case.get('missing_information', []), ensure_ascii=False)}

Please explain:

1. The possible legal issues involved.
2. The relevant rights and obligations.
3. Possible legal remedies and available options.
4. Practical next steps.
5. Any important facts that need clarification.

IMPORTANT:
- Use retrieved legal documents for specific statutory claims.
- Do not invent sections, Acts, deadlines or citations.
- Clearly distinguish established facts from allegations.
- If the knowledge base does not support a legal claim,
  explicitly state that it could not be verified.
- Do not guarantee any legal outcome.
"""

                    try:

                        with st.spinner(
                            "Checking relevant legal provisions..."
                        ):

                            result = answer_question(
                                question=case_context,
                                history=None,
                                language=language,
                                index=st.session_state.legal_index,
                                chunks=st.session_state.legal_chunks
                            )

                            st.session_state.legal_position = result

                    except Exception as exc:

                        st.error(
                            f"Legal analysis failed: {exc}"
                        )

                # DISPLAY LEGAL POSITION

                if st.session_state.get("legal_position"):

                    result = st.session_state.legal_position

                    st.divider()

                    st.markdown("### Your Legal Position")

                    st.markdown(result.get("answer", ""))

                    st.caption(
                        "Answer mode: " + result.get("mode", "Legal analysis")
                    )

                    display_sources(result.get("sources", []))

                    st.caption(
                        "This is AI-generated legal information, "
                        "not a substitute for advice from a qualified lawyer."
                    )


        # --------------------------------------------------
        # PREPARE LEGAL DOCUMENTS
        # --------------------------------------------------

        if st.session_state.active_feature == "Prepare Legal Documents":

            st.markdown("### Generate your legal document")

            st.write(
                "Create a structured legal draft using the facts "
                "and information from your analyzed case."
            )

            if not st.session_state.get("case_analysis"):

                st.info("First use Build My Case to organize your situation.")

            else:

                setup_col, draft_col = st.columns([1, 1.6], gap="large")

                with setup_col:

                    document_type = st.selectbox(
                        "Select document type",
                        [
                            "Legal Notice",
                            "Consumer Complaint",
                            "Reply to Legal Notice",
                            "Application / Representation"
                        ],
                        key="document_type"
                    )

                    additional_instructions = st.text_area(
                        "Additional instructions (optional)",
                        placeholder=(
                            "For example: Include a demand for return "
                            "of my security deposit."
                        ),
                        height=140,
                        key="document_instructions"
                    )

                    if st.button(
                        "Generate Document",
                        use_container_width=True
                    ):

                        try:

                            with st.spinner("Preparing your legal document..."):

                                document = generate_legal_document(
                                    case_analysis=st.session_state.case_analysis,
                                    document_type=document_type,
                                    language=language,
                                    legal_position=st.session_state.get(
                                        "legal_position"
                                    ),
                                    additional_instructions=additional_instructions
                                )

                                st.session_state.generated_document = {
                                    "type": document_type,
                                    "content": document
                                }

                        except Exception as exc:

                            st.error(f"Document generation failed: {exc}")

                # DISPLAY GENERATED DOCUMENT

                with draft_col:

                    if st.session_state.get("generated_document"):

                        generated = st.session_state.generated_document

                        st.markdown(
                            f"### {generated['type']} — Draft"
                        )

                        edited_document = st.text_area(
                            "Review and edit your document",
                            value=generated["content"],
                            height=330,
                            key="edited_legal_document"
                        )

                        st.download_button(
                            label="Download Document",
                            data=edited_document,
                            file_name="NyayaMitra_Legal_Draft.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                        st.caption(
                            "This is an AI-generated draft. Verify all facts, "
                            "legal provisions and procedural requirements. "
                            "Have it reviewed by a qualified lawyer before use."
                        )

                    else:

                        st.info(
                            "Your editable draft will appear here "
                            "after you click Generate Document."
                        )

# ==================================================
# CHAT VIEW
# ==================================================

else:

    prompt = None

    # --------------------------------------------------
    # CHAT HISTORY + LIVE RESPONSES (scrolls inside its own panel)
    # --------------------------------------------------

    chat_area = st.container(key="chat_area")

    with chat_area:

        if not st.session_state.messages:

            st.markdown(
                '<div class="welcome">'
                '<h2>How can I help with your legal question?</h2>'
                '<p>Ask in English or हिंदी. I will use legal sources '
                'whenever they are needed.</p>'
                '</div>',
                unsafe_allow_html=True
            )

            suggestions = [
                "What are my rights as a tenant if my landlord refuses to return my deposit?",
                "How do I file a consumer complaint against a seller?",
                "What is a legal notice and when should I send one?",
                "How can I file an FIR, and what if the police refuse?",
            ]

            with st.container(key="suggestions"):
                sug_cols = st.columns(2, gap="small")
                for i, text in enumerate(suggestions):
                    with sug_cols[i % 2]:
                        if st.button(text, key=f"suggestion_{i}", use_container_width=True):
                            prompt = text

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                display_sources(message.get("sources", []))

                if message.get("mode"):
                    st.caption("Answer mode: " + message["mode"])

                if (
                    message["role"] == "assistant"
                    and st.session_state.debug_mode
                ):

                    display_retrieval_debug(
                        message.get("debug_results", [])
                    )

    # --------------------------------------------------
    # MESSAGE BAR: message · send · voice · new chat
    # --------------------------------------------------

    with st.container(key="input_row"):
        form_col, mic_col, new_col = st.columns(
            [13, 1, 1.5], vertical_alignment="center", gap="small"
        )

        with form_col:
            with st.form("chat_message_form", clear_on_submit=True, border=False):
                input_col, send_col = st.columns([12, 1], vertical_alignment="center")
                with input_col:
                    typed_prompt = st.text_input(
                        "Message",
                        placeholder="Type your legal question here...",
                        label_visibility="collapsed"
                    )
                with send_col:
                    send_clicked = st.form_submit_button("➤", type="primary")

        with mic_col:
            with st.popover("🎙️", help="Ask using your voice", use_container_width=True):
                audio_file = st.audio_input("Record your question")
                if audio_file is not None:
                    if st.button("Transcribe and send", key="send_voice", use_container_width=True):
                        with st.spinner("Transcribing..."):
                            try:
                                prompt = transcribe_audio(audio_file)
                            except Exception as exc:
                                st.error(f"Voice transcription failed: {exc}")

        with new_col:
            new_chat_clicked = st.button(
                "＋ New Chat",
                key="new_chat_input_btn",
                use_container_width=True,
                help="Clear the current chat and case workspace"
            )

    if new_chat_clicked:
        st.session_state.messages = []
        st.session_state.active_feature = None
        st.session_state.case_analysis = None
        st.session_state.legal_position = None
        st.session_state.generated_document = None
        st.session_state.pop("edited_legal_document", None)
        st.session_state.pop("case_narrative", None)
        st.session_state.pop("document_instructions", None)
        st.rerun()

    st.markdown(
        '<div class="fine-print">General legal information, not legal advice. '
        'AI-generated answers may contain errors — consult a qualified lawyer '
        'before acting.</div>',
        unsafe_allow_html=True
    )

    if send_clicked and typed_prompt.strip():
        prompt = typed_prompt

    # --------------------------------------------------
    # PROCESS USER QUESTION
    # --------------------------------------------------

    if prompt:

        prompt = prompt.strip()

        if prompt:

            # Store user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            # Render new messages inside the chat panel
            with chat_area:

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):

                    with st.spinner("Thinking..."):

                        try:

                            result = answer_question(
                                question=prompt,
                                history=st.session_state.messages[:-1],
                                language=language,
                                index=st.session_state.legal_index,
                                chunks=st.session_state.legal_chunks
                            )

                            answer = result["answer"]
                            sources = result.get("sources", [])
                            mode = result.get("mode", "General")

                            # RETRIEVAL DEBUGGING
                            debug_results = []

                            if (
                                st.session_state.debug_mode
                                and sources
                            ):

                                try:

                                    debug_results = retrieve_debug(
                                        query=prompt,
                                        index=st.session_state.legal_index,
                                        chunks=st.session_state.legal_chunks,
                                        top_k=5
                                    )

                                except Exception as debug_error:

                                    st.warning(
                                        f"Retrieval debugging failed: {debug_error}"
                                    )

                            # DISPLAY RESPONSE
                            st.markdown(answer)

                            display_sources(sources)

                            st.caption("Answer mode: " + mode)

                            if st.session_state.debug_mode:
                                display_retrieval_debug(debug_results)

                            # SAVE ASSISTANT RESPONSE
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "mode": mode,
                                "debug_results": debug_results
                            })

                        except Exception as exc:

                            error = str(exc)

                            st.error(f"Something went wrong: {error}")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error
                            })

            # Refresh conversation after response
            st.rerun()

    # Keep the newest message in view inside the chat panel
    components.html(
        """
        <script>
            const p = window.parent;
            const doc = p.document;

            // Keep the newest message in view
            const panel = doc.querySelector('.st-key-chat_area');
            if (panel) { panel.scrollTop = panel.scrollHeight; }

            // Add a scroll-to-latest control.
            // Installed once in the parent page so it survives reruns.
            if (!p.__nmAutoRecord) {
                p.__nmAutoRecord = true;
                const code = `
                    const sb = document.createElement('button');
                    sb.id = 'nm-scroll-btn';
                    sb.textContent = '↓';
                    sb.title = 'Scroll to latest message';
                    sb.style.cssText = 'position:fixed;right:2.2rem;bottom:125px;width:40px;height:40px;'
                        + 'border-radius:50%;border:none;background:#167d8d;color:#fff;font-size:1.2rem;'
                        + 'cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.25);display:none;z-index:9999;';
                    sb.onclick = () => {
                        const pn = document.querySelector('.st-key-chat_area');
                        if (pn) pn.scrollTo({ top: pn.scrollHeight, behavior: 'smooth' });
                    };
                    document.body.appendChild(sb);
                    setInterval(() => {
                        const pn = document.querySelector('.st-key-chat_area');
                        sb.style.display = (pn && pn.scrollHeight - pn.scrollTop - pn.clientHeight > 120)
                            ? 'block' : 'none';
                    }, 300);

                `;
                const s = doc.createElement('script');
                s.textContent = code;
                doc.body.appendChild(s);
            }
        </script>
        """,
        height=0
    )