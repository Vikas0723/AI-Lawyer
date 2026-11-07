
import os
import re
import json

import streamlit as st
from groq import Groq
from sentence_transformers import util

from modules.rag import retrieve, embedding_model


# ==================================================
# 1. LEGAL KEYWORDS
# ==================================================

LEGAL_TERMS = [
    "law", "legal", "lawyer", "court", "police", "fir",
    "notice", "act", "section", "rights", "landlord",
    "tenant", "rent", "security deposit", "salary",
    "wages", "employer", "termination", "consumer",
    "refund", "complaint", "eviction", "agreement",
    "contract", "harassment", "divorce", "property",
    "cheating", "fraud", "cybercrime", "bail",
    "arrest", "accused", "warrant", "punishment",
    "theft", "robbery", "murder", "assault",
    "bns", "bnss", "ipc", "crpc", "constitution",
    "fundamental rights", "criminal", "offence",
    "offense", "prosecution", "judgment", "judge",
    "court order", "legal action",

    "बिना वेतन", "कानून", "कानूनी", "वकील",
    "किराया", "मकान मालिक", "वेतन", "तनख्वाह",
    "रिफंड", "शिकायत", "अधिकार", "नोटिस",
    "धारा", "अदालत", "पुलिस", "उपभोक्ता",
    "संविधान", "एफआईआर", "संपत्ति", "तलाक",
    "साइबर अपराध", "जमानत", "गिरफ्तारी",
    "आरोपी", "चोरी", "डकैती", "हत्या",
    "सजा", "अपराध", "न्यायालय", "मुकदमा"
]


# ==================================================
# 2. HYBRID LEGAL QUERY CLASSIFIER
# ==================================================

LEGAL_EXAMPLES = [
    "What are my legal rights?",
    "My landlord is refusing to return my deposit.",
    "My employer has not paid my salary.",
    "Someone stole my mobile phone.",
    "How do I file an FIR?",
    "Can I file a consumer complaint?",
    "What punishment is prescribed for theft?",
    "Explain Section 303 of BNS.",
    "My property is being illegally occupied.",
    "What are my fundamental rights?",
    "Can an accused person get bail?",
    "Can police arrest someone without a warrant?",
    "मेरा मकान मालिक मेरा पैसा वापस नहीं कर रहा है।",
    "मेरे नियोक्ता ने वेतन नहीं दिया है।",
    "मेरे मोबाइल की चोरी हो गई है।",
    "एफआईआर कैसे दर्ज कराएं?",
    "उपभोक्ता शिकायत कैसे दर्ज करें?",
    "भारतीय न्याय संहिता में चोरी की सजा क्या है?",
    "क्या आरोपी को जमानत मिल सकती है?"
]

GENERAL_EXAMPLES = [
    "Explain artificial intelligence.",
    "What is machine learning?",
    "Tell me about the history of India.",
    "How does a computer work?",
    "Explain photosynthesis.",
    "What is the weather today?",
    "How can I improve my programming skills?",
    "मुझे पायथन प्रोग्रामिंग समझाओ।",
    "आर्टिफिशियल इंटेलिजेंस क्या है?",
    "भारत का इतिहास बताओ।"
]


@st.cache_resource
def get_classifier_embeddings():

    model = embedding_model()

    legal_vectors = model.encode(
        LEGAL_EXAMPLES,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    general_vectors = model.encode(
        GENERAL_EXAMPLES,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return legal_vectors, general_vectors


def keyword_classifier(question):

    q = question.lower()

    if any(term in q for term in LEGAL_TERMS):
        return "legal"

    return None


def semantic_classifier(question):

    legal_vectors, general_vectors = get_classifier_embeddings()

    model = embedding_model()

    query_vector = model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    legal_score = float(
        util.cos_sim(query_vector, legal_vectors).max()
    )

    general_score = float(
        util.cos_sim(query_vector, general_vectors).max()
    )

    if legal_score >= 0.55 and legal_score > general_score:
        return "legal"

    if general_score >= 0.55 and general_score > legal_score:
        return "general"

    return None


# ==================================================
# 3. GROQ CLIENT
# ==================================================

def _client():

    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return Groq(api_key=key)


def llm_classifier(question):

    client = _client()

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {
                "role": "system",
                "content": """
Classify the user's question into exactly one category.

LEGAL:
Questions about laws, rights, crimes, police, courts,
employment disputes, rental disputes, contracts,
consumer disputes, government legal procedures,
or situations requiring legal guidance.

GENERAL:
Ordinary educational, technical, informational,
creative, or conversational questions without a
legal issue.

Return only valid JSON:
{"category": "legal"}
or
{"category": "general"}
"""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0,
        max_tokens=50
    )

    content = response.choices[0].message.content.strip()

    try:

        match = re.search(r"\{.*\}", content, re.DOTALL)

        if match:

            result = json.loads(match.group())

            category = result.get("category", "").lower()

            if category in ("legal", "general"):
                return category

    except (json.JSONDecodeError, AttributeError):
        pass

    return "legal"


def classify_question(question):

    # Layer 1: Keyword detection
    result = keyword_classifier(question)

    if result == "legal":
        return "legal"

    # Layer 2: Semantic classification
    result = semantic_classifier(question)

    if result is not None:
        return result

    # Layer 3: Groq fallback
    return llm_classifier(question)


# ==================================================
# 4. FOLLOW-UP QUESTION DETECTION
# ==================================================

FOLLOWUP_PATTERNS = [

    # English contextual references
    r"\bit\b",
    r"\bthis\b",
    r"\bthat\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\bhe\b",
    r"\bshe\b",
    r"\bhim\b",
    r"\bher\b",
    r"\bitself\b",
    r"\bthere\b",
    r"\bthose\b",
    r"\bthese\b",
    r"\bsame\b",
    r"\bagain\b",
    r"\balso\b",
    r"\binstead\b",
    r"\bprevious\b",
    r"\bearlier\b",
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\bwhat if\b",
    r"\band if\b",
    r"\bthen what\b",
    r"\bin that case\b",
    r"\bcan he\b",
    r"\bcan she\b",
    r"\bdoes it\b",
    r"\bis it\b",
    r"\bexplain more\b",
    r"\btell me more\b",
    r"\bsecond time\b",
    r"\brepeat\b",
    r"\bmore details\b",
    r"\bwhy so\b",

    # Hindi contextual references
    r"इसमें",
    r"उसमें",
    r"इसका",
    r"उसका",
    r"इसकी",
    r"उसकी",
    r"इसके",
    r"उसके",
    r"ऐसे में",
    r"फिर",
    r"दोबारा",
    r"और अगर",
    r"तो क्या",
    r"क्या ऐसा",
    r"उसी",
    r"इस मामले",
    r"इस स्थिति",
    r"इसके बाद",
    r"क्या होगा",
    r"कितनी सजा",
    r"उसे",
    r"वह व्यक्ति",
    r"उस व्यक्ति",
    r"क्या इसमें",
    r"क्या उसके",
    r"क्या उसके लिए",
    r"जमानत मिल सकती है",
    r"फिर से",
]


def needs_context_resolution(question, history):

    if not history:
        return False

    q = question.lower().strip()

    # Very short replies generally depend on previous context
    if len(q.split()) <= 3:
        return True

    # Check English and Hindi follow-up patterns
    for pattern in FOLLOWUP_PATTERNS:

        if re.search(pattern, q, re.IGNORECASE):
            return True

    return False


# ==================================================
# 5. FOLLOW-UP QUESTION RESOLVER
# ==================================================

def resolve_followup(question, history=None):

    """
    Convert context-dependent questions into standalone questions.

    Example:
    Previous: What is the punishment for theft under BNS?
    Current: What if it happens again?

    Resolved:
    What is the punishment for repeat theft under BNS?
    """

    if not history:
        return question

    if not needs_context_resolution(question, history):
        return question

    # Keep recent conversation only
    recent_history = (history or [])[-8:]

    conversation = []

    for item in recent_history:

        role = item.get("role")

        content = item.get("content", "")

        if role not in ("user", "assistant"):
            continue

        if not content:
            continue

        # Limit individual message size
        content = content[:1500]

        conversation.append({
            "role": role,
            "content": content
        })

    if not conversation:
        return question

    client = _client()

    messages = [
        {
            "role": "system",
            "content": """
You are a follow-up question resolver for NyayaMitra,
an Indian legal information assistant.

Your task is to understand the conversation and rewrite
the latest user question into a standalone question.

RULES:

1. Use previous conversation only when the latest question
   depends on it.

2. Resolve references such as:
   it, this, that, he, she, them, again, what about,
   what if, and similar Hindi expressions.

3. Preserve the original meaning and the user's intent.

4. Preserve relevant legal context, including the Act,
   offence, person, situation and jurisdiction when
   established in the conversation.

5. Do not invent facts or assume missing circumstances.

6. If the latest question is a completely new topic,
   return it unchanged.

7. If the user asks to explain the previous answer in
   another language, preserve the legal topic and
   incorporate the requested language.

8. If the context is insufficient to resolve the
   question safely, return the original question.

9. Do not answer the legal question. Only rewrite it.

10. Keep the resolved question in the same language
    as the latest user question.

Return ONLY valid JSON in this format:

{
    "standalone_question": "rewritten question"
}
"""
        }
    ]

    messages.extend(conversation)

    messages.append({
        "role": "user",
        "content": question
    })

    try:

        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=messages,
            temperature=0,
            max_tokens=200
        )

        content = (
            response.choices[0].message.content or ""
        ).strip()

        # Parse JSON response
        match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if not match:
            return question

        result = json.loads(match.group())

        resolved = result.get(
            "standalone_question",
            ""
        ).strip()

        # Validate response
        if not resolved:
            return question

        if len(resolved) > 1000:
            return question

        return resolved

    except Exception as exc:

        print("Follow-up resolution failed:", exc)

        # Never break the main assistant if resolver fails
        return question


# ==================================================
# 6. SYSTEM PROMPTS
# ==================================================

SYSTEM_GENERAL = """
You are NyayaMitra, a helpful conversational assistant.

Answer general, non-legal questions clearly and naturally.

If the question involves Indian legal rights, laws,
procedures, remedies, or interpretation, handle it
as a legal question.

Maintain conversation context when answering follow-up questions.

Match the user's language unless a response language is specified.
"""


SYSTEM_LEGAL = """
You are NyayaMitra, an Indian legal information assistant
designed for ordinary citizens.

Your task is to explain legal information in simple language.

STRICT RULES:

1. Use ONLY the supplied retrieved legal context for specific
   statutory claims, section numbers, deadlines, procedures,
   and case-law claims.

2. Never invent an Act, section, judgment, deadline, or citation.

3. Cite sources using the document name and page number
   provided in the retrieved context.

4. If the retrieved context is insufficient, clearly say:

   "I could not verify this specific legal position from
   the available knowledge base."

5. Do not pretend that a document supports a claim if it does not.

6. Distinguish general legal information from professional legal advice.

7. Ask concise follow-up questions when important facts are missing.

8. Never guarantee a legal outcome.

9. For urgent danger, advise contacting local emergency services
   or the appropriate authorities.

10. Do not request unnecessary sensitive personal information.

11. Understand the user's previous conversation when answering
    follow-up questions.

12. Do not treat previous assistant answers as verified legal evidence.
    Use the retrieved legal context for statutory claims.

Use simple language and structure the response as:

- Understanding your situation
- Relevant legal information
- Possible next steps
- Sources
"""


# ==================================================
# 7. MAIN ANSWER FUNCTION
# ==================================================

def answer_question(
    question,
    history=None,
    language="Auto-detect",
    index=None,
    chunks=None
):

    # ----------------------------------------------
    # STEP 1: RESOLVE FOLLOW-UP QUESTION
    # ----------------------------------------------

    resolved_question = resolve_followup(
        question=question,
        history=history
    )

    print("Original question:", question)
    print("Resolved question:", resolved_question)

    # ----------------------------------------------
    # STEP 2: CLASSIFY RESOLVED QUESTION
    # ----------------------------------------------

    legal = classify_question(resolved_question)

    sources = []
    context = ""

    # ----------------------------------------------
    # STEP 3: LEGAL RAG RETRIEVAL
    # ----------------------------------------------

    if legal == "legal":

        retrieved = []

        if index is not None and chunks:

            retrieved = retrieve(
                resolved_question,
                index,
                chunks,
                top_k=5
            )

        if retrieved:

            sources = [
                item["metadata"]
                for item in retrieved
            ]

            context_parts = []

            for item in retrieved:

                metadata = item["metadata"]

                context_parts.append(
                    f"""
DOCUMENT: {metadata.get('title', 'Unknown')}
PAGE: {metadata.get('page', 'Unknown')}
SOURCE: {metadata.get('source', 'Unknown')}

LEGAL TEXT:
{item['text']}
"""
                )

            context = "\n\n".join(context_parts)

        else:

            context = """
No relevant legal passages were retrieved from the
available knowledge base.

Do not provide specific statutory claims, section numbers,
deadlines, or procedures from your pretrained knowledge.

Explain that the available sources are insufficient.
"""

    # ----------------------------------------------
    # STEP 4: LANGUAGE INSTRUCTION
    # ----------------------------------------------

    if language == "English":

        lang_instruction = "Respond only in English."

    elif language == "हिंदी":

        lang_instruction = "उत्तर केवल हिंदी में दें।"

    else:

        lang_instruction = (
            "Respond in the same language used by the user."
        )

    # ----------------------------------------------
    # STEP 5: SYSTEM PROMPT
    # ----------------------------------------------

    system = (
        SYSTEM_LEGAL
        if legal == "legal"
        else SYSTEM_GENERAL
    )

    if legal == "legal":

        system += (
            "\n\nRETRIEVED LEGAL KNOWLEDGE BASE:\n"
            + context
        )

    system += "\n\n" + lang_instruction

    # ----------------------------------------------
    # STEP 6: CONVERSATION HISTORY
    # ----------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system
        }
    ]

    for item in (history or [])[-8:]:

        if item.get("role") in ("user", "assistant"):

            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

    # Preserve original user question
    messages.append({
        "role": "user",
        "content": question
    })

    # ----------------------------------------------
    # STEP 7: GROQ RESPONSE
    # ----------------------------------------------

    client = _client()

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2 if legal == "legal" else 0.6,
        max_tokens=1200
    )

    answer = (
        response.choices[0].message.content
        or "I couldn't generate a response."
    )

    # ----------------------------------------------
    # STEP 8: RETURN RESULT
    # ----------------------------------------------

    return {
        "answer": answer,

        "mode": (
            "Legal knowledge assistant (RAG)"
            if legal == "legal"
            else "General conversation"
        ),

        "sources": sources,

        "resolved_question": resolved_question
    }
