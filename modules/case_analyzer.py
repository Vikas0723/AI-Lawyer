
import os
import json

from groq import Groq


def parse_case_json(content, client, model):

    content = content.strip()

    # Remove Markdown code fences
    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "").strip()

    # Try parsing the original response
    try:
        return json.loads(content)

    except json.JSONDecodeError:

        # Ask Groq to repair malformed JSON
        repair_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the supplied text into valid JSON. "
                        "Preserve all information. Do not add facts. "
                        "Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0,
            max_tokens=2500
        )

        repaired = (
            repair_response.choices[0].message.content or ""
        ).strip()

        if repaired.startswith("```"):
            repaired = repaired.replace("```json", "", 1)
            repaired = repaired.replace("```", "").strip()

        return json.loads(repaired)


def analyze_case(narrative, language="Auto-detect"):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    if language == "English":
        language_instruction = "Respond only in English."

    elif language == "हिंदी":
        language_instruction = "उत्तर केवल हिंदी में दें।"

    else:
        language_instruction = (
            "Respond in the same language as the user's narrative."
        )

    system_prompt = f"""
You are NyayaMitra's Case Analyzer.

Your job is to convert an unstructured user story into
a structured and organized case summary.

Do not provide legal advice or invent facts.

IMPORTANT RULES:

1. Use only facts explicitly provided by the user.
2. Never assume dates, events, people or evidence.
3. Clearly distinguish confirmed facts from disputed claims.
4. If a date is unknown, write "Not specified".
5. Identify evidence the user says they already have.
6. Suggest useful evidence they could collect.
7. Ask relevant questions where information is missing.
8. Do not invent legal provisions or citations.
9. Preserve the user's original meaning.

Return ONLY valid JSON in this format:

{{
    "case_summary": "Short summary of the situation",

    "parties_involved": [
        "Person or organization and their role"
    ],

    "timeline": [
        {{
            "date": "Date or Not specified",
            "event": "What happened"
        }}
    ],

    "key_facts": [
        "Fact explicitly stated by the user"
    ],

    "disputed_facts": [
        "Claims that appear disputed or uncertain"
    ],

    "evidence_available": [
        "Evidence the user says they possess"
    ],

    "evidence_to_collect": [
        "Potentially useful evidence to collect"
    ],

    "missing_information": [
        "Questions that need clarification"
    ]
}}

{language_instruction}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": narrative
            }
        ],
        temperature=0.2,
        max_tokens=2500
    )

    # Extract model response
    content = (
        response.choices[0].message.content or ""
    ).strip()

    # Parse JSON, repairing it if necessary
    return parse_case_json(
        content,
        client,
        model
    )
