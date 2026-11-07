
import os
from groq import Groq


def generate_legal_document(
    case_analysis,
    document_type,
    language="Auto-detect",
    legal_position=None,
    additional_instructions=""
):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

    document_instructions = {
        "Legal Notice": """
Prepare a formal legal notice draft.

Include:
- Sender and recipient placeholders
- Subject
- Background and facts
- Relevant demands or requested action
- Reasonable response-period placeholder
- Consequences of non-compliance, phrased cautiously
- Signature and contact placeholders

Do not invent a statutory deadline or claim that a specific
notice period is legally mandatory unless verified.
""",

        "Consumer Complaint": """
Prepare a consumer complaint draft.

Include:
- Appropriate consumer forum placeholder
- Complainant and opposite party details
- Facts in chronological order
- Nature of grievance
- Evidence and supporting documents
- Relief requested
- Verification and signature placeholders

Do not invent jurisdiction, case numbers, fees or statutory provisions.
""",

        "Reply to Legal Notice": """
Prepare a formal reply to a legal notice.

Include:
- Reference to the notice
- Preliminary response
- Point-wise response to allegations, where information is available
- Facts and clarifications
- Documents relied upon
- Requested action or position
- Signature and contact placeholders

Do not admit or deny facts that the user has not clarified.
""",

        "Application / Representation": """
Prepare a formal application or representation.

Include:
- Recipient authority placeholder
- Subject
- Applicant details
- Facts and background
- Grounds for the request
- Specific relief requested
- List of supporting documents
- Date, place and signature placeholders

Do not invent procedural requirements or statutory provisions.
"""
    }

    if language == "English":
        language_instruction = "Write the document in English."
    elif language == "हिंदी":
        language_instruction = "दस्तावेज़ हिंदी में तैयार करें।"
    else:
        language_instruction = (
            "Write in the same language as the case information."
        )

    case_context = f"""
CASE SUMMARY:
{case_analysis.get('case_summary', '')}

PARTIES:
{case_analysis.get('parties_involved', [])}

TIMELINE:
{case_analysis.get('timeline', [])}

KEY FACTS:
{case_analysis.get('key_facts', [])}

DISPUTED FACTS:
{case_analysis.get('disputed_facts', [])}

AVAILABLE EVIDENCE:
{case_analysis.get('evidence_available', [])}

MISSING INFORMATION:
{case_analysis.get('missing_information', [])}
"""

    legal_context = ""

    if legal_position:
        legal_context = legal_position.get("answer", "")

    system_prompt = f"""
You are NyayaMitra's Legal Document Drafting Assistant.

Prepare a clear, professional draft based on the supplied
case information.

DOCUMENT TYPE:
{document_type}

DOCUMENT REQUIREMENTS:
{document_instructions[document_type]}

CASE INFORMATION:
{case_context}

LEGAL POSITION ALREADY EXPLAINED:
{legal_context}

STRICT RULES:

1. Never invent facts, dates, addresses, names or evidence.
2. Use [Insert ...] placeholders for missing details.
3. Do not treat disputed allegations as established facts.
4. Do not invent Acts, sections, judgments or legal citations.
5. Use legal provisions only if supported by the supplied
   legal-position information.
6. If legal support is uncertain, do not present it as verified.
7. Do not guarantee any legal outcome.
8. Keep the document formal, clear and logically structured.
9. Do not claim that this draft has been reviewed by a lawyer.
10. Do not include unnecessary personal information.

Additional user instructions:
{additional_instructions or "None"}

{language_instruction}

Return only the document draft.
Begin with:

DRAFT — FOR REVIEW
"""

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    "Prepare the requested document using "
                    "the supplied case information."
                )
            }
        ],
        temperature=0.2,
        max_tokens=2500
    )

    return (
        response.choices[0].message.content
        or "Document generation failed."
    ).strip()
