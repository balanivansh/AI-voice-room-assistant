import re
from dataclasses import dataclass
from typing import Optional
from src.logger import get_logger

logger = get_logger("roxstar_voice_assistant.utils.guardrails")


@dataclass
class GuardrailResult:
    is_safe: bool
    violation_type: Optional[str]
    sanitized_text: str
    refusal_message: Optional[str]


# Patterns for PII Redaction
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
)
AADHAAR_PATTERN = re.compile(
    r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b"
)
PHONE_PATTERN = re.compile(
    r"(?:\+91[\s\-]?)?[6-9]\d{9}\b|\b(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b"
)

# Patterns for Jailbreak & Prompt Injection
JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"(?:reveal|display|output|show|print)\s+(?:your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bdan\s+mode\b", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"(?:override|bypass)\s+safety", re.IGNORECASE),
]

# Patterns for Toxic Speech & Profanity (English & Hindi/Hinglish in Latin and Devanagari)
PROFANITY_PATTERNS = [
    # English slurs & profanity
    re.compile(r"\b(?:bitch|bastard|fuck|fucking|fucker|shit|asshole|cunt|motherfucker|dick|pussy|retard)\b", re.IGNORECASE),
    # Hindi/Hinglish in Latin script
    re.compile(r"\b(?:bhenchod|bhinchod|bc|madarchod|mc|gandu|gaandu|chutiya|chutiye|bhosdike|bhosadike|saale|saala|harami|kamina|randi)\b", re.IGNORECASE),
    # Hindi/Hinglish in Devanagari script
    re.compile(r"(?:गांडू|गांड|चूतिया|चूतिये|भोसड़ीके|भोसडीके|मादरचोद|बहनचोद|हरामी|कमीने|साले|रंडी)"),
]


def redact_pii(text: str) -> str:
    """Mask sensitive PII (email, credit card, Aadhaar, phone numbers) into tokens."""
    if not text:
        return text

    # Redact Email
    text = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    # Redact Credit Card
    text = CREDIT_CARD_PATTERN.sub("[CREDIT_CARD_REDACTED]", text)
    # Redact Aadhaar Card
    text = AADHAAR_PATTERN.sub("[AADHAAR_REDACTED]", text)
    # Redact Phone Number
    text = PHONE_PATTERN.sub("[PHONE_REDACTED]", text)

    return text


def apply_guardrails(text: str) -> GuardrailResult:
    """Evaluate input text against safety policies (PII, Jailbreak, Profanity).

    Args:
        text: Input utterance string.

    Returns:
        GuardrailResult containing safety status, violation type, sanitized text, and refusal message.
    """
    if not text:
        return GuardrailResult(
            is_safe=True,
            violation_type=None,
            sanitized_text="",
            refusal_message=None,
        )

    # 1. Check for Jailbreak & Prompt Injection
    for pattern in JAILBREAK_PATTERNS:
        if pattern.search(text):
            logger.warning("guardrail_jailbreak_detected", text=text)
            return GuardrailResult(
                is_safe=False,
                violation_type="jailbreak",
                sanitized_text=text,
                refusal_message="Kripya karke safety rules aur system prompts bypass karne ki koshish na karein. Main is request ka uttar nahi de sakti.",
            )

    # 2. Check for Toxic Speech & Profanity
    for pattern in PROFANITY_PATTERNS:
        if pattern.search(text):
            logger.warning("guardrail_profanity_detected", text=text)
            return GuardrailResult(
                is_safe=False,
                violation_type="profanity",
                sanitized_text=text,
                refusal_message="Aapki bhasha abhadra hai. Kripya shisht bhasha ka prayog karein, taaki hum aage baat kar sakein.",
            )

    # 3. Apply PII Redaction
    sanitized_text = redact_pii(text)

    return GuardrailResult(
        is_safe=True,
        violation_type=None,
        sanitized_text=sanitized_text,
        refusal_message=None,
    )
