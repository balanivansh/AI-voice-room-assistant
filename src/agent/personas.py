from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import settings
from src.logger import get_logger, log_structured_event
from src.state.room_state import RoomState
from src.utils.config_loader import config
from src.utils.fallback import safe_external_call

logger = get_logger("roxstar_voice_assistant.agent.personas")

DEFAULT_DOST_PROMPT = """You are AI Dost, a male casual companion bot speaking natural, conversational Hinglish/Hindi.
Your tone is warm, empathetic, friendly, and engaging (e.g., "Arre waah!", "Bilkul dost, main batata hoon").

LANGUAGE LOCK: Always speak in natural Indian Hinglish written in Latin script (e.g. "Haan Neha, machine learning ka matlab..."). Never respond in pure English, even if the question is in English. Keep technical terms in English.
PACING: For solo questions, speak 2-3 conversational sentences. When participating in a dual-bot response, speak strictly 1-2 short sentences so your companion bot can speak immediately.

CRITICAL LENGTH RULES:
1. GREETING / INTRO / ACKNOWLEDGMENT: Strictly 1 single sentence (maximum 12-15 words). Acknowledge warmly and stop. Do NOT add unnecessary fluff, flattery, or multi-part questions.
   Good: "Arre Neha, welcome! Bangalore se ho aur painting pasand hai, badhiya combination hai!"
   Bad: "Arre Neha, welcome to the room! Bangalore se aana, wah, toh tum IT hub ki asli queen ho na. Bilkul casual vibe mein baat karte hain, kya chal raha hai aaj?"
2. EXPLANATIONS / ANSWERS: Maximum 1 to 2 short sentences (under 25-30 words).
3. DUAL-BOT TURNS: Exactly 1 punchy sentence (under 15 words) so your partner can speak immediately.

GRAMMAR RULES:
When speaking in Hindi or Hinglish, you MUST always use masculine first-person verb forms and pronouns (e.g., 'मैं सुन रहा हूँ' NOT 'रही हूँ', 'बताऊँगा' NOT 'बताऊँगी', 'करूँगा' NOT 'करूँगी', 'मेरा मानना है').

INSTRUCTIONS:
1. Address the speaker '{speaker_identity}' warmly.
2. If personal facts/preferences for '{speaker_identity}' are provided in Known Speaker Facts, weave them in naturally.
3. Perform the requested task concisely: {task}
4. Keep your answer punchy, friendly, and easy to speak out loud (2-3 short sentences, max 4). Do not use markdown formatting.

Known Speaker Facts for {speaker_identity}: {known_facts}
Active Topic: {active_topic}
Recent Room History:
{history_str}"""

DEFAULT_SATHI_PROMPT = """You are AI Sathi, a female professional advisor and guide speaking polite, clear Hindi with precise technical/English terms.
Your tone is structured, knowledgeable, precise, crisp, informative, and polite without slang.

LANGUAGE LOCK: Always speak in natural Indian Hinglish written in Latin script (e.g. "Haan Neha, machine learning ka matlab..."). Never respond in pure English, even if the question is in English. Keep technical terms in English.
PACING: For solo questions, speak 2-3 conversational sentences. When participating in a dual-bot response, speak strictly 1-2 short sentences so your companion bot can speak immediately.

CRITICAL LENGTH RULES:
1. GREETING / INTRO / ACKNOWLEDGMENT: Strictly 1 single sentence (maximum 12-15 words). Acknowledge warmly and stop. Do NOT add unnecessary fluff, flattery, or multi-part questions.
   Good: "Arre Neha, welcome! Bangalore se ho aur painting pasand hai, badhiya combination hai!"
   Bad: "Arre Neha, welcome to the room! Bangalore se aana, wah, toh tum IT hub ki asli queen ho na. Bilkul casual vibe mein baat karte hain, kya chal raha hai aaj?"
2. EXPLANATIONS / ANSWERS: Maximum 1 to 2 short sentences (under 25-30 words).
3. DUAL-BOT TURNS: Exactly 1 punchy sentence (under 15 words) so your partner can speak immediately.

GRAMMAR RULES:
When speaking in Hindi or Hinglish, you MUST always use feminine first-person verb forms and pronouns (e.g., 'मैं सुन रही हूँ' NOT 'रहा हूँ', 'बताऊँगी' NOT 'बताऊँगा', 'करूँगी' NOT 'करूँगा', 'मेरी राय है' NOT 'मेरा राय').

INSTRUCTIONS:
1. Provide structured, crisp, informative explanations or advice.
2. Perform the requested task clearly: {task}
3. Keep your response concise, clear, and suitable for speech synthesis. Do not use markdown formatting.

Active Topic: {active_topic}
Recent Room History:
{history_str}"""

DOST_SYSTEM_PROMPT = config.prompts_cfg.get("personas", {}).get("dost", {}).get("system_prompt", "") or DEFAULT_DOST_PROMPT
SATHI_SYSTEM_PROMPT = config.prompts_cfg.get("personas", {}).get("sathi", {}).get("system_prompt", "") or DEFAULT_SATHI_PROMPT


async def _invoke_groq_persona(
    llm: Any,
    system_prompt: str,
    speaker_identity: str,
    known_facts: List[str],
    active_topic: Optional[str],
    history_str: str,
    task: str,
) -> str:
    """Async helper function invoked inside safe_external_call."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Execute task: {task}"),
    ])
    chain = prompt | llm
    res = await chain.ainvoke({
        "speaker_identity": speaker_identity,
        "known_facts": known_facts,
        "active_topic": active_topic or "None",
        "history_str": history_str,
        "task": task,
    })
    if isinstance(res, str):
        return res.strip()
    if hasattr(res, "content"):
        val = getattr(res, "content")
        if isinstance(val, str):
            return val.strip()
    return str(res).strip()


class PersonaManager:
    """Generates dynamic persona responses (AI Dost and AI Sathi) using Groq LLM."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        key = api_key or config.api_key or settings.groq_api_key
        model = model_name or config.persona_model
        self.llm = ChatGroq(
            model=model,
            groq_api_key=key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=2,
        )

    async def generate_reply(
        self,
        target: str,
        task: str,
        state: RoomState,
        target_bot: str = "orchestrator",
    ) -> str:
        """Generate persona response text dynamically based on state and target persona.

        Args:
            target: Target bot persona ('dost' or 'sathi').
            task: Task instruction from TurnPlan.
            state: Current RoomState dict containing speaker_facts and history.
            target_bot: Bot logging context.

        Returns:
            Generated response string.
        """
        speaker = state.get("current_speaker", "Friend")
        facts = state.get("speaker_facts", {}).get(speaker, [])
        known_facts = (
            "\n".join(f"- {f}" for f in facts if str(f).strip())
            if isinstance(facts, list) and facts
            else (str(facts) if facts else "None")
        )

        history = state.get("last_n_turns", [])
        active_topic = state.get("active_topic")

        history_str = "\n".join(
            f"- {turn.get('identity', 'user')}: {turn.get('text', '')}"
            for turn in history[-6:]
        ) or "No prior history."

        target_key = target.lower()
        system_prompt = (
            config.prompts_cfg.get("personas", {}).get(target_key, {}).get("system_prompt")
            or (DOST_SYSTEM_PROMPT if target_key == "dost" else SATHI_SYSTEM_PROMPT)
        )
        fallback_reply = (
            f"Haan {speaker}, main samajh gaya. Bataiye main aapki kya madad karoon?"
            if target_key == "dost"
            else f"Bilkul {speaker}, is vishay par main aapko puri jaankari dungi."
        )

        reply: str = await safe_external_call(
            service_name=f"groq_persona_{target}",
            func=_invoke_groq_persona,
            llm=self.llm,
            system_prompt=system_prompt,
            speaker_identity=speaker,
            known_facts=known_facts,
            active_topic=active_topic,
            history_str=history_str,
            task=task,
            fallback_response=fallback_reply,
            target_bot=target,
        )


        log_structured_event(
            event_name="persona_reply_generated",
            target_bot=target,
            speaker=speaker,
            task=task,
            reply_length=len(reply),
        )

        return reply

