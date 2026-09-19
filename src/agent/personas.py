from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import settings
from src.logger import get_logger, log_structured_event
from src.state.room_state import RoomState
from src.utils.config_loader import config
from src.utils.fallback import safe_external_call

logger = get_logger("roxstar_voice_assistant.agent.personas")

DOST_SYSTEM_PROMPT = config.prompts_cfg.get("personas", {}).get("dost", {}).get("system_prompt", "")
SATHI_SYSTEM_PROMPT = config.prompts_cfg.get("personas", {}).get("sathi", {}).get("system_prompt", "")


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
            else f"Bilkul {speaker}, is vishay par main aapko puri jaankari dunga."
        )

        reply: str = await safe_external_call(
            service_name=f"groq_persona_{target}",
            func=_invoke_groq_persona,
            llm=self.llm,
            system_prompt=system_prompt,
            speaker_identity=speaker,
            known_facts=facts,
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

