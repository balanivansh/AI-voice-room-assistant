import re
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import settings
from src.logger import get_logger, log_structured_event
from src.state.room_state import RouterOutput, TurnPlan
from src.utils.config_loader import config
from src.utils.fallback import safe_external_call
from src.utils.guardrails import apply_guardrails

logger = get_logger("roxstar_voice_assistant.agent.router")



ISOLATED_CALLOUT = re.compile(
    r'^\s*(?:ai\s+)?(?:dost|sathi|दोस्त|साथी|सार्थी)\s*[\.\?\!\,]?\s*$',
    re.IGNORECASE,
)



def normalize_phonetic_aliases(text: str) -> str:
    """Normalize phonetic STT transcript variations for assistant names."""
    if not text:
        return text
    # Only substitute when explicitly prefixed by "AI", "arre", or "yahi"
    text = re.sub(
        r'\b(?:ai|arre|yahi)\s+(?:saath?\s*hi|saathi|saal|साथ\s*ही)\b',
        'AI Sathi',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\b(?:ai|arre)\s+(?:dosth?|do|दो)\b',
        'AI Dost',
        text,
        flags=re.IGNORECASE,
    )
    return text




ROUTER_SYSTEM_PROMPT = config.prompts_cfg.get("router", {}).get("system_prompt", "")


async def _invoke_groq_router(
    llm_with_structure: Any,
    speaker_identity: str,
    known_facts: List[str],
    active_topic: Optional[str],
    interrupted_topics_str: str,
    history_str: str,
    latest_text: str,
) -> RouterOutput:
    """Async helper function passed into safe_external_call."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT or config.prompts_cfg.get("router", {}).get("system_prompt", "")),
        ("human", "{latest_text}"),
    ])
    chain = prompt | llm_with_structure
    return await chain.ainvoke({
        "speaker_identity": speaker_identity,
        "known_facts": known_facts,
        "active_topic": active_topic or "None",
        "interrupted_topics_str": interrupted_topics_str,
        "history_str": history_str,
        "latest_text": latest_text,
    })


class RouterAgent:
    """Unified Extraction and Routing Agent powered by Groq LLM."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        key = api_key or config.api_key or settings.groq_api_key
        model = model_name or config.router_model
        self.llm = ChatGroq(
            model=model,
            groq_api_key=key,
            base_url=config.base_url,
            temperature=config.router_temperature,
            max_tokens=config.router_max_tokens,
            max_retries=2,
        )
        self.structured_llm = self.llm.with_structured_output(RouterOutput, method="json_mode")

    async def route_utterance(
        self,
        speaker_identity: str,
        text: str,
        speaker_facts: Dict[str, List[str]],
        active_topic: Optional[str],
        history: List[dict],
        interrupted_topics: Optional[List[dict]] = None,
        target_bot: str = "orchestrator",
    ) -> RouterOutput:
        """Route an incoming utterance, extracting facts and producing turn plans.

        Args:
            speaker_identity: Identity of active speaker.
            text: Utterance text content.
            speaker_facts: Dictionary of all known speaker facts.
            active_topic: Current room active topic/entity.
            history: List of recent conversation turn dicts.
            interrupted_topics: List of recent interrupted topics for context recovery.
            target_bot: Bot log identifier.

        Returns:
            RouterOutput object containing facts, entity, and turn plans.
        """
        # 1. Execute guardrails FIRST at top of router entry
        guard_res = apply_guardrails(text)
        if not guard_res.is_safe:
            logger.warning(
                "security_guardrail_triggered",
                violation=guard_res.violation_type,
                speaker_identity=speaker_identity,
            )
            return RouterOutput(
                new_speaker_facts=[],
                salient_entity="safety_policy",
                turns=[
                    TurnPlan(
                        target="sathi",
                        task=guard_res.refusal_message or "Aapki request safety policy ke khilaaf hai.",
                        confidence=1.0,
                        reason=f"guardrail_refusal_{guard_res.violation_type}",
                    )
                ],
            )

        # 2. Use sanitized_text for downstream normalization and context building
        sanitized_text = guard_res.sanitized_text
        normalized_text = normalize_phonetic_aliases(sanitized_text)

        if ISOLATED_CALLOUT.match(normalized_text.strip()):
            logger.info("ignoring_isolated_wake_callout", text=normalized_text)
            return RouterOutput(
                new_speaker_facts=[],
                salient_entity=None,
                turns=[],
            )

        # 3. Format history and context AFTER sanitization
        known_facts = speaker_facts.get(speaker_identity, [])
        history_str = "\n".join(
            f"- {turn.get('identity', 'user')}: {turn.get('text', '')}"
            for turn in history[-10:]
        ) or "No prior history."

        interrupted_list = interrupted_topics or []
        interrupted_topics_str = "\n".join(
            f"- {item.get('speaker', 'user')}: {item.get('topic', '')}"
            for item in interrupted_list[-3:]
        ) or "None."




        fallback_output = RouterOutput(
            new_speaker_facts=[],
            salient_entity=active_topic,
            turns=[
                TurnPlan(
                    target="dost",
                    task="Respond to user input casually.",
                    confidence=0.5,
                    reason="fallback_safe_external_call",
                )
            ],
        )

        output: RouterOutput = await safe_external_call(
            service_name="groq_router_inference",
            func=_invoke_groq_router,
            llm_with_structure=self.structured_llm,
            speaker_identity=speaker_identity,
            known_facts=known_facts,
            active_topic=active_topic,
            interrupted_topics_str=interrupted_topics_str,
            history_str=history_str,
            latest_text=normalized_text,
            fallback_response=fallback_output,
            target_bot=target_bot,
        )

        # Alias normalization post-processor validation
        lower_text = normalized_text.lower()
        dost_aliases = ["ai dosth", "dosth", "ai dost", "dost", "दोस्त"]
        sathi_aliases = ["ai saathi", "saathi", "ai sathi", "sathi", "साथी", "साथ ही"]

        has_dost_alias = any(alias in lower_text for alias in dost_aliases)
        has_sathi_alias = any(alias in lower_text for alias in sathi_aliases)

        if output and output.turns:
            if has_sathi_alias and not has_dost_alias:
                for turn in output.turns:
                    if turn.target != "no_reply":
                        turn.target = "sathi"
            elif has_dost_alias and not has_sathi_alias:
                for turn in output.turns:
                    if turn.target != "no_reply":
                        turn.target = "dost"

            # Merge consecutive turn plans with the same target bot
            if len(output.turns) > 1:
                merged_turns: List[TurnPlan] = []
                for turn in output.turns:
                    if not merged_turns:
                        merged_turns.append(turn)
                    else:
                        last_turn = merged_turns[-1]
                        if last_turn.target == turn.target and turn.target != "no_reply":
                            last_turn.task = f"{last_turn.task} And also: {turn.task}"
                        else:
                            merged_turns.append(turn)
                output.turns = merged_turns

            # Append conciseness directive based on turn count and task type
            greeting_ack_keywords = ("greet", "acknowledg", "welcome", "fact", "hello", "hi", "intro")
            if len(output.turns) > 1:
                for turn in output.turns:
                    if turn.target != "no_reply":
                        if any(kw in turn.task.lower() for kw in greeting_ack_keywords):
                            turn.task += " Acknowledge in 1 single short sentence (under 12 words)."
                        else:
                            turn.task += " Keep response strictly to 1-2 short sentences (under 25 words) for a fast handoff to the companion bot."
            elif len(output.turns) == 1:
                for turn in output.turns:
                    if turn.target != "no_reply":
                        if any(kw in turn.task.lower() for kw in greeting_ack_keywords):
                            turn.task += " Acknowledge in 1 single short sentence (under 12 words)."
                        else:
                            turn.task += " Keep response to 2-3 natural sentences (under 45 words)."

        log_structured_event(
            event_name="router_decision",
            target_bot=target_bot,
            speaker_identity=speaker_identity,
            num_facts=len(output.new_speaker_facts),
            salient_entity=output.salient_entity,
            turns_count=len(output.turns),
        )

        return output
