from typing import Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

BotTarget = Literal["dost", "sathi", "no_reply"]
TurnReason = Literal[
    "direct_address",
    "knowledge_query",
    "context_continuation",
    "multi_speaker_chain",
    "stop_command",
    "incomplete_thought",
    "fallback_safe_external_call",
]


class TurnPlan(BaseModel):
    """Structured plan for an individual bot turn execution."""

    target: BotTarget = Field(
        default="dost",
        description="Target bot ('dost', 'sathi', or 'no_reply').",
    )
    task: str = Field(
        default="",
        description="Actionable instruction for the target persona without nested quotes",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score float between 0.0 and 1.0 reflecting certainty.",
    )
    reason: Optional[TurnReason] = Field(
        default="direct_address",
        description="Standard reason enum code: direct_address, knowledge_query, context_continuation, multi_speaker_chain, stop_command, or incomplete_thought.",
    )


class RouterOutput(BaseModel):
    """Combined output from LLM for fact extraction, entity tracking, and turn routing."""

    new_speaker_facts: List[str] = Field(
        default_factory=list,
        description="Personal facts or preferences stated by the active speaker.",
    )
    salient_entity: Optional[str] = Field(
        default=None,
        description="Key entity, person, or topic discussed for pronoun resolution.",
    )
    turns: List[TurnPlan] = Field(
        default_factory=list,
        description="Ordered list of turn plans to execute for this input.",
    )


class RoomState(TypedDict):
    """LangGraph room state tracking conversation context and history."""

    speaker_facts: Dict[str, List[str]]
    rolling_summary: str
    last_n_turns: List[dict]
    active_topic: Optional[str]
    pending_turns: List[TurnPlan]
    current_speaker: str
    current_utterance: Optional[dict]
    interrupted_topics: Optional[List[dict]]
