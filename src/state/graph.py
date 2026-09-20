import json
import os
import time
from typing import Dict, List, Optional
from langgraph.graph import END, START, StateGraph
from src.agent.router import RouterAgent
from src.audio.transcription_handler import UtteranceNormalizer
from src.logger import get_logger
from src.state.room_state import RoomState, TurnPlan

logger = get_logger("roxstar_voice_assistant.state.graph")


def create_initial_room_state() -> RoomState:
    """Instantiate a clean initial RoomState structure."""
    return {
        "speaker_facts": {},
        "rolling_summary": "",
        "last_n_turns": [],
        "active_topic": None,
        "pending_turns": [],
        "current_speaker": "",
        "current_utterance": None,
        "interrupted_topics": [],
    }


class RoomStateGraphManager:
    """Manages the LangGraph state machine for multi-speaker room context."""

    def __init__(
        self,
        router_agent: Optional[RouterAgent] = None,
        cache_file: str = ".state_cache.json",
    ) -> None:
        self.router_agent = router_agent or RouterAgent()
        self.cache_file = cache_file
        self.state: RoomState = create_initial_room_state()
        self._hydrate_from_cache()
        self.graph = self._build_graph()

    def _hydrate_from_cache(self) -> None:
        """Hydrate speaker facts from local disk cache if available."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cached_facts = json.load(f)
                    if isinstance(cached_facts, dict):
                        valid_facts = {}
                        for speaker, facts in cached_facts.items():
                            if isinstance(facts, list):
                                clean_facts = [
                                    str(fact) for fact in facts
                                    if isinstance(fact, str) and fact.strip() and "mock" not in fact.lower()
                                ]
                                if clean_facts:
                                    valid_facts[speaker] = clean_facts
                        self.state["speaker_facts"].update(valid_facts)
                        logger.info("state_cache_hydrated", facts_count=len(valid_facts))
        except Exception as exc:
            logger.warning("state_cache_hydration_error", error=str(exc))

    def _save_to_cache(self) -> None:
        """Persist current speaker facts to local disk cache."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.state.get("speaker_facts", {}), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("state_cache_save_error", error=str(exc))

    def _build_graph(self):
        """Construct the LangGraph state transition graph."""
        builder = StateGraph(RoomState)

        builder.add_node("ingestion_node", self._ingestion_node)
        builder.add_node("router_node", self._router_node)
        builder.add_node("queue_node", self._queue_node)

        builder.add_edge(START, "ingestion_node")
        builder.add_edge("ingestion_node", "router_node")
        builder.add_edge("router_node", "queue_node")
        builder.add_edge("queue_node", END)

        return builder.compile()

    def _ingestion_node(self, state: RoomState) -> dict:
        """Ingest new utterance, update turn history capped at 15 items."""
        utt = state.get("current_utterance")
        if not utt:
            return {}

        raw_text = utt.get("text", "")
        from src.utils.guardrails import apply_guardrails
        sanitized_text = apply_guardrails(raw_text).sanitized_text

        turns = list(state.get("last_n_turns", []))
        turns.append({
            "identity": utt["identity"],
            "text": sanitized_text,
            "source": utt["source"],
            "timestamp": utt["timestamp"],
            "bot_replies": [],
        })

        # Cap history to last 15 turns
        if len(turns) > 15:
            turns = turns[-15:]

        return {
            "last_n_turns": turns,
            "current_speaker": utt["identity"],
        }


    async def _router_node(self, state: RoomState) -> dict:
        """Execute LLM router node for fact extraction and turn planning."""
        utt = state.get("current_utterance")
        if not utt:
            return {"pending_turns": []}

        speaker_identity = state.get("current_speaker", utt["identity"])
        facts_dict = dict(state.get("speaker_facts", {}))
        history = state.get("last_n_turns", [])
        active_topic = state.get("active_topic")
        interrupted_topics = state.get("interrupted_topics", [])

        output = await self.router_agent.route_utterance(
            speaker_identity=speaker_identity,
            text=utt["text"],
            speaker_facts=facts_dict,
            active_topic=active_topic,
            history=history,
            interrupted_topics=interrupted_topics,
        )

        # Deep copy / retrieve existing facts for the speaker
        current_speaker_facts = list(self.state.get("speaker_facts", {}).get(speaker_identity, []))

        # Append new facts without introducing duplicates
        for fact in (output.new_speaker_facts or []):
            cleaned_fact = str(fact).strip()
            if cleaned_fact and cleaned_fact not in current_speaker_facts:
                current_speaker_facts.append(cleaned_fact)

        # Ensure state dictionary is updated properly
        if "speaker_facts" not in self.state:
            self.state["speaker_facts"] = {}
        self.state["speaker_facts"][speaker_identity] = current_speaker_facts

        # Persist updated facts to disk cache
        self._save_to_cache()

        new_topic = output.salient_entity or active_topic

        return {
            "speaker_facts": self.state["speaker_facts"],
            "active_topic": new_topic,
            "pending_turns": output.turns,
        }


    def _queue_node(self, state: RoomState) -> dict:
        """Queue node making pending_turns available for consumption."""
        logger.info(
            "graph_queue_node",
            pending_turns_count=len(state.get("pending_turns", [])),
            active_topic=state.get("active_topic"),
        )
        return {"pending_turns": state.get("pending_turns", [])}

    def record_interrupted_turn(self, speaker: str) -> None:
        """Record an interrupted turn topic in room state for router context preservation."""
        active_topic = self.state.get("active_topic")
        if active_topic:
            topics = list(self.state.get("interrupted_topics", []))
            topics.append({
                "speaker": speaker,
                "topic": active_topic,
                "timestamp": time.time(),
            })
            if len(topics) > 5:
                topics = topics[-5:]
            self.state["interrupted_topics"] = topics
            logger.info("interrupted_topic_preserved", speaker=speaker, topic=active_topic)

    async def process_utterance(self, utterance: UtteranceNormalizer) -> List[TurnPlan]:
        """Process an incoming utterance through the graph state machine.

        Args:
            utterance: Normalized input utterance.

        Returns:
            List of TurnPlan objects generated for this turn.
        """
        self.state["current_utterance"] = utterance.to_dict()

        # Run compiled LangGraph
        result_state = await self.graph.ainvoke(self.state)
        self.state.update(result_state)

        return self.state.get("pending_turns", [])
