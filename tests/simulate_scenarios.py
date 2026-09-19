import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Add workspace directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.router import RouterAgent
from src.audio.transcription_handler import UtteranceNormalizer
from src.state.graph import RoomStateGraphManager


async def run_simulation() -> None:
    """Run simulated multi-speaker conversation turns through LangGraph state machine with Groq LLM inference."""
    print("=" * 75)
    print("      ROXSTAR VOICE ASSISTANT - STAGE 3 LLM ROUTING SIMULATION      ")
    print("=" * 75)

    manager = RoomStateGraphManager()

    scenarios = [
        {"identity": "Rahul", "text": "AI kya hota hai?", "source": "voice"},
        {
            "identity": "Rahul",
            "text": "Shah Rukh Khan ke baare mein batao.",
            "source": "voice",
        },
        {"identity": "Rahul", "text": "Unki koi famous movie batao.", "source": "voice"},
        {"identity": "Priya", "text": "Thoda aur simple batao.", "source": "voice"},
        {
            "identity": "Rahul",
            "text": "Mera naam Rahul hai aur mujhe cricket pasand hai.",
            "source": "voice",
        },
        {
            "identity": "Rahul",
            "text": "Maine apne baare mein kya bataya tha?",
            "source": "voice",
        },
        {
            "identity": "Rahul",
            "text": "AI Dost, tum answer karo. AI Sathi, baad mein ek example dena.",
            "source": "text",
        },
    ]

    for idx, turn in enumerate(scenarios, 1):
        print(f"\n--- [TURN {idx}] Speaker: {turn['identity']} ({turn['source']}) ---")
        print(f"Utterance: \"{turn['text']}\"")

        utt = UtteranceNormalizer(
            identity=turn["identity"],
            text=turn["text"],
            source=turn["source"],
        )

        turns_plan = await manager.process_utterance(utt)

        speaker_facts = manager.state.get("speaker_facts", {}).get(
            turn["identity"], []
        )
        active_topic = manager.state.get("active_topic")

        print(f"Active Topic/Entity: {active_topic}")
        print(f"Speaker Facts ({turn['identity']}): {speaker_facts}")
        print(f"Generated Turn Plans ({len(turns_plan)}):")
        for plan in turns_plan:
            print(
                f"  -> Target: [{plan.target}] | Task: '{plan.task}' | "
                f"Confidence: {plan.confidence} | Reason: '{plan.reason}'"
            )
        await asyncio.sleep(0.5)

    print("\n" + "=" * 75)
    print("                     SIMULATION COMPLETE                     ")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_simulation())
