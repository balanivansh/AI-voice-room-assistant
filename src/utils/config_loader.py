import os
from pathlib import Path
from typing import Any, Dict
import yaml
from src.config import settings


class AppConfig:
    """Centralized configuration loader for models and system prompts from YAML files."""

    def __init__(
        self,
        models_path: str | Path | None = None,
        prompts_path: str | Path | None = None,
    ) -> None:
        project_root = Path(__file__).parent.parent.parent
        self.models_file = Path(models_path) if models_path else project_root / "configs" / "models.yaml"
        self.prompts_file = Path(prompts_path) if prompts_path else project_root / "configs" / "prompts.yaml"

        self.models_cfg: Dict[str, Any] = self._load_yaml(self.models_file)
        self.prompts_cfg: Dict[str, Any] = self._load_yaml(self.prompts_file)

        self.active_profile_name: str = self.models_cfg.get("active_profile", "groq")
        profiles: Dict[str, Any] = self.models_cfg.get("profiles", {})
        self.active_profile: Dict[str, Any] = profiles.get(self.active_profile_name, {})

        # Active profile properties
        self.base_url: str = self.active_profile.get("base_url", "https://api.groq.com/openai/v1")
        self.api_key_env: str = self.active_profile.get("api_key_env", "GROQ_API_KEY")
        self.api_key: str = os.getenv(self.api_key_env) or settings.groq_api_key or ""
        
        self.router_model: str = self.active_profile.get("router_model", "qwen/qwen3.8-27b")
        self.persona_model: str = self.active_profile.get("persona_model", "qwen/qwen3.8-27b")
        self.temperature: float = float(self.active_profile.get("temperature", 0.7))
        self.max_tokens: int = int(self.active_profile.get("max_tokens", 160))
        self.router_temperature: float = float(self.active_profile.get("router_temperature", 0.0))
        self.router_max_tokens: int = int(self.active_profile.get("router_max_tokens", 150))

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Safely load and parse YAML file."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_prompt(self, path_keys: list[str]) -> str:
        """Helper to fetch prompt string by key path."""
        node = self.prompts_cfg
        for k in path_keys:
            if isinstance(node, dict):
                node = node.get(k, {})
            else:
                return ""
        return str(node) if isinstance(node, str) else ""


# Global singleton configuration instance
config = AppConfig()
