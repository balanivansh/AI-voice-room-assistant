import os
import pytest
from src.utils.config_loader import AppConfig, config


def test_app_config_singleton():
    assert config is not None
    assert isinstance(config, AppConfig)


def test_app_config_models_cfg():
    profiles = config.models_cfg.get("profiles", {})
    assert config.active_profile_name in profiles


    assert config.base_url is not None
    assert config.router_model is not None
    assert config.persona_model is not None
    assert config.temperature is not None
    assert config.max_tokens is not None



def test_app_config_prompts_cfg():
    assert "router" in config.prompts_cfg
    assert "personas" in config.prompts_cfg
    assert "system_prompt" in config.prompts_cfg["router"]
    assert "dost" in config.prompts_cfg["personas"]
    assert "sathi" in config.prompts_cfg["personas"]
    assert "system_prompt" in config.prompts_cfg["personas"]["dost"]
    assert "system_prompt" in config.prompts_cfg["personas"]["sathi"]
