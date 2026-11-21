"""配置与 CLI 冒烟测试。"""

from __future__ import annotations

import pytest

from mini_agent import cli
from mini_agent.config import Settings, load_settings
from mini_agent.errors import ConfigError
from mini_agent.providers import create_provider
from mini_agent.providers.fake import FakeProvider

ENV_NAMES = [
    "MINI_AGENT_PROVIDER",
    "MINI_AGENT_API_KEY",
    "MINI_AGENT_BASE_URL",
    "MINI_AGENT_MODEL",
    "MINI_AGENT_MAX_STEPS",
    "MINI_AGENT_TEMPERATURE",
]


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """清空相关环境变量，并让默认 .env 指向空目录。"""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("mini_agent.config.PROJECT_ROOT", tmp_path)
    return tmp_path


def test_defaults(isolated_env):
    """验证在没有环境变量和 .env 时，配置使用代码默认值。"""
    settings = load_settings()

    assert settings.provider == "deepseek"
    assert settings.api_key is None
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-chat"
    assert settings.max_steps == 8
    assert settings.temperature == 0.3


def test_shell_env_wins_over_dotenv(isolated_env, monkeypatch):
    """验证进程环境变量优先于 .env 文件中的同名配置。"""
    env_file = isolated_env / ".env"
    env_file.write_text("MINI_AGENT_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MINI_AGENT_API_KEY", "from-shell")

    settings = load_settings(env_file=env_file)

    assert settings.api_key == "from-shell"


def test_dotenv_with_bom_is_readable(isolated_env):
    """验证带 UTF-8 BOM 的 .env 能正确读取，不会把 BOM 当成键名。"""
    env_file = isolated_env / ".env"
    env_file.write_text("MINI_AGENT_API_KEY=from-bom\n", encoding="utf-8-sig")

    settings = load_settings(env_file=env_file)

    assert settings.api_key == "from-bom"


def test_repr_hides_api_key():
    """验证 Settings 的 repr 不会泄露 API key。"""
    settings = Settings(provider="deepseek", api_key="sk-secret")

    assert "sk-secret" not in repr(settings)


def test_fake_provider_needs_no_key():
    """验证 fake provider 无需 API key 即可创建。"""
    provider = create_provider(Settings(provider="fake"))

    assert isinstance(provider, FakeProvider)


def test_real_provider_requires_key():
    """验证真实 provider 缺少 API key 时在请求前快速失败。"""
    with pytest.raises(ConfigError, match="MINI_AGENT_API_KEY"):
        create_provider(Settings(provider="deepseek", api_key=None))


def test_invalid_provider_rejected(isolated_env, monkeypatch):
    """验证不支持的 provider 名称会被配置校验拒绝。"""
    monkeypatch.setenv("MINI_AGENT_PROVIDER", "unknown")

    with pytest.raises(ConfigError):
        load_settings()


def test_invalid_max_steps_rejected(isolated_env, monkeypatch):
    """验证非法 max_steps 会被配置校验拒绝。"""
    monkeypatch.setenv("MINI_AGENT_MAX_STEPS", "0")

    with pytest.raises(ConfigError):
        load_settings()


def test_cli_fake_one_shot(isolated_env, capsys):
    """验证 CLI 单次模式在 fake provider 下正常输出并返回退出码 0。"""
    code = cli.main(["--provider", "fake", "你好"])

    captured = capsys.readouterr()
    assert code == 0
    assert "你好" in captured.out


def test_cli_fake_repl_two_turns(isolated_env, monkeypatch, capsys):
    """验证 REPL 能连续处理两轮输入，并支持 /history 和 /exit。"""
    inputs = iter(["第一轮", "第二轮", "/history", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    code = cli.main(["--provider", "fake"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.count("assistant>") == 2
    assert "第一轮" in captured.out
    assert "assistant" in captured.out
