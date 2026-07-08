import importlib
import os
from pathlib import Path
from types import SimpleNamespace


def test_run_all_entry_imports_without_missing_symbols():
    module = importlib.import_module("new_experiments.all.run_all")
    assert hasattr(module, "main")


def test_run_all_entry_delegates_resource_loading_to_runner():
    source = Path("new_experiments/all/run_all.py").read_text(encoding="utf-8")

    assert "runner.load_resources()" not in source


def test_paper_defaults_match_required_local_models():
    from new_experiments.core import RetrievalConfig

    config = RetrievalConfig()
    assert config.sample_size == 10
    assert config.use_api_reranker is False


def test_paper_graph_expansion_uses_k3_as_b0_seed_count():
    from new_experiments.core import EvidenceUnit, PaperExperimentRunner, RetrievalConfig

    config = RetrievalConfig(k3=7)
    runner = PaperExperimentRunner(config)
    units = [
        EvidenceUnit(id=f"unit::{idx}", title=f"Title {idx}", content="evidence", score=1.0)
        for idx in range(9)
    ]

    assert runner.graph_seed_titles(units) == [f"Title {idx}" for idx in range(7)]


def test_local_reranker_defaults_to_sentence_transformers_backend(monkeypatch):
    import src.retrievers.reranker as reranker_module

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class HangingFlagReranker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("FlagEmbedding backend must be opt-in for local paper runs")

    monkeypatch.setattr(reranker_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(reranker_module, "FLAG_EMBEDDING_AVAILABLE", True)
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(reranker_module, "FlagReranker", HangingFlagReranker)
    monkeypatch.setattr(
        reranker_module,
        "get_config",
        lambda: SimpleNamespace(rerank_model="BAAI/bge-reranker-base"),
    )
    monkeypatch.delenv("RERANK_BACKEND", raising=False)

    reranker = reranker_module.Reranker()

    assert reranker.get_model_info()["backend"] == "sentence_transformers"


def test_local_reranker_prefers_downloaded_model_directory(tmp_path, monkeypatch):
    import src.retrievers.reranker as reranker_module

    local_model = tmp_path / "models" / "BAAI" / "bge-reranker-base"
    local_model.mkdir(parents=True)
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name_or_path, *args, **kwargs):
            captured["model"] = model_name_or_path

    monkeypatch.setattr(reranker_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reranker_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(reranker_module, "FLAG_EMBEDDING_AVAILABLE", False)
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(
        reranker_module,
        "get_config",
        lambda: SimpleNamespace(rerank_model="BAAI/bge-reranker-base"),
    )

    reranker = reranker_module.Reranker()

    assert captured["model"] == str(local_model)
    assert reranker.get_model_info()["model_name"] == str(local_model)


def test_deepseek_env_aliases_are_honored(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    import src.utils.config as config_module

    config_module.get_config.cache_clear()
    config = config_module.get_config()

    assert config.openai_api_key == "test-key"
    assert config.openai_api_base == "https://api.deepseek.com"
    assert config.llm_model == "deepseek-v4-flash"


def test_deepseek_env_takes_precedence_over_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "bad-openai-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    import src.utils.config as config_module

    config_module.get_config.cache_clear()
    config = config_module.get_config()

    assert config.openai_api_key == "deepseek-key"
    assert config.openai_api_base == "https://api.deepseek.com"
    assert config.llm_model == "deepseek-v4-flash"


def test_deepseek_dotenv_values_are_honored(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "bad-openai-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-3.5-turbo")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=dotenv-deepseek-key\n"
        "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
        "DEEPSEEK_MODEL=deepseek-v4-flash\n",
        encoding="utf-8",
    )

    from src.utils.config import Config

    config = Config(_env_file=str(env_file))

    assert config.openai_api_key == "dotenv-deepseek-key"
    assert config.openai_api_base == "https://api.deepseek.com"
    assert config.llm_model == "deepseek-v4-flash"


def test_deepseek_generate_honors_explicit_sampling_and_token_limits():
    from src.llms.base_client import Message
    from src.llms.deepseek_client import DeepSeekClient

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "deepseek-v4-flash"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    response = client.generate(
        [Message(role="user", content="hello")],
        temperature=0.12,
        max_tokens=17,
    )

    assert response.success is True
    assert response.content == "ok"
    assert captured["temperature"] == 0.12
    assert captured["max_tokens"] == 17


def test_deepseek_stream_generate_uses_streaming_response_and_yields_chunks():
    from src.llms.base_client import Message
    from src.llms.deepseek_client import DeepSeekClient

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="he"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="llo"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
            ]

    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "deepseek-v4-flash"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    chunks = list(
        client.stream_generate(
            [Message(role="user", content="hello")],
            temperature=0.2,
            max_tokens=9,
        )
    )

    assert chunks == ["he", "llo"]
    assert captured["stream"] is True
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 9
