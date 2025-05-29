try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback parser
    yaml = None


def _read_env(path: str) -> str:
    with open(path) as f:
        return f.read()


def test_envs_require_ruff() -> None:
    dev_env_text = _read_env("dev-environment.yml")
    base_env_text = _read_env("environment.yml")
    assert "ruff" in dev_env_text
    assert "ruff" in base_env_text
