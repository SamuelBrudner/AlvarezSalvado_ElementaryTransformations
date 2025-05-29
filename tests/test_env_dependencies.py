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


def test_envs_require_imageio() -> None:
    dev_env_text = _read_env("dev-environment.yml")
    base_env_text = _read_env("environment.yml")
    assert "imageio" in dev_env_text
    assert "imageio" in base_env_text


def test_envs_require_imageio_ffmpeg() -> None:
    dev_env_text = _read_env("dev-environment.yml")
    base_env_text = _read_env("environment.yml")
    assert "imageio-ffmpeg" in dev_env_text
    assert "imageio-ffmpeg" in base_env_text


def test_envs_require_opencv() -> None:
    dev_env_text = _read_env("dev-environment.yml")
    base_env_text = _read_env("environment.yml")
    assert "opencv-python" in dev_env_text
    assert "opencv-python" in base_env_text


def test_pyproject_requires_opencv() -> None:
    import tomllib

    with open("pyproject.toml", "rb") as f:
        project = tomllib.load(f)

    deps: list[str] = project.get("project", {}).get("dependencies", [])
    assert any(dep.split("==")[0] == "opencv-python" for dep in deps)
def test_envs_require_ipykernel() -> None:
    dev_env_text = _read_env("dev-environment.yml")
    base_env_text = _read_env("environment.yml")
    assert "ipykernel" in dev_env_text
    assert "ipykernel" in base_env_text
    