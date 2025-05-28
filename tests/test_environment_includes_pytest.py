try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback parser
    yaml = None


def _parse_deps() -> list[str]:
    if yaml is not None:
        with open("environment.yml") as f:
            env = yaml.safe_load(f)
            if isinstance(env, dict):
                entries = env.get("dependencies", [])
                if isinstance(entries, list):
                    return [str(d).split("=")[0] for d in entries]
            return []
    deps: list[str] = []
    with open("environment.yml") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                deps.append(line[2:].split("=")[0])
    return deps


def test_environment_includes_pytest() -> None:
    dependencies = _parse_deps()
    assert "pytest" in dependencies
