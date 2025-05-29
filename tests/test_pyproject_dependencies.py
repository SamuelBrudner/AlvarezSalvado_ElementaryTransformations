import tomllib


def test_pyproject_includes_ipykernel() -> None:
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = {dep.split("=")[0] for dep in data.get("project", {}).get("dependencies", [])}
    assert "ipykernel" in deps
