# Repository Contribution Guidelines

- Create the development environment using `setup_env.sh --dev` if the script exists.
- Always run `pytest -q` before committing changes.
- Place **all** configuration files in the top-level `configs/` directory.
- Each subproject should keep its configs in its own subfolder under `configs/`.
- Name config files using snake_case with the subproject prefix, e.g. `navigation_model_default.yaml`.
- Write SLURM logs to `slurm_logs/<jobname>/` using the `<jobname>_logs` pattern.
- Use Conventional Commits for commit messages.
- Prefer verbose logging to aid debugging and understanding of code behaviour.
- Follow test-driven development; tests should validate results rather than implementation details.
