AGENTS.md

These guidelines govern the MATLAB + Python codebase for the Alvarez-Salvado Elementary Transformations project. They ensure consistent workflows across two languages, reproducible environments, and long-term discoverability of analyses and figures.

Last updated: 2025-05-23

⸻

1  Commit Discipline

Rule	Detail
Conventional Commits	Prefix every commit: feat:, fix:, docs:, style:, refactor:, test:, chore:.
Atomicity	Keep commits small and focused; one logical change per commit.
Message style	Imperative mood (“Add unit test” not “Added”); body explains why, references issues/PRs.


⸻

2  MATLAB Environment

Item	Practice
Version	MATLAB R2021a or newer (tested up to R2024b).
startup.m	Lives at repo root; adds Code/ and tests/ to the path, configures preferences.
Toolboxes	List mandatory toolboxes in DEPENDENCIES.md with min versions.
External libs	Clone to third_party/matlab/; load via startup.m.
Dependency graph	Generate with matlab.codetools.requiredFilesAndProducts and save to reference/deps/matlab_deps_<date>.json.


⸻

3  Python Environment
	1.	environment.yml – high-level spec.
	2.	conda-lock.yml – pin exact versions (conda-lock lock).
	3.	Developers run:

./setup_env.sh --dev        # creates ./dev-env
conda run --prefix ./dev-env pytest -q   # run tests

Pre-commit hooks invoke the environment automatically via conda run --prefix ./dev-env.

⸻

4  Code Quality

Language	Tools
MATLAB	Code Analyzer (mlint) ≥ score 90; Style Guidelines v2.0; use function argument validation.
Python	Ruff (ruff check .), Black, isort, mypy (--strict), interrogate for docstrings.
Pre-commit	.pre-commit-config.yaml runs all linters + tests; CI refuses non-passing commits.

Include example usage in every MATLAB help block and Python docstring.

⸻

5  Testing

Language	Framework	Location
MATLAB	matlab.unittest	tests/matlab/ with Test suffix classes
Python	pytest + hypothesis	tests/python/

	•	Goal: ≥ 80 % combined coverage (line + branch) for critical code.
	•	CI matrix tests MATLAB + Python on Linux and macOS runners.

⸻

6  Data & Figure Management

6.1  FAIR Data
	•	Raw data never edited; stored under data/raw/.
	•	Processed artefacts → data/processed/, tracked by DVC.
	•	Metadata tables (data_dictionary.csv, experiment manifests) live in metadata/.

6.2  Figure Registry

Figures are saved via Python helper src/utils/save_figure.py or MATLAB helper Code/utils/saveFigure.m, which:
	1.	Writes the image into figures/<year>/<purpose>/….
	2.	Creates side-car YAML with caption, params, commit SHA, and data hash.
	3.	Logs action via loguru (Python) or fprintf (MATLAB).

A GitHub Action verifies every binary plot has a YAML; binaries go through Git LFS.

⸻

7  Documentation

Medium	Tool
README	High-level overview, quickstart, directory map.
Python API	Sphinx + numpydoc; hosted via GitHub Pages.
MATLAB	Publish key scripts to HTML (publish.m) and place in docs/matlab/.
Design decisions	docs/decisions/<YYYY-MM-DD>_<slug>.md (ADR pattern).
Citations	CITATION.cff + codemeta.json.
Figure gallery	Auto-generated (scripts/build_fig_gallery.py) from YAML registry.


⸻

8  Development Workflow

main
 ├─ feature/<topic>
 │    └─ PR → review → squash-merge ↩︎
 └─ release/<version>

	1.	Branch off main.
	2.	Implement, commit, lint, and add/adjust tests.
	3.	Run pytest + MATLAB tests + pre-commit.
	4.	Push, open PR, request review.
	5.	Address feedback, update docs.
	6.	CI must be green → squash-merge.

⸻

9  Versioning & Releases
	•	Semantic Versioning (vMAJOR.MINOR.PATCH) tags.
	•	CHANGELOG.md follows Keep a Changelog template.
	•	Release checklist:
	•	Docs build passes.
	•	conda-lock.yml current.
	•	dvc push completes.
	•	Zenodo deposition triggered (GitHub webhook).

⸻

10  Performance & Optimisation
	•	MATLAB: use the Profiler; vectorise; pre-allocate.
	•	Python: profile with snakeviz; prefer NumPy array ops; consider Numba.
	•	Document bottlenecks in code comments.

⸻

11  Dependencies & Platforms
	•	MATLAB toolboxes enumerated in DEPENDENCIES.md with minima.
	•	Python deps pinned via conda-lock.yml.
	•	CI tests on Ubuntu-22.04, macOS-13 runners.

⸻

12  Security
	•	No secrets in Git – use environment variables or GitHub Secrets.
	•	Validate external inputs; sanitise file paths.
	•	Principle of least privilege for file & network access.

⸻

13  Automations (ClickUp ↔ Slack ↔ GitHub)

Nightly GitHub Action calls tools/clickup_sync.py:
	•	Sync closed PRs → “Done” tasks.
	•	Cache audit JSON under reference/status/.

⸻

End of AGENTS.md