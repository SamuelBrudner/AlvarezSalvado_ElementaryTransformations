# AGENTS.md

These instructions apply to the entire repository.

## 1. Commit Discipline
- **Conventional Commits** are mandatory (e.g., `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`).
- Keep commits small, focused, and atomic.
- Write clear, descriptive commit messages in the imperative mood.
- Reference relevant issues or pull requests when applicable.

## 2. Environment Setup
- Use `conda` for environment management with an `environment.yml` file.
- The environment should be installed in the project directory (e.g., `.env`).
- All Python commands must be run within the activated environment.
- Dependencies should be explicitly versioned in `environment.yml`.
- A `conda-lock.yml` file should pin exact versions for reproducibility.

## 3. Code Quality & Testing
- Follow PEP 8 style guidelines for Python code.
- Write unit tests for all new functionality using `pytest`.
- Maintain test coverage above 90%.
- Use type hints throughout the codebase.
- Document all public functions and classes with docstrings.

## 4. Pre-commit Hooks
- Install pre-commit hooks with `pre-commit install`.
- Hooks must include:
  - `ruff` for linting
  - `black` for code formatting
  - `isort` for import sorting
  - `mypy` for static type checking
  - `interrogate` for docstring coverage
  - `pre-commit-hooks` for trailing whitespace and other checks
- CI will run `pre-commit run --all-files` on all pull requests.

## 5. Documentation
- Maintain up-to-date documentation in the `docs/` directory.
- Use Markdown for all documentation.
- Keep the `README.md` up-to-date with:
  - Project description and purpose
  - Installation instructions
  - Basic usage examples
  - Contribution guidelines
  - License information
- Document all major design decisions in `docs/decisions/`.

## 6. Data Management
- Follow FAIR principles for all data (Findable, Accessible, Interoperable, Reusable).
- Store raw data in `data/raw/` and never modify it directly.
- Store processed data in `data/processed/`.
- Document all data processing steps in `notebooks/`.
- Provide a `CITATION.cff` file for proper attribution.

## 7. Development Workflow
1. Create a new branch for each feature or bugfix.
2. Write tests for new functionality.
3. Implement the feature or fix.
4. Ensure all tests pass.
5. Update documentation as needed.
6. Submit a pull request for review.
7. Address all review comments.
8. Squash and merge when approved.

## 8. Versioning & Releases
- Use Semantic Versioning (MAJOR.MINOR.PATCH).
- Create a changelog entry for each release.
- Tag releases with `vMAJOR.MINOR.PATCH`.
- Update version numbers in all relevant files.

## 9. Continuous Integration
- All tests must pass before merging to main.
- Code coverage must not decrease.
- Documentation builds must succeed.
- Type checking must pass.

## 10. Security
- Never commit sensitive information (API keys, passwords, etc.).
- Use environment variables for configuration.
- Keep dependencies up-to-date and audit for vulnerabilities.
- Follow the principle of least privilege for all services.

---

*Last updated: May 21, 2025*
