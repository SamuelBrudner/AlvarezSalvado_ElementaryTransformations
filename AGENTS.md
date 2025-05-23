# AGENTS.md

These instructions apply to the entire MATLAB and Python codebase for the
Alvarez-Salvado Elementary Transformations project.

## 1. Commit Discipline
- **Conventional Commits** are mandatory (e.g., `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`).
- Keep commits small, focused, and atomic.
- Write clear, descriptive commit messages in the imperative mood.
- Reference relevant issues or pull requests when applicable.

## 2. MATLAB Environment Setup
- Use MATLAB R2021a or later for compatibility.
- Maintain a `startup.m` file in the root directory for path configuration.
- Document all MATLAB toolboxes required in `README.md`.
- Use MATLAB's built-in dependency analysis tools to track function dependencies.
- Keep a record of any external MATLAB toolboxes used.

## 3. Python Environment Setup
- Maintain `environment.yml` describing required packages.
- Generate `conda-lock.yml` using `conda-lock lock` to pin exact versions.
- Developers should create the environment with `setup_env.sh --dev` and run
  Python commands with `conda run --prefix ./dev-env`.

## 4. Code Organization
- Organize code into logical directories (e.g., `Code/` for main scripts, `tests/` for test files).
- Follow MATLAB's naming conventions:
  - Function names: `camelCase`
  - Script names: `lowercase_with_underscores.m`
  - Class names: `PascalCase`
- Keep functions focused on a single task.
- Use MATLAB's function validation blocks for input validation.

## 5. Code Quality
- Use the MATLAB Code Analyzer (mlint) to check code quality.
- Follow the MATLAB Style Guidelines 2.0.
- Document all functions using MATLAB's help text format.
- Include example usage in function help text.
- Use meaningful variable names that indicate purpose and units where applicable.
- Configure pre-commit with hooks such as Black, Flake8 or Ruff, MyPy and
  pytest for Python code.

## 6. Testing
- Write unit tests using MATLAB's testing framework.
- Place test files in the `tests/` directory.
- Name test files with a `test_` prefix.
- Aim for high test coverage, especially for critical functions.
- Run the full test suite (MATLAB and Python) locally before each commit.
- Pre-commit should automatically execute the tests; ensure they pass.

## 7. Data Management
- Follow FAIR principles (Findable, Accessible, Interoperable, Reusable).
- Store raw data separately from processed data.
- Use `.mat` files for MATLAB-specific data storage.
- Document data formats and structures in `docs/data_formats.md`.
- Include a `data_dictionary.csv` for all data files.
- Provide `CITATION.cff` and `codemeta.json` in the repository root for
  citability and rich metadata. Consider a `metadata/` directory for additional
  tables.

-## 8. Documentation
- Maintain a comprehensive `README.md` with:
  - Project overview
  - Setup instructions
  - Usage examples
  - Directory structure
- Use MATLAB's built-in publishing tools for generating documentation.
- Document all major design decisions in `docs/decisions/`.
 - Include a `CITATION.cff` file for proper attribution.
 - For larger documentation efforts consider using MkDocs or Sphinx.

## 9. Development Workflow
1. Create a feature branch from `main`.
2. Implement changes with clear, focused commits.
3. Write or update tests for new functionality.
4. Run all tests to ensure nothing is broken. Pre-commit hooks will fail the
   commit if tests do not pass.
5. Update documentation as needed.
6. Submit a pull request for code review.
7. Address all review comments.
8. Merge to `main` when approved.

## 10. Version Control
- Use Semantic Versioning (MAJOR.MINOR.PATCH).
- Tag releases with `vMAJOR.MINOR.PATCH`.
- Update version numbers in relevant files.
- Maintain a `CHANGELOG.md` following Keep a Changelog format.

## 11. Performance & Optimization
- Profile code using MATLAB's Profiler before optimizing.
- Pre-allocate arrays when possible.
- Use vectorized operations instead of loops when appropriate.
- Document any performance considerations in the code.

## 12. Dependencies
- List all MATLAB toolboxes and external dependencies in `DEPENDENCIES.md`.
- Include minimum version requirements.
- Document any platform-specific considerations.

## 13. Security
- Never commit sensitive information (API keys, credentials).
- Use environment variables for configuration.
- Follow the principle of least privilege for all system interactions.
- Validate all external inputs.

---

*Last updated: May 22, 2025*
