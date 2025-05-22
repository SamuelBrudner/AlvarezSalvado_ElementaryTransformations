# AGENTS.md

These instructions apply to the entire MATLAB codebase for the Alvarez-Salvado Elementary Transformations project.

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

## 3. Code Organization
- Organize code into logical directories (e.g., `Code/` for main scripts, `tests/` for test files).
- Follow MATLAB's naming conventions:
  - Function names: `camelCase`
  - Script names: `lowercase_with_underscores.m`
  - Class names: `PascalCase`
- Keep functions focused on a single task.
- Use MATLAB's function validation blocks for input validation.

## 4. Code Quality
- Use the MATLAB Code Analyzer (mlint) to check code quality.
- Follow the MATLAB Style Guidelines 2.0.
- Document all functions using MATLAB's help text format.
- Include example usage in function help text.
- Use meaningful variable names that indicate purpose and units where applicable.

## 5. Testing
- Write unit tests using MATLAB's testing framework.
- Place test files in the `tests/` directory.
- Name test files with a `test_` prefix.
- Aim for high test coverage, especially for critical functions.
- Run all tests before committing changes.

## 6. Data Management
- Follow FAIR principles (Findable, Accessible, Interoperable, Reusable).
- Store raw data separately from processed data.
- Use `.mat` files for MATLAB-specific data storage.
- Document data formats and structures in `docs/data_formats.md`.
- Include a `data_dictionary.csv` for all data files.

## 7. Documentation
- Maintain a comprehensive `README.md` with:
  - Project overview
  - Setup instructions
  - Usage examples
  - Directory structure
- Use MATLAB's built-in publishing tools for generating documentation.
- Document all major design decisions in `docs/decisions/`.
- Include a `CITATION.cff` file for proper attribution.

## 8. Development Workflow
1. Create a feature branch from `main`.
2. Implement changes with clear, focused commits.
3. Write or update tests for new functionality.
4. Run all tests to ensure nothing is broken.
5. Update documentation as needed.
6. Submit a pull request for code review.
7. Address all review comments.
8. Merge to `main` when approved.

## 9. Version Control
- Use Semantic Versioning (MAJOR.MINOR.PATCH).
- Tag releases with `vMAJOR.MINOR.PATCH`.
- Update version numbers in relevant files.
- Maintain a `CHANGELOG.md` following Keep a Changelog format.

## 10. Performance & Optimization
- Profile code using MATLAB's Profiler before optimizing.
- Pre-allocate arrays when possible.
- Use vectorized operations instead of loops when appropriate.
- Document any performance considerations in the code.

## 11. Dependencies
- List all MATLAB toolboxes and external dependencies in `DEPENDENCIES.md`.
- Include minimum version requirements.
- Document any platform-specific considerations.

## 12. Security
- Never commit sensitive information (API keys, credentials).
- Use environment variables for configuration.
- Follow the principle of least privilege for all system interactions.
- Validate all external inputs.

---

*Last updated: May 22, 2025*
