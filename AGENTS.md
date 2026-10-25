# Rules

- **Minimize code above all else.**
- Read and follow `.agents/skills/ponytail/SKILL.md`.
- Prefer existing packages, pretrained models, and built-in functions.
- Avoid reinventing functionality, unnecessary abstractions, and boilerplate.
- No comments unless essential.
- No `try/except`.
- Use the Python `expression` package's `Result`, `Ok`, and `Error` for expected failures.
- Let unexpected errors propagate. Never silently suppress errors.
- Use **marimo** for reactive interfaces and caching.
- Optimize for **16 GB RAM + GTX 1660**.
- Preserve scientific accuracy and reproducibility.
- **Minimum code. Maximum functionality.**
