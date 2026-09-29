# project-template

Python project template: uv, ruff, mypy, pytest, pre-commit, and CI preconfigured.

## Quickstart

1. On GitHub, click **Use this template** to create a new repo.
2. Clone it and run:

```bash
   make setup
   make test
   make lint
```

## After creating a new project

- [ ] Rename `src/app/` to your package name
- [ ] Update `name` in `pyproject.toml`
- [ ] Replace this README

## Tooling

| Tool | Purpose |
|---|---|
| uv | Dependencies and virtual env |
| ruff | Linting and formatting |
| mypy | Type checking |
| pytest | Tests |
| pre-commit | Runs ruff before each commit |