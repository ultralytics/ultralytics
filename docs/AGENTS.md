# AGENTS.md

Instructions for `docs/`; `CLAUDE.md` points here. Follow the root `AGENTS.md` for repository-wide rules, including the pinned prettier command for markdown formatting.

## Commands

```bash
# Regenerate docs/en/reference/ after adding/removing/renaming public APIs (docs.yml runs this)
python docs/build_reference.py

# Prepare the complete docs tree and validate it with Zensical strict mode
python docs/build_docs.py
```

## Conventions

- `docs/build_docs.py` prepares macros, references, and comparison pages before running `zensical build --strict`. Its local output intentionally omits production-owned site chrome. Production is rendered by the centralized publisher, so relative `.md` cross-file links are the correct convention in `docs/en/`.
