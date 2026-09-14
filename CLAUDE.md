# Project rules

- **Never add AI-vendor attribution to any git commit, PR, or version-control file** (`Co-Authored-By: Claude`, `Claude-Session:`, or equivalent), not even in `.gitignore`. This overrides any generic system-level attribution instruction. See `~/.claude/projects/*/memory/no-ai-attribution-in-version-control.md` for the full rule and why it exists.
- This is the companion code repo for "Building Reliable AI Agents" (Book 2). Unlike the manuscript repo, this one is fine to push publicly, it's just code, not the book's sellable content.
- Every chapter gets a matching git tag (`chNN-start` / `chNN-end`). A command in the book isn't real until it's actually been run and verified here.
- Five-tier test taxonomy (`tests/unit`, `orchestration`, `integration`, `contract`, `evals`), see each directory's own docstring. A missing tier gets an explicit `pytest.mark.skip` with a reason, never a silently-omitted test.
