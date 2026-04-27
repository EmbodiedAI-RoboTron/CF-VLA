# Changelog

All notable changes to this project are documented in this file.

## 2026-04-24

### Changed
- Restored `scripts/train_pytorch.py` as an executable training entrypoint.
- Renamed hyphenated names in code/config scope to underscore variants.
- Replaced hardcoded local absolute paths with portable relative/env-based settings.

### Security
- Removed in-repo heavyweight checkpoint snapshots and local release weight folders.
- Kept checkpoint publishing scripts while enforcing code-repo vs model-repo separation.

### Added
- Added CI workflow at `.github/workflows/ci.yml` for ruff syntax checks, compile checks, and smoke tests.
- Added smoke tests in `tests/test_smoke.py`.
- Added citation metadata in `CITATION.cff`.
