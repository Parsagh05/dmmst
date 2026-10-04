# Version 1 — bug fixes and the first full study (archived)

**Status:** finished. Kept for reference only — version 2 redoes everything.

## What this was
After fixing the bugs from version 0, we ran 13 notebooks covering most of the paper:
the ranking losses, synthetic scenarios, METABRIC/SUPPORT, the end-to-end LLM, a
simulated EHR sequence dataset, and several follow-up checks.
The details of each notebook are in [README.md](README.md).

Results: [results/version_1/](../../results/version_1/) · report: `results/report_latex/`

## What we learned (and why there is a version 2)
- The code fixes made here are kept (they live in the code, not in these notebooks).
- But the study was built step by step, so it was not one clean, complete test of the
  paper: some parts of the paper were never tested (for example the regression task
  head, §2.4), the protocol changed along the way, and not every loss combination was run.
- Version 2 therefore starts again from scratch with one fixed, standard protocol.
  **Version 1 numbers are not used in the paper.**
