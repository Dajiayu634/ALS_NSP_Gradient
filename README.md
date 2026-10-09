# ALS connectome study: public analysis code and methods

This repository provides actual downstream analysis scripts, method documentation and references. Participant-level data and study result tables are not included in the minimal public code release.

NSP computation is referenced to Wang et al. and subsequent disease-method publications. Gradient computation is referenced to BrainSpace. Their source code is not redistributed. Study-specific settings remain documented in `docs/METHODS.md`.

## Included analyses
- Adjusted NSP outcome comparisons, mean-FC sensitivity, conventional within/between-network connectivity and graph-outcome comparisons.
- Clinical partial Spearman correlations, missing-data handling, BH-FDR and participant bootstrap intervals.
- Structural-gradient network inference with DWI RMS and sensitivity settings, using supplied gradients.
- Education subgroup reanalysis, clinical motion/duration sensitivity and cached-spin spatial inference.
- Motion/education descriptive comparisons and main/network/spatial plotting.

See `docs/RUN_ANALYSES.md` for private input prerequisites, execution order and important limitations. See `docs/CODE_PROVENANCE.json` for provenance. Scripts are study-specific and retain the 145-person cohort checks. The generic `statistics.py` is supplementary, not the actual study pipeline.

The previously prepared aggregate CSVs are not part of this minimal code release. No real clinical data, participant matrices, time series or identity maps are included. Data-access arrangements and licensing remain subject to author/institutional confirmation; see `DATA_AVAILABILITY.md` and `RELEASE_CHECKLIST.md`.
