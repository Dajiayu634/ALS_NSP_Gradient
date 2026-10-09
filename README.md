# ALS connectome study: public analysis code and methods

This public repository contains study-specific downstream analysis and plotting scripts, method documentation, references, reproducibility metadata, and an example configuration. It does not include raw imaging, participant-level clinical or covariate data, participant-level derived matrices, parcellation files, or aggregate result tables.

NSP methodology is referenced to Wang et al. and the cited disease-method publications. Gradient methodology is referenced to BrainSpace. Upstream NSP, BrainSpace, CATO, and FreeSurfer source code is not redistributed. Study-specific settings are documented in [METHODS.md](METHODS.md) and [REFERENCES.md](REFERENCES.md).

## Included analyses

- Adjusted NSP outcome comparisons, mean-FC sensitivity, conventional within/between-network connectivity, and graph-outcome comparisons.
- Clinical partial Spearman correlations, missing-data handling, BH-FDR, and participant bootstrap intervals.
- Structural-gradient network inference with DWI RMS and sensitivity settings, using supplied private gradients.
- Education subgroup reanalysis, clinical motion/duration sensitivity, and cached-spin spatial inference.
- Motion/education descriptive comparisons and main, network, and spatial plotting.

See [RUN_ANALYSES.md](RUN_ANALYSES.md) for private input prerequisites, execution order, and limitations. See [CODE_PROVENANCE.json](CODE_PROVENANCE.json) for the relationship between the public scripts and their source analyses. The scripts retain study-specific cohort checks and require authorized private inputs; they are not a self-contained raw-to-results pipeline.

## Public-data scope

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the code-only data-sharing statement. No participant-level data, identity maps, parcellation files, or aggregate result tables are included in this repository.
