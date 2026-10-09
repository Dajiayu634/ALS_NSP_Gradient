# Data dictionary

## Proposed controlled-access package (not included)
| Field | Definition |
|---|---|
| release_id | Newly assigned participant token, shared across modalities; no original identifier or identity key |
| SC_NOS_raw | 68x68 reconstructed streamline-count matrix; preserve original diagonal and symmetry conventions in accompanying metadata |
| SC_NOS_volume_normalized | NOS/sqrt(Vi*Vj); distinct from raw counts |
| parcel_volume_mm3 | Regional gray-matter volumes used for normalization |
| FC_signed_r | Signed Pearson FC; record retained-frame scope explicitly |
| FC_128_signed_r | Signed Pearson FC from first 128 retained frames |
| NSP_input_absolute_r | Absolute FC_128_signed_r, with diagonal retained |
| group | HC or ALS; binary contrast ALS minus HC |
| age, sex, education | Model covariates; coding, units and missingness documented in approved package |
| meanFD_mm | Pre-censoring fMRI mean FD, excluding initial zero |
| mean_relative_RMS_mm | DWI consecutive-volume RMS, excluding initial reference zero |
| clinical outcomes | Only approved variables required for intended analyses; missing values explicitly marked |

This is a target schema, not a claim that these files have already been prepared or approved. Raw NOS diagonal conventions, parcel names/order, network assignments, gradient templates/orientation, and NSP calibration factors must be supplied and verified before an exact-reproduction package is offered. Group-specific calibration may require controlled derived targets rather than raw time series.

## Planned public aggregate CSV schema
`feature`, `parcel`, `network`, `scope`, `index`: analysis labels, not participant identifiers. `input_scheme`, `model`, `setting`: the specific analysis variant. `n`, `n_HC`, `n_ALS`, `df`: sample sizes and residual degrees of freedom. `beta`: adjusted group coefficient (ALS minus HC); `se`: standard error; `t`: test statistic; `p`: unadjusted probability; `q`: BH-adjusted probability within the stated family; `ci_low`, `ci_high`: pointwise 95% limits. Mean/SD fields describe groups. Correlation fields describe aggregate associations; interpret p/q and covariates with each row's labels.

No public aggregate result CSV is included in this release. If aggregate tables are added in a future release, provide a separate inventory describing columns, analysis variants, and the primary-result filters.


