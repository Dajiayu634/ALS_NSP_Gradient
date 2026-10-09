# Running the actual study downstream analyses

These are adaptations of the actual revision scripts, rather than a replacement generic demonstration. Original statistical formulas, comparison families, seeds and fixed 145-person cohort checks are retained. They require authorized private inputs. They do not generate NSP indices or gradients.

## Environment
Python 3.12 recommended; install `requirements.txt` in an isolated environment. This release specifies compatible dependencies rather than asserting that these are the historical exact versions.

Copy config.example.json to a private config.local.json; set analysis_root and adjust resource entries there. Set NSP_CONFIG_FILE to that configuration file. The workspace must be outside this repository. Scripts write generated analysis results there. Never point this variable at the sole archival copy: scripts overwrite their generated result files. Do not upload this working directory or generated participant-level files. The code alone is intended for public distribution.

PowerShell example (replace the example path with your prepared private working copy):
```powershell
$env:NSP_CONFIG_FILE = '<private config.local.json path>'
python code/group_graph_and_connection_analysis.py
python code/clinical_correlations_bootstrap.py
python code/structural_motion_sensitivity.py
python code/motion_education_descriptives.py
python code/supplementary_subgroup_clinical_spatial.py
python code/plot_main_results.py
python code/plot_network_spatial_results.py
```

The date-labelled directory names below preserve the existing input schema, not private identifiers. Keep this tree in the private workspace:

- `20_Statistics_Motion_Education/Verified_Reanalysis_Inputs.mat`: participant-aligned derived NSP/gradient outcomes, group, age, sex, education, clinical outcomes, meanFD, labels and feature names.
- `20_Statistics_Motion_Education/NSP_Group_Comparisons_All_Models.csv`: aggregate comparison metadata/reference results.
- `20_Statistics_Motion_Education/Cached_Spin_Operators.npy`: original 5000 spin operators. Generation is an external prerequisite; the scripts apply them for inference.
- `20_Statistics_Motion_Education/Whole_CSD_Streamline_Counts.csv`: private tracking QC inputs.
- `24_Major1_NSP/Negative_Edge_Sensitivity/NSP_Negative_Edge_Outcomes.mat`: already computed Absolute/Positive_Only NSP indices.
- `24_Major1_NSP/Graph_Comparison/Graph_Comparison_Inputs_Results.mat`: already computed PC/Q/node measures. The public code performs their statistical comparisons; it does not yet reconstruct those measures from matrices.
- A sibling `TS_data.mat` above the analysis root: original aligned regional time series required by the connection-strength comparisons. Matrices alone are insufficient for this script as written. No time series are distributed.
- `10_DWI_QC_Check/ExtractedResults/20261001_212003/DWI_QC_Summary.csv`: DWI RMS and consistency flags.
- `09_Functional_QC_Check` and other motion CSV paths: exact filenames are shown in the descriptive script.
- `47_Major1_Simple_Model_20261007/Major2_Analysis/Major2_Gradients.mat` and `Gradient_Network_Results.csv`: externally computed gradient variants and existing FC results; the structural script recalculates the SC model using RMS.
- `22_Major2_Gradient_Robustness/SC_NOS_Sparsity_Sensitivity`, `SC_Distance_Attenuation_Sensitivity`, `SC_Tracking_Sensitivity`: precomputed gradient sensitivity inputs.
- `16_SC_Weight_Gradient_Results/20261002_161806/FA_Gradients.mat` and `17_SC_Network_Comparison/Network_Analysis_Inputs.mat`: precomputed aligned FA gradients.
- `48_subsample_sup/education_matched_subgroup`: approved matching pairs and comparison metadata. Matching itself is an external prerequisite; the code rechecks the supplied pairs and compares subgroup models.
- Clinical panel selection uses the prior analysis CSV/feature definitions named in the clinical script; selection is preserved rather than rediscovered by significance.
- Surface plotting uses nilearn/atlas resources and may download public template surfaces. Review the plotting scripts for exact resource paths.

`CODE_PROVENANCE.json` records original script hashes and MAT field names observed in the code. `original_index` is a historical alignment field, not a public disclosure of any actual record values. An authorized release may replace it with consistent nonidentifying tokens, but must preserve indexing needed by the time-series script. Do not publish the matching map.

## Scope and limitations
The fixed cohort checks deliberately prevent these scripts from being mistaken for a general analysis package. Upstream preprocessing, NSP calibration, gradient computation, graph metric computation, education matching and spin generation are prerequisites, not claimed to be reproduced by these scripts. The retained 13/six clinical-score families and FDR boundaries are study-specific. Some plots display individual observations: generated plots and files require a separate disclosure review.

Public syntax/privacy checks do not establish successful numerical reproduction. Runtime and numerical comparison against authorized study inputs must be completed before describing this as a fully verified reproduction package.

