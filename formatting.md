Paper Polishing Instructions (IEEE Format Compliant)

(Improve visual appeal, clarity, structure, and professionalism)

Agent — you will enhance the appearance, readability, and polish of the IEEE paper draft.
Execute the steps in order. After each step, append a line to logs/agent_paper_edit.log describing the change.

STEP 1 — Improve Title Block (IEEE-safe Enhancement)

Open main.tex.

Modify the title to include a short descriptive subtitle on the next line, following IEEE rules for line breaks:

\title{Predictive Design of Perovskite Materials for Solar Cell Applications\\
\large A High-Accuracy Machine Learning Pipeline for Bandgap Engineering}


Do not change font, spacing, or title style beyond allowable IEEE formatting.

Log completion.

STEP 2 — Rewrite Abstract for Better Flow and Structure

Rewrite the abstract into six logically separated sentences, maintaining a single paragraph as IEEE requires.

The abstract must include:

Motivation sentence (PV → costly DFT)

Contribution sentence (end-to-end ML pipeline)

Dataset sentence (double perovskites, MP API, curated features)

Key quantitative results (F22 + F10 metrics)

Interpretability statement (SHAP)

Impact sentence (rapid screening and candidate discovery)

Ensure transitions:
“Specifically,” “In particular,” “To this end,” “We demonstrate that…”

Save revised abstract and log.

STEP 3 — Insert a Pipeline Diagram (Fig. 1)

Generate a simple horizontal workflow block diagram image with labels:

Materials Project API → Data Cleaning → Feature Engineering →
Feature Selection (F4–F24) → Model Training → Evaluation → Candidate Screening


Save as figures/pipeline_diagram.png.

Insert into LaTeX after Introduction Section as:

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/pipeline_diagram.png}
\caption{Overview of the machine-learning pipeline used for bandgap prediction and screening of perovskite materials.}
\label{fig:pipeline}
\end{figure}


Ensure IEEE-compatible figure size: full-column width ≤ 3.5 inches.

Log completion.

STEP 4 — Add Feature-Subset Performance Table (F4 → F24)

Create a table summarizing all F-subset model performances:

Feature Set	# Features	R²	MAE (eV)	RMSE (eV)
F4	4	0.6755	...	...
...	...	...	...	...
F22	22	0.7620	...	...
F24	24	0.7615	...	...

Save as a LaTeX table in main.tex under “Methods → Feature Selection”.

Use IEEE table formatting (top rule, midrule, bottom rule).

Log completion.

STEP 5 — Improve Section Headings for Clarity

Modify section intros to add 1–2 sentences of motivation each (IEEE-appropriate):

Add at beginning of each major section:

Section II (Related Work)
Add: “We situate our work within existing machine-learning methods for bandgap prediction, perovskite modeling, and interpretable materials informatics.”

Section III (Methods)
Add: “This section outlines the end-to-end machine-learning pipeline used for perovskite bandgap prediction, from data acquisition to model selection.”

Section IV (Results)
Add: “We evaluate our regression and classification models using standard metrics and analyze performance across feature subsets.”

Do not exceed 2 sentences per section.
Log completion.

STEP 6 — Add Callout Sentences (IEEE-friendly)

For each major result (F22 model, classifier accuracy, SHAP insight), add a single boldface callout sentence at the beginning of the paragraph:

Example:

\textbf{Our F22 LightGBM model achieves R² = 0.88 and MAE = 0.35 eV.}


Add 3–5 such callouts in the Results section to increase readability.
Log completion.

STEP 7 — Add a SHAP Figure and Caption

Generate SHAP summary plot (F22 model).

Save as figures/shap_summary.png.

Insert figure near Results section:

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/shap_summary.png}
\caption{SHAP summary plot showing the dominant thermodynamic and electronic factors influencing the predicted bandgap.}
\label{fig:shap}
\end{figure}


Log completion.

STEP 8 — Improve Text Flow with IEEE-style Transitional Connectors

Automatically scan the introduction and methods sections.
Insert transition connectors where needed:

“Specifically,”

“In particular,”

“To this end,”

“Moreover,”

“Furthermore,”

“Consequently,”

Follow IEEE editorial rules: no more than 1 connector per paragraph.
Log the locations modified.

STEP 9 — Refine Conclusion Section

Rewrite the Conclusion to include:

One sentence summarizing contributions

One sentence summarizing performance gains

One sentence highlighting interpretability

One sentence stating practical impact (screening candidates)

One sentence identifying future work

Save revisions and log.

STEP 10 — Rebuild PDF and Generate Diff Summary

Compile LaTeX into PDF.

Compare old vs new PDF using a simple diff tool (latexdiff or PDF MD5 hashes).

Save difference summary to results/paper_diff_summary.txt.

Log finalization.