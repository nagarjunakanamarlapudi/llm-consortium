# LLM Consortium — IEEE paper (LaTeX / arXiv source)

LaTeX source for the IEEE conference paper, built from `../LLM_Consortium_IEEE_Paper.docx`
(commit `73e30ab`) and formatted for arXiv submission.

## Files
- `main.tex` — the complete paper. `\documentclass[conference]{IEEEtran}`; bibliography is inline
  via `thebibliography` (no `.bib`/`.bbl` needed).
- `figures/image1.png … image9.png` — the 9 figures (extracted from the docx; Fig. 1 = workflow
  topologies, … Fig. 9 = three-evaluator cross-validation).
- `main.pdf` — local compiled preview (11 pages, US Letter).

## Compile
```
tectonic -X compile main.tex        # what was used here (tectonic 0.16.9)
# or: pdflatex main.tex  (run twice so cross-references resolve)
```

## Submitting to arXiv
arXiv does **not** accept `.docx`, and **rejects PDFs generated from LaTeX** — it wants the source.
Upload the LaTeX source: either `../arxiv-submission.zip` (already packaged) or `main.tex` +
the `figures/` folder. PNG figures are fine with pdfLaTeX. Primary category suggestion: `cs.SE`
(cross-list `cs.AI`). arXiv compiles the source on its end.

## Notes
- Author block: Nagarjuna Kanamarlapudi & Praveen K (both Fremont, USA) with emails.
- Figures are auto-numbered 1–9; the source docx skipped "Fig. 2" — fixed here via `\ref`.
- The docx is the editable Word source; this LaTeX tree is the arXiv deliverable. Keep them in sync
  if the paper changes.
