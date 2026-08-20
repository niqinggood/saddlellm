# Overleaf layout guide

## Upload and main file

Upload the complete `paper/` directory to Overleaf and select `main.tex` as the
main document. The current document class is a neutral two-column draft, not an
official conference template. Once the venue is fixed, replace the document
class and venue-specific bibliography settings using the official template;
keep the section files and figure paths unchanged.

## Planned paper structure

1. Introduction
2. Related Work
3. Preliminaries and Problem Formulation
4. Methodology
5. Experiments
6. Discussion
7. Limitations
8. Conclusion
9. Appendices

Experiments must remain a separate section. Discussion interprets evidence and
must not substitute for reporting results.

## Figure placement

| Figure | Environment | Width | Preferred placement |
|---|---|---:|---|
| Problem/method overview | `figure*` | `0.96\textwidth` | Top of page 2 |
| RTG example | `figure` | `0.96\columnwidth` | Beside the RTG subsection |
| Transfer heatmap | `figure*` | `0.94\textwidth` | Start of main results |
| DRC prediction | `figure` | `0.96\columnwidth` | Immediately after prediction setup |
| Capability flow | `figure*` | `0.94\textwidth` | Top of capability-results page |

Use `[t]` for conference floats. Avoid `[H]`, repeated `\vspace`, or manual page
breaks during drafting. A two-column `figure*` usually appears at the top of a
later page, so place its LaTeX block approximately half a page before the ideal
visual location. Refer to every figure in the prose before or near its first
appearance.

## Figure sizing

- Single-column plot: `\columnwidth`, normally 45--60 mm tall.
- Double-column plot: `\textwidth`, normally 55--75 mm tall.
- Use the same font family and notation as the paper.
- Target at least 7--8 pt figure text at final size.
- Use consistent panel labels `(a)`, `(b)`, and `(c)`.
- Put essential interpretation in the caption; captions should be understandable
  without searching the main text.

## Tables

- Use `booktabs`; do not use vertical rules.
- Right-align numbers and keep decimal precision consistent.
- State whether values are means, standard deviations, or confidence intervals.
- Bold only according to a declared rule; do not manually highlight convenient
  cells.
- Keep complete per-benchmark/per-seed tables in the appendix if the main table
  would become unreadable.

## Draft placeholders

Missing figures compile as framed placeholders through
`\figureorplaceholder`. Unavailable results appear in red through
`\resultplaceholder`. Before submission:

1. replace all result placeholders;
2. set `\paperdraftfalse` in `macros.tex`;
3. search the compiled PDF and source for `placeholder`, `TODO`, `--`, and red
   text;
4. confirm that no float covers text or crosses the page boundary;
5. check the PDF at 100% zoom and in grayscale.

## Recommended page emphasis

- Introduction: 10--12%
- Related Work: 8--10%
- Preliminaries: 8--10%
- Method: 20--25%
- Experiments: 35--40%
- Discussion: 6--8%
- Limitations: 4--5%
- Conclusion: 2--3%

Do not shrink figure text or table fonts to compensate for an overlong paper.
Move secondary results and implementation details to the appendix first.

