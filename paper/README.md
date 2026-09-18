# Working Paper, Version 2

- [Read the PDF](paper.pdf).
- [Read the generated Markdown](paper.md).
- [Edit the manuscript source](manuscript.md).
- [Inspect generated numeric claims](claims.json).

`build.mjs` reads the frozen v1 and v2 aggregates, public-data diagnostics and
the separate real-MCP validation summary. Tables and key claims come from these data;
unknown placeholders fail the build. Rebuilding never calls a language model.

```sh
npm ci --prefix paper
npx --prefix paper playwright install chromium
.venv/bin/python paper/figures.py
.venv/bin/python paper/figures_v2.py
npm run pdf --prefix paper
```

The renderer bundles math fonts and figures into an offline HTML document, then
uses headless Chromium for PDF generation. It checks math, loaded images, and
mobile document width; it does not touch the user's browser tabs. The HTML and
preview screenshots remain local. The PDF and Markdown are public artifacts.

This is an empirical working draft, not a peer-reviewed publication. Version 2
adds complete-return controls, BFCL-derived group-held-out selection, two actual
local models and policy-contrast support analysis. It does not claim a new DR
estimator, established theoretical priority, production safety or official BFCL
leaderboard results. Negative and corrected findings remain part of the paper.

Preparation note: code and prose were developed with AI assistance and checked
against executable tests and generated results. Author identity, affiliation,
licensing and any eventual submission metadata have not been invented here.
