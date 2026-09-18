# Working Paper, Version 1

- [Read the PDF](paper.pdf).
- [Read the generated Markdown](paper.md).
- [Edit the manuscript source](manuscript.md).
- [Inspect generated numeric claims](claims.json).

`build.mjs` reads `../artifacts/v1/aggregate.json` and the separate real-MCP
validation summary. Tables and key numeric claims are inserted from these data;
unknown placeholders fail the build. Rebuilding never calls a language model.

```sh
npm ci --prefix paper
npx --prefix paper playwright install chromium
.venv/bin/python paper/figures.py
npm run pdf --prefix paper
```

The renderer bundles math fonts and figures into an offline HTML document, then
uses headless Chromium for PDF generation. It checks math, loaded images, and
mobile document width; it does not touch the user's browser tabs. The HTML and
preview screenshots remain local. The PDF and Markdown are public artifacts.

This is a first empirical working draft, not a peer-reviewed publication. It
does not claim a new DR estimator, validated production safety, or results on
external agent benchmarks. The negative ranking and conservative-abstention
results are part of the paper, not placeholders awaiting more favorable runs.

Preparation note: code and prose were developed with AI assistance and checked
against executable tests and generated results. Author identity, affiliation,
licensing and any eventual submission metadata have not been invented here.
