# Submission Readiness Review

Reviewed on 2026-09-18. Target selected by the author: arXiv preprint first.
This is a submission preparation record, not evidence of submission or acceptance.

## Assessment

**Version-2 update:** the original concerns below are retained as review history.
Complete realized-return and component controls are now implemented, and one
initial DR advantage reverses under the stronger direct baseline. Public BFCL-
derived function selection with group-disjoint splits and two local Qwen2.5
baselines provides independent evidence, but not official BFCL execution scores
or enterprise deployment validation. Disagreement-support analysis adds a precise
proposition, executable identification bounds and fallback ablations; it does not
establish that this mathematical observation is new to the broader literature.
See [EXPERIMENTS_V2.md](EXPERIMENTS_V2.md) and [v2 results](artifacts/v2/tables.md).

### Original Version-1 Review

The artifact is stronger as a reproducible engineering study than as a new
algorithm paper. Exact logging, complete-action authorization, transport parity,
negative results and source-linked tables are useful. They do not by themselves
establish a new learning method or generalizable findings for enterprise agents.

The most important scientific concerns in version 1 were:

1. **External validity.** All outcomes come from eleven hand-designed tools and
   a finite synthetic generator. There is no independent benchmark, LLM planner,
   large candidate catalog or deployment dataset. See manuscript Sections 6 and 8.
2. **Comparator limitation.** The direct model predicts two outcome components
   and subtracts nominal costs; the DR correction uses realized total utility.
   Revocation and failures change actual charges/latency/access. A direct model
   of complete realized utility is a missing control. See `learning.py`
   `action_penalty`, `Learner.fit` and `Learner.rank`. The manuscript now states
   this limitation explicitly; no frozen result has been changed.
3. **Modest research novelty.** DR, deficient support and deterministic permission
   gates are established ideas. The paper should be presented as a diagnostic
   empirical study, not a newly invented estimator or new security theorem.
4. **Selection and statistical uncertainty.** Five seeds, limited calibration,
   shared templates and no matched-coverage risk comparison do not support a
   universal method ranking. The negative conservative-policy result is useful,
   but it is not proof that calibrated abstention in general fails.

Recommendation: a bounded preprint is reasonable after author review. A relevant
workshop or empirical/reproducibility track is more plausible than a flagship
new-method paper in its present form. TMLR is in scope for empirical insights but
not automatically suitable: its criteria explicitly distinguish generalizable
findings from an educational reimplementation. No acceptance probability is claimed.

Remaining priorities before a serious peer-reviewed submission are a fully
external executable workload, broader/stronger model comparisons, semantic family
generalization and fixed-policy OPE stress tests reporting bias, variance and
interval coverage. The complete-return baseline and function-name-group holdout
are now addressed in separately numbered experiments. V1 is not rewritten.

## Prepared arXiv Package

The author has explicitly confirmed the final publication spelling **Jiapeng Li**,
affiliation **Microsoft**, and sole authorship. The earlier draft spelling
`jiapengli` has been replaced in both the manuscript and submission metadata.
No email, ORCID or employer endorsement has been invented.

The author selected the [arXiv non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
and confirmed review of version 2, responsibility for its content, agreement to
the AI-assistance disclosure, and the right to publish with the stated identity
and affiliation. These are author-provided confirmations, not independently
verified employer approval. License selection is not a grant to arXiv until the
author completes its submission process.

```sh
npm run pdf --prefix paper
node paper/prepare-arxiv.mjs
.venv/bin/python paper/check_submission.py
```

The local `submission/` directory contains one upload PDF, a plain-text abstract,
form metadata, a PDF preflight report and a status manifest. Upload only the PDF
for a PDF-only arXiv submission, not the JSON/status files. The PDF is rendered
from Markdown/HTML, not from a hidden TeX source. The original experimental
artifacts and all reported numeric results remain unchanged.

Proposed primary category: `cs.LG` (machine learning). Moderators may reclassify
it. arXiv is a moderated preprint repository, not peer review or journal acceptance.

## Current External Status

- An authenticated arXiv author account is not available to this submission workflow.
- Accessing <https://arxiv.org/user> returns the login page.
- The registration page requires an email, username, password and verification.
- No account was created and no credentials were collected.
- No file has been uploaded to arXiv, no final submission was made, and no
  submission number or arXiv identifier exists yet.

The author must complete registration and email verification directly at
<https://arxiv.org/user/register>. A first submission may require subject-area
endorsement. An institutional email alone does not guarantee endorsement under
the current policy. Do not impersonate an author or mass-email endorsers.

Before clicking the final submission button, the author must review the processed
PDF and metadata and accept the submission agreement on the website. The final
author list, license preference, paper review and AI-assistance disclosure are
confirmed; account verification, any required endorsement, website agreement
acceptance and final submission are not. Preparation does not automatically
accept legal terms or press the final submission button.

## Verified Requirements

- [arXiv submission guide](https://info.arxiv.org/help/submit/index.html)
- [PDF-only requirements](https://info.arxiv.org/help/submit_pdf.html)
- [Registration](https://info.arxiv.org/help/registerhelp.html)
- [Endorsement](https://info.arxiv.org/help/endorsement.html)
- [Moderation and generative-AI disclosure](https://info.arxiv.org/help/moderation/index.html)
- [Submission agreement and license choices](https://info.arxiv.org/help/policies/submission_agreement.html)
- [TMLR acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
- [TMLR author requirements](https://jmlr.org/tmlr/author-guide.html)

The public identity-bearing PDF is not an anonymous TMLR submission. A later
TMLR submission would require its approved format, anonymous supplementary
material, complete author profiles and declarations. Do not submit the same work
simultaneously to incompatible archival peer-reviewed venues.

## Originality Review

A pre-submission review compared the final manuscript against the eleven central
cited papers using normalized contiguous-token matching. After excluding the
References section, no source shared a contiguous sequence of eight or more
normalized words with the manuscript. Matches in the complete manuscript were
8--14 words and were paper titles in the bibliography. Exact searches for four
distinctive manuscript phrases produced no result in public GitHub code search
outside this repository; general web-search coverage was incomplete.

Repository documentation shares short result summaries with the paper. The
longest measured internal overlap was ten normalized words, consisting of a
numeric result statement or method configuration. This is self-repetition within
the same research artifact, not evidence of copying another author's prose.

The review also identified two attribution-distance risks and corrected them:
the introduction now cites the contextual-bandit/logged-feedback framing [1,2]
at first use, and the contrast-support proposition directly identifies its
relationship to deficient-support and baseline-restriction work [3,4].

This review is not a Turnitin/iThenticate report, does not search every book,
paywalled article, webpage or unpublished manuscript, and cannot certify zero
plagiarism. The human author remains responsible for originality, accurate
quotation, citation and rights to all submitted material. arXiv performs its own
text-overlap and moderation checks after submission.
