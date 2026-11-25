---
paths:
  - "docs/**"
  - "README.md"
---

# Writing style (docs)

1. Project log in English, verdicts included (`**Verdict: success**`). Fact first, then exact numbers: 3 decimals,
   Δ with sign, 95% CI in brackets, arrows for change: `0.021 → 0.274, Δ +0.253 [+0.204, +0.303]`.
   Every result has its W&B run / artifact link and commit hash.
2. Facts and hypotheses apart: a hypothesis is marked "Likely (not tested): …".
3. No judgement words ("significant", "great", "huge"). A difference is supported only by its CI.
4. Describe the current state: no dates in titles, no "removed X" entries, update the description instead.
5. Run reports (`docs/sft-*.md`) share one structure: setup → criteria table → verdict → what went wrong.
   Tables over long paragraphs.
