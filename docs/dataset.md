# Dataset plan

Based on [COBOL-Coder](https://arxiv.org/html/2604.03986v1) (Dau et al., 2026). The paper uses GPT-4o;
we use open models only (Qwen itself) because we can't call external APIs.

## Sources

| # | Source | Method | Paper result | Status |
|---|---|---|---|---|
| 1 | Real COBOL: The Stack v1 dedup COBOL subset + X-COBOL (GitHub) | MinHash dedup → compile with GnuCOBOL → failing programs get up to 3 repair attempts by Qwen, using the compiler errors | ~31K programs | **Start here** |
| 2 | Instruction data | Generate a task description for each program; rejection sampling keeps solutions that compile and pass tests | — | Planned |

## Skipped

- **Knowledge Q&A (textbooks):** the books are licensed, and the set adds little for code generation.
- **Synthetic COBOL from Java (Stack v2):** dropped; the paper's ~173K Java→COBOL pairs. Revisit only if the baseline shows data size is the bottleneck.
- **Translation as a task:** out of scope unless we also evaluate on COBOL-JavaTrans.

## Always

- Remove anything that overlaps COBOLEval (and COBOLCodeBench, if added) before training.
- Evaluate on COBOLEval (146 tasks) and add COBOLCodeBench (46 harder tasks), so our results compare directly with the paper's.
