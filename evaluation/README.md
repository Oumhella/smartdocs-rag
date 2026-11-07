# SmartDocs evaluation

## Retrieval benchmark

`datasets/sickness_policy.json` contains 30 manually labeled questions. Each
entry records an expected answer, source filename, page, and a passage that
must appear in a retrieved chunk for that result to count as relevant. The
benchmark uses the persistent Chroma collection already used by SmartDocs. If
the source is not indexed, the runner indexes `data/mpe-sickness-absence-policy.pdf`
when that file is available.

From the project root in PowerShell, run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.evaluation.run --top-k 4
```

The command writes `evaluation/results/latest.json`. The report includes the
per-question rank, scores, timing, and previews of the retrieved chunks.

### Metrics

- **Hit Rate@K:** fraction of questions with the expected source, page, and
  labeled passage in the top K. `26/30` means `0.867` at K=4.
- **MRR@K:** mean reciprocal rank of the first matching result. A rank-1 hit
  contributes `1`, rank 2 contributes `0.5`, and a miss contributes `0`.
- **Mean retrieval latency:** average local search and query-embedding time per
  question after one unscored query warms the embedding model. It excludes
  model initialization, LLM generation, and network latency.

### Current run

The checked-in report records 30 questions at K=4: Hit Rate@4 **0.867**,
MRR@4 **0.783**, and mean warmed retrieval latency **19.87 ms**. Twenty-six questions
matched the labeled passage; four did not. The source PDF is not currently in
`data/`, so this run used its existing local Chroma index. This is a small,
single-document benchmark and a machine-specific baseline, not a general
quality claim.

## Reliability checks

Run the automated checks with:

```powershell
python -m unittest discover -s tests -v
```

The checks cover empty and corrupted uploads, duplicate uploads, long-document
chunk bounds, low-relevance questions avoiding the LLM, injection text being
marked as untrusted context, and the metric calculations.

## Limitations

- Gold labels were written for one example policy document. Questions and
  passages may not represent other domains or document layouts.
- A result counts only when the exact normalized gold passage occurs in the
  retrieved chunk. Paraphrases can be relevant but still score as misses.
- Hit Rate@K and MRR evaluate retrieval, not whether generated answers are
  faithful, complete, or relevant. Those need human ratings or a separate answer
  evaluation design.
- Prompt-injection checks verify that the instruction and document text are
  passed with an untrusted-data boundary. They do not prove a live LLM will
  resist every injection.
- Duplicate-upload checks verify identical filename and bytes. Renamed copies
  are treated as separate documents. Documents indexed before content hashes
  were added are not recognized as duplicates until they are indexed again.
- Empty/corrupt upload and long-text checks are local reliability tests; they
  do not measure memory use on very large real-world files.
- Retrieval latency excludes model initialization, answer generation, and full
  browser-to-API response time.
- The latest report can depend on local `data/chroma/`. Reproduce it from a
  clean environment by providing the source PDF at the path above and using the
  same embedding model and chunking settings.
