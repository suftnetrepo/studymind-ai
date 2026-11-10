# Staging environment settings (AI backend)

Set these in the **staging** environment only. Code defaults are unchanged, so production behaviour is unchanged until
you choose to set them there.

| Variable | Staging value | Default | Notes |
|---|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.20` | `0.25` | Measured, see below. Re-tune on real staging documents. |
| `RETRIEVAL_VECTOR_K` | `100` | `100` | Vector candidate depth. Keeps keyword hits inside the vector result so every hit has a real similarity. |
| `OPENAI_CHAT_MODEL` | **confirm** | `gpt-4.1-mini` | A local backend `.env` sets `gpt-4o`. Run `python scripts/print_model_config.py` in staging. |
| `OPENAI_LLM_MAX_RETRIES` / `OPENAI_EMBED_MAX_RETRIES` | `1` / `1` | `3` / `10` | Bounds paid retries on failure. |
| `OPENAI_TIMEOUT_SECONDS` | `60` | `60` | |

`scripts/print_model_config.py` prints the effective models, prices, retrieval settings and worst-case per-request
reservations without any secret, and exits 1 if the chat model is not in the price table (it would then be priced as
the most expensive model).

## Retrieval threshold

Similarity is cosine similarity between the question and a chunk. Chunks that cover several topics score lower, so the
right threshold depends on chunk size. Measured on a synthetic corpus (5 documents, 3 modules, real OpenAI embeddings,
real Typesense; 104 queries: 50 relevant, 36 off-topic, 18 about another module):

| τ | recall, 512-token chunks (1-2 per doc) | recall, 128-token chunks (6-9 per doc) | out-of-module questions wrongly grounded (512 / 128) | wrong-document share of citations (512 / 128) |
|---|---|---|---|---|
| 0.15 | 100% | 100% | 28% / 39% | 31% / 15% |
| **0.20** | **94%** | **100%** | **17% / 22%** | **17% / 8%** |
| 0.25 | 84% | 100% | 0% / 6% | 2% / 4% |

Isolation held at every threshold (0 chunks from another module in 104 queries x 5 thresholds x 2 corpora). Off-topic
questions (sports, cooking...) retrieved nothing at 0.15 and above. Recommendation: **0.20**. Choose 0.25 only if a
spurious citation is judged worse than a missing one and chunks are small.

An earlier defect made the threshold ineffective: a chunk matched by keyword only had no vector distance and was given an
invented similarity of 0.75, so any keyword hit bypassed the threshold (41% of off-topic questions were "grounded"
whatever the threshold). It is fixed: such hits now score 0.0.
