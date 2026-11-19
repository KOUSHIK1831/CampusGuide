# CampusGuide RAG upgrade

## Request flow

CampusGuide stores uploads, the document index, the Groq key, and chat history in the visitor's Streamlit session. An upload change invalidates the index and its chat. Only pretrained model resources are shared across sessions.

The first question is searched directly. Subsequent turns can use one LangChain and Groq call to resolve references using up to four successful turns. The rewrite sees bounded dialogue text and the current question. The answer call receives the original question, the search query for reference resolution, and the ranked excerpts as evidence. If rewriting fails or a changed query retrieves nothing, CampusGuide searches the original question.

Hybrid retrieval ranks cosine and BM25 results independently and combines them with reciprocal rank fusion. Strong lexical overlap or an exact identifier can surface a passage below the cosine cutoff. Exact duplicates from the same provenance are collapsed. An optional CPU cross-encoder scores a bounded set of candidates and keeps the original ranking when scoring fails.

The UI clears streamed draft text after completion, refusal, or provider failure. A completed answer needs in-range citation references before it enters the history used for rewriting. Excerpts and explicit policy metadata accompany accepted references. The prompt asks for competing citations when available documents disagree and authority remains unclear. Historical applicability and conflict interpretation depend on the passages delivered to the model.

## Benchmark method

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
python scripts/evaluate.py --backend faiss --reranker-model --details
```

[The fixture manifest](../evaluation/demo_cases.json) lists the local source snapshots and fictional documents. It contains 22 cases across 10 files, including five negatives, three follow-ups, and two policy comparison or conflict cases. Labels are repository-authored examples with explicit expected excerpts. The archived files are snapshots, and the fictional fixtures describe invented policies. There was no independent human review.

The comparison uses one index and the same cutoff and top-k across methods. The evaluator matches each labeled excerpt and source, reports the first relevant rank and coverage of all expected evidence, and records empty retrievals on negative queries. The report includes hashes, model IDs, package versions, runtime settings, and case-level diagnostics. The three supplied standalone queries are evaluated separately as reference queries. They are written into the fixture and give no measurement of Groq's rewrite quality.

## Measured results

The FAISS run used 21 chunks, top-k 4, and a dense cutoff of 0.30. It ran on Windows with Python 3.12.10, FAISS 1.16.0, NumPy 1.26.4, Sentence Transformers 5.7.0, Transformers 5.23.0, and PyTorch 2.15.0+cpu with two CPU threads. The encoder was `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243` and the cross-encoder was `cross-encoder/ms-marco-MiniLM-L2-v2` at revision `1b5cd67b`.

| Strategy | Evidence hit@4 | Full evidence@4 | MRR@4 | Negative queries with zero hits | Warm p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense cosine | 13/17 | 13/17 | 0.6078 | 2/5 | 34.00 / 39.99 ms |
| Dense + BM25 | 13/17 | 13/17 | 0.6471 | 2/5 | 34.37 / 53.23 ms |
| Dense + BM25 + cross-encoder | 13/17 | 13/17 | 0.6471 | 2/5 | 108.21 / 480.39 ms |

- Hybrid retrieval moved the archived makeup-grade excerpt from rank 2 to rank 1 and the lost-ID excerpt from rank 3 to rank 2. The academic-advising question missed in all three runs.
- Reranking changed individual ranks and matched the aggregate hybrid scores. It ran on 19 of 25 queries, including the supplied follow-ups. Six queries produced zero candidates. Reranking is disabled by default.
- All three methods retrieved full evidence for both policy comparison or conflict cases. On follow-ups, all three methods found evidence for 0/3 original questions and 3/3 supplied standalone queries.
- Three of five negative queries returned passages in every strategy. Empty retrieval and final refusal are separate measurements. The NumPy run reproduced the FAISS ranking metrics. The latency values are from one warm CPU run over this small corpus.

## Verification scope

The automated tests use mocked provider responses, including streamed events through the real LangChain and Groq client libraries. The offline benchmark uses local model inference with no provider key. The test suite covers citation reference checks, and the benchmark measures retrieval. Accuracy of generated claims, model handling of policy conflicts, runtime rewrite quality, and live Groq latency require separate evaluation.

## Résumé wording

> Built CampusGuide, a conversational document Q&A app using LangChain, Groq, Sentence Transformers and FAISS. Implemented BM25 and dense rank fusion, optional CPU reranking, and streamed answers with source-reference checks. Created a reproducible 22-case retrieval benchmark covering follow-ups, conflicting policies, and out-of-scope questions.

## API references

- [LangChain ChatGroq integration and streaming API](https://reference.langchain.com/python/langchain-groq/chat_models/ChatGroq)
- [LangChain chat prompt templates](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)
- [Groq reasoning parameters for GPT-OSS](https://console.groq.com/docs/reasoning)
- [Sentence Transformers cross-encoder inference](https://sbert.net/docs/package_reference/cross_encoder/model.html)
