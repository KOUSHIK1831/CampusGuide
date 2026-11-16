# Copilot review follow-up

This branch addresses four comments on [PR #1](https://github.com/KOUSHIK1831/CampusGuide/pull/1). The original PR merged before Copilot reviewed it.

| Finding | Resolution | Regression test |
| --- | --- | --- |
| Concurrent rebuild mismatches search metadata | Publish an index and metadata together as one immutable snapshot; searches retain that snapshot even during rebuilds. Each store locks save/load. | `test_search_keeps_old_index_with_old_metadata_during_rebuild` |
| Cached auto-indexing failure | Already fixed in BYOK redesign: no cached index, explicit per-session upload processing. Only the pretrained model is cached. | Existing upload/session isolation tests |
| Source citations omit supporting evidence | Show the same 1,250-character passage given to Groq, not a 400-character excerpt. | `test_citations_include_all_passage_text_sent_to_llm` |
| Malformed DOCX aborts indexing | Existing parser converts corrupt DOCX to `ValueError`. CLI ingestion skips bad files. BYOK upload intentionally rejects the whole invalid batch so users do not chat against incomplete uploads. | `test_bad_docx_does_not_abort_directory_ingestion`, `test_bad_docx_upload_rejects_entire_batch_without_partial_index` |

## Boundaries

- Snapshot consistency is enforced within each `VectorStore` instance. Independent CLI processes concurrently writing to the same output path still require coordination.
- No API secrets are added to source or tests.
- A valid citation number does not guarantee factual entailment.
- This PR stays unmerged until CI and the new Copilot review are checked and the owner approves.
