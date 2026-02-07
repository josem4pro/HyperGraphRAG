# CLAUDE.md - HyperGraphRAG

**Repo:** josem4pro/HyperGraphRAG (fork of HKUDS/HyperGraphRAG)
**Version:** 1.0.6
**License:** MIT
**Paper:** https://arxiv.org/abs/2503.21322 (NeurIPS 2025)

---

## Description

Retrieval-Augmented Generation system using **hypergraph-structured knowledge representation** instead of traditional entity-relation triples. Preserves complete knowledge fragments as hyperedges, maintaining semantic integrity of complex multi-entity relationships.

**Jose's fork extensions:**
- Multi-LLM backend support: Gemini (GCP Vertex AI), Groq (GPT-OSS-120B, Qwen3-32B, Kimi-K2), expanded Ollama
- Contextual.ai reranker integration (second-layer noise filtering)
- Multi-model benchmark infrastructure (`run_benchmark_master.py`, per-chapter extraction)
- GraphRAG-Bench integration (novel/medical datasets, 6 evaluation metrics)

---

## Commands

### Setup
```bash
conda create -n hypergraphrag python=3.11
conda activate hypergraphrag
pip install -r requirements.txt
export OPENAI_API_KEY='your_key'
```

### Basic Usage
```bash
python script_construct.py          # Document ingestion (build KG)
python script_query.py              # Query the knowledge graph
python script_ollama_local.py       # Local Ollama model setup
python ingest_paper.py              # PDF paper ingestion
```

### Benchmark (Jose's additions)
```bash
python run_benchmark_master.py --llm groq --reranker contextual  # Master benchmark
python run_benchmark_master.py --compare-all                      # Compare all models
python run_chapters_gemini3.py                                    # Per-chapter extraction (Gemini 3)
python run_chapters_groq.py                                       # Per-chapter extraction (Groq)
python run_all_chapters_glm.py                                    # Per-chapter extraction (GLM)
python compare_chapter1.py                                        # Cross-model comparison
```

### Evaluation Pipeline
```bash
cd evaluation
python script_insert.py --cls hypertension                                            # Build KG
python script_hypergraphrag.py --data_source hypertension                             # Retrieve
python get_generation.py --data_sources hypertension --methods HyperGraphRAG          # Generate
CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source hypertension --method HyperGraphRAG  # Score
python see_score.py --data_source hypertension --method HyperGraphRAG                 # View
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Primary Language | Python (84 files, ~22,879 LOC) |
| Total Files | 294 |
| Total LOC | ~136,692 (including JSON data) |
| Git Commits | 56 on main |
| Python Version | 3.11 |
| Core Module | 9 .py files, ~4,700 LOC |

---

## Structure

```
HyperGraphRAG/
|-- hypergraphrag/                    # CORE MODULE (~4,700 LOC)
|   |-- hypergraphrag.py              # Main class: insert/query orchestrator (549 LOC)
|   |-- operate.py                    # Entity extraction, kg_query(), context building (1,201 LOC)
|   |-- llm.py                        # All LLM integrations (1,535 LOC -- largest file)
|   |-- base.py                       # Abstract bases: BaseVectorStorage, BaseKVStorage, BaseGraphStorage
|   |-- storage.py                    # Default impls: JsonKV, NanoVectorDB, NetworkX
|   |-- prompt.py                     # LLM prompt templates
|   |-- utils.py                      # Token processing, hashing, caching, async utils
|   |-- rerank.py                     # Contextual.ai reranker (simple wrapper)
|   |-- contextual_reranker.py        # Advanced reranker with instruction presets (325 LOC)
|   |-- kg/                           # KG backend plugins
|       |-- neo4j_impl.py, oracle_impl.py (759 LOC), tidb_impl.py, chroma_impl.py, milvus_impl.py, mongo_impl.py
|-- evaluation/                       # Evaluation pipeline (upstream original)
|   |-- hypergraphrag/                # DUPLICATE of core module (evaluation isolation)
|   |-- simcse/                       # SimCSE semantic similarity scoring
|   |-- script_insert.py, script_hypergraphrag.py, get_generation.py, get_score.py
|-- GraphRAG-Bench/                   # Benchmark framework (Jose)
|   |-- Datasets/Corpus/              # novel.json, medical.json
|   |-- Datasets/Questions/           # novel_questions, medical_questions
|   |-- Evaluation/metrics/           # answer_accuracy, context_recall, coverage, faithfulness, rouge
|-- chapters_gemini25_flash/          # Extraction results: Gemini 2.5 Flash (22 chapters)
|-- chapters_gemini3_flash/           # Extraction results: Gemini 3 Flash
|-- chapters_glm47_flash/             # Extraction results: GLM 4.7 Flash
|-- chapters_groq_gpt120b/            # Extraction results: Groq GPT-OSS 120B
|-- hypergraphrag_workspace/          # Working data (KG graphs, VDB, KV stores)
|-- IA-BACKGROUND/                    # Research docs (Claude Code, Gemini, NotebookLM, Perplexity)
|-- run_benchmark_master.py           # Master benchmark runner (410 LOC)
|-- test_reranker.py, test_ab_reranker.py  # Ad-hoc test scripts
```

---

## Architecture

### Core Pipeline
- **Entry Point:** `HyperGraphRAG` class in `hypergraphrag.py` (dataclass config, async-first)
- **Ingestion:** `insert()` -> `chunking_by_token_size()` -> `extract_entities()` -> store KV + VDB + Graph
- **Query:** `query()` -> `kg_query()` -> retrieval (hybrid) -> optional reranking -> LLM answer
- **Query Modes:** local (entity-based), global (hyperedge-based), hybrid (both), naive (direct chunks)

### LLM Integrations (`llm.py` - 1,535 LOC)
Original: OpenAI, Azure OpenAI, Bedrock, HuggingFace, Ollama, ZhiPu
Jose added: `gemini_gcp_complete()`, `gemini_complete()`, `groq_complete_if_cache()`, `groq_gpt_oss_120b_complete()`, `groq_qwen3_32b_complete()`, `groq_kimi_k2_complete()`, `GroqRateLimiter`, `MultiModel`

### Pluggable Storage
- **KV:** JSON (default), Oracle, MongoDB, TiDB
- **Vector:** NanoVectorDB (default), Milvus, Chroma, TiDB
- **Graph:** NetworkX (default), Neo4j, Oracle
- Lazy-loaded via `lazy_external_import()` pattern

### Code Patterns
- **Async-first:** `insert()`/`ainsert()`, `query()`/`aquery()`, `delete_by_entity()`/`adelete_by_entity()`
- **Prompt delimiters:** `<SEP>` (field), `<|>` (value), `##` (record), `<|COMPLETE|>` (done)
- **Caching:** MD5 hash-based LLM response cache (`compute_args_hash()`), file-based JSON persistence

### Key Configuration Defaults
```python
chunk_token_size=1200, chunk_overlap_token_size=100
entity_extract_max_gleaning=2, entity_summary_to_max_tokens=500
llm_model_max_token_size=32768, enable_llm_cache=True
node_embedding_algorithm="node2vec"
```

---

## Dependencies

- **PyTorch 2.3.0** -- Deep learning
- **NetworkX** -- Default graph storage
- **OpenAI/Ollama/Gemini/Groq APIs** -- LLM providers
- **tiktoken** -- Token counting (gpt-4o-mini model)
- **nano-vectordb** -- Default lightweight vector storage
- **tenacity** -- Retry logic with exponential backoff
- **Contextual.ai API** -- Reranker (Jose's addition)

---

## Risks / Known Issues

| Risk | Severity | Details |
|------|----------|---------|
| Code Duplication | HIGH | `evaluation/hypergraphrag/` is a full copy of core module (11 files). Changes require manual sync. |
| Hardcoded Paths & Credentials | MEDIUM | Placeholder API keys in scripts, hardcoded `/home/jose/` paths, `~/.env` manual read. |
| Missing Test Infrastructure | MEDIUM | No pytest, no CI/CD. Test files are ad-hoc experiment runners without assertions. |
| LLM Module Monolith | MEDIUM | `llm.py` at 1,535 LOC with 40+ functions/classes for all providers. No separation. |
| Experiment Data in Repo | LOW | 150+ chapter result JSONs, checkpoint chunks, large PNGs/PDFs tracked in git. |
| Query Mode Limitation | LOW | `aquery()` only handles "hybrid" mode despite QueryParam supporting local/global/naive. |
| Python Version Mismatch | LOW | README specifies 3.11, but .pyc files compiled with 3.12. |
