#!/usr/bin/env python3
"""
HyperGraphRAG Runner for GraphRAG-Bench Evaluation (Ollama Version)

This script processes the GraphRAG-Bench datasets using HyperGraphRAG
with local Ollama models (NO API keys required).

Usage:
    # Run with gpt-oss:20b (default)
    python run_hypergraphrag.py --subset novel --sample 3

    # Full benchmark
    python run_hypergraphrag.py --subset novel --output_file ../results/hypergraphrag_novel.json
"""

import asyncio
import os
import sys
import argparse
import json
import logging
import time
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path to import hypergraphrag
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hypergraphrag import HyperGraphRAG, QueryParam
from hypergraphrag.llm import ollama_model_complete, ollama_embed
from hypergraphrag.utils import wrap_embedding_func_with_attrs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Mapping from benchmark question types to evaluation types
QUESTION_TYPE_MAP = {
    "Fact Retrieval": "type1",
    "Complex Reasoning": "type2",
    "Contextual Summarize": "type3",
    "Creative Generation": "type4"
}

# Dataset paths relative to GraphRAG-Bench directory
SUBSET_PATHS = {
    "medical": {
        "corpus": "Datasets/Corpus/medical.json",
        "questions": "Datasets/Questions/medical_questions.json"
    },
    "novel": {
        "corpus": "Datasets/Corpus/novel.json",
        "questions": "Datasets/Questions/novel_questions.json"
    }
}


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Load corpus data from JSON file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_questions(questions_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSON file."""
    with open(questions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_questions_by_type(questions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group questions by their type for evaluation output format."""
    grouped = defaultdict(list)
    for q in questions:
        q_type = QUESTION_TYPE_MAP.get(q.get("question_type", ""), "type1")
        grouped[q_type].append(q)
    return grouped


def create_ollama_embedding_func(embed_model: str = "embeddinggemma", embedding_dim: int = 768):
    """Create embedding function for Ollama."""
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=2048)
    async def ollama_embedding(texts, **kwargs):
        return await ollama_embed(texts, embed_model=embed_model, **kwargs)
    return ollama_embedding


def initialize_hypergraphrag(
    working_dir: str,
    model_name: str = "gpt-oss:20b",
    embedding_model: str = "embeddinggemma",
    enable_cache: bool = True,
    ollama_host: str = "http://localhost:11434"
) -> HyperGraphRAG:
    """Initialize HyperGraphRAG instance with Ollama configuration."""

    os.makedirs(working_dir, exist_ok=True)

    # Create embedding function
    embedding_func = create_ollama_embedding_func(embedding_model)

    rag = HyperGraphRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=model_name,
        embedding_func=embedding_func,
        llm_model_kwargs={"host": ollama_host},
        enable_llm_cache=enable_cache,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=2,
        llm_model_max_async=4,
    )

    logger.info(f"Initialized HyperGraphRAG with Ollama")
    logger.info(f"  LLM: {model_name}")
    logger.info(f"  Embeddings: {embedding_model}")
    logger.info(f"  Working dir: {working_dir}")

    return rag


def ingest_corpus(rag: HyperGraphRAG, corpus_data: List[Dict]) -> None:
    """Ingest corpus documents into HyperGraphRAG."""
    for item in corpus_data:
        corpus_name = item.get("corpus_name", "Unknown")
        context = item.get("context", "")

        if not context:
            logger.warning(f"Empty context for corpus: {corpus_name}")
            continue

        logger.info(f"Ingesting corpus: {corpus_name} ({len(context)} characters)")
        rag.insert(context)
        logger.info(f"Successfully ingested: {corpus_name}")


def run_queries(
    rag: HyperGraphRAG,
    questions: List[Dict],
    mode: str = "hybrid",
    top_k: int = 60,
    sample_limit: int = None
) -> Dict[str, List[Dict]]:
    """
    Run queries for all questions and return results grouped by type.

    Args:
        rag: HyperGraphRAG instance
        questions: List of question dictionaries
        mode: Query mode (hybrid, local, global)
        top_k: Number of top results to retrieve
        sample_limit: Limit number of questions per type (for testing)

    Returns:
        Results grouped by question type in evaluation format
    """
    results = defaultdict(list)

    # Group questions by type
    grouped = group_questions_by_type(questions)

    for q_type, q_list in grouped.items():
        # Apply sample limit if specified
        if sample_limit and sample_limit < len(q_list):
            q_list = q_list[:sample_limit]

        logger.info(f"Processing {len(q_list)} questions of {q_type}")

        for q in tqdm(q_list, desc=f"Querying {q_type}"):
            try:
                # Configure query parameters
                param = QueryParam(
                    mode=mode,
                    top_k=top_k
                )

                # Execute query
                start_time = time.time()
                response = rag.query(q["question"], param=param)
                query_time = time.time() - start_time

                # Handle response format
                if isinstance(response, tuple):
                    answer, context = response
                else:
                    answer = str(response)
                    context = ""

                # Format result for evaluation
                result = {
                    "id": q.get("id", ""),
                    "question": q["question"],
                    "source": q.get("source", ""),
                    "context": context if isinstance(context, str) else str(context),
                    "evidence": q.get("evidence", []),
                    "question_type": q.get("question_type", ""),
                    "generated_answer": str(answer),
                    "gold_answer": q.get("answer", ""),
                    "query_time_seconds": round(query_time, 2)
                }

                results[q_type].append(result)

            except Exception as e:
                logger.error(f"Error processing question {q.get('id', 'unknown')}: {e}")
                # Add error result
                results[q_type].append({
                    "id": q.get("id", ""),
                    "question": q["question"],
                    "source": q.get("source", ""),
                    "context": "",
                    "evidence": q.get("evidence", []),
                    "question_type": q.get("question_type", ""),
                    "generated_answer": f"[ERROR: {str(e)}]",
                    "gold_answer": q.get("answer", ""),
                    "query_time_seconds": 0
                })

    return dict(results)


def save_results(results: Dict, output_path: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")

    # Print summary
    total = sum(len(v) for v in results.values())
    logger.info(f"Total questions processed: {total}")
    for q_type, items in results.items():
        logger.info(f"  {q_type}: {len(items)} questions")


def main():
    parser = argparse.ArgumentParser(
        description="Run HyperGraphRAG on GraphRAG-Bench datasets (Ollama)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset selection
    parser.add_argument(
        "--subset",
        required=True,
        choices=["medical", "novel"],
        help="Dataset subset to process"
    )

    # Paths
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Base directory for GraphRAG-Bench (auto-detected if not specified)"
    )
    parser.add_argument(
        "--working_dir",
        default="./hypergraphrag_workspace",
        help="Working directory for HyperGraphRAG cache"
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output file path (default: results/hypergraphrag_{subset}.json)"
    )

    # Model configuration (Ollama)
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Ollama LLM model name"
    )
    parser.add_argument(
        "--embed_model",
        default="embeddinggemma",
        help="Ollama embedding model"
    )
    parser.add_argument(
        "--ollama_host",
        default="http://localhost:11434",
        help="Ollama server URL"
    )

    # Query configuration
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "local", "global", "naive"],
        help="Query mode"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=60,
        help="Number of top results to retrieve"
    )

    # Other options
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of questions per type (for testing)"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable LLM response caching"
    )
    parser.add_argument(
        "--skip_ingest",
        action="store_true",
        help="Skip corpus ingestion (use existing index)"
    )

    args = parser.parse_args()

    # Determine base directory
    if args.base_dir:
        base_dir = args.base_dir
    else:
        # Auto-detect: script is in Examples/, base is parent
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get paths for selected subset
    paths = SUBSET_PATHS[args.subset]
    corpus_path = os.path.join(base_dir, paths["corpus"])
    questions_path = os.path.join(base_dir, paths["questions"])

    # Verify files exist
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found: {corpus_path}")
        sys.exit(1)
    if not os.path.exists(questions_path):
        logger.error(f"Questions file not found: {questions_path}")
        sys.exit(1)

    # Set output path
    if args.output_file:
        output_path = args.output_file
    else:
        results_dir = os.path.join(base_dir, "results")
        output_path = os.path.join(results_dir, f"hypergraphrag_{args.subset}.json")

    # Create working directory specific to subset
    working_dir = os.path.join(args.working_dir, args.subset)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("HyperGraphRAG - GraphRAG-Bench Evaluation (Ollama)")
    logger.info("=" * 60)
    logger.info(f"Subset: {args.subset}")
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Questions: {questions_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Working dir: {working_dir}")
    logger.info(f"LLM Model: {args.model}")
    logger.info(f"Embed Model: {args.embed_model}")
    logger.info(f"Query mode: {args.mode}")
    logger.info(f"Sample per type: {args.sample or 'ALL'}")
    logger.info("=" * 60)

    # Initialize HyperGraphRAG with Ollama
    rag = initialize_hypergraphrag(
        working_dir=working_dir,
        model_name=args.model,
        embedding_model=args.embed_model,
        enable_cache=not args.no_cache,
        ollama_host=args.ollama_host
    )

    # Ingest corpus (unless skipped)
    if not args.skip_ingest:
        logger.info("Loading and ingesting corpus...")
        corpus_data = load_corpus(corpus_path)
        ingest_corpus(rag, corpus_data)
    else:
        logger.info("Skipping corpus ingestion (using existing index)")

    # Load questions
    logger.info("Loading questions...")
    questions = load_questions(questions_path)
    logger.info(f"Loaded {len(questions)} questions")

    # Run queries
    logger.info("Running queries...")
    results = run_queries(
        rag=rag,
        questions=questions,
        mode=args.mode,
        top_k=args.top_k,
        sample_limit=args.sample
    )

    # Save results
    save_results(results, output_path)

    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Results saved to: {output_path}")
    logger.info("")
    logger.info("To evaluate results, run:")
    logger.info(f"  python -m Evaluation.generation_eval --data_file {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
