#!/usr/bin/env python3
"""
Benchmark HyperGraphRAG con Novel-30752 (corpus ya indexado)
Con logging detallado pregunta por pregunta.
"""

import os
import sys
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hypergraphrag import HyperGraphRAG, QueryParam
from hypergraphrag.llm import ollama_model_complete, ollama_embed
from hypergraphrag.utils import wrap_embedding_func_with_attrs

# Configuración
WORKING_DIR = "./hypergraphrag_workspace/novel"
MODEL = "gpt-oss:20b"
EMBED_MODEL = "embeddinggemma"
SAMPLE_PER_TYPE = 3  # 3 preguntas por tipo

print("=" * 70)
print("BENCHMARK: HyperGraphRAG + Novel-30752")
print("=" * 70)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Modelo: {MODEL}")
print(f"Sample por tipo: {SAMPLE_PER_TYPE}")
print("=" * 70)

# Crear embedding function
@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=2048)
async def ollama_embedding(texts, **kwargs):
    return await ollama_embed(texts, embed_model=EMBED_MODEL, **kwargs)

# Inicializar RAG (usando índice existente)
print("\n[1/4] Inicializando HyperGraphRAG...")
rag = HyperGraphRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name=MODEL,
    embedding_func=ollama_embedding,
    llm_model_kwargs={"host": "http://localhost:11434"},
    enable_llm_cache=True,
)
print(f"      Workspace: {WORKING_DIR}")
print("      [OK] RAG inicializado con índice existente")

# Cargar preguntas filtradas
print("\n[2/4] Cargando preguntas de Novel-30752...")
questions_path = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'Questions', 'novel_30752_questions.json')
with open(questions_path) as f:
    all_questions = json.load(f)

# Agrupar por tipo
from collections import defaultdict
by_type = defaultdict(list)
type_map = {
    "Fact Retrieval": "type1",
    "Complex Reasoning": "type2",
    "Contextual Summarize": "type3",
    "Creative Generation": "type4"
}
for q in all_questions:
    qtype = type_map.get(q.get("question_type", ""), "type1")
    by_type[qtype].append(q)

print(f"      Total preguntas: {len(all_questions)}")
for t, qs in sorted(by_type.items()):
    print(f"      {t}: {len(qs)} disponibles, usaremos {min(SAMPLE_PER_TYPE, len(qs))}")

# Ejecutar queries
print("\n[3/4] Ejecutando queries...")
print("=" * 70)

results = defaultdict(list)
total_questions = 0
total_time = 0
query_num = 0

for qtype in ["type1", "type2", "type3", "type4"]:
    questions = by_type[qtype][:SAMPLE_PER_TYPE]

    if not questions:
        print(f"\n[{qtype}] Sin preguntas disponibles, saltando...")
        continue

    type_names = {
        "type1": "Fact Retrieval",
        "type2": "Complex Reasoning",
        "type3": "Contextual Summarize",
        "type4": "Creative Generation"
    }

    print(f"\n{'─' * 70}")
    print(f"[{qtype}] {type_names[qtype]} ({len(questions)} preguntas)")
    print("─" * 70)

    for i, q in enumerate(questions, 1):
        query_num += 1
        total_questions += 1

        print(f"\n  Q{query_num} ({qtype}/{i}):")
        print(f"  Pregunta: {q['question'][:80]}{'...' if len(q['question']) > 80 else ''}")
        print(f"  Procesando...", end=" ", flush=True)

        start = time.time()
        try:
            param = QueryParam(mode="hybrid", top_k=60)
            response = rag.query(q["question"], param=param)
            elapsed = time.time() - start
            total_time += elapsed

            if isinstance(response, tuple):
                answer, context = response
            else:
                answer = str(response)
                context = ""

            print(f"OK ({elapsed:.1f}s)")

            # Mostrar preview de respuesta
            answer_preview = str(answer)[:200].replace('\n', ' ')
            print(f"  Respuesta: {answer_preview}{'...' if len(str(answer)) > 200 else ''}")

            # Mostrar gold answer para comparación
            gold_preview = q.get('answer', '')[:100].replace('\n', ' ')
            print(f"  Gold: {gold_preview}{'...' if len(q.get('answer', '')) > 100 else ''}")

            results[qtype].append({
                "id": q.get("id", ""),
                "question": q["question"],
                "source": q.get("source", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": str(answer),
                "gold_answer": q.get("answer", ""),
                "query_time_seconds": round(elapsed, 2),
                "evidence": q.get("evidence", [])
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s)")
            print(f"  Error: {str(e)[:100]}")

            results[qtype].append({
                "id": q.get("id", ""),
                "question": q["question"],
                "source": q.get("source", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": f"[ERROR: {str(e)}]",
                "gold_answer": q.get("answer", ""),
                "query_time_seconds": round(elapsed, 2),
                "evidence": q.get("evidence", [])
            })

# Guardar resultados
print("\n" + "=" * 70)
print("[4/4] Guardando resultados...")

output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'hypergraphrag_novel30752_sample.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dict(results), f, indent=2, ensure_ascii=False)

print(f"      Archivo: {output_path}")

# Resumen final
print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print(f"Preguntas procesadas: {total_questions}")
print(f"Tiempo total queries: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Tiempo promedio/query: {total_time/total_questions:.1f}s" if total_questions > 0 else "N/A")
print()
print("Por tipo:")
for qtype in ["type1", "type2", "type3", "type4"]:
    if results[qtype]:
        type_time = sum(r["query_time_seconds"] for r in results[qtype])
        print(f"  {qtype}: {len(results[qtype])} preguntas, {type_time:.1f}s total")

print()
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
