"""
A/B Test REAL: HyperGraphRAG con vs sin Contextual.ai Reranker
SIN cache de LLM para comparación justa.
"""
import os
import time
import json

# Cargar .env ANTES de imports
env_path = os.path.expanduser("~/.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip('"\'')

ORIGINAL_API_KEY = os.environ.get("CONTEXTUAL_API_KEY", "")

from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import ollama_model_complete, ollama_embed
from hypergraphrag.utils import wrap_embedding_func_with_attrs
import hypergraphrag.rerank as rerank_module

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=2048)
async def ollama_embedding_local(texts, **kwargs):
    return await ollama_embed(texts, embed_model="embeddinggemma", **kwargs)

def create_rag():
    """Crear instancia de HyperGraphRAG SIN cache."""
    return HyperGraphRAG(
        working_dir="expr/gpt_oss_test",
        llm_model_func=ollama_model_complete,
        llm_model_name="gpt-oss:20b",
        embedding_func=ollama_embedding_local,
        llm_model_kwargs={"host": "http://localhost:11434"},
        enable_llm_cache=False,  # DESACTIVADO para test justo
    )

# Queries NUEVAS - no cacheadas
QUERIES = [
    "¿Cómo maneja HyperGraphRAG las relaciones n-arias que involucran más de 3 entidades?",
    "¿Qué diferencia hay entre el retrieval local y global en HyperGraphRAG?",
    "¿Por qué las hyperedges preservan mejor el contexto semántico que los triples tradicionales?",
]

results = []

print("=" * 70)
print("A/B TEST REAL: HyperGraphRAG con vs sin Reranker (CACHE OFF)")
print("=" * 70)
print(f"API Key configurada: {'✅' if ORIGINAL_API_KEY else '❌'}")
print(f"LLM Cache: ❌ DESACTIVADO")
print(f"Queries a testear: {len(QUERIES)}")
print("=" * 70)

for i, query in enumerate(QUERIES, 1):
    print(f"\n{'='*70}")
    print(f"QUERY {i}/{len(QUERIES)}: {query[:60]}...")
    print("="*70)

    # === TEST A: CON RERANKER (sin cache) ===
    print("\n🅰️  TEST A: CON Contextual.ai Reranker (cache OFF)")
    print("-" * 50)

    rerank_module.CONTEXTUAL_API_KEY = ORIGINAL_API_KEY

    rag_a = create_rag()
    start_a = time.time()
    response_a = rag_a.query(query)
    time_a = time.time() - start_a

    print(f"⏱️  Tiempo: {time_a:.1f}s")
    print(f"📝 Respuesta A ({len(response_a)} chars):\n{response_a[:500]}...")

    # === TEST B: SIN RERANKER (sin cache) ===
    print("\n🅱️  TEST B: SIN Reranker (cache OFF)")
    print("-" * 50)

    rerank_module.CONTEXTUAL_API_KEY = ""

    rag_b = create_rag()
    start_b = time.time()
    response_b = rag_b.query(query)
    time_b = time.time() - start_b

    print(f"⏱️  Tiempo: {time_b:.1f}s")
    print(f"📝 Respuesta B ({len(response_b)} chars):\n{response_b[:500]}...")

    # === COMPARACIÓN ===
    print("\n📊 COMPARACIÓN QUERY", i)
    print("-" * 50)

    len_a = len(response_a)
    len_b = len(response_b)
    time_diff = time_a - time_b
    # Compare first 500 chars to check similarity
    same_start = response_a[:500].strip() == response_b[:500].strip()

    print(f"| Métrica | Con Reranker (A) | Sin Reranker (B) |")
    print(f"|---------|------------------|------------------|")
    print(f"| Tiempo | {time_a:.1f}s | {time_b:.1f}s |")
    print(f"| Longitud | {len_a} chars | {len_b} chars |")
    print(f"| Δ Tiempo | {time_diff:+.1f}s (overhead reranker) |")
    print(f"| Inicio similar? | {'SÍ ⚠️' if same_start else 'NO ✅ (diferentes)'} |")

    results.append({
        "query_num": i,
        "query": query,
        "response_a_with_reranker": response_a,
        "response_b_without_reranker": response_b,
        "time_a": round(time_a, 2),
        "time_b": round(time_b, 2),
        "len_a": len_a,
        "len_b": len_b,
        "same_start": same_start,
    })

# Restaurar API key
rerank_module.CONTEXTUAL_API_KEY = ORIGINAL_API_KEY

# === RESUMEN FINAL ===
print("\n" + "="*70)
print("RESUMEN A/B TEST (CACHE DESACTIVADO)")
print("="*70)

total_time_a = sum(r["time_a"] for r in results)
total_time_b = sum(r["time_b"] for r in results)
avg_len_a = sum(r["len_a"] for r in results) / len(results)
avg_len_b = sum(r["len_b"] for r in results) / len(results)
similar_count = sum(1 for r in results if r["same_start"])

print(f"""
| Métrica              | Con Reranker (A) | Sin Reranker (B) | Diferencia    |
|----------------------|------------------|------------------|---------------|
| Tiempo total         | {total_time_a:.1f}s            | {total_time_b:.1f}s            | {total_time_a - total_time_b:+.1f}s (overhead) |
| Tiempo promedio      | {total_time_a/len(results):.1f}s            | {total_time_b/len(results):.1f}s            | {(total_time_a-total_time_b)/len(results):+.1f}s/query |
| Longitud promedio    | {avg_len_a:.0f} chars       | {avg_len_b:.0f} chars       | {avg_len_a - avg_len_b:+.0f}          |
| Respuestas similares | {similar_count}/{len(results)}              |                  |               |
""")

print("DETALLE POR QUERY:")
print("-" * 70)
for r in results:
    status = "⚠️ similar" if r["same_start"] else "✅ diferente"
    print(f"Q{r['query_num']}: A={r['time_a']:.1f}s | B={r['time_b']:.1f}s | Δt={r['time_a']-r['time_b']:+.1f}s | {status}")

# Guardar resultados completos
output_file = "ab_test_reranker_nocache_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n💾 Resultados guardados en: {output_file}")

print("\n" + "="*70)
print("PARA EVALUACIÓN MANUAL DE CALIDAD")
print("="*70)
print("Revisa ab_test_reranker_nocache_results.json para comparar respuestas completas.")
print("Criterios sugeridos:")
print("  - ¿Cuál respuesta es más precisa?")
print("  - ¿Cuál cubre mejor los conceptos clave?")
print("  - ¿Cuál tiene mejor estructura?")
print("="*70)
