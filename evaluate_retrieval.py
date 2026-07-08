"""
PASSO 4 do pipeline: AVALIAÇÃO DE RETRIEVAL.

Corre as 50 perguntas do eval_dataset.jsonl contra os três métodos de
retrieval (BM25, dense, hybrid/RRF) e calcula métricas comparáveis:

  - Recall@k (k=1,3,5,10): dos chunks relevantes anotados, que fração
    apareceu no top-k? (a métrica principal de "encontrou ou não encontrou")
  - MRR (Mean Reciprocal Rank): em média, quão cedo aparece o primeiro
    chunk relevante? (1.0 = sempre em 1º; 0.5 = tipicamente em 2º; etc.)

Notas de desenho:
  - Perguntas out_of_scope (sem relevant_chunk_ids) são EXCLUÍDAS destas
    métricas de retrieval — não há "chunk certo" para encontrar. Serão
    usadas mais tarde na avaliação de geração (abstenção correcta).
  - Os resultados são guardados em eval_results/retrieval_results.json,
    incluindo o detalhe por pergunta (não só os agregados), para permitir
    análise de erros pergunta a pergunta.

Uso:
  python evaluate_retrieval.py
  python evaluate_retrieval.py --only q003 q016    # só perguntas específicas (debug)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from hybrid_search import load_engines, bm25_ranking, dense_ranking, rrf_fuse

BASE_DIR = Path(__file__).parent
EVAL_DATASET_PATH = BASE_DIR / "eval_dataset.jsonl"
RESULTS_DIR = BASE_DIR / "eval_results"

K_VALUES = [1, 3, 5, 10]
RECALL_CANDIDATES = 50  # candidatos por método (consistente com hybrid_search)


def load_eval_dataset():
    questions = []
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def get_rankings_for_query(engines, query: str):
    """Devolve os três rankings (listas de chunk_ids, melhor primeiro) para uma query."""
    bm25_ids = bm25_ranking(engines, query, n=RECALL_CANDIDATES)
    dense_ids = dense_ranking(engines, query, n=RECALL_CANDIDATES)

    fused_points = rrf_fuse([bm25_ids, dense_ids])
    hybrid_ids = [cid for cid, _ in sorted(fused_points.items(), key=lambda kv: kv[1], reverse=True)]

    return {"bm25": bm25_ids, "dense": dense_ids, "hybrid": hybrid_ids}


def recall_at_k(ranking: list, relevant_ids: list, k: int) -> float:
    """Fração dos chunks relevantes que aparecem no top-k do ranking."""
    if not relevant_ids:
        return None
    top_k = set(ranking[:k])
    found = sum(1 for rid in relevant_ids if rid in top_k)
    return found / len(relevant_ids)


def reciprocal_rank(ranking: list, relevant_ids: list) -> float:
    """1/posição do PRIMEIRO chunk relevante encontrado (0 se nenhum no ranking)."""
    if not relevant_ids:
        return None
    for position, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def evaluate(engines, questions):
    per_question = []
    aggregates = {method: defaultdict(list) for method in ["bm25", "dense", "hybrid"]}

    evaluable = [q for q in questions if q["relevant_chunk_ids"]]
    skipped = [q["id"] for q in questions if not q["relevant_chunk_ids"]]
    print(f"A avaliar {len(evaluable)} perguntas (excluídas {len(skipped)} out_of_scope: {skipped})\n")

    for q in evaluable:
        rankings = get_rankings_for_query(engines, q["question"])
        q_result = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "methods": {},
        }

        for method, ranking in rankings.items():
            metrics = {}
            for k in K_VALUES:
                r = recall_at_k(ranking, q["relevant_chunk_ids"], k)
                metrics[f"recall@{k}"] = r
                aggregates[method][f"recall@{k}"].append(r)
            rr = reciprocal_rank(ranking, q["relevant_chunk_ids"])
            metrics["reciprocal_rank"] = rr
            aggregates[method]["mrr"].append(rr)
            # guarda o top-10 para análise de erros posterior
            metrics["top10"] = ranking[:10]
            q_result["methods"][method] = metrics

        per_question.append(q_result)
        print(f"  {q['id']} ({q['category']}) avaliada")

    # médias agregadas
    summary = {}
    for method, metric_lists in aggregates.items():
        summary[method] = {
            metric: round(sum(values) / len(values), 4)
            for metric, values in metric_lists.items()
        }

    return summary, per_question


def print_summary(summary):
    methods = ["bm25", "dense", "hybrid"]
    metrics = [f"recall@{k}" for k in K_VALUES] + ["mrr"]

    print("\n" + "=" * 62)
    print(f"{'métrica':<12}" + "".join(f"{m:>16}" for m in methods))
    print("-" * 62)
    for metric in metrics:
        row = f"{metric:<12}"
        best = max(summary[m][metric] for m in methods)
        for m in methods:
            val = summary[m][metric]
            marker = " *" if val == best else "  "
            row += f"{val:>14.4f}{marker}"
        print(row)
    print("=" * 62)
    print("(* = melhor valor da linha)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="IDs de perguntas específicas (debug)")
    args = parser.parse_args()

    questions = load_eval_dataset()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
        print(f"Modo debug: só {[q['id'] for q in questions]}")

    engines = load_engines()
    summary, per_question = evaluate(engines, questions)

    print_summary(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "retrieval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, ensure_ascii=False, indent=2)
    print(f"\nResultados detalhados guardados em: {output_path}")


if __name__ == "__main__":
    main()