"""
PASSO 6 do pipeline: AVALIAÇÃO FINAL com o reranker incluído.

Igual ao evaluate_retrieval.py, mas adiciona um QUARTO método: hybrid+rerank
(o pipeline completo). Compara os quatro lado a lado:
  bm25 | dense | hybrid | hybrid_rerank

Assim vê-se, com números, se o reranker melhora de facto sobre a baseline
(hybrid e BM25) nas mesmas 41 perguntas avaliáveis.

Nota de desenho: o reranker recebe os 50 candidatos do hybrid e reordena-os.
Como o reranker é lento (lê 50 pares de texto por pergunta, em CPU), esta
avaliação demora bastante mais que a anterior — conta com vários minutos.

Uso:
  python evaluate_retrieval_v2.py
  python evaluate_retrieval_v2.py --only q003 q047   # debug de perguntas específicas
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from rerank import load_all, hybrid_candidates, rerank
from hybrid_search import bm25_ranking, dense_ranking, rrf_fuse

BASE_DIR = Path(__file__).parent
EVAL_DATASET_PATH = BASE_DIR / "eval_dataset.jsonl"
RESULTS_DIR = BASE_DIR / "eval_results"

K_VALUES = [1, 3, 5, 10]
RECALL_CANDIDATES = 50


def load_eval_dataset():
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_all_rankings(engines, query: str):
    """Devolve os quatro rankings (listas de chunk_ids, melhor primeiro)."""
    bm25_ids = bm25_ranking(engines, query, n=RECALL_CANDIDATES)
    dense_ids = dense_ranking(engines, query, n=RECALL_CANDIDATES)

    fused = rrf_fuse([bm25_ids, dense_ids])
    hybrid_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]

    # hybrid+rerank: pega nos candidatos do hybrid e reordena com o cross-encoder
    candidates = hybrid_ids[:RECALL_CANDIDATES]
    reranked = rerank(engines, query, candidates)
    rerank_ids = [cid for cid, _ in reranked]

    return {
        "bm25": bm25_ids,
        "dense": dense_ids,
        "hybrid": hybrid_ids,
        "hybrid_rerank": rerank_ids,
    }


def recall_at_k(ranking, relevant_ids, k):
    if not relevant_ids:
        return None
    top_k = set(ranking[:k])
    return sum(1 for rid in relevant_ids if rid in top_k) / len(relevant_ids)


def reciprocal_rank(ranking, relevant_ids):
    if not relevant_ids:
        return None
    for position, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def evaluate(engines, questions):
    per_question = []
    methods = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    aggregates = {m: defaultdict(list) for m in methods}

    evaluable = [q for q in questions if q["relevant_chunk_ids"]]
    skipped = [q["id"] for q in questions if not q["relevant_chunk_ids"]]
    print(f"A avaliar {len(evaluable)} perguntas (excluídas {len(skipped)} out_of_scope)\n")

    for q in evaluable:
        rankings = get_all_rankings(engines, q["question"])
        q_result = {"id": q["id"], "category": q["category"], "difficulty": q["difficulty"],
                    "relevant_chunk_ids": q["relevant_chunk_ids"], "methods": {}}

        for method, ranking in rankings.items():
            metrics = {}
            for k in K_VALUES:
                r = recall_at_k(ranking, q["relevant_chunk_ids"], k)
                metrics[f"recall@{k}"] = r
                aggregates[method][f"recall@{k}"].append(r)
            rr = reciprocal_rank(ranking, q["relevant_chunk_ids"])
            metrics["reciprocal_rank"] = rr
            aggregates[method]["mrr"].append(rr)
            metrics["top10"] = ranking[:10]
            q_result["methods"][method] = metrics

        per_question.append(q_result)
        print(f"  {q['id']} ({q['category']}) avaliada")

    summary = {}
    for method, metric_lists in aggregates.items():
        summary[method] = {metric: round(sum(v) / len(v), 4) for metric, v in metric_lists.items()}

    return summary, per_question


def print_summary(summary):
    methods = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    metrics = [f"recall@{k}" for k in K_VALUES] + ["mrr"]

    print("\n" + "=" * 78)
    print(f"{'métrica':<12}" + "".join(f"{m:>16}" for m in methods))
    print("-" * 78)
    for metric in metrics:
        row = f"{metric:<12}"
        best = max(summary[m][metric] for m in methods)
        for m in methods:
            val = summary[m][metric]
            marker = " *" if val == best else "  "
            row += f"{val:>14.4f}{marker}"
        print(row)
    print("=" * 78)
    print("(* = melhor valor da linha)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    questions = load_eval_dataset()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
        print(f"Modo debug: {[q['id'] for q in questions]}")

    engines = load_all()  # inclui o reranker
    summary, per_question = evaluate(engines, questions)

    print_summary(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "retrieval_results_with_rerank.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, ensure_ascii=False, indent=2)
    print(f"\nResultados detalhados guardados em: {output_path}")


if __name__ == "__main__":
    main()