"""
PASSO 5 do pipeline: CROSS-ENCODER RERANKER.

O reranker recebe os ~50 candidatos do hybrid search e reordena-os lendo
a query e cada chunk JUNTOS, par a par. Diferença fundamental face ao
dense retrieval:

  - Dense (bi-encoder): query e chunks são convertidos em vectores
    SEPARADAMENTE (os chunks foram embutidos na indexação, sem saber que
    query viria). A comparação é entre vectores pré-calculados — rápida,
    mas "cega" à interacção entre os textos.
  - Cross-encoder: recebe o par (query, chunk) como UMA entrada única e
    produz um score de relevância. O modelo "lê" os dois textos em
    conjunto e capta interacções finas (a query pergunta X, este chunk
    responde exactamente X?). Muito mais preciso, mas muito mais lento —
    por isso só se aplica aos ~50 candidatos, nunca ao corpus todo.

Modelo: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90MB, treinado no dataset
MS MARCO de pares pergunta-passagem; standard em tutoriais e produção
leve).

Uso:
  python rerank.py                          # corre os testes de fumo
  ou, interactivamente:
    from rerank import load_all, search_with_rerank
    engines = load_all()
    search_with_rerank(engines, "How many employees does NVIDIA have?", top_k=5)
"""

import argparse

from sentence_transformers import CrossEncoder

from hybrid_search import load_engines, bm25_ranking, dense_ranking, rrf_fuse

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RECALL_CANDIDATES = 50   # candidatos por método antes da fusão
RERANK_POOL = 50         # quantos candidatos do hybrid entram no reranker


def load_all():
    """Carrega os engines do hybrid (BM25 + dense + Chroma) e o cross-encoder."""
    engines = load_engines()
    print(f"A carregar reranker ({RERANKER_MODEL})...")
    engines["reranker"] = CrossEncoder(RERANKER_MODEL)
    print("Reranker pronto.\n")
    return engines


def hybrid_candidates(engines, query: str, pool_size: int = RERANK_POOL):
    """Corre o hybrid search e devolve os top pool_size chunk_ids (o funil de recall)."""
    bm25_ids = bm25_ranking(engines, query, n=RECALL_CANDIDATES)
    dense_ids = dense_ranking(engines, query, n=RECALL_CANDIDATES)
    fused = rrf_fuse([bm25_ids, dense_ids])
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [cid for cid, _ in ranked[:pool_size]]


def rerank(engines, query: str, candidate_ids: list):
    """
    Reordena os candidatos com o cross-encoder.
    Constrói pares (query, texto_do_chunk) e pede um score para cada par.
    Devolve lista de (chunk_id, score) ordenada por score decrescente.
    """
    pairs = [(query, engines["chunk_texts"][cid]) for cid in candidate_ids]
    scores = engines["reranker"].predict(pairs)
    scored = list(zip(candidate_ids, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def search_with_rerank(engines, query: str, top_k: int = 5):
    """Pipeline completo: hybrid (recall) -> cross-encoder (precision) -> top_k."""
    candidates = hybrid_candidates(engines, query)
    reranked = rerank(engines, query, candidates)

    results = []
    for chunk_id, score in reranked[:top_k]:
        results.append({
            "chunk_id": chunk_id,
            "rerank_score": round(float(score), 4),
            "pos_before_rerank": candidates.index(chunk_id) + 1,
            "text_preview": engines["chunk_texts"][chunk_id][:150],
        })
    return results


def print_results(results, title: str):
    print(f"--- {title} ---")
    for i, r in enumerate(results, start=1):
        print(f"{i:2d}. score={r['rerank_score']:>8.4f}  (era {r['pos_before_rerank']}º no hybrid)  {r['chunk_id']}")
        print(f"     {r['text_preview']}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    engines = load_all()

    if args.query:
        results = search_with_rerank(engines, args.query, top_k=args.top_k)
        print_results(results, f"Hybrid + rerank: '{args.query}'")
        return

    # Teste 1: o caso q047 onde o hybrid tinha DILUÍDO o acerto do BM25
    # (BM25 1º, dense fora do top 50, hybrid empurrou para fora do top 10).
    # Previsão: o reranker deve trazer o chunk0002 de volta ao topo.
    results = search_with_rerank(engines, "What is NVIDIA's I.R.S. Employer Identification Number?", top_k=5)
    print_results(results, "Teste 1 - q047 IRS number (hybrid tinha falhado; esperado: chunk0002 no topo)")

    # Teste 2: a paráfrase da Groq SEM a palavra 'Groq' - o caso que nem BM25
    # nem dense nem hybrid resolveram. O teste mais difícil do projecto.
    results = search_with_rerank(engines, "AI chip startup NVIDIA licensed inference technology from", top_k=5)
    print_results(results, "Teste 2 - paráfrase Groq (o caso mais difícil; nenhum método resolveu até agora)")

    # Teste 3: sanity check - caso fácil que todos acertavam deve continuar certo
    results = search_with_rerank(engines, "How much goodwill was recorded in the Groq deal?", top_k=5)
    print_results(results, "Teste 3 - sanity check q018 (todos acertavam; esperado: chunk0318 no topo)")


if __name__ == "__main__":
    main()