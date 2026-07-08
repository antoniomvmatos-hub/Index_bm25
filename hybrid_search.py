"""
PASSO 3 do pipeline: HYBRID SEARCH — fusão dos rankings BM25 + dense via RRF.
 
Reciprocal Rank Fusion (RRF) combina duas listas ordenadas usando as
POSIÇÕES (não os scores) de cada chunk em cada lista:
 
    pontos(chunk) = soma, por cada lista onde aparece, de  1 / (K + posição)
 
- Usa posições porque os scores de BM25 (0 a ~16+) e de dense (-1 a 1)
  estão em escalas incomparáveis — posições são sempre comparáveis.
- K=60 é a constante do paper original (Cormack et al. 2009); suaviza a
  diferença entre posições próximas do topo.
 
Pré-requisitos (têm de existir, criados pelos passos 1 e 2):
  indexes/bm25_index.pkl
  indexes/chroma_db/  (collection "nvidia_10k_chunks")
 
Uso:
  python hybrid_search.py                      # corre os testes de fumo
  ou, interactivamente:
    from hybrid_search import load_engines, hybrid_search
    engines = load_engines()
    hybrid_search(engines, "How many employees does NVIDIA have?", top_k=10)
"""
 
import json
import pickle
import argparse
from pathlib import Path
 
import chromadb
from sentence_transformers import SentenceTransformer
 
from index_bm25 import simple_tokenize
 
BASE_DIR = Path(__file__).parent
BM25_INDEX_PATH = BASE_DIR / "indexes" / "bm25_index.pkl"
CHROMA_DIR = BASE_DIR / "indexes" / "chroma_db"
COLLECTION_NAME = "nvidia_10k_chunks"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
 
RRF_K = 60          # constante do paper original
RECALL_PER_METHOD = 50  # quantos candidatos pedir a CADA método antes de fundir
 
 
def load_engines():
    """Carrega tudo o que é preciso, uma vez, para depois pesquisar quantas
    vezes quisermos sem recarregar. Devolve um dicionário com as 4 peças."""
    print("A carregar índice BM25...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
 
    print(f"A carregar modelo de embeddings ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
 
    print("A ligar ao Chroma...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
 
    # mapa chunk_id -> texto, para mostrar previews nos resultados
    chunk_texts = {c["chunk_id"]: c["text"] for c in bm25_data["chunks"]}
 
    print("Pronto.\n")
    return {
        "bm25": bm25_data["bm25"],
        "chunks": bm25_data["chunks"],
        "model": model,
        "collection": collection,
        "chunk_texts": chunk_texts,
    }
 
 
def bm25_ranking(engines, query: str, n: int = RECALL_PER_METHOD):
    """Devolve lista de chunk_ids ordenada por relevância BM25 (melhor primeiro)."""
    tokenized = simple_tokenize(query)
    scores = engines["bm25"].get_scores(tokenized)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [engines["chunks"][i]["chunk_id"] for i in ranked]
 
 
def dense_ranking(engines, query: str, n: int = RECALL_PER_METHOD):
    """Devolve lista de chunk_ids ordenada por similaridade dense (melhor primeiro)."""
    query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
    query_emb = engines["model"].encode([query_with_prefix], normalize_embeddings=True)
    results = engines["collection"].query(
        query_embeddings=query_emb.tolist(),
        n_results=n,
    )
    return results["ids"][0]
 
 
def rrf_fuse(rankings: list, k: int = RRF_K):
    """
    Recebe uma lista de rankings (cada um é uma lista de chunk_ids ordenada,
    melhor primeiro) e devolve um dicionário chunk_id -> pontos RRF.
    """
    points = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            points[chunk_id] = points.get(chunk_id, 0.0) + 1.0 / (k + position)
    return points
 
 
def hybrid_search(engines, query: str, top_k: int = 10):
    """Pipeline completo: BM25 + dense -> RRF -> top_k combinado."""
    bm25_ids = bm25_ranking(engines, query)
    dense_ids = dense_ranking(engines, query)
 
    fused_points = rrf_fuse([bm25_ids, dense_ids])
    ranked = sorted(fused_points.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
 
    results = []
    for chunk_id, points in ranked:
        # posição em cada ranking individual (ou None se fora do recall desse método)
        pos_bm25 = bm25_ids.index(chunk_id) + 1 if chunk_id in bm25_ids else None
        pos_dense = dense_ids.index(chunk_id) + 1 if chunk_id in dense_ids else None
        results.append({
            "chunk_id": chunk_id,
            "rrf_points": round(points, 5),
            "pos_bm25": pos_bm25,
            "pos_dense": pos_dense,
            "text_preview": engines["chunk_texts"].get(chunk_id, "")[:150],
        })
    return results
 
 
def print_results(results, title: str):
    print(f"--- {title} ---")
    for i, r in enumerate(results, start=1):
        print(f"{i:2d}. rrf={r['rrf_points']:.5f}  bm25_pos={r['pos_bm25']}  dense_pos={r['pos_dense']}  {r['chunk_id']}")
        print(f"     {r['text_preview']}")
    print()
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Query única a pesquisar (senão corre os testes de fumo)")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
 
    engines = load_engines()
 
    if args.query:
        results = hybrid_search(engines, args.query, top_k=args.top_k)
        print_results(results, f"Hybrid search: '{args.query}'")
        return
 
    # Teste de fumo 1: o caso do chunk0060 (BM25 acertou em 1º, dense deixou em 7º)
    # Previsão: chunk0060 no topo do ranking combinado.
    results = hybrid_search(engines, "How many employees did NVIDIA have at the end of fiscal year 2026?", top_k=5)
    print_results(results, "Teste 1 - caso 'employees' (esperado: chunk0060 no topo)")
 
    # Teste de fumo 2: a paráfrase da Groq SEM a palavra 'Groq'
    # (dense aguentou-se tematicamente, BM25 deve ser inútil aqui)
    results = hybrid_search(engines, "AI chip startup NVIDIA licensed inference technology from", top_k=5)
    print_results(results, "Teste 2 - paráfrase Groq sem a palavra 'Groq'")
 
    # Teste de fumo 3: caso fácil para ambos (com a palavra 'Groq')
    results = hybrid_search(engines, "Groq goodwill", top_k=5)
    print_results(results, "Teste 3 - 'Groq goodwill' (ambos acertavam; esperado: chunk0318 destacado)")
 
 
if __name__ == "__main__":
    main()