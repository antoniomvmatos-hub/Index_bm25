"""
PASSO 1 do pipeline: índice BM25 (sparse/keyword retrieval).

Isto é deliberadamente o primeiro passo porque:
  - Não precisa de download de modelo (corre instantaneamente)
  - Não precisa de GPU/muita RAM
  - É fácil de replicar localmente com um subconjunto pequeno dos chunks
    (basta mudar MAX_CHUNKS abaixo, ou passar um número por argumento)

Uso:
  python3 index_bm25.py            # usa todos os 396 chunks
  python3 index_bm25.py --max 20   # usa só os primeiros 20 chunks (para testares rápido)
"""

import json
import re
import pickle
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).parent / "chunks" / "chunks_nvidia.jsonl"
INDEX_OUTPUT_PATH = Path(__file__).parent / "indexes" / "bm25_index.pkl"


def simple_tokenize(text: str):
    """
    Tokenização simples: lowercase + separar por não-alfanuméricos.
    BM25 não precisa de nada mais sofisticado (não é embeddings semânticos);
    o que importa é ter as MESMAS palavras-chave da query e do documento.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def load_chunks(max_chunks: int = None):
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    if max_chunks:
        chunks = chunks[:max_chunks]
    return chunks


def build_bm25_index(chunks):
    tokenized_corpus = [simple_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def search_bm25(bm25, chunks, query: str, top_k: int = 5):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = [
        {"chunk_id": chunks[i]["chunk_id"], "score": float(scores[i]), "text_preview": chunks[i]["text"][:150]}
        for i in ranked_indices
    ]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Limitar a N primeiros chunks (para testes rápidos)")
    args = parser.parse_args()

    chunks = load_chunks(max_chunks=args.max)
    print(f"A indexar {len(chunks)} chunks com BM25...")

    bm25 = build_bm25_index(chunks)

    INDEX_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(INDEX_OUTPUT_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"Índice BM25 guardado em: {INDEX_OUTPUT_PATH}")

    # teste rápido de fumo: uma query que sabemos que devia acertar em cheio
    print("\n--- Teste de fumo: query 'Groq acquisition goodwill' ---")
    results = search_bm25(bm25, chunks, "Groq goodwill", top_k=3)
    for r in results:
        print(f"  score={r['score']:.2f}  {r['chunk_id']}")
        print(f"    {r['text_preview']}")


if __name__ == "__main__":
    main()
