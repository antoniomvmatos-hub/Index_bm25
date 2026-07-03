"""
PASSO 2 do pipeline: índice DENSE (embeddings semânticos) com Chroma.
 
Ao contrário do BM25 (que casa palavras exactas), este passo converte cada
chunk num vector numérico que representa o SIGNIFICADO do texto. Duas
frases com palavras diferentes mas o mesmo significado ficam com vectores
"próximos" no espaço matemático.
 
Uso:
  python3 index_embeddings.py            # usa todos os 396 chunks
  python3 index_embeddings.py --max 20   # usa só os primeiros 20 (teste rápido)
"""
 
import json
import argparse
from pathlib import Path
 
from sentence_transformers import SentenceTransformer
import chromadb
 
CHUNKS_PATH = Path(__file__).parent / "chunks" / "chunks_nvidia.jsonl"
CHROMA_DIR = Path(__file__).parent / "indexes" / "chroma_db"
COLLECTION_NAME = "nvidia_10k_chunks"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
 
 
def load_chunks(max_chunks: int = None):
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    if max_chunks:
        chunks = chunks[:max_chunks]
    return chunks
 
 
def build_embedding_index(chunks, model: SentenceTransformer, client: chromadb.ClientAPI):
    # se já existir uma collection com este nome (de um run anterior), apaga-a
    # primeiro - evita duplicar/misturar dados de runs diferentes (ex: --max 20
    # seguido de run completo).
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
 
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    # Chroma exige metadata só com tipos simples (str/int/float/bool);
    # 'breadcrumb' pode ser None nalguns chunks (capa do documento) - Chroma
    # não aceita None em metadata, por isso convertemos para string vazia.
    metadatas = [{"breadcrumb": c.get("breadcrumb") or ""} for c in chunks]
 
    print(f"A gerar embeddings para {len(texts)} chunks (modelo: {MODEL_NAME})...")
    # normalize_embeddings=True: essencial para o bge-large - o modelo foi
    # treinado para usar similaridade de cosseno, que só é bem comparável
    # entre vectores normalizados (todos com o mesmo "comprimento").
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=16,
    )
 
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    return collection
 
 
def search_dense(collection, model: SentenceTransformer, query: str, top_k: int = 5):
    # a bge-large foi treinada com um prefixo específico para QUERIES
    # (não para documentos) - omitir isto degrada a qualidade da pesquisa.
    # É uma particularidade deste modelo, não uma regra universal de embeddings.
    query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = model.encode([query_with_prefix], normalize_embeddings=True)[0]
 
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
 
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "chunk_id": results["ids"][0][i],
            # Chroma devolve "distância" (menor = mais parecido); convertemos
            # para "similaridade" (maior = mais parecido) para ser directamente
            # comparável, na cabeça, ao score do BM25 (onde maior = melhor).
            "similarity": 1 - results["distances"][0][i],
            "text_preview": results["documents"][0][i][:150],
        })
    return formatted
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Limitar a N primeiros chunks (para testes rápidos)")
    args = parser.parse_args()
 
    chunks = load_chunks(max_chunks=args.max)
 
    print(f"A carregar o modelo {MODEL_NAME} (primeira vez: faz download, ~1.3GB)...")
    model = SentenceTransformer(MODEL_NAME)
 
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
 
    collection = build_embedding_index(chunks, model, client)
    print(f"Índice guardado em: {CHROMA_DIR} (collection: {COLLECTION_NAME})")
 
    print("\n--- Teste de fumo: query 'Groq goodwill' (mesma query do BM25) ---")
    results = search_dense(collection, model, "Groq goodwill", top_k=3)
    for r in results:
        print(f"  similarity={r['similarity']:.3f}  {r['chunk_id']}")
        print(f"    {r['text_preview']}")
 
    print("\n--- Teste extra: paráfrase SEM a palavra 'Groq' ---")
    results2 = search_dense(collection, model, "AI chip startup NVIDIA licensed inference technology from", top_k=3)
    for r in results2:
        print(f"  similarity={r['similarity']:.3f}  {r['chunk_id']}")
        print(f"    {r['text_preview']}")
 
 
if __name__ == "__main__":
    main()