"""
PASSO 12: O PIPELINE COMPLETO — pergunta entra, resposta VERIFICADA sai.

Este é o script que cose todas as peças construídas até agora:

  pergunta
     v
  retrieval (BM25 + dense -> RRF -> reranker)      [passos 3-6]
     v
  build_prompt (etiquetas [S1] + regras)           [passo 8]
     v
  gerador (Ollama)                                 [passo 9]
     v
  verify_citations (estrutural: a etiqueta existe?) [passo 10]
     v
  verify_semantic (LettuceDetect: o chunk prova?)   [passo 11]
     v
  resposta + veredicto combinado

DECISÃO DE DESENHO — porque as duas camadas correm em CADEIA e não em
paralelo: a estrutural é instantânea (só regex) e apanha problemas que
tornam a semântica impossível. Se uma afirmação cita [S7] (inexistente), não
há chunk nenhum para o LettuceDetect ler. Correr a barata primeiro também
poupa trabalho ao caro.

VEREDICTO COMBINADO (a política de decisão):
  - ABSTAINED  -> o modelo recusou-se a responder. Correcto se a pergunta
                  estava fora do contexto; é medido na avaliação.
  - VERIFIED   -> passou nas duas camadas. Resposta fundamentada.
  - FLAGGED    -> passou a estrutural mas falhou a semântica: as citações
                  existem, mas o conteúdo não é suportado pela fonte. É o
                  caso perigoso — parece bem formada mas mente.
  - REJECTED   -> falhou a estrutural: citações inventadas ou afirmações sem
                  fonte. Nem chega a ser avaliada semanticamente.

Uso:
  python answer.py --query "How much goodwill was recorded in the Groq deal?"
  python answer.py --query "..." --model qwen2.5:7b --top_k 5
  python answer.py --query "..." --json      # saída estruturada
"""

import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from build_prompt import build_prompt
from generate import generate, check_server, DEFAULT_MODEL
from verify_citations import verify_structure, is_abstention
from verify_semantic import load_detector, load_chunk_texts, verify_per_citation


def combine_verdicts(structural: dict, semantic: dict) -> str:
    """
    Política de decisão a partir dos dois relatórios.

    Nota: a ordem dos testes importa. A abstenção vem primeiro porque não é
    erro nenhum — é o comportamento desejado quando a resposta não está no
    contexto, e não faz sentido avaliar citações numa recusa.
    """
    if structural.get("abstained"):
        return "ABSTAINED"

    if not structural["verdict"].startswith("PASS"):
        return "REJECTED"

    if semantic and not semantic["verdict"].startswith("PASS"):
        return "FLAGGED"

    return "VERIFIED"


def answer_and_verify(query: str, engines, detector, chunk_texts: dict,
                      model: str = DEFAULT_MODEL, top_k: int = 3) -> dict:
    """
    O pipeline completo numa chamada. Recebe os recursos já carregados
    (engines, detector, chunk_texts) para poder ser chamado em ciclo na
    avaliação sem recarregar modelos de cada vez.
    """
    from rerank import hybrid_candidates, rerank

    # 1-2. RETRIEVAL: funil largo + reordenação fina
    candidates = hybrid_candidates(engines, query)
    reranked = rerank(engines, query, candidates)
    top_ids = [cid for cid, _ in reranked[:top_k]]

    by_id = {c["chunk_id"]: c for c in engines["chunks"]}
    top_chunks = [by_id[cid] for cid in top_ids]

    # 3. PROMPT: etiquetas + regras
    prompt, label_map = build_prompt(query, top_chunks)

    # 4. GERAÇÃO
    answer = generate(prompt, model=model)

    # 5. VERIFICAÇÃO ESTRUTURAL (barata, corre sempre)
    structural = verify_structure(answer, label_map)

    # 6. VERIFICAÇÃO SEMÂNTICA (cara, só se valer a pena)
    #    Salta-se em dois casos: abstenção (não há afirmações) e rejeição
    #    estrutural (as citações não são de confiança para começar).
    semantic = None
    if not structural.get("abstained") and structural["verdict"].startswith("PASS"):
        semantic = verify_per_citation(detector, query, answer, label_map, chunk_texts)

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": top_ids,
        "label_map": label_map,
        "structural": structural,
        "semantic": semantic,
        "verdict": combine_verdicts(structural, semantic),
    }


def print_result(result: dict):
    VERDICT_NOTE = {
        "VERIFIED": "resposta fundamentada nas fontes citadas",
        "FLAGGED": "citações válidas MAS conteúdo não suportado pela fonte",
        "REJECTED": "citações inventadas ou afirmações sem fonte",
        "ABSTAINED": "o modelo recusou-se a responder (correcto se fora do contexto)",
    }

    print("=" * 72)
    print(f"PERGUNTA: {result['query']}")
    print("-" * 72)
    print(f"RESPOSTA:\n{result['answer']}")
    print("-" * 72)
    print(f"VEREDICTO: {result['verdict']} — {VERDICT_NOTE[result['verdict']]}")
    print("=" * 72)

    print("\nFONTES RECUPERADAS:")
    for label, chunk_id in result["label_map"].items():
        print(f"  [{label}] {chunk_id}")

    s = result["structural"]
    print(f"\nCAMADA 1 (estrutural): {s['verdict']}")
    for c in s["claims"]:
        print(f"   [{c['status']}] {c['claim'][:65]}")
        if c["fabricated"]:
            print(f"      !! etiquetas inexistentes: {c['fabricated']}")

    sem = result["semantic"]
    if sem is None:
        print("\nCAMADA 2 (semântica): não corrida (abstenção ou rejeição estrutural)")
    else:
        print(f"\nCAMADA 2 (semântica): {sem['verdict']}")
        for c in sem["claims"]:
            print(f"   [{c['status']}] {c['claim'][:65]}")
            for src in c["per_source"]:
                for sp in src["spans"]:
                    print(f"      !! [{src['source_label']}] não suporta (conf={sp['confidence']:.3f}): '{sp['text'][:50]}'")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str,
                        default="How much goodwill was recorded in the Groq deal?")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="saída em JSON")
    args = parser.parse_args()

    if not check_server():
        return

    from rerank import load_all

    print("A carregar o pipeline (retrieval + reranker + detector)...")
    engines = load_all()
    detector = load_detector()
    chunk_texts = load_chunk_texts()
    print("Pronto.\n")

    result = answer_and_verify(args.query, engines, detector, chunk_texts,
                               model=args.model, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()