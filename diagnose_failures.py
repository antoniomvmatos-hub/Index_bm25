"""
DIAGNOSTICO: as falhas numericas sao de RETRIEVAL ou de GERACAO?

Le o generation_results.json + o eval_dataset.jsonl e cruza-os. Para cada
pergunta que NAO deu VERIFIED, verifica uma coisa simples mas decisiva:

  o(s) chunk(s) anotado(s) como correcto(s) no dataset chegaram aos chunks
  que o modelo recebeu?

Dois desfechos possiveis, com accoes opostas:
  - CHUNK CERTO AUSENTE  -> falha de RETRIEVAL. O modelo inventou porque nao
                            recebeu o numero. Trocar de gerador NAO resolve.
  - CHUNK CERTO PRESENTE -> falha de GERACAO. O numero estava no contexto e o
                            modelo errou na mesma. Um gerador melhor podia
                            resolver.

Nao corre modelos nem Ollama — so le ficheiros. Instantaneo.

Uso:
  python diagnose_failures.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
RESULTS = BASE / "eval_results" / "generation_results.json"
DATASET = BASE / "eval_dataset.jsonl"


def load():
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    dataset = {}
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            dataset[q["id"]] = q
    return results, dataset


def main():
    results, dataset = load()
    rows = results["per_question"]

    print(f"Gerador avaliado: {results.get('model')}\n")
    print("=" * 78)
    print("FALHAS QUE NAO SAO ABSTENCAO: o chunk certo chegou ao modelo?")
    print("=" * 78)

    buckets = defaultdict(list)

    for r in rows:
        # so interessa investigar o que correu mal e NAO era para abster
        if r["verdict"] in ("VERIFIED", "ABSTAINED"):
            continue
        if r["expected_abstention"]:
            continue  # out_of_scope tratam-se a parte

        q = dataset[r["id"]]
        gold = set(q["relevant_chunk_ids"])
        got = set(r["retrieved_chunks"])
        hit = gold & got

        if not gold:
            estado = "SEM_GOLD"   # nao deveria acontecer para respondiveis
        elif hit:
            estado = "CHUNK_PRESENTE"   # -> falha de geracao
        else:
            estado = "CHUNK_AUSENTE"    # -> falha de retrieval

        buckets[estado].append(r["id"])

        print(f"\n{r['id']} ({r['category']}) — veredicto {r['verdict']}")
        print(f"  P: {q['question'][:66]}")
        print(f"  Resposta: {r['answer'][:70].strip()}")
        print(f"  Chunk(s) certo(s): {sorted(gold)}")
        print(f"  Recebeu          : {r['retrieved_chunks']}")
        if hit:
            print(f"  >> CHUNK CERTO PRESENTE ({sorted(hit)}) -> falha de GERACAO")
        else:
            print(f"  >> CHUNK CERTO AUSENTE -> falha de RETRIEVAL")

    print("\n" + "=" * 78)
    print("VEREDICTO DO DIAGNOSTICO")
    print("=" * 78)
    n_ret = len(buckets["CHUNK_AUSENTE"])
    n_gen = len(buckets["CHUNK_PRESENTE"])
    total = n_ret + n_gen
    print(f"  Falha de RETRIEVAL (chunk certo nao chegou): {n_ret}  {buckets['CHUNK_AUSENTE']}")
    print(f"  Falha de GERACAO   (chunk certo estava la) : {n_gen}  {buckets['CHUNK_PRESENTE']}")
    if buckets["SEM_GOLD"]:
        print(f"  Sem gold anotado (rever dataset)           : {buckets['SEM_GOLD']}")

    print()
    if total == 0:
        print("  Nenhuma falha nao-abstencao para investigar.")
    elif n_ret > n_gen:
        print("  >> A MAIORIA sao falhas de RETRIEVAL. Trocar de gerador (ex: qwen)")
        print("     NAO resolveria estes casos — o numero nem chega ao contexto.")
        print("     Accao com mais retorno: melhorar o retrieval nas numericas")
        print("     (ex: o chunking parte tabelas financeiras? o top_k=3 e' curto?).")
    elif n_gen > n_ret:
        print("  >> A MAIORIA sao falhas de GERACAO. O numero estava no contexto e o")
        print("     modelo errou. Um gerador melhor (qwen2.5:7b) pode valer a pena.")
    else:
        print("  >> Empate. Provavelmente ha os dois problemas — ver caso a caso.")
    print("=" * 78)


if __name__ == "__main__":
    main()