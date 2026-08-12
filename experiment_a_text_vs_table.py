"""
EXPERIENCIA A: a diferenca texto-vs-tabela e' real ou acaso?

Nao corre o pipeline — le o generation_results.json + eval_dataset.jsonl +
chunks_nvidia.jsonl e classifica cada pergunta pela NATUREZA DA FONTE:

  - TABELA: pelo menos um dos chunks gold contem o marcador [TABELA]
  - TEXTO : nenhum chunk gold e' tabela

Depois compara a taxa de sucesso (VERIFIED, ou o chunk gold ter sido
recuperado) entre os dois grupos. Se as perguntas-tabela falharem muito mais
que as perguntas-texto, a diferenca e' ESTRUTURAL, nao acaso.

Isto formaliza a observacao informal "factual 85% vs numeric 41%": a divisao
certa nao e' factual/numeric (uma categoria de tema), e' texto/tabela (a
natureza fisica da fonte). Ha perguntas numericas cuja resposta esta em prosa
e perguntas factuais cuja resposta esta em tabela — esta analise separa-as
pelo que interessa.

Uso:
  python experiment_a_text_vs_table.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
RESULTS = BASE / "eval_results" / "generation_results.json"
DATASET = BASE / "eval_dataset.jsonl"
CHUNKS = BASE / "chunks" / "chunks_nvidia.jsonl"


def load():
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    dataset = {}
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            dataset[q["id"]] = q
    chunk_is_table = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_is_table[c["chunk_id"]] = "[TABELA]" in c["text"]
    return results, dataset, chunk_is_table


def classify_source(gold_ids, chunk_is_table):
    """TABELA se algum chunk gold e' tabela; TEXTO caso contrario."""
    if not gold_ids:
        return "sem_gold"
    if any(chunk_is_table.get(cid, False) for cid in gold_ids):
        return "TABELA"
    return "TEXTO"


def main():
    results, dataset, chunk_is_table = load()
    rows = results["per_question"]

    print(f"Gerador: {results.get('model')}\n")

    stats = {"TEXTO": {"total": 0, "verified": 0, "retrieved": 0},
             "TABELA": {"total": 0, "verified": 0, "retrieved": 0}}
    detail = {"TEXTO": [], "TABELA": []}

    for r in rows:
        if r["expected_abstention"]:
            continue  # out_of_scope nao entram nesta comparacao
        q = dataset[r["id"]]
        gold = q["relevant_chunk_ids"]
        src = classify_source(gold, chunk_is_table)
        if src == "sem_gold":
            continue

        stats[src]["total"] += 1
        verified = r["verdict"] == "VERIFIED"
        retrieved = bool(set(gold) & set(r["retrieved_chunks"]))
        stats[src]["verified"] += verified
        stats[src]["retrieved"] += retrieved
        detail[src].append((r["id"], r["category"], r["verdict"], retrieved))

    print("=" * 74)
    print("SUCESSO POR NATUREZA DA FONTE (perguntas respondiveis)")
    print("=" * 74)
    print(f"{'fonte':<8}{'perguntas':>11}{'gold recuperado':>18}{'VERIFIED':>12}")
    print("-" * 74)
    for src in ["TEXTO", "TABELA"]:
        s = stats[src]
        if s["total"] == 0:
            continue
        ret = f"{s['retrieved']}/{s['total']} ({s['retrieved']/s['total']:.0%})"
        ver = f"{s['verified']}/{s['total']} ({s['verified']/s['total']:.0%})"
        print(f"{src:<8}{s['total']:>11}{ret:>18}{ver:>12}")
    print("=" * 74)

    # a conclusao
    t, tab = stats["TEXTO"], stats["TABELA"]
    if t["total"] and tab["total"]:
        ret_text = t["retrieved"] / t["total"]
        ret_table = tab["retrieved"] / tab["total"]
        print("\nRETRIEVAL (o gold chegou ao top-3?):")
        print(f"  texto : {ret_text:.0%}")
        print(f"  tabela: {ret_table:.0%}")
        if ret_text - ret_table > 0.25:
            print(f"\n  >> As perguntas cuja resposta esta em TABELA sao recuperadas")
            print(f"     muito pior ({ret_table:.0%} vs {ret_text:.0%}). A diferenca e'")
            print(f"     ESTRUTURAL: o retrieval falha em tabelas, nao em numeros por si.")
            print(f"     Isto valida a Experiencia B (serializar tabelas em prosa).")
        else:
            print("\n  >> A diferenca nao e' grande — a hipotese tabular fica em duvida.")

    print("\nDETALHE (id | categoria | veredicto | gold recuperado?):")
    for src in ["TEXTO", "TABELA"]:
        print(f"\n  --- {src} ---")
        for qid, cat, verdict, ret in detail[src]:
            mark = "recuperado" if ret else "PERDIDO"
            print(f"    {qid} {cat:<10} {verdict:<10} {mark}")


if __name__ == "__main__":
    main()