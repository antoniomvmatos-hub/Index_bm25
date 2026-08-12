"""
ANÁLISE DE ERROS da avaliação de geração.

Lê o eval_results/generation_results.json e responde às perguntas que o
resumo agregado não responde:

  1. Onde é que o sistema falha? (quebra por categoria de pergunta)
  2. Onde é que os dois avaliadores discordam, e para que lado?
  3. Que perguntas out_of_scope foram respondidas indevidamente?
  4. Quais as respostas concretas dos casos problemáticos?

Corre sobre o ficheiro já gerado — não repete a avaliação (que demora ~26min).

Uso:
  python analyse_generation.py
  python analyse_generation.py --show-answers    # mostra o texto das respostas
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_PATH = Path(__file__).parent / "eval_results" / "generation_results.json"


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def breakdown_by_category(rows):
    print("=" * 72)
    print("1. VEREDICTOS POR CATEGORIA")
    print("=" * 72)

    by_cat = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_cat[r["category"]][r["verdict"]] += 1

    verdicts = ["VERIFIED", "FLAGGED", "REJECTED", "ABSTAINED"]
    print(f"{'categoria':<14}" + "".join(f"{v:>11}" for v in verdicts) + f"{'% VERIF':>10}")
    print("-" * 72)

    for cat in sorted(by_cat):
        counts = by_cat[cat]
        total = sum(counts.values())
        # nas out_of_scope, ABSTAINED é o resultado desejado — a taxa de
        # VERIFIED não faz sentido como "qualidade"
        rate = "n/a" if cat == "out_of_scope" else f"{counts['VERIFIED']/total:.0%}"
        print(f"{cat:<14}" + "".join(f"{counts[v]:>11}" for v in verdicts) + f"{rate:>10}")
    print()


def disagreements(rows):
    print("=" * 72)
    print("2. ONDE OS DOIS AVALIADORES DISCORDAM")
    print("=" * 72)

    own_ok_judge_no, own_no_judge_ok = [], []
    for r in rows:
        if not r.get("judge"):
            continue
        own_grounded = r["verdict"] == "VERIFIED"
        judge_grounded = r["judge"]["decision"] == "GROUNDED"
        if own_grounded and not judge_grounded:
            own_ok_judge_no.append(r)
        elif not own_grounded and judge_grounded:
            own_no_judge_ok.append(r)

    print(f"\nNOSSO=VERIFIED mas JUIZ=NOT_GROUNDED  ({len(own_ok_judge_no)} casos)")
    print("  -> o nosso verificador é mais PERMISSIVO")
    for r in own_ok_judge_no:
        print(f"    {r['id']} ({r['category']}): {r['question'][:52]}")

    print(f"\nNOSSO=FLAGGED/REJECTED mas JUIZ=GROUNDED  ({len(own_no_judge_ok)} casos)")
    print("  -> o nosso verificador é mais ESTRITO")
    for r in own_no_judge_ok:
        print(f"    {r['id']} ({r['category']}): {r['question'][:52]}")

    total_judged = sum(1 for r in rows if r.get("judge"))
    n_disagree = len(own_ok_judge_no) + len(own_no_judge_ok)
    print(f"\n  Concordância: {total_judged - n_disagree}/{total_judged} "
          f"({(total_judged - n_disagree)/total_judged:.0%})")
    if len(own_ok_judge_no) > len(own_no_judge_ok) * 2:
        print("  >> O nosso verificador é sistematicamente mais permissivo.")
        print("     Causa provável: o limiar do LettuceDetect (0.9) foi calibrado")
        print("     para EVITAR falsos positivos em afirmações verdadeiras — o que")
        print("     necessariamente o inclina para 'suportado'. Ver PROJECT_LOG 18.8.")
    print()


def out_of_scope_failures(rows, show_answers=False):
    print("=" * 72)
    print("3. OUT_OF_SCOPE RESPONDIDAS (deviam ter-se abstido)")
    print("=" * 72)

    failures = [r for r in rows if r["expected_abstention"] and not r["abstained"]]
    if not failures:
        print("  Nenhuma. O sistema absteve-se em todas.\n")
        return

    for r in failures:
        judge = r["judge"]["decision"] if r.get("judge") else "n/a"
        print(f"\n  {r['id']} -> veredicto {r['verdict']} | juiz {judge}")
        print(f"    Pergunta: {r['question']}")
        print(f"    Resposta: {r['answer'][:160]}")
        if r["verdict"] == "VERIFIED":
            print("    !! O GUARDRAIL APROVOU. Este é o pior caso: resposta")
            print("       fundamentada no texto citado, mas a pergunta não tinha")
            print("       resposta no documento. Faithfulness != correctness:")
            print("       o modelo respondeu a OUTRA pergunta, com fontes válidas.")
    print()


def worst_cases(rows, show_answers=False):
    print("=" * 72)
    print("4. CASOS REJECTED (falha estrutural: citações inventadas ou sem fonte)")
    print("=" * 72)
    rejected = [r for r in rows if r["verdict"] == "REJECTED"]
    for r in rejected:
        print(f"\n  {r['id']} ({r['category']}): {r['question'][:60]}")
        if show_answers:
            print(f"    Resposta: {r['answer'][:200]}")
    if not show_answers and rejected:
        print("\n  (usa --show-answers para ver o texto das respostas)")
    print()


def timing(rows):
    times = [r["seconds"] for r in rows if "seconds" in r]
    if not times:
        return
    print("=" * 72)
    print("5. TEMPOS")
    print("=" * 72)
    print(f"  media: {sum(times)/len(times):.1f}s | min: {min(times):.0f}s | max: {max(times):.0f}s")
    abst = [r["seconds"] for r in rows if r["abstained"]]
    resp = [r["seconds"] for r in rows if not r["abstained"]]
    if abst and resp:
        print(f"  abstencoes: {sum(abst)/len(abst):.1f}s  (mais rapidas — resposta curta")
        print(f"              e as camadas de verificacao nao chegam a correr)")
        print(f"  respostas : {sum(resp)/len(resp):.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-answers", action="store_true")
    args = parser.parse_args()

    data = load_results()
    rows = data["per_question"]

    print(f"\nGerador: {data.get('model')} | Juiz: {data.get('judge_model', 'n/a')}")
    print(f"Perguntas avaliadas: {len(rows)}\n")

    breakdown_by_category(rows)
    disagreements(rows)
    out_of_scope_failures(rows, args.show_answers)
    worst_cases(rows, args.show_answers)
    timing(rows)


if __name__ == "__main__":
    main()