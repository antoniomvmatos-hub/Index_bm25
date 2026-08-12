"""
PASSO 13: AVALIAÇÃO DA GERAÇÃO.

Corre o pipeline completo (answer.py) sobre as 50 perguntas do eval_dataset e
mede três coisas:

  1. FAITHFULNESS — as respostas estão fundamentadas nas fontes?
     Medida de DUAS formas independentes, de propósito:
       (a) verificador próprio: % de respostas VERIFIED
       (b) LLM-as-judge: um segundo prompt ao LLM a perguntar "esta resposta
           está fundamentada neste contexto?"
     Se concordarem, ganha-se confiança nas duas. Se discordarem, aprende-se
     onde cada uma falha. A (a) é o sistema a avaliar-se a si próprio — daí
     a necessidade da (b), que é o que RAGAS e TruLens usam.

  2. ABSTENTION ACCURACY — das perguntas out_of_scope, em quantas se absteve
     correctamente? (deve ser alta)

  3. OVER-ABSTENTION — das perguntas respondíveis, em quantas se absteve
     quando DEVIA ter respondido? (deve ser baixa)

Porque a 3ª métrica importa: um sistema que se abstém sempre teria 100% na 2ª
e seria completamente inútil. As duas medem-se sempre em conjunto, como
precision e recall.

NOTA SOBRE O LLM-AS-JUDGE: usa o mesmo modelo que gerou a resposta. Isto tem
um viés conhecido (modelos tendem a aprovar o próprio output) e está
registado nas limitações. O ideal seria um modelo diferente e maior.

Uso:
  python evaluate_generation.py
  python evaluate_generation.py --limit 5           # teste rápido
  python evaluate_generation.py --model qwen2.5:7b
  python evaluate_generation.py --no-judge          # salta o LLM-as-judge
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from answer import answer_and_verify
from generate import generate, check_server, DEFAULT_MODEL
from verify_citations import ABSTENTION_PHRASE

BASE_DIR = Path(__file__).parent
EVAL_DATASET_PATH = BASE_DIR / "eval_dataset.jsonl"
RESULTS_DIR = BASE_DIR / "eval_results"

# O juiz é um modelo SEPARADO e maior. Validado com diagnose_judge.py:
# llama3.2:3b -> 2/4 (diz sempre NÃO, zero discriminação)
# qwen2.5:7b  -> 4/4
DEFAULT_JUDGE_MODEL = "qwen2.5:7b"

JUDGE_PROMPT = """You are a strict fact-checker. Your job is to decide whether an ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Decide: is EVERY factual claim in the ANSWER supported by the CONTEXT above?
- Ignore citation labels like [S1] when judging; judge only the content.
- If the answer states a number, that exact number must appear in the context.
- If any claim is not supported, or contradicts the context, the answer is NOT grounded.

Reply with ONE word only: GROUNDED or NOT_GROUNDED"""


def load_eval_dataset():
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge_faithfulness(question: str, answer: str, chunks_text: list,
                       model: str = DEFAULT_MODEL) -> dict:
    """
    LLM-as-judge: pergunta ao LLM se a resposta está fundamentada.

    O 'model' aqui é DELIBERADAMENTE separado do modelo do gerador. Descoberta
    que motivou a separação: o llama3.2:3b gera respostas correctas mas não
    consegue julgá-las — no diagnóstico respondeu 'NOT_GROUNDED' a tudo (2/4,
    exactamente a pontuação de quem diz sempre não). O qwen2.5:7b acerta 4/4.
    JULGAR É MAIS DIFÍCIL QUE GERAR — daí o RAGAS e o TruLens usarem modelos
    de topo como juízes.

    Devolve a decisão e o texto cru, porque LLMs pequenos nem sempre
    obedecem ao "ONE word only" — e queremos poder auditar isso depois em
    vez de o esconder.
    """
    context = "\n\n---\n\n".join(chunks_text)
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    raw = generate(prompt, model=model)

    upper = raw.upper()
    # ordem importa: "NOT_GROUNDED" contém "GROUNDED"
    if "NOT_GROUNDED" in upper or "NOT GROUNDED" in upper:
        decision = "NOT_GROUNDED"
    elif "GROUNDED" in upper:
        decision = "GROUNDED"
    else:
        decision = "UNPARSEABLE"

    return {"decision": decision, "raw": raw[:200]}


def evaluate(questions, engines, detector, chunk_texts, model, judge_model=None,
             use_judge=True):
    """
    'model' gera as respostas; 'judge_model' julga-as. São separados de
    propósito (ver judge_faithfulness). Se judge_model for None, usa o mesmo.
    """
    judge_model = judge_model or model
    per_question = []
    counts = defaultdict(int)

    for i, q in enumerate(questions, start=1):
        t0 = time.time()
        is_out_of_scope = not q["relevant_chunk_ids"]

        result = answer_and_verify(q["question"], engines, detector, chunk_texts,
                                   model=model)
        abstained = result["verdict"] == "ABSTAINED"

        # --- métricas de abstenção ---
        if is_out_of_scope:
            counts["out_of_scope_total"] += 1
            if abstained:
                counts["correct_abstention"] += 1      # bom: não sabia e disse
            else:
                counts["hallucinated_out_of_scope"] += 1  # mau: inventou
        else:
            counts["answerable_total"] += 1
            if abstained:
                counts["over_abstention"] += 1          # mau: sabia e calou-se
            else:
                counts["answered"] += 1

        # --- faithfulness (a): o nosso verificador ---
        # Conta-se SEPARADAMENTE para respondíveis e out_of_scope. Bug apanhado
        # na primeira corrida completa: contar todos os VERIFIED no numerador
        # mas só as respondíveis no denominador mistura populações — as
        # out_of_scope respondidas (q030, q046) inflacionavam a métrica. As duas
        # faithfulness (nossa e do juiz) têm de olhar para a MESMA população.
        counts[f"verdict_{result['verdict']}"] += 1
        if not is_out_of_scope and not abstained:
            counts[f"answerable_verdict_{result['verdict']}"] += 1

        # --- faithfulness (b): LLM-as-judge ---
        # só faz sentido em respostas de facto dadas (numa abstenção não há
        # afirmações para julgar)
        judge = None
        if use_judge and not abstained:
            chunks_used = [chunk_texts[cid] for cid in result["label_map"].values()]
            judge = judge_faithfulness(q["question"], result["answer"], chunks_used,
                                       judge_model)
            counts[f"judge_{judge['decision']}"] += 1
            # mesma população da faithfulness própria, para a comparação ser justa
            if not is_out_of_scope:
                counts[f"answerable_judge_{judge['decision']}"] += 1

        elapsed = time.time() - t0
        per_question.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": result["answer"],
            "verdict": result["verdict"],
            "judge": judge,
            "expected_abstention": is_out_of_scope,
            "abstained": abstained,
            "retrieved_chunks": result["retrieved_chunks"],
            "seconds": round(elapsed, 1),
        })

        judge_str = f" | judge={judge['decision']}" if judge else ""
        print(f"  [{i}/{len(questions)}] {q['id']} ({q['category']}) "
              f"-> {result['verdict']}{judge_str}  [{elapsed:.0f}s]")

    return counts, per_question


def compute_summary(counts: dict) -> dict:
    def pct(num, den):
        return round(num / den, 4) if den else None

    # POPULAÇÃO COMUM às duas métricas de faithfulness: perguntas respondíveis
    # que foram de facto respondidas. As out_of_scope respondidas são um
    # problema DIFERENTE (falha de abstenção) e contam-se à parte — misturá-las
    # aqui inflacionava a métrica (bug da 1ª corrida).
    answered = counts.get("answered", 0)
    verified = counts.get("answerable_verdict_VERIFIED", 0)
    judged_grounded = counts.get("answerable_judge_GROUNDED", 0)
    judged_total = judged_grounded + counts.get("answerable_judge_NOT_GROUNDED", 0)

    return {
        # faithfulness (a) — verificador próprio
        "faithfulness_own": pct(verified, answered),
        # faithfulness (b) — LLM-as-judge, MESMA população
        "faithfulness_judge": pct(judged_grounded, judged_total),
        "judge_unparseable": counts.get("judge_UNPARSEABLE", 0),
        # abstenção
        "abstention_accuracy": pct(counts.get("correct_abstention", 0),
                                   counts.get("out_of_scope_total", 0)),
        "over_abstention_rate": pct(counts.get("over_abstention", 0),
                                    counts.get("answerable_total", 0)),
        # falha grave: respondeu a uma pergunta sem resposta no documento
        "hallucinated_out_of_scope": counts.get("hallucinated_out_of_scope", 0),
        # distribuição de veredictos (todas as 50)
        "verdicts": {k.replace("verdict_", ""): v
                     for k, v in counts.items()
                     if k.startswith("verdict_")},
        "totals": {
            "answerable": counts.get("answerable_total", 0),
            "out_of_scope": counts.get("out_of_scope_total", 0),
            "answered": answered,
            "judged": judged_total,
        },
    }


def print_summary(s: dict):
    print("\n" + "=" * 72)
    print("RESULTADOS DA AVALIAÇÃO DE GERAÇÃO")
    print("=" * 72)

    print("\nFAITHFULNESS (as respostas estão fundamentadas?)")
    own = s["faithfulness_own"]
    judge = s["faithfulness_judge"]
    print(f"  (a) verificador próprio : {own if own is not None else 'n/a'}")
    print(f"  (b) LLM-as-judge        : {judge if judge is not None else 'n/a'}")
    if own is not None and judge is not None:
        diff = abs(own - judge)
        concordancia = "concordam" if diff < 0.1 else "DISCORDAM — vale a pena investigar"
        print(f"      -> diferença {diff:.3f} ({concordancia})")
    if s["judge_unparseable"]:
        print(f"      (juiz deu {s['judge_unparseable']} resposta(s) não interpretável(is))")

    print("\nABSTENÇÃO")
    print(f"  Abstention accuracy  : {s['abstention_accuracy']}   "
          f"(das {s['totals']['out_of_scope']} out_of_scope, quantas recusou bem)")
    print(f"  Over-abstention rate : {s['over_abstention_rate']}   "
          f"(das {s['totals']['answerable']} respondíveis, quantas recusou mal)")
    if s["hallucinated_out_of_scope"]:
        print(f"  !! {s['hallucinated_out_of_scope']} pergunta(s) out_of_scope foram RESPONDIDAS")
        print("     (devia ter-se abstido — se o veredicto delas foi VERIFIED, o")
        print("      guardrail carimbou uma resposta que não devia existir)")
    print("  Nota: as duas leem-se juntas. Abster-se sempre daria 1.0 na primeira")
    print("        e 1.0 na segunda — sistema inútil.")

    print("\nDISTRIBUIÇÃO DE VEREDICTOS")
    for verdict, n in sorted(s["verdicts"].items(), key=lambda kv: -kv[1]):
        print(f"  {verdict:12s} {n}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="modelo que GERA as respostas")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="modelo que JULGA as respostas (deve ser maior; "
                             "validar com diagnose_judge.py antes de confiar)")
    parser.add_argument("--limit", type=int, default=None,
                        help="avalia só as primeiras N perguntas (teste rápido)")
    parser.add_argument("--no-judge", action="store_true",
                        help="salta o LLM-as-judge (mais rápido)")
    args = parser.parse_args()

    if not check_server():
        return

    questions = load_eval_dataset()
    if args.limit:
        questions = questions[:args.limit]

    n_oos = sum(1 for q in questions if not q["relevant_chunk_ids"])
    print(f"A avaliar {len(questions)} perguntas "
          f"({len(questions) - n_oos} respondíveis, {n_oos} out_of_scope)")
    print(f"Gerador: {args.model}")
    print(f"Juiz   : {'(desligado)' if args.no_judge else args.judge_model}\n")

    from rerank import load_all
    from verify_semantic import load_detector, load_chunk_texts

    print("A carregar o pipeline...")
    engines = load_all()
    detector = load_detector()
    chunk_texts = load_chunk_texts()
    print("Pronto.\n")

    t0 = time.time()
    counts, per_question = evaluate(questions, engines, detector, chunk_texts,
                                    args.model, judge_model=args.judge_model,
                                    use_judge=not args.no_judge)
    total_min = (time.time() - t0) / 60

    summary = compute_summary(counts)
    print_summary(summary)
    print(f"\nTempo total: {total_min:.1f} minutos")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "generation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "judge_model": args.judge_model,
                   "summary": summary, "per_question": per_question},
                  f, ensure_ascii=False, indent=2)
    print(f"Resultados detalhados: {out}")


if __name__ == "__main__":
    main()