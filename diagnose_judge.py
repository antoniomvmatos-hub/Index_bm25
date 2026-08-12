"""
DIAGNÓSTICO DO LLM-AS-JUDGE.

Porquê este script: na primeira corrida da avaliação, o juiz respondeu
NOT_GROUNDED a 5 em 5 respostas — incluindo respostas obviamente correctas
("42,000 employees", que está literalmente no chunk). Um juiz que rejeita
tudo é tão inútil como um que aceita tudo (a mesma lição do verificador NLI).

Antes de confiar num juiz, valida-se o juiz. Este script faz o que devíamos
ter feito à partida: dá-lhe casos de resposta CONHECIDA e vê se acerta.

Testa três coisas:
  1. ACERTA em casos óbvios? (resposta correcta -> GROUNDED,
     resposta inventada -> NOT_GROUNDED)
  2. VIÉS DE POSIÇÃO: se trocarmos a ordem das opções no prompt
     ("GROUNDED or NOT_GROUNDED" vs "NOT_GROUNDED or GROUNDED"), a decisão
     muda? Se mudar, o modelo está só a repetir a última opção que leu —
     não está a julgar nada.
  3. Um prompt mais simples ajuda? Modelos pequenos afogam-se em instruções
     longas.

Uso:
  python diagnose_judge.py
  python diagnose_judge.py --model qwen2.5:7b
"""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generate import generate, check_server, DEFAULT_MODEL

# --- os três prompts a comparar ---------------------------------------

# V1: o original (o que falhou)
PROMPT_V1 = """You are a strict fact-checker. Your job is to decide whether an ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Decide: is EVERY factual claim in the ANSWER supported by the CONTEXT above?
- Ignore citation labels like [S1] when judging; judge only the content.
- If the answer states a number, that exact number must appear in the context.
- If any claim is not supported, or contradicts the context, the answer is NOT grounded.

Reply with ONE word only: GROUNDED or NOT_GROUNDED"""

# V2: igual ao V1 mas com a ORDEM DAS OPÇÕES TROCADA.
# Se as decisões mudarem só por causa disto, é viés de posição.
PROMPT_V2 = PROMPT_V1.replace(
    "Reply with ONE word only: GROUNDED or NOT_GROUNDED",
    "Reply with ONE word only: NOT_GROUNDED or GROUNDED"
)

# V3: simples e directo. Modelos pequenos lidam mal com listas de regras.
# Pede SIM/NÃO em vez de etiquetas compostas (NOT_GROUNDED contém GROUNDED,
# o que também dificulta ao modelo produzir a etiqueta certa).
PROMPT_V3 = """Read the DOCUMENT and the STATEMENT.

DOCUMENT:
{context}

STATEMENT: {answer}

Is the STATEMENT true according to the DOCUMENT?
Answer YES or NO."""


def parse_grounded(raw: str) -> str:
    """Parser para os prompts V1/V2 (etiquetas GROUNDED/NOT_GROUNDED)."""
    u = raw.upper()
    # ordem importa: 'NOT_GROUNDED' contém 'GROUNDED'
    if "NOT_GROUNDED" in u or "NOT GROUNDED" in u:
        return "NO"
    if "GROUNDED" in u:
        return "YES"
    return "?"


def parse_yesno(raw: str) -> str:
    """Parser para o prompt V3 (YES/NO)."""
    u = raw.strip().upper()
    if u.startswith("NO") or " NO " in f" {u} ":
        return "NO"
    if u.startswith("YES") or " YES " in f" {u} ":
        return "YES"
    return "?"


# --- casos de teste com resposta conhecida ----------------------------

CONTEXT_EMPLOYEES = (
    "As of the end of fiscal year 2026, we had approximately 42,000 employees "
    "in 38 countries; 31,000 were engaged in research and development and "
    "11,000 were engaged in sales, marketing, operations, and administrative "
    "positions. More than 80% of our employees are in technical roles."
)

CONTEXT_GROQ = (
    "In December 2025, we entered into a non-exclusive license agreement with "
    "Groq, Inc., or Groq, for its language processing unit technology and hired "
    "certain Groq employees. We recorded $14.4 billion of goodwill and a $2.5 "
    "billion developed technology intangible asset. Total consideration consists "
    "of $13.0 billion paid at closing and $4 billion payable within one year."
)

CONTEXT_INCORPORATION = (
    "Headquartered in Santa Clara, California, NVIDIA was incorporated in "
    "California in April 1993 and reincorporated in Delaware in April 1998."
)

# Casos ordenados do fácil para o difícil. Os quatro primeiros são o teste
# mínimo (o modelo discrimina de todo?); os restantes são os casos realistas
# onde um juiz fraco se desmascara.
CASES = [
    # --- básicos: o modelo discrimina? ---
    ("verdadeiro simples", CONTEXT_EMPLOYEES,
     "How many employees did NVIDIA have?",
     "NVIDIA had approximately 42,000 employees.", "YES"),

    ("verdadeiro, outro numero", CONTEXT_EMPLOYEES,
     "How many employees work in R&D?",
     "31,000 employees work in research and development.", "YES"),

    ("numero inventado", CONTEXT_EMPLOYEES,
     "How many employees did NVIDIA have?",
     "NVIDIA had approximately 99,000 employees.", "NO"),

    ("facto ausente", CONTEXT_EMPLOYEES,
     "What does NVIDIA do with its profits?",
     "NVIDIA donates all its profits to charity.", "NO"),

    # --- difíceis: onde um juiz fraco se desmascara ---
    ("numero real mas do sitio errado", CONTEXT_GROQ,
     "How much goodwill was recorded in the Groq deal?",
     "NVIDIA recorded $13.0 billion of goodwill in the Groq deal.", "NO"),

    ("meia verdade (1a parte certa, 2a inventada)", CONTEXT_GROQ,
     "What did NVIDIA record in the Groq deal?",
     "NVIDIA recorded $14.4 billion of goodwill and acquired all of Groq's "
     "customer contracts.", "NO"),

    ("correferencia: 'we' vs 'NVIDIA'", CONTEXT_EMPLOYEES,
     "How many countries does NVIDIA operate in?",
     "NVIDIA operates in 38 countries.", "YES"),

    ("fundamentado mas responde a pergunta errada", CONTEXT_INCORPORATION,
     "In which U.S. state is NVIDIA incorporated?",
     "NVIDIA was incorporated in California.", "YES"),
]

# NOTA sobre o último caso: a resposta é FUNDAMENTADA (a frase está no texto)
# mas ERRADA para a pergunta (a resposta certa é Delaware, por causa da
# reincorporação). Esperamos YES porque estamos a medir FAITHFULNESS, não
# correcção. É uma distinção real e uma limitação do que este pipeline mede:
#   faithfulness       -> "isto está no documento?"
#   answer correctness -> "isto responde bem à pergunta?"
# Este caso serve para verificar que o juiz percebe a diferença. Se disser NO,
# está a julgar correcção em vez de fundamentação — o que estragaria a métrica.


def run_variant(name: str, template: str, parser, model: str) -> dict:
    print(f"--- {name} ---")
    acertos = 0
    respostas = []
    for caso, context, question, answer, esperado in CASES:
        prompt = template.format(context=context, question=question, answer=answer)
        raw = generate(prompt, model=model)
        got = parser(raw)
        ok = got == esperado
        acertos += ok
        respostas.append(got)
        print(f"  [{'OK ' if ok else 'ERRO'}] {caso:22s} esperado={esperado:3s} "
              f"obtido={got:3s}  (cru: {raw[:40]!r})")
    print(f"  Acertos: {acertos}/{len(CASES)}\n")
    return {"acertos": acertos, "respostas": respostas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not check_server():
        return

    print(f"A diagnosticar o juiz: {args.model}\n")

    v1 = run_variant("V1: prompt original (GROUNDED or NOT_GROUNDED)",
                     PROMPT_V1, parse_grounded, args.model)
    v2 = run_variant("V2: ordem das opcoes TROCADA (NOT_GROUNDED or GROUNDED)",
                     PROMPT_V2, parse_grounded, args.model)
    v3 = run_variant("V3: prompt simples (YES or NO)",
                     PROMPT_V3, parse_yesno, args.model)

    print("=" * 66)
    print("DIAGNOSTICO")
    print("=" * 66)
    n = len(CASES)
    print(f"  V1 (original)  : {v1['acertos']}/{n}  {v1['respostas']}")
    print(f"  V2 (invertido) : {v2['acertos']}/{n}  {v2['respostas']}")
    print(f"  V3 (simples)   : {v3['acertos']}/{n}  {v3['respostas']}")
    print()

    # Linha de base do inútil: quantos acertaria quem dissesse sempre a mesma
    # coisa? Se uma variante não bater isto, não está a discriminar nada.
    n_yes = sum(1 for c in CASES if c[4] == "YES")
    n_no = n - n_yes
    baseline = max(n_yes, n_no)
    print(f"  Linha de base (dizer sempre a mesma coisa): {baseline}/{n}")
    for nome, v in [("V1", v1), ("V2", v2), ("V3", v3)]:
        unicas = set(v["respostas"])
        if len(unicas) == 1:
            print(f"  >> {nome} respondeu SEMPRE '{unicas.pop()}' — zero discriminacao,")
            print(f"     os acertos sao coincidencia (relogio parado).")

    # Viés de posição: a ordem das opções mudou as decisões E na direcção
    # esperada? No V2 a última opção lida é GROUNDED (=YES), por isso viés de
    # posição empurraria para MAIS YES. Se as mudanças forem noutro sentido,
    # é instabilidade, não viés.
    print()
    if v1["respostas"] == v2["respostas"]:
        print("  >> Sem vies de posicao: trocar a ordem nao mudou nada.")
    else:
        mudou_para_yes = sum(1 for a, b in zip(v1["respostas"], v2["respostas"])
                             if a != b and b == "YES")
        mudou_para_no = sum(1 for a, b in zip(v1["respostas"], v2["respostas"])
                            if a != b and b == "NO")
        print(f"  >> A ordem mudou {mudou_para_yes + mudou_para_no} decisao(oes): "
              f"{mudou_para_yes} para YES, {mudou_para_no} para NO.")
        if mudou_para_yes > mudou_para_no:
            print("     Direccao consistente com VIES DE POSICAO (a ultima opcao")
            print("     lida no V2 e' GROUNDED, e as decisoes foram para YES).")
        else:
            print("     Direccao NAO consistente com vies de posicao — as decisoes")
            print("     foram para o lado oposto ao esperado. Isto e' INSTABILIDADE")
            print("     em casos concretos, nao vies sistematico.")

    print()
    melhor = max([("V1", v1), ("V2", v2), ("V3", v3)], key=lambda x: x[1]["acertos"])
    print(f"  >> Melhor variante: {melhor[0]} com {melhor[1]['acertos']}/{n}")
    if melhor[1]["acertos"] <= baseline:
        print("     NAO supera a linha de base -> este modelo nao serve para julgar.")
        print("     Caminho: modelo maior SO para o juiz.")
    elif melhor[1]["acertos"] < n:
        print("     Supera a linha de base mas nao acerta em tudo. Ver acima que")
        print("     casos falhou — pode ser aceitavel se forem os dificeis.")
    else:
        print("     Acerta em TODOS os casos, incluindo os dificeis. Juiz utilizavel.")
    print("=" * 66)


if __name__ == "__main__":
    main()