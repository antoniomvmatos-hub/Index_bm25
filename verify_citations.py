"""
PASSO 7 (bloco 2): CITATION VERIFICATION — camada estrutural (Abordagem 1).

Recebe a resposta gerada pelo LLM + o label_map do build_prompt, e audita-a:

  1. Parte a resposta em afirmações (claims), tipicamente uma por frase.
  2. Para cada afirmação, extrai as citações marcadas: [S1], [S2][S4], etc.
  3. Verifica três coisas:
     - CITAÇÃO INVENTADA: cita um label que não existe no contexto (ex: [S9]
       quando só havia S1..S3). Sintoma clássico de alucinação.
     - AFIRMAÇÃO SEM FONTE: uma afirmação factual sem nenhuma citação.
     - ABSTENÇÃO: a resposta é exactamente a frase de recusa -> não é erro,
       é o comportamento correcto para perguntas fora do contexto.

Esta camada NÃO verifica se o chunk citado SUPORTA de facto a afirmação —
isso é a camada semântica (NLI), que vem a seguir. Aqui só se verifica a
ESTRUTURA das citações. A distinção importa: um LLM pode citar [S1]
correctamente formatado e na mesma inventar o número.

Testável sem gerador nenhum: escrevem-se respostas à mão (uma boa, uma com
citação inventada, uma sem fontes) e vê-se o verificador apanhá-las.

Uso:
  python verify_citations.py     # corre os testes com respostas de exemplo
"""

import re

ABSTENTION_PHRASE = "I cannot answer this question based on the provided context."

# frases que não são afirmações factuais e por isso não precisam de citação
# (o LLM às vezes acrescenta uma abertura ou fecho inofensivo)
NON_FACTUAL_PATTERNS = [
    r"^(here|below|the following)\b",
    r"^(in summary|to summarise|to summarize|overall)\b",
    r"^(note that|please note)\b",
]


def is_abstention(answer: str) -> bool:
    """Detecta a frase FIXA de recusa. É por isso que o prompt exige uma
    frase literal: torna esta verificação trivial e sem ambiguidade."""
    return answer.strip().rstrip(".").lower() == ABSTENTION_PHRASE.strip().rstrip(".").lower()


def split_into_claims(answer: str) -> list:
    """
    Parte a resposta em afirmações. Heurística: uma por frase.

    Nota honesta de limitação: partir por ponto final é simplista — falha em
    abreviaturas ("Inc.", "U.S.") e em números decimais ("$14.4 billion").
    Por isso a regex exige que o ponto seja seguido de espaço + maiúscula, e
    protegemos os casos mais comuns antes de partir.
    """
    protected = answer
    # protege pontos que NÃO terminam frases
    for abbr in ["Inc.", "Corp.", "U.S.", "Ltd.", "No.", "Jan.", "Dec."]:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    # protege decimais: $14.4 -> $14<DOT>4
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    # parte em pontos seguidos de espaço e maiúscula (ou fim de string)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

    claims = []
    for p in parts:
        claim = p.replace("<DOT>", ".").strip()
        if claim:
            claims.append(claim)
    return claims


def extract_citations(claim: str) -> list:
    """Extrai os labels citados numa afirmação: 'texto [S1][S3].' -> ['S1','S3']"""
    return re.findall(r"\[([A-Z]\d+)\]", claim)


def is_factual_claim(claim: str) -> bool:
    """
    Decide se uma afirmação precisa de citação.

    ORDEM IMPORTA. A regra do tamanho (que era a primeira versão) era frágil:
    foi escrita a pensar em prosa ("In summary, ...") mas o modelo real
    respondeu em telegrama — "$14.4 billion [S1]." tem 3 palavras e era
    classificada como NON_FACTUAL, atravessando as DUAS camadas de
    verificação sem ser verificada. Bug apanhado ao ligar o gerador real.

    A hierarquia correcta:
      1. Cita uma fonte -> é factual por definição (o LLM está a afirmar algo
         e a atribuir-lhe origem). O tamanho é irrelevante.
      2. Frase de enquadramento sem citação -> não é facto novo.
      3. Contém números -> quase de certeza um facto (e se não citar, deve
         ser marcada UNCITED).
      4. Só então o tamanho serve de último critério.
    """
    # 1. cita uma fonte -> factual, ponto final
    if extract_citations(claim):
        return True

    stripped = claim.strip().lower()

    # 2. frases de enquadramento sem citação não são factos novos
    for pattern in NON_FACTUAL_PATTERNS:
        if re.match(pattern, stripped):
            return False

    # 3. contém números -> quase de certeza um facto
    if re.search(r"\d", stripped):
        return True

    # 4. fallback: curtas, sem citação e sem números, raramente são factos
    return len(stripped.split()) >= 4


def verify_structure(answer: str, label_map: dict) -> dict:
    """
    Auditoria estrutural completa. Devolve um relatório com o veredicto por
    afirmação e um resumo agregado.
    """
    if is_abstention(answer):
        return {
            "abstained": True,
            "claims": [],
            "summary": {"total_claims": 0, "ok": 0, "fabricated_citation": 0, "uncited": 0},
            "verdict": "ABSTENTION (comportamento correcto se a resposta não estava no contexto)",
        }

    valid_labels = set(label_map.keys())
    claims_report = []
    counts = {"ok": 0, "fabricated_citation": 0, "uncited": 0}

    for claim_text in split_into_claims(answer):
        citations = extract_citations(claim_text)
        factual = is_factual_claim(claim_text)

        fabricated = [c for c in citations if c not in valid_labels]

        if fabricated:
            status = "FABRICATED_CITATION"
            counts["fabricated_citation"] += 1
        elif factual and not citations:
            status = "UNCITED"
            counts["uncited"] += 1
        elif not factual:
            status = "NON_FACTUAL (não exige citação)"
        else:
            status = "OK"
            counts["ok"] += 1

        claims_report.append({
            "claim": claim_text,
            "citations": citations,
            "resolved_chunks": [label_map.get(c) for c in citations if c in valid_labels],
            "fabricated": fabricated,
            "status": status,
        })

    problems = counts["fabricated_citation"] + counts["uncited"]
    verdict = "PASS" if problems == 0 else f"FAIL ({problems} problema(s) estrutural(is))"

    return {
        "abstained": False,
        "claims": claims_report,
        "summary": {"total_claims": len(claims_report), **counts},
        "verdict": verdict,
    }


def print_report(report: dict, title: str):
    print(f"--- {title} ---")
    print(f"Veredicto: {report['verdict']}")
    if report["abstained"]:
        print()
        return
    for i, c in enumerate(report["claims"], start=1):
        print(f"  {i}. [{c['status']}]")
        print(f"     Afirmação: {c['claim'][:90]}")
        if c["citations"]:
            print(f"     Cita: {c['citations']} -> {c['resolved_chunks']}")
        if c["fabricated"]:
            print(f"     !! Labels inexistentes: {c['fabricated']}")
    print(f"  Resumo: {report['summary']}")
    print()


if __name__ == "__main__":
    # label_map igual ao que o build_prompt.py produziu para a query da Groq
    label_map = {
        "S1": "nvidia_10k_fy2026__chunk0318",
        "S2": "nvidia_10k_fy2026__chunk0330",
        "S3": "nvidia_10k_fy2026__chunk0314",
    }

    # CASO 1: resposta bem comportada
    good = "NVIDIA recorded $14.4 billion of goodwill in the Groq deal [S1]. The goodwill was allocated to the Compute & Networking reporting unit [S1][S2]."
    print_report(verify_structure(good, label_map), "Caso 1: resposta correcta (esperado: PASS)")

    # CASO 2: cita um label que não existe (alucinação clássica)
    fabricated = "NVIDIA recorded $14.4 billion of goodwill [S1]. The deal also included a $9 billion cash payment [S7]."
    print_report(verify_structure(fabricated, label_map), "Caso 2: citação inventada [S7] (esperado: FAIL)")

    # CASO 3: afirmação factual sem qualquer fonte
    uncited = "NVIDIA recorded $14.4 billion of goodwill [S1]. Jensen Huang personally approved the transaction."
    print_report(verify_structure(uncited, label_map), "Caso 3: afirmação sem citação (esperado: FAIL)")

    # CASO 4: abstenção correcta
    print_report(verify_structure(ABSTENTION_PHRASE, label_map), "Caso 4: abstenção (esperado: ABSTENTION)")