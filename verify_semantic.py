"""
PASSO 7 (bloco 3, v3): CITATION VERIFICATION — camada semântica com LettuceDetect.

HISTÓRICO DAS TENTATIVAS (ver PROJECT_LOG para o detalhe):
  v1 — NLI genérico (cross-encoder/nli-deberta-v3-base): FALHOU.
       Com o chunk inteiro como premissa dizia 'neutral' a tudo; com frases
       isoladas apanhava a mentira do $99B mas marcava a afirmação VERDADEIRA
       como UNSUPPORTED (falso negativo). O modelo foi treinado em SNLI/MNLI
       (frases curtas do dia-a-dia), não em documentos financeiros densos.
  v2 — HHEM (vectara/hallucination_evaluation_model): INVIÁVEL.
       O modelo traz código próprio (trust_remote_code) escrito para uma
       versão antiga do transformers; rebenta com AttributeError
       ('all_tied_weights_keys') e os pesos nem carregam bem. Abandonado.
  v3 — LettuceDetect (esta): a ferramenta certa para o trabalho.

PORQUÊ O LETTUCEDETECT:
  - Treinado no RAGTruth (18k exemplos anotados de alucinação em RAG) —
    exactamente a nossa tarefa, ao contrário do NLI genérico.
  - Baseado em ModernBERT: contexto até 4k tokens. Resolve directamente a
    limitação que matou a v1 (premissas longas).
  - pip install limpo, licença MIT, sem trust_remote_code — a razão exacta
    pela qual a v2 rebentou.
  - Pequeno (150M base / 396M large) e rápido em CPU.
  - BÓNUS: detecção ao nível do TOKEN. Não diz só "esta frase é má" — diz
    QUE PALAVRAS são inventadas, com um score de confiança.

DIFERENÇA DE API face às versões anteriores:
  O LettuceDetect recebe um triplo (context, question, answer) e devolve os
  spans alucinados da answer. Precisa da PERGUNTA, que as versões anteriores
  não usavam.

Dois modos implementados:
  - MODO NATURAL: context = todos os chunks citados, answer = resposta toda.
    É a forma como o modelo foi desenhado para ser usado.
  - MODO POR-CITAÇÃO: para cada afirmação, context = APENAS o chunk que ela
    cita. Mais rigoroso — é o que combina de facto a Abordagem 1 (marcação)
    com a Abordagem 2 (verificação semântica): não basta a afirmação estar
    suportada por ALGUM chunk, tem de estar suportada pelo chunk que CITOU.
    Só este modo apanha o "citar a fonte errada".

Uso:
  pip install lettucedetect -U
  python verify_semantic.py
"""

import json
import re
from pathlib import Path

from lettucedetect.models.inference import HallucinationDetector

from verify_citations import split_into_claims, extract_citations, is_abstention, is_factual_claim

# ATENÇÃO ao nome: 'lettucedect', sem o segundo 'e'. É um typo no repositório
# oficial deles, mas é o nome real do modelo no Hugging Face.
LETTUCE_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"   # 150M
# alternativa mais precisa e mais pesada:
# LETTUCE_MODEL = "KRLabsOrg/lettucedect-large-modernbert-en-v1"  # 396M

CHUNKS_PATH = Path(__file__).parent / "chunks" / "chunks_nvidia.jsonl"

# um span com confiança abaixo disto é ignorado (ruído do modelo)
CONFIDENCE_THRESHOLD = 0.9


def load_detector():
    print(f"A carregar LettuceDetect ({LETTUCE_MODEL})...")
    return HallucinationDetector(method="transformer", model_path=LETTUCE_MODEL)


def load_chunk_texts():
    """Mapa chunk_id -> texto. A camada estrutural nunca precisou disto;
    esta precisa, porque vai LER os chunks."""
    texts = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            texts[c["chunk_id"]] = c["text"]
    return texts


def strip_citations(claim: str) -> str:
    """Remove os marcadores [S1][S2] antes de dar ao modelo. O modelo nunca
    viu marcadores destes no treino — deixá-los só o confunde, e além disso
    deslocaria as posições dos spans que ele devolve."""
    return re.sub(r"\s*\[[A-Z]\d+\]", "", claim).strip()


def safe(text: str) -> str:
    """
    Normaliza caracteres unicode 'invisíveis' do HTML da SEC que o cp1252 do
    Windows não sabe codificar ao escrever para ficheiro. É a MESMA família de
    problema do non-breaking space que nos mordeu no parser: aqui aparece o
    non-breaking hyphen em palavras como 'non-exclusive'.
    """
    replacements = {
        "\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\xa0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("ascii", errors="replace").decode("ascii")


def detect_spans(detector, contexts: list, question: str, answer: str) -> list:
    """
    Chama o LettuceDetect e devolve os spans alucinados acima do limiar.
    Cada span: {'start': int, 'end': int, 'text': str, 'confidence': float}
    """
    preds = detector.predict(
        context=contexts,
        question=question,
        answer=answer,
        output_format="spans",
    )
    return [p for p in preds if p.get("confidence", 0) >= CONFIDENCE_THRESHOLD]


def verify_natural(detector, question: str, answer: str, label_map: dict,
                   chunk_texts: dict) -> dict:
    """
    MODO NATURAL: dá ao modelo todos os chunks do contexto de uma vez e a
    resposta inteira. É como o LettuceDetect foi desenhado para ser usado.
    Limitação: não distingue "suportado pelo chunk citado" de "suportado por
    algum outro chunk" — não apanha o caso de citar a fonte errada.
    """
    if is_abstention(answer):
        return {"abstained": True, "spans": [], "verdict": "ABSTENTION"}

    contexts = [chunk_texts[cid] for cid in label_map.values()]
    clean_answer = strip_citations(answer)
    spans = detect_spans(detector, contexts, question, clean_answer)

    verdict = "PASS" if not spans else f"FAIL ({len(spans)} span(s) nao suportado(s))"
    return {"abstained": False, "spans": spans, "answer": clean_answer, "verdict": verdict}


def verify_per_citation(detector, question: str, answer: str, label_map: dict,
                        chunk_texts: dict) -> dict:
    """
    MODO POR-CITAÇÃO: para cada afirmação, testa APENAS contra o chunk que ela
    cita. É aqui que as duas abordagens se combinam de facto — exige que a
    afirmação esteja suportada pela fonte que INVOCOU, não por qualquer uma.

    Optimização herdada da Abordagem 1: como o LLM já nos disse [S1], fazemos
    1 comparação por afirmação em vez de a testar contra os 396 chunks.
    """
    if is_abstention(answer):
        return {"abstained": True, "claims": [], "summary": {"supported": 0, "unsupported": 0},
                "verdict": "ABSTENTION"}

    claims_report = []
    counts = {"supported": 0, "unsupported": 0}

    for claim_text in split_into_claims(answer):
        citations = extract_citations(claim_text)
        cited_valid = [c for c in citations if c in label_map]

        # sem citação válida -> a camada estrutural já tratou disso
        if not cited_valid or not is_factual_claim(claim_text):
            continue

        hypothesis = strip_citations(claim_text)

        # basta UM dos chunks citados suportar a afirmação
        per_source = []
        supported_by_any = False
        for label in cited_valid:
            chunk_id = label_map[label]
            spans = detect_spans(detector, [chunk_texts[chunk_id]], question, hypothesis)
            per_source.append({
                "source_label": label,
                "chunk_id": chunk_id,
                "spans": spans,
            })
            if not spans:
                supported_by_any = True

        status = "SUPPORTED" if supported_by_any else "UNSUPPORTED"
        counts["supported" if supported_by_any else "unsupported"] += 1

        claims_report.append({"claim": hypothesis, "status": status, "per_source": per_source})

    problems = counts["unsupported"]
    verdict = "PASS" if problems == 0 else f"FAIL ({problems} afirmacao(oes) nao fundamentada(s))"
    return {"abstained": False, "claims": claims_report, "summary": counts, "verdict": verdict}


def print_natural(report: dict, title: str):
    print(f"--- {title} ---")
    print(f"Veredicto: {report['verdict']}")
    if not report.get("abstained"):
        print(f"  Resposta: {safe(report['answer'])}")
        for s in report["spans"]:
            print(f"  !! ALUCINADO (conf={s['confidence']:.3f}): '{safe(s['text'])}'")
    print()


def print_per_citation(report: dict, title: str):
    print(f"--- {title} ---")
    print(f"Veredicto: {report['verdict']}")
    for c in report["claims"]:
        print(f"  [{c['status']}] {safe(c['claim'][:70])}")
        for s in c["per_source"]:
            if s["spans"]:
                for sp in s["spans"]:
                    print(f"      vs [{s['source_label']}]: alucinado (conf={sp['confidence']:.3f}) -> '{safe(sp['text'])}'")
            else:
                print(f"      vs [{s['source_label']}]: tudo suportado")
    print()


if __name__ == "__main__":
    QUESTION = "How much goodwill was recorded in the Groq deal?"

    label_map = {
        "S1": "nvidia_10k_fy2026__chunk0318",
        "S2": "nvidia_10k_fy2026__chunk0330",
        "S3": "nvidia_10k_fy2026__chunk0314",
    }

    chunk_texts = load_chunk_texts()
    detector = load_detector()
    print("Pronto.\n")

    CASES = [
        ("Caso 1: numero correcto (esperado: SUPPORTED)",
         "NVIDIA recorded $14.4 billion of goodwill in the Groq deal [S1]."),
        ("Caso 2: $99B com fonte real (esperado: UNSUPPORTED)",
         "NVIDIA recorded $99 billion of goodwill in the Groq deal [S1]."),
        ("Caso 3: afirmacao fora do chunk (esperado: UNSUPPORTED)",
         "NVIDIA will donate all its profits to charity [S1]."),
        ("Caso 4: chunk errado (esperado: UNSUPPORTED no modo por-citacao)",
         "NVIDIA recorded $14.4 billion of goodwill in the Groq deal [S2]."),
        ("Caso 5: numero do OUTRO chunk (esperado: UNSUPPORTED)",
         "NVIDIA recorded $15.6 billion of goodwill in the Groq deal [S1]."),
    ]

    print("#" * 70)
    print("# MODO NATURAL: contexto = todos os chunks, resposta inteira")
    print("#" * 70 + "\n")
    for title, answer in CASES:
        print_natural(verify_natural(detector, QUESTION, answer, label_map, chunk_texts), title)

    print("#" * 70)
    print("# MODO POR-CITACAO: cada afirmacao vs. APENAS o chunk que citou")
    print("#" * 70 + "\n")
    for title, answer in CASES:
        print_per_citation(verify_per_citation(detector, QUESTION, answer, label_map, chunk_texts), title)