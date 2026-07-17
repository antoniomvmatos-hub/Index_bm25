"""
PASSO 7 (bloco 1): MONTAGEM DO PROMPT.

Pega na pergunta + chunks recuperados pelo pipeline e monta o texto que
será enviado ao LLM gerador. Não precisa de nenhum LLM para ser testado —
podes correr isto e LER o prompt montado, que é o ponto: o prompt é a peça
onde se ganha ou perde a disciplina de citação.

Decisões de desenho (Abordagem 1 - citações explícitas por marcação):
  - Cada chunk entra no contexto com um rótulo curto e estável: [S1], [S2]...
    (não o chunk_id completo, que é longo e o LLM tende a truncar/errar).
    Guardamos o mapa S1 -> chunk_id à parte, para a verificação depois.
  - O prompt instrui explicitamente: responder SÓ com base no contexto,
    citar a fonte de CADA afirmação, e admitir ignorância se o contexto não
    tiver a resposta (isto é o que ativa a abstenção nas out_of_scope).
  - O breadcrumb de cada chunk é incluído: dá ao LLM contexto de secção
    ("isto vem de Item 8 > Note 2 - Groq"), o que ajuda a responder melhor.

Uso:
  python build_prompt.py                      # mostra o prompt de exemplo
  python build_prompt.py --query "..."        # monta o prompt para uma query
"""

import argparse
import sys

# O texto do 10-K traz caracteres unicode "invisíveis" da SEC (ex: \u2011,
# non-breaking hyphen, em "non-exclusive"). No terminal do VS Code isto passa
# (UTF-8), mas ao redireccionar para ficheiro (> prompt.txt) o Windows usa
# cp1252 e rebenta com UnicodeEncodeError.
#
# AQUI NÃO se pode substituir os caracteres por ASCII (como se faz nos
# relatórios do verify_semantic.py): este texto vai para o LLM e queremos o
# conteúdo intacto. A solução é forçar o stdout a UTF-8.
#
# Nota: é a MESMA família de problema do \xa0 que apanhámos no parser
# (PROJECT_LOG 3.4) — a raiz seria normalizar isto no clean_text() ao criar
# os chunks, mas isso obrigaria a re-indexar tudo. Fica como dívida técnica.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_INSTRUCTIONS = """You are a financial analyst assistant answering questions about SEC filings.

RULES — follow all of them strictly:

1. Answer ONLY using the information in the CONTEXT below. Never use outside knowledge, even if you are confident it is correct.
2. After EVERY factual claim, cite the source label it came from, in square brackets. Example: "Revenue was $215,938 million [S3]."
3. If a claim draws on more than one source, cite all of them: "... [S1][S4]".
4. If the CONTEXT does not contain enough information to answer, reply exactly: "I cannot answer this question based on the provided context." Do not guess, do not fill gaps, do not apologise at length.
5. Be concise. Do not restate the question. Do not add caveats that are not in the context.
6. Never cite a source label that does not appear in the CONTEXT."""


def format_context(chunks: list) -> tuple:
    """
    Recebe uma lista de dicts de chunk (com chunk_id, breadcrumb, text) e
    devolve (texto_do_contexto, mapa_label_para_chunk_id).

    O mapa é essencial: o LLM vai citar [S1], mas a verificação precisa de
    saber que [S1] == "nvidia_10k_fy2026__chunk0318".
    """
    blocks = []
    label_map = {}

    for i, chunk in enumerate(chunks, start=1):
        label = f"S{i}"
        label_map[label] = chunk["chunk_id"]

        breadcrumb = chunk.get("breadcrumb") or "(no section)"
        blocks.append(f"[{label}] (from: {breadcrumb})\n{chunk['text']}")

    return "\n\n---\n\n".join(blocks), label_map


def build_prompt(query: str, chunks: list) -> tuple:
    """Monta o prompt completo. Devolve (prompt, label_map)."""
    context, label_map = format_context(chunks)

    prompt = f"""{SYSTEM_INSTRUCTIONS}

CONTEXT:

{context}

---

QUESTION: {query}

ANSWER (remember: cite a source label after every claim):"""

    return prompt, label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="How much goodwill was recorded in the Groq deal?")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    # importa aqui (não no topo) para o script poder ser importado sem
    # carregar os modelos pesados do pipeline
    from rerank import load_all, hybrid_candidates, rerank

    engines = load_all()

    candidates = hybrid_candidates(engines, args.query)
    reranked = rerank(engines, args.query, candidates)
    top_ids = [cid for cid, _ in reranked[:args.top_k]]

    # reconstrói os dicts completos dos chunks escolhidos
    by_id = {c["chunk_id"]: c for c in engines["chunks"]}
    top_chunks = [by_id[cid] for cid in top_ids]

    prompt, label_map = build_prompt(args.query, top_chunks)

    print("=" * 70)
    print(prompt)
    print("=" * 70)
    print("\nMAPA DE LABELS (para a verificação de citações):")
    for label, chunk_id in label_map.items():
        print(f"  {label} -> {chunk_id}")


if __name__ == "__main__":
    main()