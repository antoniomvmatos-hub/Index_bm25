"""
EXPERIENCIA C: CONTEXTUAL RETRIEVAL para tabelas.

O PROBLEMA (provado nas seccoes 23-25): as tabelas nao sao recuperadas. A
Experiencia B (serializar em prosa) falhou porque dar palavras a UMA tabela
da'-as a TODAS — 58 tabelas ficam parecidas e o problema real e'
DESAMBIGUACAO (distinguir a tabela/linha/coluna certa), nao vocabulario.

A SOLUCAO (metodo da Anthropic, adaptado ao ambiente local): antes de indexar,
prefixar cada chunk-tabela com um resumo curto que diz O QUE A TABELA E' e
ONDE ENCAIXA no documento. Isso da' a cada tabela uma IDENTIDADE que a
distingue das outras — ataca a desambiguacao, que a serializacao nao tocava.

  Serializacao (B): "Revenue was $215,938..."          <- palavras que TODAS tem
  Contextual (C):   "This is the consolidated income
                     statement from NVIDIA's FY2026 10-K.
                     Revenue was $215,938..."           <- identidade que SO esta tem

COMO O CONTEXTO E' GERADO (combinando 3 fontes, para reduzir alucinacao —
o modelo local nao ve o documento inteiro, so 2MB nao cabem):
  1. BREADCRUMB — ancora de VERDADE (extraida pelo parser, nao inventada)
  2. A TABELA inteira com os seus cabecalhos
  3. Os CHUNKS DA MESMA SECCAO — contexto local fiavel
O prompt instrui o modelo a ANCORAR-SE no breadcrumb e a nao inventar.

DECISOES (validadas com o utilizador):
  - So os 58 chunks-tabela sao contextualizados (nao os 396) — a hipotese e'
    sobre tabelas, e mantem a experiencia controlada e rapida (~15 min).
  - Modelo gerador do contexto: qwen2.5:7b (melhor a seguir instrucoes que o
    llama3b; corre uma vez na indexacao, nao e' tempo-critico).
  - Uma AMOSTRA dos contextos gerados e' impressa para inspecao HUMANA antes
    de confiar — rede de seguranca contra contexto errado.

Uso:
  python experiment_c_contextual.py            # gera e grava
  python experiment_c_contextual.py --sample 10  # so mostra 10 exemplos, nao grava
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generate import generate, check_server

BASE = Path(__file__).parent
CHUNKS_IN = BASE / "chunks" / "chunks_nvidia.jsonl"
CHUNKS_OUT = BASE / "chunks" / "chunks_nvidia_contextual.jsonl"

CONTEXT_MODEL = "qwen2.5:7b"

# O prompt que gera o contexto. Ancorado no breadcrumb (verdade), instruido a
# nao inventar. Pede 1-2 frases — curto de proposito: o objectivo e' dar
# IDENTIDADE ao chunk, nao reescrever a tabela.
CONTEXT_PROMPT = """You are labeling a table from a financial document (a SEC 10-K filing) so it can be found by search.

The table belongs to this section of the document:
{breadcrumb}

Nearby text from the same section (for context):
{neighbours}

The table itself:
{table}

Write ONE or TWO short sentences that state WHAT THIS TABLE IS and WHAT IT CONTAINS, so someone searching can identify it. Anchor your description in the section name above — do not invent facts not supported by the section name or the table. Start with "This table". Be specific about which financial statement or topic it represents.

Description:"""


def load_chunks():
    with open(CHUNKS_IN, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_field(chunk, *names, default=""):
    """Tolerante a variacoes de nome de campo entre versoes dos chunks."""
    for n in names:
        if n in chunk and chunk[n]:
            return chunk[n]
    return default


def find_section_neighbours(chunk, all_chunks, max_neighbours=4):
    """
    Devolve o texto dos chunks da MESMA seccao (mesmo breadcrumb), excluindo
    o proprio. Se nao houver breadcrumb, cai para os vizinhos por posicao.
    """
    bc = get_field(chunk, "breadcrumb", "section")
    cid = get_field(chunk, "chunk_id", "id")

    same_section = []
    if bc:
        for c in all_chunks:
            if get_field(c, "chunk_id", "id") == cid:
                continue
            if get_field(c, "breadcrumb", "section") == bc:
                same_section.append(get_field(c, "text"))

    # fallback: vizinhos por indice, se a seccao nao deu nada
    if not same_section:
        ids = [get_field(c, "chunk_id", "id") for c in all_chunks]
        try:
            i = ids.index(cid)
            for j in (i - 1, i + 1, i - 2, i + 2):
                if 0 <= j < len(all_chunks):
                    same_section.append(get_field(all_chunks[j], "text"))
        except ValueError:
            pass

    # limita o tamanho para caber na janela e nao diluir
    text = "\n---\n".join(same_section[:max_neighbours])
    return text[:2000] if text else "(no additional section context available)"


def extract_table(text):
    """Isola o bloco [TABELA] do resto do texto do chunk."""
    idx = text.find("[TABELA]")
    if idx == -1:
        return text[:1500]
    return text[idx:idx + 1500]


def generate_context(chunk, all_chunks):
    breadcrumb = get_field(chunk, "breadcrumb", "section", default="(unknown section)")
    neighbours = find_section_neighbours(chunk, all_chunks)
    table = extract_table(get_field(chunk, "text"))

    prompt = CONTEXT_PROMPT.format(
        breadcrumb=breadcrumb, neighbours=neighbours, table=table
    )
    context = generate(prompt, model=CONTEXT_MODEL).strip()
    # limpeza: as vezes o modelo repete "Description:" ou poe aspas
    context = context.replace("Description:", "").strip().strip('"').strip()
    return context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="so gera e mostra N exemplos, sem gravar (inspecao)")
    args = parser.parse_args()

    if not check_server():
        return

    all_chunks = load_chunks()
    tables = [c for c in all_chunks if "[TABELA]" in get_field(c, "text")]
    print(f"Total de chunks: {len(all_chunks)}")
    print(f"Chunks com tabela: {len(tables)}")
    print(f"Modelo gerador de contexto: {CONTEXT_MODEL}\n")

    if args.sample:
        tables = tables[:args.sample]
        print(f"=== MODO INSPECAO: {len(tables)} exemplos (NAO grava) ===\n")

    contextualized = []
    for i, chunk in enumerate(all_chunks, start=1):
        if "[TABELA]" not in get_field(chunk, "text"):
            contextualized.append(chunk)
            continue
        if args.sample and chunk not in tables:
            contextualized.append(chunk)
            continue

        context = generate_context(chunk, all_chunks)
        cid = get_field(chunk, "chunk_id", "id")

        print(f"[{cid}]")
        print(f"  breadcrumb: {get_field(chunk, 'breadcrumb', 'section')[:70]}")
        print(f"  -> contexto gerado: {context[:160]}")
        print()

        if not args.sample:
            new_chunk = dict(chunk)
            new_chunk["text"] = context + "\n\n" + chunk["text"]
            new_chunk["generated_context"] = context  # guarda para auditoria
            contextualized.append(new_chunk)

    if args.sample:
        print("=== INSPECAO terminada. Se os contextos fazem sentido, corre sem --sample. ===")
        return

    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        for c in contextualized:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Gravado: {CHUNKS_OUT}")
    print("\nPROXIMOS PASSOS:")
    print("  1. Guardar baseline (se ainda nao guardado):")
    print("     Copy-Item chunks/chunks_nvidia.jsonl chunks/chunks_nvidia_baseline.jsonl")
    print("  2. Activar o contextual:")
    print("     Copy-Item chunks/chunks_nvidia_contextual.jsonl chunks/chunks_nvidia.jsonl")
    print("  3. Reconstruir indices: python index_bm25.py ; python index_embeddings.py")
    print("  4. Reavaliar: python evaluate_retrieval_v2.py")
    print("  5. Comparar MRR com baseline (0.549) e serializado (0.547)")


if __name__ == "__main__":
    main()