"""
EXPERIENCIA B: serializar tabelas em prosa para o retrieval as encontrar.

O PROBLEMA (provado nas seccoes 23-24): uma tabela como
    [TABELA]
    Year Ended
    Jan 25, 2026 | Jan 26, 2025 | Jan 28, 2024
    Revenue | $ | 215,938 | $ | 130,497 | $ | 60,922
tem a palavra "revenue" 1x. Os chunks de prosa que FALAM sobre receita tem-na
3-6x. O BM25 conta palavras -> a prosa ganha a tabela que CONTEM o numero. E o
embedding de uma grelha de numeros nao casa com uma pergunta em linguagem.

A SOLUCAO: reescrever cada linha da tabela em prosa que preserva os numeros
mas lhes da' linguagem. A linha acima vira:
    "Revenue was $215,938 for the year ended Jan 25, 2026; $130,497 for the
     year ended Jan 26, 2025; $60,922 for the year ended Jan 28, 2024."
Agora "revenue" aparece com contexto, os anos estao por extenso, e a pergunta
"what was total revenue in fiscal 2026" tem palavras para casar.

NAO MEXE NO PARSER. Le o chunks_nvidia.jsonl existente, reescreve so os 64
chunks com [TABELA], e grava um chunks_nvidia_serialized.jsonl novo. Assim
podem coexistir os dois e a comparacao e' directa (mesmo tudo, so muda a
serializacao das tabelas).

DEPOIS de correr isto:
  1. reconstruir os indices sobre o ficheiro novo (ver instrucoes no fim)
  2. reavaliar e comparar com o baseline

Uso:
  python experiment_b_serialize_tables.py
"""

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
CHUNKS_IN = BASE / "chunks" / "chunks_nvidia.jsonl"
CHUNKS_OUT = BASE / "chunks" / "chunks_nvidia_serialized.jsonl"


def looks_like_header(line: str) -> bool:
    """Uma linha de cabecalho tem datas/anos e poucos ou nenhuns numeros
    grandes. Heuristica: contem um ano (20xx) ou 'Year Ended'."""
    if "Year Ended" in line or "Fiscal" in line:
        return True
    if re.search(r"\b20\d{2}\b", line) and not re.search(r"\d{3,}", line.replace("20", "")):
        return True
    return False


def clean_cells(cells: list) -> list:
    """Remove celulas vazias e simbolos soltos ($, %) que a extracao deixou."""
    out = []
    for c in cells:
        c = c.strip()
        if c and c not in ("$", "%", "|"):
            out.append(c)
    return out


def serialize_table(table_text: str) -> str:
    """
    Converte um bloco [TABELA] em prosa. Objectivo duplo:
      1. preservar TODOS os numeros e rotulos (nada se perde);
      2. maximizar as PALAVRAS-CHAVE que o retrieval precisa — repetir o
         rotulo ("Revenue") e ancorar o periodo mais recente ("fiscal year
         2026" / "in the most recent period"), sem encher de repeticoes inuteis.

    Formato por linha:
        "Revenue was $215,938 in the period ending Jan 25, 2026
         (Revenue), compared to $130,497 and $60,922 in prior periods."
    O "(Revenue)" repetido de proposito da' peso ao BM25 no rotulo, que e' o
    termo que a pergunta usa.
    """
    lines = [l.strip() for l in table_text.split("\n") if l.strip()]
    lines = [l for l in lines if l != "[TABELA]"]

    periods = []
    data_lines = []
    for line in lines:
        if "|" in line and looks_like_header(line):
            periods = clean_cells(line.split("|"))
        elif "|" in line:
            data_lines.append(line)
        else:
            if looks_like_header(line):
                extra = clean_cells(re.split(r"\s{2,}|\t", line))
                periods.extend(p for p in extra if p not in periods)
            # linhas sem barras e sem cara de cabecalho sao subtitulos -> ignora

    most_recent = periods[0] if periods else None

    sentences = []
    for line in data_lines:
        cells = clean_cells(line.split("|"))
        if not cells:
            continue
        label = cells[0]
        values = cells[1:]
        if not values:
            continue

        # re-anexa o simbolo $ que a extracao separou (heuristica: valores que
        # parecem montantes financeiros, i.e. tem digitos e virgulas/pontos)
        def money(v):
            return f"${v}" if re.match(r"^\(?\d[\d,\.]*\)?$", v) else v

        vals = [money(v) for v in values]
        first = vals[0]
        rest = vals[1:]

        if most_recent:
            s = f"{label} was {first} in {most_recent} ({label})"
            if rest:
                s += f", compared to {', '.join(rest)} in prior periods"
            s += "."
        else:
            s = f"{label}: {', '.join(vals)} ({label})."
        sentences.append(s)

    if not sentences:
        return table_text

    header = "Financial data table."
    if periods:
        header = f"Financial data for periods {', '.join(periods)}."
    return header + " " + " ".join(sentences)


def process_chunk_text(text: str) -> str:
    """Encontra blocos [TABELA] no texto e substitui-os pela versao em prosa,
    mantendo o texto nao-tabular intacto."""
    if "[TABELA]" not in text:
        return text

    # parte o texto nos blocos [TABELA]. Um bloco vai do [TABELA] ate ao
    # proximo paragrafo em branco duplo ou fim.
    parts = re.split(r"(\[TABELA\].*?)(?=\n\n|\Z)", text, flags=re.DOTALL)
    rebuilt = []
    for p in parts:
        if p.startswith("[TABELA]"):
            rebuilt.append(serialize_table(p))
        else:
            rebuilt.append(p)
    return "".join(rebuilt)


def main():
    with open(CHUNKS_IN, encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f]

    n_tables = 0
    examples = []
    for c in chunks:
        if "[TABELA]" in c["text"]:
            before = c["text"]
            c["text"] = process_chunk_text(c["text"])
            n_tables += 1
            if len(examples) < 2:
                examples.append((c["chunk_id"], before, c["text"]))

    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Chunks processados: {len(chunks)}")
    print(f"Chunks com tabela serializados: {n_tables}")
    print(f"Gravado em: {CHUNKS_OUT}")

    for cid, before, after in examples:
        print(f"\n{'='*70}\n{cid} — ANTES:\n{'='*70}")
        print(before[:400])
        print(f"\n{cid} — DEPOIS:\n{'-'*70}")
        print(after[:400])

    print(f"\n{'='*70}")
    print("PROXIMOS PASSOS (correr a mao):")
    print("="*70)
    print("As experiencias precisam de indices SEPARADOS para nao destruir o")
    print("baseline. Sugestao: apontar os scripts de indice ao ficheiro novo.")
    print()
    print("  1. Reconstruir indices sobre chunks_nvidia_serialized.jsonl")
    print("     (temporariamente troca o caminho nos index_*.py, ou copia o")
    print("      ficheiro para chunks_nvidia.jsonl DEPOIS de guardar o original)")
    print("  2. python evaluate_generation.py --no-judge  (compara faithfulness_own)")
    print("  3. python experiment_a_text_vs_table.py  (ve se o retrieval de")
    print("     tabelas subiu dos 44%)")


if __name__ == "__main__":
    main()