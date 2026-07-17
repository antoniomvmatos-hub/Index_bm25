"""
PASSO 8: O GERADOR — a peça que finalmente escreve a resposta.

Até agora o pipeline terminava num prompt com a linha "ANSWER:" vazia. Esta
peça preenche-a: recebe o prompt do build_prompt.py, entrega-o a um LLM, e
devolve o texto gerado.

DESENHO AGNÓSTICO (decisão registada no PROJECT_LOG 16.4):
  A função generate(prompt) é uma "ficha" onde encaixa qualquer LLM. Hoje
  está ligada ao Ollama (local, grátis); trocar para uma API seria escrever
  outra função com a mesma assinatura. O resto do pipeline (verificação,
  avaliação) não sabe nem quer saber qual está a ser usada.

DECISÃO IMPORTANTE — temperature=0:
  A "temperatura" controla a aleatoriedade do LLM. Com temperature>0, a mesma
  pergunta dá respostas diferentes de cada vez. Isso é bom para escrita
  criativa e PÉSSIMO para nós, por duas razões:
    1. Avaliação: se as respostas mudam a cada corrida, os números da
       avaliação não são reprodutíveis nem comparáveis.
    2. Factualidade: queremos que o modelo copie os números do contexto, não
       que "explore alternativas criativas" ao escrever $14.4 billion.
  Com temperature=0 o modelo escolhe sempre o token mais provável. Não é
  100% determinístico (há detalhes de hardware), mas é o mais perto que se
  consegue.

O servidor do Ollama corre em segundo plano no localhost:11434 — não é
preciso ter o comando 'ollama' no PATH.

Uso:
  python generate.py --query "How much goodwill was recorded in the Groq deal?"
  python generate.py --query "..." --model qwen2.5:7b
"""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ollama

DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"


def generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """
    A INTERFACE AGNÓSTICA. Recebe um prompt, devolve texto.

    Para trocar de gerador, escreve-se outra função com esta assinatura
    (ex: generate_openai, generate_anthropic) e troca-se a chamada. Nada
    mais no pipeline muda.
    """
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature},
    )
    return response["response"].strip()


def check_server() -> bool:
    """Confirma que o servidor do Ollama está a responder, com mensagem útil
    em vez de um traceback críptico."""
    try:
        models = ollama.list()
        names = [m.get("model") or m.get("name") for m in models.get("models", [])]
        print(f"Ollama a responder. Modelos disponíveis: {names}")
        return True
    except Exception as e:
        print(f"ERRO: não consigo falar com o Ollama em {OLLAMA_HOST}")
        print(f"       {type(e).__name__}: {e}")
        print("       Confirma que o servidor está a correr:")
        print("       Invoke-RestMethod http://localhost:11434/api/tags")
        return False


def answer_question(query: str, model: str = DEFAULT_MODEL, top_k: int = 3) -> dict:
    """
    O PIPELINE COMPLETO, ponta a ponta, pela primeira vez:
      retrieval (hybrid + rerank) -> build_prompt -> gerador -> resposta

    Devolve tudo o que as camadas de verificação vão precisar a seguir:
    a resposta, o label_map (para traduzir [S1] -> chunk_id), e a query.
    """
    # importados aqui e não no topo: assim o generate() pode ser usado
    # isoladamente (ex: como LLM-as-judge) sem carregar os modelos pesados
    # do retrieval.
    from rerank import load_all, hybrid_candidates, rerank
    from build_prompt import build_prompt

    engines = load_all()

    candidates = hybrid_candidates(engines, query)
    reranked = rerank(engines, query, candidates)
    top_ids = [cid for cid, _ in reranked[:top_k]]

    by_id = {c["chunk_id"]: c for c in engines["chunks"]}
    top_chunks = [by_id[cid] for cid in top_ids]

    prompt, label_map = build_prompt(query, top_chunks)

    print(f"A gerar com {model} (pode demorar)...\n")
    answer = generate(prompt, model=model)

    return {
        "query": query,
        "answer": answer,
        "label_map": label_map,
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str,
                        default="How much goodwill was recorded in the Groq deal?")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--show-prompt", action="store_true",
                        help="mostra também o prompt enviado ao modelo")
    args = parser.parse_args()

    if not check_server():
        return

    result = answer_question(args.query, model=args.model, top_k=args.top_k)

    if args.show_prompt:
        print("=" * 70)
        print(result["prompt"])
        print("=" * 70 + "\n")

    print("=" * 70)
    print(f"PERGUNTA: {result['query']}")
    print("-" * 70)
    print(f"RESPOSTA:\n{result['answer']}")
    print("=" * 70)
    print("\nMAPA DE LABELS (para a verificação):")
    for label, chunk_id in result["label_map"].items():
        print(f"  {label} -> {chunk_id}")


if __name__ == "__main__":
    main()