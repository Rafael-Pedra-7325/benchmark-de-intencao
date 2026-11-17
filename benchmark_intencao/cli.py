"""Linha de comando: python -m benchmark_intencao [opções]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import dados as mod_dados
from .benchmark import executar_modelo
from .config import DADOS_PADRAO, MODELO_EMBEDDINGS, MODELOS_DISPONIVEIS, RAIZ, SAIDA_PADRAO, SEMENTE
from .modelos import criar
from .relatorio import gerar_relatorio


def montar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmark_intencao",
        description="Compara TF-IDF+LR, embeddings e LLM zero-shot em acurácia, latência e custo.",
    )
    p.add_argument("--modelos", nargs="+", default=list(MODELOS_DISPONIVEIS), choices=MODELOS_DISPONIVEIS,
                   help="modelos a executar (padrão: todos; os indisponíveis são pulados com aviso)")
    p.add_argument("--dados", default=str(DADOS_PADRAO), help="CSV com colunas texto,intencao[,grupo,particao]")
    p.add_argument("--saida", default=str(SAIDA_PADRAO), help="pasta dos resultados")
    p.add_argument("--max-teste", type=int, default=0,
                   help="limita o nº de linhas de teste (amostra estratificada); útil para reduzir o custo do LLM")
    p.add_argument("--llm-modelo", default=None, help="modelo do LLM (padrão: env LLM_MODELO ou config.py)")
    p.add_argument("--modelo-embeddings", default=MODELO_EMBEDDINGS)
    p.add_argument("--preco-entrada", type=float, default=None, help="USD por milhão de tokens de entrada")
    p.add_argument("--preco-saida", type=float, default=None, help="USD por milhão de tokens de saída")
    p.add_argument("--semente", type=int, default=SEMENTE)
    p.add_argument("--estrito", action="store_true", help="falha (exit 1) se algum modelo pedido não rodar")
    return p


def main(argv=None) -> int:
    a = montar_parser().parse_args(argv)
    caminho = Path(a.dados)
    if not caminho.exists():
        print(f"{caminho} não existe; gerando o dataset sintético padrão...")
        caminho.parent.mkdir(parents=True, exist_ok=True)
        mod_dados.gerar_dataset(a.semente).to_csv(caminho, index=False)

    df = mod_dados.carregar(caminho, a.semente)
    treino = df[df["particao"] == "treino"].reset_index(drop=True)
    teste = df[df["particao"] == "teste"].reset_index(drop=True)
    if treino.empty or teste.empty:
        print("erro: partição de treino ou teste vazia", file=sys.stderr)
        return 2
    teste = mod_dados.amostrar_teste(teste, a.max_teste, a.semente).reset_index(drop=True)
    rotulos = sorted(df["intencao"].unique())
    print(f"treino={len(treino)} teste={len(teste)} intenções={len(rotulos)}")

    resultados = []
    for chave in a.modelos:
        print(f"-> {chave}", flush=True)
        modelo = criar(chave, a.llm_modelo, a.modelo_embeddings)
        r = executar_modelo(modelo, treino, teste, a.preco_entrada, a.preco_saida)
        if r.status == "ok":
            s = r.resumo
            print(f"   acurácia={s['acuracia']:.3f} f1={s['f1_macro']:.3f} p50={s['latencia_p50_ms']:.2f}ms")
        else:
            print(f"   {r.status.upper()}: {r.motivo}")
        resultados.append(r)

    try:
        fonte = str(caminho.resolve().relative_to(RAIZ))
    except ValueError:
        fonte = str(caminho)
    caminho_rel = gerar_relatorio(resultados, treino, teste, rotulos, Path(a.saida), {"dados": fonte})
    print(f"relatório: {caminho_rel}")
    if a.estrito and any(r.status != "ok" for r in resultados):
        return 1
    if not any(r.status == "ok" for r in resultados):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
