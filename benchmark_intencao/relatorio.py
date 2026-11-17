"""Geração dos artefatos: metricas.csv, predicoes.csv, relatorio.md e gráficos (PNG)."""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn

from .metricas import maiores_confusoes


def _fmt_custo(v):
    if v is None:
        return "n/d"
    return "0 (local)" if v == 0 else f"{v:.4f}"


def salvar_tabelas(resultados, saida: Path):
    ok = [r for r in resultados if r.status == "ok"]
    linhas = []
    for r in ok:
        linhas.append({"modelo": r.nome, "chave": r.chave, **r.resumo})
    pd.DataFrame(linhas).to_csv(saida / "metricas.csv", index=False)
    pred = [
        pd.DataFrame({"modelo": r.chave, "texto": r.textos, "grupo": r.grupos,
                      "real": r.y_true, "previsto": r.y_pred, "latencia_ms": r.latencias_ms})
        for r in ok
    ]
    if pred:
        pd.concat(pred).to_csv(saida / "predicoes.csv", index=False)


def _graficos(resultados, rotulos, saida: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return []
    ok = [r for r in resultados if r.status == "ok"]
    arquivos = []
    if ok:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        nomes = [r.chave for r in ok]
        acc = np.array([r.resumo["acuracia"] for r in ok])
        err = np.array([[a - r.resumo["acuracia_ic_inf"], r.resumo["acuracia_ic_sup"] - a] for a, r in zip(acc, ok)]).T
        a1.bar(nomes, acc, yerr=err, capsize=4, color="#4C78A8")
        a1.set_ylim(0, 1.05)
        a1.set_title("Acurácia (IC 95% por grupo)")
        a2.bar(nomes, [r.resumo["latencia_p50_ms"] for r in ok], color="#F58518")
        a2.set_yscale("log")
        a2.set_title("Latência p50 por predição (ms, escala log)")
        fig.tight_layout()
        fig.savefig(saida / "comparacao.png", dpi=130)
        plt.close(fig)
        arquivos.append("comparacao.png")
    for r in ok:
        cm = confusion_matrix(r.y_true, r.y_pred, labels=rotulos).astype(float)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(rotulos)))
        ax.set_yticks(range(len(rotulos)))
        ax.set_xticklabels(rotulos, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(rotulos, fontsize=8)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        ax.set_title(f"Matriz de confusão (normalizada por linha) — {r.chave}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        nome = f"matriz_confusao_{r.chave}.png"
        fig.savefig(saida / nome, dpi=120)
        plt.close(fig)
        arquivos.append(nome)
    return arquivos


def gerar_relatorio(resultados, treino, teste, rotulos, saida: Path, opcoes: dict):
    saida.mkdir(parents=True, exist_ok=True)
    salvar_tabelas(resultados, saida)
    imagens = _graficos(resultados, rotulos, saida)
    ok = [r for r in resultados if r.status == "ok"]

    L = []
    L.append("# Relatório do benchmark de intenções\n")
    L.append(f"Gerado em {datetime.now():%Y-%m-%d %H:%M} · Python {platform.python_version()} · "
             f"scikit-learn {sklearn.__version__} · {platform.system()} {platform.machine()}\n")
    L.append("## Dados\n")
    L.append(f"- Treino: **{len(treino)}** linhas ({treino['grupo'].nunique()} frases-base)")
    L.append(f"- Teste: **{len(teste)}** linhas ({teste['grupo'].nunique()} frases-base, nunca vistas no treino)")
    L.append(f"- Intenções: **{len(rotulos)}**")
    L.append(f"- Fonte: `{opcoes.get('dados')}`")
    L.append("- ⚠️ Com o dataset sintético padrão, o número efetivo de exemplos de teste independentes é o de "
             "frases-base (não o de linhas). Por isso os intervalos de confiança são calculados por grupo "
             "e tendem a ser largos.\n")

    L.append("## Resultados\n")
    if ok:
        L.append("| Modelo | Acurácia (IC 95%) | F1-macro | Latência p50 (ms) | p95 (ms) | Custo / 1k pred. (USD) | Treino (s) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in ok:
            s = r.resumo
            L.append(f"| {r.nome} | {s['acuracia']:.3f} ({s['acuracia_ic_inf']:.3f}–{s['acuracia_ic_sup']:.3f}) "
                     f"| {s['f1_macro']:.3f} | {s['latencia_p50_ms']:.2f} | {s['latencia_p95_ms']:.2f} "
                     f"| {_fmt_custo(s['custo_por_1k_usd'])} | {s['tempo_treino_s']:.2f} |")
        L.append("")
        if "comparacao.png" in imagens:
            L.append("![comparação](comparacao.png)\n")
    else:
        L.append("Nenhum modelo foi executado com sucesso.\n")

    L.append("## Notas de leitura\n")
    L.append("- **Latência**: tempo de UMA predição isolada (batch=1), na máquina que rodou o benchmark. "
             "Para o LLM inclui rede e fila da API; para modelos locais, só inferência.")
    L.append("- **Custo**: modelos locais mostram 0 porque não têm custo por chamada — custo de infraestrutura "
             "e de treino não está incluído. Para o LLM, o custo vem dos tokens reais reportados pela API "
             "× os preços informados em `--preco-entrada/--preco-saida` (USD por milhão de tokens); "
             "sem os preços, aparece `n/d`.")
    L.append("- **Zero-shot** não usa o conjunto de treino. Respostas fora da lista de intenções contam como erro "
             "(`invalido`).\n")

    L.append("## Configurações e erros mais comuns\n")
    for r in ok:
        L.append(f"### {r.nome}\n")
        L.append(f"{r.descricao}\n")
        if r.detalhes:
            L.append(f"- {r.detalhes}")
        if r.resumo.get("invalidas"):
            L.append(f"- respostas inválidas: {r.resumo['invalidas']}")
        if r.chave == "llm":
            L.append(f"- tokens: {r.resumo['tokens_entrada']} entrada / {r.resumo['tokens_saida']} saída")
        conf = maiores_confusoes(r.y_true, r.y_pred)
        if conf:
            L.append("- confusões mais frequentes (real → previsto):")
            for t, p, n in conf:
                L.append(f"  - `{t}` → `{p}`: {n}")
        else:
            L.append("- nenhuma confusão no conjunto de teste")
        L.append("")

    nao_ok = [r for r in resultados if r.status != "ok"]
    if nao_ok:
        L.append("## Modelos não executados\n")
        for r in nao_ok:
            L.append(f"- **{r.chave}** ({r.status}): {r.motivo}")
        L.append("")

    (saida / "relatorio.md").write_text("\n".join(L), encoding="utf-8")
    return saida / "relatorio.md"
