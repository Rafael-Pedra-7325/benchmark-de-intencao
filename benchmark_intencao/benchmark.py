"""Execução do benchmark: treino, predição amostra-a-amostra, latência, tokens e custo."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import metricas
from .modelos import Classificador, DependenciaAusente


@dataclass
class Resultado:
    chave: str
    nome: str
    descricao: str
    status: str = "ok"          # ok | pulado | falhou
    motivo: str = ""
    detalhes: str = ""
    y_true: list = field(default_factory=list)
    y_pred: list = field(default_factory=list)
    grupos: list = field(default_factory=list)
    textos: list = field(default_factory=list)
    latencias_ms: list = field(default_factory=list)
    resumo: dict = field(default_factory=dict)


def executar_modelo(modelo: Classificador, treino, teste, preco_in=None, preco_out=None, verboso=True) -> Resultado:
    r = Resultado(modelo.chave, modelo.nome, modelo.descricao)
    try:
        modelo.preparar()
    except DependenciaAusente as e:
        r.status, r.motivo = "pulado", str(e)
        return r

    try:
        t0 = time.perf_counter()
        modelo.treinar(treino["texto"].tolist(), treino["intencao"].tolist(), treino["grupo"].tolist())
        tempo_treino = time.perf_counter() - t0
        r.detalhes = modelo.detalhes()

        textos = teste["texto"].tolist()
        if modelo.aquecer and textos:
            modelo.prever(textos[0])

        tok_in = tok_out = 0
        for i, texto in enumerate(textos, 1):
            t = time.perf_counter()
            p = modelo.prever(texto)
            r.latencias_ms.append((time.perf_counter() - t) * 1000)
            r.y_pred.append(p.rotulo)
            tok_in += p.tokens_entrada
            tok_out += p.tokens_saida
            if verboso and not modelo.local and i % 50 == 0:
                print(f"    [{modelo.chave}] {i}/{len(textos)}", flush=True)
    except Exception as e:  # noqa: BLE001 - um modelo falhando não derruba o benchmark inteiro
        r.status, r.motivo = "falhou", f"{type(e).__name__}: {e}"
        return r

    r.y_true = teste["intencao"].tolist()
    r.grupos = teste["grupo"].tolist()
    r.textos = textos
    custo_total = None if modelo.local else metricas.custo_usd(tok_in, tok_out, preco_in, preco_out)
    r.resumo = metricas.resumir(r.y_true, r.y_pred, r.grupos, r.latencias_ms, custo_total, tempo_treino, modelo.local)
    r.resumo["tokens_entrada"] = tok_in
    r.resumo["tokens_saida"] = tok_out
    return r
