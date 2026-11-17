"""Métricas: acurácia, F1, latência, custo, IC por bootstrap de clusters e confusões."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def intervalo_bootstrap(acertos, grupos, n_reamostras: int = 2000, semente: int = 42, nivel: float = 0.95):
    """IC da acurácia reamostrando GRUPOS (frases-base), não linhas.

    As variações da mesma frase são quase duplicatas; reamostrar linhas daria um IC estreito demais.
    """
    acertos = np.asarray(acertos, dtype=float)
    _, inv = np.unique(np.asarray(grupos).astype(str), return_inverse=True)
    soma = np.bincount(inv, weights=acertos)
    tam = np.bincount(inv).astype(float)
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(soma), size=(n_reamostras, len(soma)))
    accs = soma[idx].sum(axis=1) / tam[idx].sum(axis=1)
    a = (1 - nivel) / 2 * 100
    lo, hi = np.percentile(accs, [a, 100 - a])
    return float(lo), float(hi)


def maiores_confusoes(y_true, y_pred, k: int = 5):
    c = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [(t, p, n) for (t, p), n in c.most_common(k)]


def custo_usd(tokens_in: int, tokens_out: int, preco_in, preco_out):
    """Preços em USD por milhão de tokens. Retorna None se algum preço não foi informado."""
    if preco_in is None or preco_out is None:
        return None
    return tokens_in / 1e6 * preco_in + tokens_out / 1e6 * preco_out


def resumir(y_true, y_pred, grupos, latencias_ms, custo_total, tempo_treino_s, local: bool):
    y_true, y_pred = list(y_true), list(y_pred)
    acertos = [t == p for t, p in zip(y_true, y_pred)]
    lo, hi = intervalo_bootstrap(acertos, grupos)
    lat = np.asarray(latencias_ms, dtype=float)
    n = len(y_true)
    if local:
        custo_1k = 0.0
    else:
        custo_1k = None if custo_total is None else custo_total / n * 1000
    return {
        "n_teste": n,
        "acuracia": accuracy_score(y_true, y_pred),
        "acuracia_ic_inf": lo,
        "acuracia_ic_sup": hi,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "latencia_p50_ms": float(np.percentile(lat, 50)),
        "latencia_p95_ms": float(np.percentile(lat, 95)),
        "custo_por_1k_usd": custo_1k,
        "tempo_treino_s": tempo_treino_s,
        "invalidas": sum(1 for p in y_pred if p == "invalido"),
    }
