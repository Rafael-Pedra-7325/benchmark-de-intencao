"""Geração, carregamento e split dos dados.

O dataset padrão é SINTÉTICO: frases-semente escritas à mão (catalogo.py) + variações
(saudações, contexto, abreviações, typos, acentos). O split é feito por GRUPO (frase-semente),
então nenhuma frase-base aparece em treino e teste ao mesmo tempo (evita vazamento).

Para usar dados reais, passe um CSV com colunas `texto` e `intencao` (e, opcionalmente,
`grupo` e `particao` com valores "treino"/"teste").
"""
from __future__ import annotations

import argparse
import random
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

from .catalogo import INTENCOES
from .config import DADOS_PADRAO, SEMENTE

ABERTURAS = ["", "", "", "oi", "olá", "bom dia", "boa noite", "oi pessoal", "e aí", "ei", "por favor,", "olá, tudo bem?"]
CONTEXTOS = ["", "", "", " #{n}", " (pedido {n})", " de hoje", " no app", " do {rest}"]
FECHAMENTOS = ["", "", "", "", " por favor", " urgente", " obrigado", " :(", "!!", "?", " preciso de ajuda", " alguém aí?"]
RESTAURANTES = ["Burger Ponto", "Pizzaria Bella", "Sushi Yama", "Açaí do Zé", "Cantina da Vó"]
ABREVIACOES = {
    "você": "vc", "porque": "pq", "por favor": "pfv", "também": "tb", "para": "pra",
    "está": "tá", "estou": "tô", "comigo": "cmg", "quero": "qro", "não": "n", "pedido": "ped",
}


def _sem_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def _typo(texto: str, rng: random.Random) -> str:
    palavras = texto.split(" ")
    candidatas = [i for i, p in enumerate(palavras) if len(p) >= 5 and p.isalpha()]
    if not candidatas:
        return texto
    i = rng.choice(candidatas)
    p = palavras[i]
    j = rng.randrange(1, len(p) - 1)
    op = rng.choice(["troca", "apaga", "duplica"])
    if op == "troca":
        p = p[:j] + p[j + 1] + p[j] + p[j + 2:]
    elif op == "apaga":
        p = p[:j] + p[j + 1:]
    else:
        p = p[:j] + p[j] + p[j:]
    palavras[i] = p
    return " ".join(palavras)


def _abreviar(texto: str, rng: random.Random) -> str:
    out = []
    for palavra in texto.split(" "):
        out.append(ABREVIACOES[palavra] if palavra in ABREVIACOES and rng.random() < 0.3 else palavra)
    return " ".join(out)


def _variacao(frase: str, rng: random.Random, limpa: bool = False) -> str:
    if limpa:
        return frase
    texto = frase
    if rng.random() < 0.5:
        texto = _abreviar(texto, rng)
    if rng.random() < 0.2:
        texto = _typo(texto, rng)
    ctx = rng.choice(CONTEXTOS).format(n=rng.randint(1000, 99999), rest=rng.choice(RESTAURANTES))
    texto = f"{rng.choice(ABERTURAS)} {texto}{ctx}{rng.choice(FECHAMENTOS)}".strip()
    if rng.random() < 0.3:
        texto = _sem_acentos(texto)
    r = rng.random()
    if r < 0.4:
        texto = texto[:1].upper() + texto[1:]
    elif r < 0.45:
        texto = texto.upper()
    return " ".join(texto.split())


def gerar_dataset(semente: int = SEMENTE, variantes: int = 10, grupos_teste: int = 4) -> pd.DataFrame:
    """Gera o dataset sintético com colunas: texto, intencao, grupo, particao."""
    rng = random.Random(semente)
    linhas = []
    for intencao, info in INTENCOES.items():
        indices = list(range(len(info["frases"])))
        random.Random(f"{semente}-{intencao}").shuffle(indices)
        teste = set(indices[:grupos_teste])
        for i, frase in enumerate(info["frases"]):
            vistos: set[str] = set()
            tentativas = 0
            while len(vistos) < variantes and tentativas < variantes * 30:
                v = _variacao(frase, rng, limpa=(tentativas == 0))
                tentativas += 1
                if v not in vistos:
                    vistos.add(v)
                    linhas.append({
                        "texto": v,
                        "intencao": intencao,
                        "grupo": f"{intencao}-{i:02d}",
                        "particao": "teste" if i in teste else "treino",
                    })
    return pd.DataFrame(linhas)


def carregar(caminho: Path | str = DADOS_PADRAO, semente: int = SEMENTE) -> pd.DataFrame:
    """Carrega um CSV e garante as colunas `grupo` e `particao`."""
    df = pd.read_csv(caminho)
    faltando = {"texto", "intencao"} - set(df.columns)
    if faltando:
        raise ValueError(f"CSV precisa das colunas 'texto' e 'intencao' (faltando: {sorted(faltando)})")
    df = df.dropna(subset=["texto", "intencao"]).reset_index(drop=True)
    df["texto"] = df["texto"].astype(str)
    df["intencao"] = df["intencao"].astype(str)

    if "particao" not in df.columns:
        rng = random.Random(semente)
        if "grupo" in df.columns:
            alvo = {}
            for _, g in df.groupby("intencao"):
                ids = sorted(g["grupo"].astype(str).unique())
                rng.shuffle(ids)
                n_teste = max(1, round(len(ids) * 0.3))
                alvo.update({gid: "teste" for gid in ids[:n_teste]})
            df["particao"] = df["grupo"].astype(str).map(alvo).fillna("treino")
        else:
            warnings.warn(
                "CSV sem 'grupo' nem 'particao': split aleatório estratificado. Se houver frases "
                "quase idênticas, o teste pode estar vazado e as métricas ficam otimistas.",
                stacklevel=2,
            )
            df["grupo"] = [f"linha-{i}" for i in range(len(df))]
            df["particao"] = "treino"
            for _, g in df.groupby("intencao"):
                idx = list(g.index)
                rng.shuffle(idx)
                df.loc[idx[: max(1, round(len(idx) * 0.3))], "particao"] = "teste"
    elif "grupo" not in df.columns:
        df["grupo"] = [f"linha-{i}" for i in range(len(df))]
    return df


def amostrar_teste(teste: pd.DataFrame, n: int, semente: int = SEMENTE) -> pd.DataFrame:
    """Amostra estratificada por intenção (n linhas no total, aproximadamente)."""
    if n <= 0 or n >= len(teste):
        return teste
    por_classe = max(1, n // teste["intencao"].nunique())
    partes = [g.sample(min(len(g), por_classe), random_state=semente) for _, g in teste.groupby("intencao")]
    return pd.concat(partes).sort_index()


def main() -> None:
    p = argparse.ArgumentParser(description="Gera o dataset sintético de intenções.")
    p.add_argument("--saida", default=str(DADOS_PADRAO))
    p.add_argument("--semente", type=int, default=SEMENTE)
    p.add_argument("--variantes", type=int, default=10, help="variações por frase-semente")
    a = p.parse_args()
    df = gerar_dataset(a.semente, a.variantes)
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.saida, index=False)
    print(f"{len(df)} linhas | {df['intencao'].nunique()} intenções | "
          f"treino={int((df.particao == 'treino').sum())} teste={int((df.particao == 'teste').sum())} -> {a.saida}")


if __name__ == "__main__":
    main()
