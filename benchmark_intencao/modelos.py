"""Os quatro classificadores comparados.

Interface comum: `preparar()` (verifica dependências), `treinar(textos, rotulos, grupos)` e
`prever(texto) -> Predicao` (uma amostra por vez, para medir latência de forma comparável).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .catalogo import descricoes
from .config import MODELO_EMBEDDINGS, MODELO_LLM, SEMENTE


class DependenciaAusente(RuntimeError):
    """O modelo não pode rodar neste ambiente (pacote, modelo baixável ou chave de API ausentes)."""


@dataclass
class Predicao:
    rotulo: str
    tokens_entrada: int = 0
    tokens_saida: int = 0


class Classificador:
    chave = ""
    nome = ""
    descricao = ""
    aquecer = True  # fazer uma predição descartável antes de medir latência
    local = True    # False = chamada de API externa

    def preparar(self) -> None:
        """Levanta DependenciaAusente se o modelo não puder rodar."""

    def treinar(self, textos, rotulos, grupos=None) -> None:
        raise NotImplementedError

    def prever(self, texto: str) -> Predicao:
        raise NotImplementedError

    def detalhes(self) -> str:
        return ""


def _validacao_cruzada(grupos, rotulos):
    """GroupKFold quando há grupos suficientes (evita variações da mesma frase em folds distintos)."""
    if grupos is not None and len(set(grupos)) >= 3:
        return GroupKFold(n_splits=3), {"groups": list(grupos)}
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMENTE), {}


class TfidfRegLog(Classificador):
    chave = "tfidf"
    nome = "TF-IDF + Regressão Logística"
    descricao = "Baseline clássico: n-gramas de palavra ou caractere + LR, com busca de hiperparâmetros (GroupKFold)."

    def treinar(self, textos, rotulos, grupos=None):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(strip_accents="unicode", sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ])
        grade = [
            {"tfidf__analyzer": ["word"], "tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [1, 10, 100]},
            {"tfidf__analyzer": ["char_wb"], "tfidf__ngram_range": [(2, 4), (3, 5)], "clf__C": [1, 10, 100]},
        ]
        cv, extra = _validacao_cruzada(grupos, rotulos)
        busca = GridSearchCV(pipe, grade, cv=cv, scoring="f1_macro", n_jobs=1)
        busca.fit(list(textos), list(rotulos), **extra)
        self.pipe = busca.best_estimator_
        self.melhores = busca.best_params_
        self.cv_f1 = float(busca.best_score_)

    def prever(self, texto):
        return Predicao(str(self.pipe.predict([texto])[0]))

    def detalhes(self):
        p = self.melhores
        return (f"melhor config: analyzer={p['tfidf__analyzer']}, ngram={p['tfidf__ngram_range']}, "
                f"C={p['clf__C']} (F1-macro CV por grupo = {self.cv_f1:.3f})")


class LsaRegLog(Classificador):
    chave = "lsa"
    nome = "LSA (TF-IDF + SVD) + Regressão Logística"
    descricao = ("Representação densa local (Análise Semântica Latente), sem baixar modelo. "
                 "Serve de 'embedding' offline quando sentence-transformers não está disponível.")

    def treinar(self, textos, rotulos, grupos=None):
        tfidf = TfidfVectorizer(strip_accents="unicode", sublinear_tf=True, ngram_range=(1, 2))
        X = tfidf.fit_transform(list(textos))
        n = max(2, min(100, X.shape[1] - 1))
        self.pipe = Pipeline([
            ("tfidf", tfidf),
            ("svd", TruncatedSVD(n_components=n, random_state=SEMENTE)),
            ("norm", Normalizer(copy=False)),
            ("clf", LogisticRegression(C=10, max_iter=2000)),
        ])
        self.pipe.fit(list(textos), list(rotulos))
        self.n_componentes = n

    def prever(self, texto):
        return Predicao(str(self.pipe.predict([texto])[0]))

    def detalhes(self):
        return f"{self.n_componentes} componentes SVD, C=10"


class EmbeddingsRegLog(Classificador):
    chave = "embeddings"
    nome = "Embeddings (sentence-transformers) + Regressão Logística"
    descricao = "Encoder multilíngue pré-treinado congelado + LR treinada sobre os vetores."

    def __init__(self, modelo: str = MODELO_EMBEDDINGS):
        self.modelo = modelo
        self.encoder = None

    def preparar(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise DependenciaAusente("sentence-transformers não instalado (pip install -e '.[embeddings]')") from e
        try:
            self.encoder = SentenceTransformer(self.modelo)
        except Exception as e:  # noqa: BLE001 - download/IO pode falhar de várias formas
            raise DependenciaAusente(f"não foi possível carregar '{self.modelo}' (sem internet?): {e}") from e

    def _emb(self, textos):
        return self.encoder.encode(list(textos), normalize_embeddings=True, show_progress_bar=False)

    def treinar(self, textos, rotulos, grupos=None):
        X = self._emb(textos)
        cv, extra = _validacao_cruzada(grupos, rotulos)
        busca = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [1, 10, 100]}, cv=cv,
                             scoring="f1_macro", n_jobs=1)
        busca.fit(X, list(rotulos), **extra)
        self.clf = busca.best_estimator_
        self.melhor_c = busca.best_params_["C"]

    def prever(self, texto):
        return Predicao(str(self.clf.predict(self._emb([texto]))[0]))

    def detalhes(self):
        return f"encoder={self.modelo}, C={self.melhor_c}"


def extrair_rotulo(resposta: str, rotulos) -> str:
    """Converte a resposta livre do LLM em um dos rótulos válidos, ou 'invalido'."""
    t = (resposta or "").strip().lower().strip("`'\".,;:!*\n ")
    if t in rotulos:
        return t
    achados = [(t.find(r), -len(r), r) for r in rotulos if r in t]
    if achados:
        return sorted(achados)[0][2]
    return "invalido"


class LLMZeroShot(Classificador):
    chave = "llm"
    nome = "LLM zero-shot"
    descricao = "Sem treino: o modelo recebe a lista de intenções e classifica cada mensagem via API (Groq)."
    aquecer = False
    local = False

    def __init__(self, modelo: str | None = None, client=None, tentativas: int = 5):
        self.modelo = modelo or os.environ.get("LLM_MODELO") or MODELO_LLM
        self.client = client
        self.tentativas = tentativas
        self.nome = f"LLM zero-shot ({self.modelo})"

    def preparar(self):
        if self.client is not None:
            return
        try:
            from groq import Groq
        except ImportError as e:
            raise DependenciaAusente("pacote 'groq' não instalado (pip install -e '.[llm]')") from e
        if not os.environ.get("GROQ_API_KEY"):
            raise DependenciaAusente("variável de ambiente GROQ_API_KEY não definida")
        self.client = Groq()

    def treinar(self, textos, rotulos, grupos=None):
        # Zero-shot não aprende nada: só captura o conjunto de rótulos para montar o prompt.
        self.rotulos = sorted(set(rotulos))
        desc = descricoes()
        linhas = [f"- {r}: {desc[r]}" if r in desc else f"- {r}" for r in self.rotulos]
        self.system = (
            "Você é um classificador de intenções de atendimento ao cliente de um aplicativo de "
            "delivery de comida. Classifique a mensagem do cliente em exatamente UMA das intenções abaixo.\n\n"
            "Intenções:\n" + "\n".join(linhas) + "\n\n"
            "Responda SOMENTE com o identificador da intenção (por exemplo: " + self.rotulos[0] + "), "
            "sem explicações, sem pontuação e sem aspas."
        )

    def prever(self, texto):
        extras = {"reasoning_effort": "low"} if "gpt-oss" in self.modelo else {}
        ultimo_erro = None
        for tentativa in range(self.tentativas):
            try:
                resp = self.client.chat.completions.create(
                    model=self.modelo,
                    max_tokens=256,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self.system},
                        {"role": "user", "content": texto},
                    ],
                    **extras,
                )
                break
            except Exception as e:  # noqa: BLE001 - erros de rede/limite de taxa: tenta de novo
                ultimo_erro = e
                time.sleep(2 ** tentativa)
        else:
            raise RuntimeError(f"falha na API após {self.tentativas} tentativas: {ultimo_erro}") from ultimo_erro
        saida = (resp.choices[0].message.content or "") if getattr(resp, "choices", None) else ""
        uso = getattr(resp, "usage", None)
        return Predicao(
            extrair_rotulo(saida, self.rotulos),
            getattr(uso, "prompt_tokens", 0) or 0,
            getattr(uso, "completion_tokens", 0) or 0,
        )

    def detalhes(self):
        return f"modelo={self.modelo}, max_tokens=256, temperature=0, sem exemplos (zero-shot)"


def criar(chave: str, llm_modelo: str | None = None, modelo_embeddings: str = MODELO_EMBEDDINGS) -> Classificador:
    if chave == "tfidf":
        return TfidfRegLog()
    if chave == "lsa":
        return LsaRegLog()
    if chave == "embeddings":
        return EmbeddingsRegLog(modelo_embeddings)
    if chave == "llm":
        return LLMZeroShot(llm_modelo)
    raise ValueError(f"modelo desconhecido: {chave}")
