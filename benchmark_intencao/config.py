"""Configurações padrão do projeto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS_PADRAO = RAIZ / "data" / "intencoes.csv"
SAIDA_PADRAO = RAIZ / "resultados"

SEMENTE = 42

# Modelo de embeddings multilíngue (inclui português), ~120 MB, roda em CPU.
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

# Modelo de LLM usado no zero-shot (Groq). Pode ser trocado com --llm-modelo ou LLM_MODELO.
MODELO_LLM = "llama-3.1-8b-instant"

# Chaves aceitas em --modelos, na ordem em que aparecem no relatório.
MODELOS_DISPONIVEIS = ("tfidf", "lsa", "embeddings", "llm")
