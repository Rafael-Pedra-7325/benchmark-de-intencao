"""Interface web interativa (FastAPI) para testar os classificadores de intenção ao vivo."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from benchmark_intencao.catalogo import descricoes
from benchmark_intencao.config import DADOS_PADRAO, SAIDA_PADRAO
from benchmark_intencao.dados import carregar
from benchmark_intencao.metricas import custo_usd
from benchmark_intencao.modelos import DependenciaAusente, criar

PRECO_IN_PADRAO = float(os.environ.get("PRECO_ENTRADA_USD", "0.075"))
PRECO_OUT_PADRAO = float(os.environ.get("PRECO_SAIDA_USD", "0.30"))

MODELOS_ATIVOS: list = []
MODELOS_PULADOS: list[dict] = []
DESCRICOES = descricoes()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    df = carregar(DADOS_PADRAO)
    treino = df[df["particao"] == "treino"].reset_index(drop=True)
    textos = treino["texto"].tolist()
    rotulos = treino["intencao"].tolist()
    grupos = treino["grupo"].tolist()

    chaves = ["tfidf", "lsa", "llm"]
    if os.environ.get("HABILITAR_EMBEDDINGS", "0") == "1":
        chaves.insert(2, "embeddings")

    for chave in chaves:
        clf = criar(chave)
        try:
            clf.preparar()
            clf.treinar(textos, rotulos, grupos)
            if clf.aquecer and textos:
                clf.prever(textos[0])
            MODELOS_ATIVOS.append(clf)
        except DependenciaAusente as e:
            MODELOS_PULADOS.append({"chave": clf.chave, "nome": clf.nome, "motivo": str(e)})
    yield


app = FastAPI(title="Benchmark de Classificação de Intenções", lifespan=lifespan)

if SAIDA_PADRAO.exists():
    app.mount("/resultados", StaticFiles(directory=str(SAIDA_PADRAO)), name="resultados")


class RequisicaoClassificar(BaseModel):
    texto: str = Field(..., min_length=1, max_length=500)


@app.get("/api/info")
def info():
    metricas = []
    csv_metricas = SAIDA_PADRAO / "metricas.csv"
    if csv_metricas.exists():
        df_m = pd.read_csv(csv_metricas).where(pd.notnull, None)
        metricas = df_m.to_dict(orient="records")
    return {
        "ativos": [{"chave": m.chave, "nome": m.nome, "local": m.local} for m in MODELOS_ATIVOS],
        "pulados": MODELOS_PULADOS,
        "metricas_benchmark": metricas,
        "intencoes": DESCRICOES,
    }


@app.post("/api/classificar")
def classificar(req: RequisicaoClassificar):
    texto = req.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="Digite uma mensagem válida.")

    saida = []
    for clf in MODELOS_ATIVOS:
        t0 = time.perf_counter()
        try:
            pred = clf.prever(texto)
            lat_ms = (time.perf_counter() - t0) * 1000
            custo_1k = 0.0
            if not clf.local:
                c = custo_usd(pred.tokens_entrada, pred.tokens_saida, PRECO_IN_PADRAO, PRECO_OUT_PADRAO)
                custo_1k = round(c * 1000, 4) if c is not None else None
            saida.append({
                "chave": clf.chave,
                "nome": clf.nome,
                "local": clf.local,
                "intencao": pred.rotulo,
                "descricao_intencao": DESCRICOES.get(pred.rotulo, "Resposta fora do catálogo."),
                "latencia_ms": round(lat_ms, 2),
                "tokens_entrada": pred.tokens_entrada,
                "tokens_saida": pred.tokens_saida,
                "custo_1k_usd": custo_1k,
                "erro": None,
            })
        except Exception as e:  # noqa: BLE001
            saida.append({
                "chave": clf.chave,
                "nome": clf.nome,
                "local": clf.local,
                "intencao": "erro",
                "descricao_intencao": "",
                "latencia_ms": round((time.perf_counter() - t0) * 1000, 2),
                "tokens_entrada": 0,
                "tokens_saida": 0,
                "custo_1k_usd": None,
                "erro": str(e),
            })

    return {"texto": texto, "resultados": saida, "pulados": MODELOS_PULADOS}


HTML_PAGINA = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Benchmark de Intenções — Live Demo</title>
  <style>
    :root {
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8;
      --accent-hover: #0284c7; --amber: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }
    .container { max-width: 960px; margin: 0 auto; padding: 2rem 1rem 4rem; }
    header { margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.25rem; }
    h1 { margin: 0 0 0.4rem; font-size: 1.75rem; }
    .subtitle { color: var(--muted); margin: 0; font-size: 0.98rem; }
    .card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;
    }
    .input-row { display: flex; gap: 0.75rem; flex-wrap: wrap; }
    input[type="text"] {
      flex: 1; min-width: 240px; padding: 0.75rem 1rem; border-radius: 8px;
      border: 1px solid var(--border); background: #0f172a; color: var(--text); font-size: 1rem;
    }
    button.primary {
      padding: 0.75rem 1.4rem; border-radius: 8px; border: none;
      background: var(--accent); color: #0f172a; font-weight: 700; cursor: pointer; font-size: 0.95rem;
    }
    button.primary:hover { background: var(--accent-hover); color: #fff; }
    button.primary:disabled { opacity: 0.6; cursor: wait; }
    .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }
    .chip {
      background: #0f172a; border: 1px solid var(--border); color: var(--muted);
      padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.82rem; cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); color: var(--text); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
    .model-card {
      background: #0f172a; border: 1px solid var(--border); border-radius: 10px; padding: 1rem;
    }
    .model-name { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.4rem; }
    .intent-badge {
      display: inline-block; background: rgba(56, 189, 248, 0.15); color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.4); padding: 0.25rem 0.65rem;
      border-radius: 6px; font-family: monospace; font-size: 0.95rem; font-weight: 700;
    }
    .intent-desc { font-size: 0.85rem; color: var(--muted); margin: 0.6rem 0; min-height: 2.4em; }
    .metrics-line {
      display: flex; justify-content: space-between; font-size: 0.82rem;
      border-top: 1px solid var(--border); padding-top: 0.5rem; color: #cbd5e1;
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.75rem; }
    th, td { padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 600; }
    .chart { width: 100%; border-radius: 8px; margin-top: 1rem; border: 1px solid var(--border); }
    .notice { font-size: 0.84rem; color: var(--amber); margin-top: 0.75rem; }
    a { color: var(--accent); text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Benchmark de Classificação de Intenções</h1>
      <p class="subtitle">
        Compare <b>acurácia, latência e custo</b> entre modelos clássicos locais (TF-IDF, LSA) e LLM zero-shot (Groq) no domínio de delivery.
        · <a href="https://github.com/Rafael-Pedra-7325/benchmark-de-intencao" target="_blank">Código no GitHub</a>
      </p>
    </header>

    <div class="card">
      <div class="input-row">
        <input id="texto" type="text" placeholder="Ex.: o motoboy sumiu faz 1h e minha pizza não chega..." />
        <button id="btn" class="primary" onclick="classificar()">Classificar ao vivo</button>
      </div>
      <div class="chips">
        <span class="chip" onclick="usarExemplo(this)">o motoboy sumiu faz 1h e minha pizza não chega</span>
        <span class="chip" onclick="usarExemplo(this)">pedi sem cebola e veio cheio, quero meu dinheiro</span>
        <span class="chip" onclick="usarExemplo(this)">errei o numero do ap na entrega, da pra mudar?</span>
        <span class="chip" onclick="usarExemplo(this)">o pix caiu mas o app diz pendente</span>
        <span class="chip" onclick="usarExemplo(this)">me passa pra uma pessoa de verdade pfv</span>
      </div>
      <div id="aviso" class="notice"></div>
      <div id="saida" class="grid" style="margin-top: 1.2rem;"></div>
    </div>

    <div class="card">
      <h2 style="margin-top:0; font-size:1.2rem;">Resultados do Benchmark (110 frases-teste inéditas)</h2>
      <p style="color:var(--muted); font-size:0.9rem; margin-top:0;">
        Avaliação sem vazamento de dados (split por frase-base). Enquanto o TF-IDF responde em < 1 ms a custo zero, o LLM atinge 100% de acurácia em paráfrases inéditas.
      </p>
      <div style="overflow-x:auto;">
        <table id="tabela-bench">
          <thead>
            <tr>
              <th>Modelo</th>
              <th>Acurácia (IC 95%)</th>
              <th>F1-macro</th>
              <th>Latência p50</th>
              <th>Custo / 1k pred.</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <img class="chart" src="/resultados/comparacao.png" alt="Comparação de acurácia e latência" onerror="this.style.display='none'" />
    </div>
  </div>

  <script>
    function usarExemplo(el) {
      document.getElementById("texto").value = el.textContent;
      classificar();
    }

    async function carregarInfo() {
      const r = await fetch("/api/info");
      const data = await r.json();
      if (data.pulados && data.pulados.length) {
        document.getElementById("aviso").textContent =
          "Nota: " + data.pulados.map(p => `${p.nome} (${p.motivo})`).join(" | ");
      }
      const tbody = document.querySelector("#tabela-bench tbody");
      tbody.innerHTML = (data.metricas_benchmark || []).map(m => `
        <tr>
          <td><b>${m.modelo}</b></td>
          <td>${(m.acuracia * 100).toFixed(1)}% (${(m.acuracia_ic_inf * 100).toFixed(0)}–${(m.acuracia_ic_sup * 100).toFixed(0)}%)</td>
          <td>${m.f1_macro.toFixed(3)}</td>
          <td>${m.latencia_p50_ms.toFixed(2)} ms</td>
          <td>${Number(m.custo_por_1k_usd) === 0 ? "0 (local)" : "$" + Number(m.custo_por_1k_usd).toFixed(4)}</td>
        </tr>
      `).join("");
    }

    async function classificar() {
      const input = document.getElementById("texto");
      const btn = document.getElementById("btn");
      const texto = input.value.trim();
      if (!texto) return;
      btn.disabled = true;
      btn.textContent = "Classificando...";
      try {
        const r = await fetch("/api/classificar", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({texto})
        });
        const data = await r.json();
        const grid = document.getElementById("saida");
        grid.innerHTML = (data.resultados || []).map(res => `
          <div class="model-card">
            <div class="model-name">${res.nome}</div>
            <div class="intent-badge">${res.intencao}</div>
            <div class="intent-desc">${res.erro ? "Erro: " + res.erro : res.descricao_intencao}</div>
            <div class="metrics-line">
              <span>Latência: <b>${res.latencia_ms} ms</b></span>
              <span>Custo/1k: <b>${res.local ? "$0 (local)" : "$" + res.custo_1k_usd}</b></span>
            </div>
          </div>
        `).join("");
      } finally {
        btn.disabled = false;
        btn.textContent = "Classificar ao vivo";
      }
    }

    document.getElementById("texto").addEventListener("keydown", e => {
      if (e.key === "Enter") classificar();
    });
    carregarInfo();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGINA
