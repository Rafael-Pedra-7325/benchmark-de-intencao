import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmark_intencao import cli
from benchmark_intencao.benchmark import executar_modelo
from benchmark_intencao.dados import carregar
from benchmark_intencao.modelos import LLMZeroShot
from benchmark_intencao.metricas import custo_usd, intervalo_bootstrap, maiores_confusoes


class TestMetricas(unittest.TestCase):
    def test_ic_contem_acuracia_e_e_valido(self):
        acertos = [1, 1, 0, 1, 1, 0, 1, 1]
        grupos = ["a", "a", "b", "b", "c", "c", "d", "d"]
        lo, hi = intervalo_bootstrap(acertos, grupos, n_reamostras=500)
        self.assertLessEqual(lo, 0.75)
        self.assertGreaterEqual(hi, 0.75)
        self.assertTrue(0 <= lo <= hi <= 1)

    def test_custo(self):
        self.assertAlmostEqual(custo_usd(1_000_000, 500_000, 1.0, 4.0), 3.0)
        self.assertIsNone(custo_usd(10, 10, None, 4.0))

    def test_confusoes(self):
        r = maiores_confusoes(["a", "a", "b"], ["b", "b", "b"])
        self.assertEqual(r, [("a", "b", 2)])


class TestEndToEndOffline(unittest.TestCase):
    def test_tfidf_e_lsa_geram_artefatos(self):
        with tempfile.TemporaryDirectory() as tmp:
            codigo = cli.main(["--modelos", "tfidf", "lsa", "--saida", tmp, "--max-teste", "110"])
            self.assertEqual(codigo, 0)
            for nome in ("relatorio.md", "metricas.csv", "predicoes.csv"):
                self.assertTrue((Path(tmp) / nome).exists(), nome)
            self.assertIn("TF-IDF", (Path(tmp) / "relatorio.md").read_text(encoding="utf-8"))

    # Sem chave de API o LLM é sempre pulado: evita gastar cota real se alguém rodar os testes com a chave no ambiente.
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_modelo_indisponivel_e_pulado_sem_quebrar(self):
        os.environ.pop("GROQ_API_KEY", None)
        with tempfile.TemporaryDirectory() as tmp:
            codigo = cli.main(["--modelos", "tfidf", "llm", "--saida", tmp, "--max-teste", "55"])
            self.assertEqual(codigo, 0)  # tfidf rodou; llm foi pulado

    @mock.patch.dict(os.environ, {}, clear=False)
    def test_estrito_falha_se_modelo_pulado(self):
        os.environ.pop("GROQ_API_KEY", None)
        with tempfile.TemporaryDirectory() as tmp:
            codigo = cli.main(["--modelos", "tfidf", "llm", "--saida", tmp, "--max-teste", "55", "--estrito"])
            self.assertEqual(codigo, 1)


class _ClienteSempreStatus:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="status_pedido"))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=4),
        )


class TestCaminhoLLMComClienteFalso(unittest.TestCase):
    def test_custo_e_metricas_do_llm(self):
        df = carregar(Path(cli.DADOS_PADRAO))
        treino = df[df.particao == "treino"].reset_index(drop=True)
        teste = df[df.particao == "teste"].head(20).reset_index(drop=True)
        r = executar_modelo(LLMZeroShot("fake", client=_ClienteSempreStatus()), treino, teste,
                            preco_in=1.0, preco_out=5.0, verboso=False)
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.resumo["tokens_entrada"], 20 * 200)
        # custo/1k = (200/1e6*1 + 4/1e6*5) * 1000 = 0.22
        self.assertAlmostEqual(r.resumo["custo_por_1k_usd"], 0.22, places=6)

    def test_sem_precos_custo_e_nd(self):
        df = carregar(Path(cli.DADOS_PADRAO))
        treino = df[df.particao == "treino"].reset_index(drop=True)
        teste = df[df.particao == "teste"].head(5).reset_index(drop=True)
        r = executar_modelo(LLMZeroShot("fake", client=_ClienteSempreStatus()), treino, teste, verboso=False)
        self.assertIsNone(r.resumo["custo_por_1k_usd"])


if __name__ == "__main__":
    unittest.main()
