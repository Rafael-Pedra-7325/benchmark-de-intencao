import unittest

from benchmark_intencao.catalogo import INTENCOES
from benchmark_intencao.dados import amostrar_teste, gerar_dataset


class TestDados(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = gerar_dataset(semente=7, variantes=5)

    def test_deterministico(self):
        outro = gerar_dataset(semente=7, variantes=5)
        self.assertTrue(self.df.equals(outro))

    def test_sem_vazamento_de_grupos(self):
        treino = set(self.df[self.df.particao == "treino"].grupo)
        teste = set(self.df[self.df.particao == "teste"].grupo)
        self.assertFalse(treino & teste)

    def test_todas_intencoes_nos_dois_lados(self):
        for part in ("treino", "teste"):
            self.assertEqual(set(self.df[self.df.particao == part].intencao), set(INTENCOES))

    def test_sem_texto_duplicado_dentro_do_grupo(self):
        dup = self.df.groupby("grupo")["texto"].apply(lambda s: s.duplicated().any())
        self.assertFalse(dup.any())

    def test_amostragem_estratificada(self):
        teste = self.df[self.df.particao == "teste"]
        amostra = amostrar_teste(teste, 22, semente=1)
        self.assertEqual(set(amostra.intencao), set(INTENCOES))
        self.assertLessEqual(len(amostra), len(teste))


if __name__ == "__main__":
    unittest.main()
