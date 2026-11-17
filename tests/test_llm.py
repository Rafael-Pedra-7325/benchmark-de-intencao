import unittest
from types import SimpleNamespace

from benchmark_intencao.modelos import LLMZeroShot, extrair_rotulo

ROTULOS = ["cancelar_pedido", "reembolso", "status_pedido"]


class ClienteFalso:
    """Imita anthropic.Anthropic().messages.create sem rede."""

    def __init__(self, respostas, falhas_iniciais=0):
        self.respostas = list(respostas)
        self.falhas = falhas_iniciais
        self.chamadas = []
        self.messages = self

    def create(self, **kw):
        self.chamadas.append(kw)
        if self.falhas > 0:
            self.falhas -= 1
            raise ConnectionError("rede")
        texto = self.respostas.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=texto)],
                               usage=SimpleNamespace(input_tokens=120, output_tokens=5))


class TestExtracao(unittest.TestCase):
    def test_exato_e_com_ruido(self):
        self.assertEqual(extrair_rotulo("reembolso", ROTULOS), "reembolso")
        self.assertEqual(extrair_rotulo("  `Status_Pedido`.\n", ROTULOS), "status_pedido")

    def test_dentro_de_frase(self):
        self.assertEqual(extrair_rotulo("A intenção é cancelar_pedido.", ROTULOS), "cancelar_pedido")

    def test_invalido(self):
        self.assertEqual(extrair_rotulo("não sei", ROTULOS), "invalido")
        self.assertEqual(extrair_rotulo("", ROTULOS), "invalido")


class TestLLM(unittest.TestCase):
    def test_prever_registra_tokens(self):
        m = LLMZeroShot("modelo-x", client=ClienteFalso(["reembolso"]))
        m.preparar()
        m.treinar([], ROTULOS)
        p = m.prever("quero meu dinheiro de volta")
        self.assertEqual((p.rotulo, p.tokens_entrada, p.tokens_saida), ("reembolso", 120, 5))
        self.assertIn("reembolso", m.system)

    def test_retry_em_falha_transitoria(self):
        cliente = ClienteFalso(["status_pedido"], falhas_iniciais=1)
        m = LLMZeroShot("modelo-x", client=cliente, tentativas=3)
        m.treinar([], ROTULOS)
        import benchmark_intencao.modelos as mod
        original, mod.time.sleep = mod.time.sleep, lambda s: None
        try:
            self.assertEqual(m.prever("onde está?").rotulo, "status_pedido")
        finally:
            mod.time.sleep = original
        self.assertEqual(len(cliente.chamadas), 2)


if __name__ == "__main__":
    unittest.main()
