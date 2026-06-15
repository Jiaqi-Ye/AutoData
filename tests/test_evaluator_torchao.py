import sys
import types

from autodata.evaluation import evaluator


def test_disable_incompatible_torchao_for_peft_patches_availability(monkeypatch):
    fake_import_utils = types.SimpleNamespace(is_torchao_available=lambda: True)
    fake_lora_torchao = types.SimpleNamespace(is_torchao_available=lambda: True)

    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))
    monkeypatch.setitem(sys.modules, "peft.import_utils", fake_import_utils)
    monkeypatch.setitem(sys.modules, "peft.tuners", types.ModuleType("peft.tuners"))
    monkeypatch.setitem(sys.modules, "peft.tuners.lora", types.ModuleType("peft.tuners.lora"))
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.torchao", fake_lora_torchao)
    monkeypatch.setattr(evaluator.metadata, "version", lambda package: "0.10.0")

    patched = evaluator.disable_incompatible_torchao_for_peft()

    assert patched is True
    assert fake_import_utils.is_torchao_available() is False
    assert fake_lora_torchao.is_torchao_available() is False


def test_disable_incompatible_torchao_for_peft_leaves_new_versions(monkeypatch):
    monkeypatch.setattr(evaluator.metadata, "version", lambda package: "0.16.0")

    assert evaluator.disable_incompatible_torchao_for_peft() is False
