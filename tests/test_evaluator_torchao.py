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


def test_hide_incompatible_torchao_from_peft_metadata_restores(monkeypatch):
    def fake_version(package_name):
        if package_name == "torchao":
            return "0.10.0"
        return "1.2.3"

    monkeypatch.setattr(evaluator.metadata, "version", fake_version)

    original_version = evaluator.hide_incompatible_torchao_from_peft_metadata()

    assert original_version is fake_version
    try:
        try:
            evaluator.metadata.version("torchao")
            raised = False
        except evaluator.metadata.PackageNotFoundError:
            raised = True
        assert raised is True
        assert evaluator.metadata.version("other-package") == "1.2.3"
    finally:
        evaluator.metadata.version = original_version

    assert evaluator.metadata.version("torchao") == "0.10.0"
