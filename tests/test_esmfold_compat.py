from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from protbind_agent import esmfold_compat


def _clear_folding_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name, *_ in esmfold_compat._MODULE_PATCHES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_compatibility_gate_rejects_unreviewed_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_folding_modules(monkeypatch)
    source = tmp_path / "trunk.py"
    source.write_text("unreviewed source\n", encoding="utf-8")
    monkeypatch.setattr(
        esmfold_compat.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(source)),
    )

    with pytest.raises(RuntimeError, match="differs from the reviewed release"):
        esmfold_compat.install_fair_esm_py312_compat()


def test_compatibility_gate_rejects_early_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_folding_modules(monkeypatch)
    module_name = esmfold_compat.FAIR_ESM_TRUNK_MODULE
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    with pytest.raises(RuntimeError, match="imported before compatibility gating"):
        esmfold_compat.install_fair_esm_py312_compat()


def test_compatibility_gate_is_idempotent_for_marked_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_folding_modules(monkeypatch)
    for module_name, *_ in esmfold_compat._MODULE_PATCHES:
        module = ModuleType(module_name)
        module.__protbind_compatibility_id__ = esmfold_compat.COMPATIBILITY_ID
        monkeypatch.setitem(sys.modules, module_name, module)

    assert esmfold_compat.install_fair_esm_py312_compat() is True
