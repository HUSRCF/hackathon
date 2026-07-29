"""Exact-hash Python 3.12 compatibility shim for official fair-esm 2.0.0."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types

FAIR_ESM_TRUNK_MODULE = "esm.esmfold.v1.trunk"
FAIR_ESM_TRUNK_SHA256 = (
    "8d5c82c0b5c3422d493c2144f63d20bf23371e0142b5faf52617ba9dc1f58139"
)
FAIR_ESM_MODEL_MODULE = "esm.esmfold.v1.esmfold"
FAIR_ESM_MODEL_SHA256 = (
    "2a6ba6b1da164a666f846045251adcdb68015197585065b7f4689860076a292a"
)
COMPATIBILITY_ID = "fair-esm-2.0.0-python312-dataclass-default-factory-v1"

_MODULE_PATCHES = (
    (
        FAIR_ESM_TRUNK_MODULE,
        FAIR_ESM_TRUNK_SHA256,
        (
            (
                "from dataclasses import dataclass\n",
                "from dataclasses import dataclass, field\n",
            ),
            (
                "    structure_module: StructureModuleConfig = StructureModuleConfig()\n",
                (
                    "    structure_module: StructureModuleConfig = "
                    "field(default_factory=StructureModuleConfig)\n"
                ),
            ),
        ),
    ),
    (
        FAIR_ESM_MODEL_MODULE,
        FAIR_ESM_MODEL_SHA256,
        (
            (
                "from dataclasses import dataclass\n",
                "from dataclasses import dataclass, field\n",
            ),
            (
                "    trunk: T.Any = FoldingTrunkConfig()\n",
                "    trunk: T.Any = field(default_factory=FoldingTrunkConfig)\n",
            ),
        ),
    ),
)


def install_fair_esm_py312_compat() -> bool:
    """Load one hash-pinned fair-esm module with Python 3.12 dataclass syntax.

    fair-esm 2.0.0 predates Python 3.11's rejection of mutable dataclass
    defaults. This changes only construction of the default config object; no
    model tensor operation or checkpoint value is changed.
    """

    for module_name, expected_sha256, replacements in _MODULE_PATCHES:
        existing = sys.modules.get(module_name)
        if existing is not None:
            if getattr(existing, "__protbind_compatibility_id__", None) != (
                COMPATIBILITY_ID
            ):
                raise RuntimeError(
                    "fair-esm folding modules were imported before compatibility gating"
                )
            continue
        spec = importlib.util.find_spec(module_name)
        if spec is None or not isinstance(spec.origin, str):
            raise RuntimeError("fair-esm folding source is unavailable")
        with open(spec.origin, "rb") as handle:
            source_bytes = handle.read()
        if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
            raise RuntimeError("fair-esm folding source differs from the reviewed release")
        source = source_bytes.decode("utf-8")
        for old, new in replacements:
            if source.count(old) != 1:
                raise RuntimeError("fair-esm compatibility patch point is ambiguous")
            source = source.replace(old, new)

        module = types.ModuleType(module_name)
        module.__file__ = module_name.replace(".", "/") + ".py"
        module.__package__ = module_name.rpartition(".")[0]
        module.__loader__ = spec.loader
        module.__spec__ = spec
        module.__protbind_compatibility_id__ = COMPATIBILITY_ID
        sys.modules[module_name] = module
        try:
            exec(compile(source, module.__file__, "exec"), module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return True
