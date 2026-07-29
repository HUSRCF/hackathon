from __future__ import annotations

from radeon_agent.hardware import competition_role, extract_device_arches, extract_gfx_arches

ROCMINFO = """
  Agent 2
  *******
    Name:                    gfx1100
    Marketing Name:          Radeon PRO W7900
  Agent 3
  *******
    Name:                    gfx1100
"""


def test_extracts_each_real_gpu_agent() -> None:
    assert extract_device_arches(ROCMINFO) == ("gfx1100", "gfx1100")
    assert extract_gfx_arches(ROCMINFO) == ("gfx1100",)


def test_falls_back_to_rocm_smi_device_lines() -> None:
    output = """
GPU[0] : GFX Version: gfx1100
GPU[1] : GFX Version: gfx1201
"""
    assert extract_device_arches(output) == ("gfx1100", "gfx1201")


def test_architecture_roles_are_explicit() -> None:
    assert competition_role("gfx1100") == "primary-radeon-rdna3"
    assert competition_role("gfx1201") == "cross-verifier-radeon-rdna4"
    assert competition_role("gfx90a") == "development-only-instinct-cdna2"
    assert competition_role("gfx9999") == "unclassified"
