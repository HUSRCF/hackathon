#!/usr/bin/env python3
"""Validate the loopback HipFire endpoint without exposing paths or secrets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protbind_agent.agent_runtime import require_loopback_hipfire_url
from protbind_agent.artifacts import canonical_json_bytes
from radeon_agent.backends import HipFireBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require_loopback_hipfire_url(args.base_url)
    backend = HipFireBackend(args.base_url, timeout_seconds=10.0)
    health = backend.health()
    advertised = backend.list_models()
    if args.model not in advertised:
        raise RuntimeError(
            f"requested demo model is not advertised by HipFire: {args.model}"
        )
    receipt = {
        "schema_version": "1.0",
        "kind": "protbind.submission-demo-hipfire-preflight",
        "base_url": args.base_url,
        "loopback_only": True,
        "requested_model": args.model,
        "requested_model_advertised": True,
        "advertised_model_count": len(advertised),
        "health": health,
        "secrets_recorded": False,
    }
    args.output.write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
