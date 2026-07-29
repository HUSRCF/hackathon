"""Content-addressed artifacts with atomic writes and path-safe references."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .models import ArtifactRef

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _object_path(self, digest: str) -> Path:
        return self.root / "objects" / digest[:2] / digest[2:]

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        producer: str,
        producer_version: str = "unknown",
        source: str | None = None,
        license: str | None = None,
    ) -> ArtifactRef:
        digest = sha256_bytes(data)
        destination = self._object_path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return ArtifactRef(
            sha256=digest,
            media_type=media_type,
            size_bytes=len(data),
            producer=producer,
            producer_version=producer_version,
            source=source,
            license=license,
        )

    def put_json(
        self,
        value: Any,
        *,
        producer: str,
        producer_version: str = "unknown",
        source: str | None = None,
        license: str | None = None,
    ) -> ArtifactRef:
        return self.put_bytes(
            canonical_json_bytes(value),
            media_type="application/json",
            producer=producer,
            producer_version=producer_version,
            source=source,
            license=license,
        )

    def import_file(
        self,
        path: Path,
        *,
        media_type: str,
        producer: str = "protbind.import",
        producer_version: str = "0.1.0",
        source: str | None = None,
        license: str | None = None,
    ) -> ArtifactRef:
        safe_source = source if source is not None else f"local-import:{path.name}"
        return self.put_bytes(
            path.read_bytes(),
            media_type=media_type,
            producer=producer,
            producer_version=producer_version,
            source=safe_source,
            license=license,
        )

    def resolve(self, artifact: ArtifactRef, *, verify: bool = True) -> Path:
        path = self._object_path(artifact.sha256)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is not present: {artifact.artifact_id}")
        if verify:
            if path.stat().st_size != artifact.size_bytes:
                raise ValueError(f"artifact size mismatch: {artifact.artifact_id}")
            if sha256_file(path) != artifact.sha256:
                raise ValueError(f"artifact hash mismatch: {artifact.artifact_id}")
        return path

    def resolve_sha256(self, digest: str, *, verify: bool = True) -> Path:
        """Resolve a content commitment without disclosing ArtifactRef metadata."""

        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("artifact commitment must be a lowercase SHA-256 digest")
        path = self._object_path(digest)
        if not path.is_file():
            raise FileNotFoundError(f"artifact commitment is not present: sha256:{digest}")
        if verify and sha256_file(path) != digest:
            raise ValueError(f"artifact commitment hash mismatch: sha256:{digest}")
        return path

    def read_bytes(self, artifact: ArtifactRef, *, verify: bool = True) -> bytes:
        return self.resolve(artifact, verify=verify).read_bytes()

    def read_json(self, artifact: ArtifactRef, *, verify: bool = True) -> Any:
        return json.loads(self.read_bytes(artifact, verify=verify))
