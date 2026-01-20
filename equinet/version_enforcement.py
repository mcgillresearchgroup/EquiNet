from __future__ import annotations

from dataclasses import dataclass
from packaging.version import Version, InvalidVersion
from packaging.version import Version, InvalidVersion

def _parse(v):
    try:
        return Version(str(v))
    except (InvalidVersion, TypeError):
        return None

def assert_checkpoint_compatible(state: dict, *, strict: bool = True) -> None:
    meta = state.get("equinet_meta", {}) if isinstance(state, dict) else {}
    v = _parse(meta.get("version"))
    min_v = _parse(meta.get("min_compatible_version")) or MIN_COMPATIBLE_CHECKPOINT_VERSION

    # Old checkpoints won't have equinet_meta: fail fast.
    if v is None:
        if strict:
            raise RuntimeError(
                "Incompatible EquiNet checkpoint: missing version metadata.\n"
                "This checkpoint is likely from EquiNet v0.1.x and is NOT compatible with v0.2.x+\n"
                "(self-activity correction changed).\n"
                "Fix: retrain with EquiNet >= 0.2.0 or use EquiNet 0.1.x to load this checkpoint."
            )
        return

    if v < min_v:
        raise RuntimeError(
            f"Incompatible EquiNet checkpoint version: {v}\n"
            f"Minimum compatible checkpoint version: {min_v}\n"
            f"Current code version: {EQUINET_VERSION}\n"
            "Fix: retrain with EquiNet >= 0.2.0 or use EquiNet 0.1.x."
        )

# Your current code version (keep in sync with pyproject.toml / __init__.py)
EQUINET_VERSION = Version("0.2.0")

# Old checkpoints are incompatible with 0.2.0+ logic (self-activity changed)
MIN_COMPATIBLE_CHECKPOINT_VERSION = Version("0.2.0")


@dataclass(frozen=True)
class CompatibilityInfo:
    checkpoint_version: Version | None
    min_compatible: Version


def _parse_version(v: object) -> Version | None:
    if v is None:
        return None
    try:
        return Version(str(v))
    except InvalidVersion:
        return None


def assert_checkpoint_compatible(ckpt: dict, *, strict: bool = True) -> CompatibilityInfo:
    """
    Enforce that loaded checkpoints are compatible with this code version.
    - strict=True: missing/unparseable versions are treated as incompatible.
    """
    meta = ckpt.get("equinet_meta", {}) if isinstance(ckpt, dict) else {}
    ckpt_v = _parse_version(meta.get("version"))
    min_v = _parse_version(meta.get("min_compatible_version")) or MIN_COMPATIBLE_CHECKPOINT_VERSION

    # If missing version and strict -> fail (old models won’t have metadata)
    if ckpt_v is None:
        if strict:
            raise RuntimeError(
                "Incompatible EquiNet checkpoint: missing/invalid version metadata.\n"
                "This checkpoint is likely from EquiNet v0.1.x and is NOT compatible with v0.2.x+\n"
                "(self-activity correction semantics changed).\n\n"
                "Fix: re-train with EquiNet >= 0.2.0 or load it with EquiNet 0.1.x code."
            )
        return CompatibilityInfo(checkpoint_version=None, min_compatible=min_v)

    if ckpt_v < min_v:
        raise RuntimeError(
            f"Incompatible EquiNet checkpoint version: {ckpt_v}\n"
            f"Minimum compatible checkpoint version for this code is: {min_v}\n"
            f"Current EquiNet code version is: {EQUINET_VERSION}\n\n"
            "Reason: self-activity correction semantics changed in v0.2.x.\n"
            "Fix: re-train with EquiNet >= 0.2.0 or use EquiNet 0.1.x code to run this checkpoint."
        )

    return CompatibilityInfo(checkpoint_version=ckpt_v, min_compatible=min_v)
