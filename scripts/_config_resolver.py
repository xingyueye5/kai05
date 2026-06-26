"""
Unified config resolver for kai05-VLA scripts.

Background
----------
The repo has historically grown two parallel families of CLI entrypoints:

    *_yaml.py    — load a TrainConfig from an explicit YAML file path
                    (via `openpi.training.config_loader.load_config`)
    *.py         — load a TrainConfig from a registered config name
                    (via `openpi.training.config.get_config`)

The only real difference between paired CLIs (`serve_policy.py` vs
`serve_policy_yaml.py`, and the two `value_realtime_evaluator_*` files) is which
of these two loaders gets called. This module centralizes that choice in a
single helper so the rest of the logic doesn't have to know about it.

API contract
------------
`resolve_train_config(name_or_path)`:
    - If the argument looks like a filesystem path to an existing YAML file
      (or ends with `.yaml` / `.yml`), delegate to `config_loader.load_config`.
    - Otherwise treat it as a registered config name and delegate to
      `_config.get_config`.

Both legacy call sites continue to work unchanged — they may still call the
underlying loaders directly. This helper is opt-in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:  # avoid circular imports / heavy deps at module import time
    from openpi.training.config import TrainConfig


def _looks_like_yaml_path(s: str) -> bool:
    """Heuristic: a YAML config path either has a yaml extension or points to
    an existing file on disk."""
    lowered = s.lower()
    if lowered.endswith(".yaml") or lowered.endswith(".yml"):
        return True
    # absolute / relative path that exists on disk
    if os.sep in s or s.startswith(".") or s.startswith("/"):
        return Path(s).is_file()
    return False


def resolve_train_config(name_or_path: Union[str, "os.PathLike[str]"]) -> "TrainConfig":
    """Load a `TrainConfig` from either a registered name or a YAML path.

    Parameters
    ----------
    name_or_path :
        Either a registered training config name (e.g. ``"pi05_libero"``)
        recognized by ``openpi.training.config.get_config``, or a filesystem
        path to a YAML file consumable by
        ``openpi.training.config_loader.load_config``.

    Returns
    -------
    TrainConfig
        The fully-resolved training config object.

    Notes
    -----
    The two underlying loaders are imported lazily so this module stays cheap
    to import (useful for short-lived CLI tools).
    """
    s = str(name_or_path)
    if _looks_like_yaml_path(s):
        from openpi.training import config_loader  # noqa: WPS433  lazy import
        return config_loader.load_config(s)
    from openpi.training import config as _config  # noqa: WPS433  lazy import
    return _config.get_config(s)
