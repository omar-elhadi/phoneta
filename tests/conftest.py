"""Pytest fixtures.

Injects lightweight fake ``sounddevice`` / ``torch`` modules into
``sys.modules`` so unit tests can run without the heavy runtime dependencies
installed.  Real packages, when present, are left untouched.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _NoGrad:
    """Fake ``torch.no_grad`` context manager."""

    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _from_numpy(array):
    """Fake ``torch.from_numpy`` — returns a tensor whose ``.float()`` is itself."""
    tensor = MagicMock()
    tensor.float.return_value = tensor
    return tensor


def _install_fakes() -> None:
    if "torch" not in sys.modules:
        fake_hub = _module("torch.hub", load=MagicMock())
        fake_torch = _module("torch", hub=fake_hub, no_grad=_NoGrad, from_numpy=_from_numpy)
        sys.modules["torch"] = fake_torch
        sys.modules["torch.hub"] = fake_hub

    if "sounddevice" not in sys.modules:
        fake_sd = _module(
            "sounddevice",
            query_devices=MagicMock(return_value={"default_samplerate": 16000}),
            rec=MagicMock(),
            wait=MagicMock(),
            default=MagicMock(),
        )
        sys.modules["sounddevice"] = fake_sd


_install_fakes()
