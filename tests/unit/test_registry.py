"""Tests for the model registry."""

from __future__ import annotations

from phoneta.models.registry import (
    ModelEntry,
    all_present,
    list_models,
    missing_models,
    model_cache_dir,
    verify_checksum,
)


class TestRegistry:
    """Registry listing and presence checks."""

    def test_list_returns_all_five(self) -> None:
        models = list_models()
        names = {m.name for m in models}
        assert names == {"whisper-base", "silero-vad", "mfa-en", "mfa-fr", "espeak-ng"}

    def test_all_present(self) -> None:
        """Build a synthetic all-present registry and check all_present."""
        from phoneta.models.registry import _REGISTRY, ModelEntry

        original = _REGISTRY[:]
        try:
            _REGISTRY[:] = [
                ModelEntry(name="a", description="a", present=lambda: True),
                ModelEntry(name="b", description="b", present=lambda: True),
            ]
            assert all_present() is True
            assert missing_models() == []
        finally:
            _REGISTRY[:] = original

    def test_missing_returns_only_absent(self) -> None:
        """missing_models filters out present entries."""
        from phoneta.models.registry import _REGISTRY, ModelEntry

        original = _REGISTRY[:]
        try:
            _REGISTRY[:] = [
                ModelEntry(name="ok", description="ok", present=lambda: True),
                ModelEntry(name="nope", description="nope", present=lambda: False),
            ]
            missing = missing_models()
            assert len(missing) == 1
            assert missing[0].name == "nope"
            assert all_present() is False
        finally:
            _REGISTRY[:] = original

    def test_nothing_present(self) -> None:
        from phoneta.models.registry import _REGISTRY, ModelEntry

        original = _REGISTRY[:]
        try:
            _REGISTRY[:] = [
                ModelEntry(name="a", description="a", present=lambda: False),
                ModelEntry(name="b", description="b", present=lambda: False),
                ModelEntry(name="c", description="c", present=lambda: False),
                ModelEntry(name="d", description="d", present=lambda: False),
                ModelEntry(name="e", description="e", present=lambda: False),
            ]
            assert len(missing_models()) == 5
            assert all_present() is False
        finally:
            _REGISTRY[:] = original


class TestModelEntry:
    """ModelEntry dataclass behaviour."""

    def test_present_true(self) -> None:
        entry = ModelEntry(name="test", description="d", present=lambda: True)
        assert entry.present() is True

    def test_present_false(self) -> None:
        entry = ModelEntry(name="test", description="d", present=lambda: False)
        assert entry.present() is False


class TestChecksum:
    """File checksum verification."""

    def test_matching_checksum(self, tmp_path) -> None:
        import hashlib

        f = tmp_path / "test.bin"
        content = b"hello phoneta"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert verify_checksum(f, expected) is True

    def test_mismatched_checksum(self, tmp_path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        assert verify_checksum(f, "00" * 64) is False

    def test_missing_file(self, tmp_path) -> None:
        assert verify_checksum(tmp_path / "nope.bin", "00" * 64) is False


class TestCacheDir:
    """Model cache directory resolution."""

    def test_creates_and_returns_path(self, tmp_path, monkeypatch) -> None:
        import phoneta.models.registry as mod

        monkeypatch.setattr(mod, "_xdg_cache", lambda: tmp_path / "phoneta" / "models")
        path = model_cache_dir()
        assert path.is_dir()
        assert path.parts[-3:] == ("phoneta", "models", "") or path.parts[-2:] == (
            "phoneta",
            "models",
        )