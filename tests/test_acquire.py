"""Acquisition: URL validation, safe extraction, download caps, idempotency."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from http.client import HTTPMessage
from pathlib import Path

import pytest

from app.cricsheet import acquire as acquire_mod
from app.cricsheet.acquire import (
    acquire,
    download_archive,
    safe_extract,
    sha256_file,
    validate_url,
)
from app.cricsheet.errors import AcquireError, UnsafeArchiveError

_HOSTS = frozenset({"cricsheet.org"})


def _make_zip(path: Path, entries: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)


class _FakeResponse:
    """A urlopen-like context manager over fixed bytes."""

    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self._buf.close()


class _FakeOpener:
    """Stands in for the validating opener; returns fixed bytes."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def open(self, request: object, timeout: float = 0.0) -> _FakeResponse:
        return _FakeResponse(self._payload)


def test_validate_url_requires_https() -> None:
    with pytest.raises(AcquireError):
        validate_url("http://cricsheet.org/ipl_json.zip", _HOSTS)


def test_validate_url_host_allowlist() -> None:
    with pytest.raises(AcquireError):
        validate_url("https://evil.example/ipl_json.zip", _HOSTS)
    validate_url("https://cricsheet.org/ipl_json.zip", _HOSTS)  # allowed: no raise


def test_sha256_file(tmp_path: Path) -> None:
    target = tmp_path / "x"
    target.write_bytes(b"abc")
    assert sha256_file(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_safe_extract_good(tmp_path: Path) -> None:
    archive = tmp_path / "good.zip"
    _make_zip(archive, {"a.json": "{}", "sub/b.json": "{}"})
    out = tmp_path / "out"
    files = safe_extract(archive, out, max_total_bytes=10_000, max_entries=10)
    assert len(files) == 2
    assert (out / "a.json").exists()
    assert (out / "sub" / "b.json").exists()


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    _make_zip(archive, {"../escape.json": "{}"})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "out", max_total_bytes=10_000, max_entries=10)


def test_safe_extract_entry_cap(tmp_path: Path) -> None:
    archive = tmp_path / "many.zip"
    _make_zip(archive, {f"f{i}.json": "{}" for i in range(5)})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "out", max_total_bytes=10_000, max_entries=2)


def test_safe_extract_size_cap(tmp_path: Path) -> None:
    archive = tmp_path / "big.zip"
    _make_zip(archive, {"big.json": "x" * 1000})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "out", max_total_bytes=100, max_entries=10)


def test_safe_extract_bad_zip(tmp_path: Path) -> None:
    not_a_zip = tmp_path / "nope.zip"
    not_a_zip.write_text("plain text", encoding="utf-8")
    with pytest.raises(UnsafeArchiveError):
        safe_extract(not_a_zip, tmp_path / "out", max_total_bytes=10_000, max_entries=10)


def test_download_archive_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello-archive-bytes"
    monkeypatch.setattr(acquire_mod, "_build_opener", lambda hosts: _FakeOpener(payload))
    dest = tmp_path / "out.zip"
    result = download_archive(
        "https://cricsheet.org/ipl_json.zip", dest, allowed_hosts=_HOSTS, max_bytes=1000
    )
    assert result.read_bytes() == payload


def test_download_archive_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acquire_mod, "_build_opener", lambda hosts: _FakeOpener(b"x" * 100))
    with pytest.raises(AcquireError):
        download_archive(
            "https://cricsheet.org/ipl_json.zip",
            tmp_path / "out.zip",
            allowed_hosts=_HOSTS,
            max_bytes=10,
        )
    assert not (tmp_path / "out.zip").exists()  # partial file cleaned up


def test_redirect_to_disallowed_target_is_rejected() -> None:
    handler = acquire_mod._ValidatingRedirectHandler(_HOSTS)
    req = urllib.request.Request("https://cricsheet.org/ipl_json.zip")
    # A redirect to http:// or another host must raise, not be followed.
    with pytest.raises(AcquireError):
        handler.redirect_request(
            req, io.BytesIO(b""), 302, "Found", HTTPMessage(), "http://evil.example/x.zip"
        )
    with pytest.raises(AcquireError):
        handler.redirect_request(
            req, io.BytesIO(b""), 302, "Found", HTTPMessage(), "https://evil.example/x.zip"
        )


def test_acquire_from_local_archive_idempotent(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    _make_zip(archive, {"sample.json": "{}"})
    dest = tmp_path / "raw"
    first = acquire(
        dest_dir=dest,
        local_archive=archive,
        allowed_hosts=_HOSTS,
        max_download_bytes=1_000_000,
        max_uncompressed_bytes=1_000_000,
        max_entries=100,
    )
    assert len(first.match_files) == 1
    assert first.from_cache is False
    second = acquire(
        dest_dir=dest,
        local_archive=archive,
        allowed_hosts=_HOSTS,
        max_download_bytes=1_000_000,
        max_uncompressed_bytes=1_000_000,
        max_entries=100,
    )
    assert second.from_cache is True


def test_acquire_local_dir(tmp_path: Path) -> None:
    source = tmp_path / "jsons"
    source.mkdir()
    (source / "a.json").write_text("{}", encoding="utf-8")
    result = acquire(
        dest_dir=tmp_path / "raw",
        local_dir=source,
        allowed_hosts=_HOSTS,
        max_download_bytes=1,
        max_uncompressed_bytes=1,
        max_entries=1,
    )
    assert len(result.match_files) == 1
    assert result.from_cache is True


def test_acquire_requires_a_source(tmp_path: Path) -> None:
    with pytest.raises(AcquireError):
        acquire(
            dest_dir=tmp_path / "raw",
            allowed_hosts=_HOSTS,
            max_download_bytes=1,
            max_uncompressed_bytes=1,
            max_entries=1,
        )
