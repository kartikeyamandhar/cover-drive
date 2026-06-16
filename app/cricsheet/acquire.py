"""Acquire Cricsheet source data: download and/or safely extract an archive.

The only external surface in Phase 1. Hardened per the phase security review:
HTTPS only, host allowlist, a download size cap, and an extractor that rejects
absolute and ``..`` paths and caps uncompressed size and entry count (zip-bomb
guard). Downloaded bytes are data, never executed. Idempotent: a re-run with an
unchanged archive skips the download and extraction.

IO at the edge; pure helpers (URL validation, hashing) are separable and tested.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO

import structlog

from app.cricsheet.errors import AcquireError, UnsafeArchiveError

log = structlog.get_logger(__name__)

_META_NAME = ".acquire_meta.json"
_CHUNK = 1 << 16


@dataclass(frozen=True)
class AcquireResult:
    """Outcome of an acquire run."""

    match_files: list[Path]
    extracted_dir: Path | None
    archive_path: Path | None
    archive_sha256: str | None
    from_cache: bool


def validate_url(url: str, allowed_hosts: frozenset[str]) -> None:
    """Enforce HTTPS and a host allowlist. Raises ``AcquireError`` otherwise."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise AcquireError(f"refusing non-HTTPS url: {url!r}")
    if parsed.hostname is None or parsed.hostname not in allowed_hosts:
        raise AcquireError(f"host not in allowlist {sorted(allowed_hosts)}: {url!r}")


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target, so a redirect cannot escape the
    HTTPS-only / host-allowlist policy enforced on the initial URL."""

    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        self._allowed_hosts = allowed_hosts

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_url(newurl, self._allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener(allowed_hosts: frozenset[str]) -> urllib.request.OpenerDirector:
    """An opener whose redirects are constrained to the allowlist."""
    return urllib.request.build_opener(_ValidatingRedirectHandler(allowed_hosts))


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(
    url: str,
    dest: Path,
    *,
    allowed_hosts: frozenset[str],
    max_bytes: int,
    timeout: float = 60.0,
) -> Path:
    """Stream ``url`` to ``dest``, aborting if it exceeds ``max_bytes``.

    Raises:
        AcquireError: on a disallowed url, a network error, or an oversize body.
    """
    validate_url(url, allowed_hosts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "cricket-commentary/0.0"})
    opener = _build_opener(allowed_hosts)
    try:
        with opener.open(request, timeout=timeout) as response:  # redirects re-validated
            written = 0
            with dest.open("wb") as out:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise AcquireError(f"download exceeded max_bytes={max_bytes} for {url!r}")
                    out.write(chunk)
    except AcquireError:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise AcquireError(f"download failed for {url!r}: {exc}") from exc
    log.info("downloaded", url=url, dest=str(dest), bytes=written)
    return dest


def _safe_target(dest_root: Path, name: str) -> Path:
    """Resolve an archive member name to a path strictly under ``dest_root``."""
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise UnsafeArchiveError(f"unsafe archive entry path: {name!r}")
    target = (dest_root / name).resolve()
    if target != dest_root and dest_root not in target.parents:
        raise UnsafeArchiveError(f"archive entry escapes destination: {name!r}")
    return target


def safe_extract(
    archive: Path,
    dest_dir: Path,
    *,
    max_total_bytes: int,
    max_entries: int,
) -> list[Path]:
    """Extract ``archive`` into ``dest_dir`` safely; return extracted file paths.

    Rejects absolute and parent-traversal entries, caps the entry count and the
    total uncompressed size (zip-bomb guard), and writes only under ``dest_dir``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    written: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            if len(infos) > max_entries:
                raise UnsafeArchiveError(
                    f"archive has {len(infos)} entries > max_entries={max_entries}"
                )
            total = 0
            for info in infos:
                target = _safe_target(dest_root, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total += info.file_size
                if total > max_total_bytes:
                    raise UnsafeArchiveError(f"uncompressed size exceeds cap {max_total_bytes}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as out:
                    while True:
                        chunk = src.read(_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                written.append(target)
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError(f"not a valid zip archive: {archive}: {exc}") from exc
    log.info("extracted", archive=str(archive), dest=str(dest_dir), files=len(written))
    return written


def _read_marker(meta_path: Path) -> str | None:
    """Return the archive hash recorded by a prior successful extraction, if any."""
    if not meta_path.exists():
        return None
    try:
        return str(json.loads(meta_path.read_text(encoding="utf-8")).get("archive_sha256"))
    except (OSError, ValueError):
        return None


def _write_marker(meta_path: Path, archive_sha256: str, n_files: int) -> None:
    meta_path.write_text(
        json.dumps({"archive_sha256": archive_sha256, "n_files": n_files}, indent=2),
        encoding="utf-8",
    )


def acquire(
    *,
    dest_dir: Path,
    url: str | None = None,
    local_archive: Path | None = None,
    local_dir: Path | None = None,
    allowed_hosts: frozenset[str],
    max_download_bytes: int,
    max_uncompressed_bytes: int,
    max_entries: int,
    timeout: float = 60.0,
) -> AcquireResult:
    """Obtain match JSON files, from a local dir, a local archive, or a download.

    Idempotent: when extracting, a re-run with the same archive hash and a
    populated extraction directory is a no-op.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if local_dir is not None:
        files = sorted(local_dir.rglob("*.json"))
        log.info("using local dir", dir=str(local_dir), files=len(files))
        return AcquireResult(files, None, None, None, from_cache=True)

    if local_archive is not None:
        archive_path = local_archive
    elif url is not None:
        archive_path = download_archive(
            url,
            dest_dir / "archive.zip",
            allowed_hosts=allowed_hosts,
            max_bytes=max_download_bytes,
            timeout=timeout,
        )
    else:
        raise AcquireError("acquire needs one of local_dir, local_archive, or url")

    archive_sha = sha256_file(archive_path)
    extracted_dir = dest_dir / "extracted"
    meta_path = dest_dir / _META_NAME

    cached = _read_marker(meta_path)
    if cached == archive_sha and extracted_dir.exists() and any(extracted_dir.rglob("*.json")):
        files = sorted(extracted_dir.rglob("*.json"))
        log.info("acquire cache hit", sha256=archive_sha, files=len(files))
        return AcquireResult(files, extracted_dir, archive_path, archive_sha, from_cache=True)

    safe_extract(
        archive_path,
        extracted_dir,
        max_total_bytes=max_uncompressed_bytes,
        max_entries=max_entries,
    )
    files = sorted(extracted_dir.rglob("*.json"))
    _write_marker(meta_path, archive_sha, len(files))
    return AcquireResult(files, extracted_dir, archive_path, archive_sha, from_cache=False)
