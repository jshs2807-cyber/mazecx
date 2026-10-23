"""업로드한 zip에서 미로 JSON만 꺼낸다."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path


def read_jsons_from_zip(payload: bytes) -> list[tuple[str, bytes]]:
    """zip 안의 .json만 읽는다. 폴더 경로는 버리고 파일 이름만 남긴다."""
    found: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.lower().endswith(".json"):
                continue
            found.append((name, archive.read(info)))
    return found
