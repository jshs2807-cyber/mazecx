"""zip에서 JSON만 고르는 함수 테스트."""

from __future__ import annotations

import io
import zipfile

from mazecx.bundle import read_jsons_from_zip


def test_zip_reader_keeps_json_and_drops_other_files() -> None:
    """json 두 개와 txt 하나가 든 zip에서는 json 두 개만 읽는다."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nested/one.json", b'{"dim": 2}')
        archive.writestr("two.json", b'{"dim": 3}')
        archive.writestr("notes/readme.txt", b"ignore me")
    found = read_jsons_from_zip(buffer.getvalue())
    assert [name for name, _payload in found] == ["one.json", "two.json"]
    assert found[0][1] == b'{"dim": 2}'
