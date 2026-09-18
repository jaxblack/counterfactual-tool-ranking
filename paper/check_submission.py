"""Inspect the prepared arXiv PDF without treating local checks as acceptance."""

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject


def inspect_pdf(file_path: Path) -> dict:
    reader = PdfReader(file_path)
    if reader.is_encrypted:
        raise ValueError("submission PDF must not be encrypted")
    page_text = [page.extract_text() or "" for page in reader.pages]
    if not page_text or not all(text.strip() for text in page_text):
        raise ValueError("every page must contain machine-readable text")
    document_text = "\n".join(page_text)
    for expected in ("Jiapeng Li", "Microsoft", "Generative-AI Assistance", "References"):
        if expected not in document_text:
            raise ValueError(f"missing expected PDF content: {expected}")
    visited = set()
    fonts = set()

    def walk(value):
        if isinstance(value, IndirectObject):
            identity = (value.idnum, value.generation)
            if identity in visited:
                return
            visited.add(identity)
            value = value.get_object()
        if isinstance(value, DictionaryObject):
            if value.get("/S") in ("/JavaScript", "/Launch") or "/JS" in value or "/JavaScript" in value:
                raise ValueError("PDF contains an active action")
            if value.get("/Type") == "/Font":
                subtype = str(value.get("/Subtype"))
                name = str(value.get("/BaseFont", "unnamed"))
                if subtype == "/Type3":
                    raise ValueError(f"bitmap/Type3 font is unsuitable for submission: {name}")
                if subtype != "/Type0":
                    descriptor_ref = value.get("/FontDescriptor")
                    descriptor = descriptor_ref.get_object() if descriptor_ref is not None else {}
                    if not any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
                        raise ValueError(f"font is not embedded: {name}")
                    fonts.add(name)
            for child in value.values():
                walk(child)
        elif isinstance(value, ArrayObject):
            for child in value:
                walk(child)

    walk(reader.trailer["/Root"])
    if not fonts:
        raise ValueError("no embedded fonts found")
    return {
        "pages": len(reader.pages),
        "bytes": file_path.stat().st_size,
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        "embedded_fonts": sorted(fonts),
        "machine_readable_pages": len(page_text),
        "encrypted": False,
        "active_actions": False,
        "local_pdf_checks": "passed; arXiv's own processing and moderation remain unverified",
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    result = inspect_pdf(root / "submission" / "counterfactual-tool-ranking.pdf")
    state_path = root / "submission" / "status.json"
    state = json.loads(state_path.read_text())
    if state["sha256"] != result["sha256"]:
        raise ValueError("prepared PDF differs from the status manifest")
    (root / "submission" / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))