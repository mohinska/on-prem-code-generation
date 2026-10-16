"""Size/text filters over raw-corpus. Sets kept=False and filter_reason; never deletes rows. Run by clean.py."""
from dataclasses import replace

from onprem_codegen.data.schema import ManifestRow


def reason(content: str, cfg: dict) -> str | None:
    if "\x00" in content or "�" in content:
        return "not_text"
    n = content.count("\n") + 1
    if n < cfg["min_lines"]:
        return "too_short"
    if n > cfg["max_lines"]:
        return "too_long"
    s = [c for c in content if not c.isspace()]
    if s and sum(not c.isalnum() for c in s) / len(s) > cfg["max_nonalnum_ratio"]:
        return "high_nonalnum"
    return None


def apply_filters(rows: list[ManifestRow], contents: dict[str, str], cfg: dict) -> list[ManifestRow]:
    out = []
    for r in rows:
        why = r.filter_reason or reason(contents[r.id], cfg)
        out.append(replace(r, kept=r.kept and why is None, filter_reason=why))
    return out
