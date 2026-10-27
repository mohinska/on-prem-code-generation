import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from onprem_codegen.data import clean
from onprem_codegen import utils
from onprem_codegen.eval.execute import ExecResult, Status
from onprem_codegen.data.decontaminate import decontaminate
from onprem_codegen.data.dedup import exact_dedup, near_dedup
from onprem_codegen.data.filters import apply_filters
from onprem_codegen.data.schema import ManifestRow, raw_record, read_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]
FILTERS = {"min_lines": 5, "max_lines": 50, "max_nonalnum_ratio": 0.5}
DEDUP = {"threshold": 0.8, "num_perm": 128, "shingle": 5}


def _prog(tag: str, n: int = 12) -> str:
    """A COBOL-ish file with `n` distinct lines, made unique by `tag`."""
    return "\n".join(f"           MOVE {tag}-{i} TO WS-OUT-{i}." for i in range(n)) + "\n"


def _row(content: str, source="stack_v1", path="a.cbl") -> ManifestRow:
    rec, h, n = raw_record(source, content, "MIT", {})
    return ManifestRow(id=h, source=source, source_version="v", path=path, ext=".cbl", repo=None,
                       sha256=h, license="MIT", license_ok=True, n_bytes=n)


def _by_path(rows):
    return {r.path: r for r in rows}


# --- filters

@pytest.mark.parametrize("name, content, why", [
    ("short", "ONE\nTWO\n", "too_short"),
    ("long", "X\n" * 60, "too_long"),
    ("binary", "A\n" * 10 + "\x00", "not_text"),
    ("replacement", "A\n" * 10 + "�", "not_text"),
    ("symbols", "#$%^&*()\n" * 10, "high_nonalnum"),
    ("ok", _prog("OK"), None),
])
def test_filters_flag_each_reason_and_keep_good_files(name, content, why):
    row = _row(content, path=name)
    [out] = apply_filters([row], {row.id: content}, FILTERS)
    assert out.filter_reason == why
    assert out.kept is (why is None)


def test_filters_keep_earlier_reason():
    content = _prog("OK")
    row = _row(content)
    row.kept, row.filter_reason = False, "exact_dup"
    [out] = apply_filters([row], {row.id: content}, FILTERS)
    assert (out.kept, out.filter_reason) == (False, "exact_dup")


# --- exact dedup

def test_exact_dedup_stack_beats_xcobol_regardless_of_order():
    c = _prog("SAME")
    rows = exact_dedup([_row(c, "xcobol", "x.cbl"), _row(c, "stack_v1", "s.cbl")])
    r = _by_path(rows)
    assert r["s.cbl"].kept and not r["x.cbl"].kept
    assert r["x.cbl"].filter_reason == "exact_dup"
    assert r["x.cbl"].exact_dup_of == "stack_v1:s.cbl"


def test_exact_dedup_keeps_distinct_files():
    rows = exact_dedup([_row(_prog("A"), path="a"), _row(_prog("B"), path="b")])
    assert all(r.kept for r in rows)


# --- near dedup

def test_near_dedup_drops_near_copy_across_sources_and_keeps_distinct():
    base = _prog("BASE", 40)
    near = base + "           DISPLAY WS-EXTRA.\n"  # one extra line: Jaccard well above 0.8
    rows = [_row(base, "stack_v1", "s.cbl"), _row(near, "xcobol", "x.cbl"), _row(_prog("OTHER", 40), "xcobol", "o.cbl")]
    contents = {r.id: c for r, c in zip(rows, [base, near, _prog("OTHER", 40)])}
    r = _by_path(near_dedup(rows, contents, DEDUP))
    assert r["s.cbl"].kept and r["o.cbl"].kept
    assert not r["x.cbl"].kept and r["x.cbl"].filter_reason == "near_dup"
    assert r["x.cbl"].dup_cluster_id == r["s.cbl"].dup_cluster_id


# --- decontamination

def _bench(tmp_path, prompt):
    p = tmp_path / "bench.jsonl"
    p.write_text(json.dumps({"task_id": "CobolEval/0", "prompt": prompt, "tests": [{"test": "TEST-CASE-1"}]}) + "\n")
    return {"benchmark": str(p), "ngram": 5, "threshold": 0.5}


def test_decontaminate_drops_copy_of_benchmark_prompt(tmp_path):
    prompt = _prog("BENCH", 15)
    cfg = _bench(tmp_path, prompt)
    leak, clean_ = _row(prompt + "           STOP RUN.\n", path="leak"), _row(_prog("FINE", 15), path="fine")
    contents = {leak.id: prompt + "           STOP RUN.\n", clean_.id: _prog("FINE", 15)}
    rows, report = decontaminate([leak, clean_], contents, cfg)
    r = _by_path(rows)
    assert r["leak"].decontam_hit and not r["leak"].kept and r["leak"].filter_reason == "decontam"
    assert r["fine"].kept and not r["fine"].decontam_hit
    assert [e["path"] for e in report] == ["leak"] and report[0]["task_id"] == "CobolEval/0"


def test_decontaminate_ignores_comment_only_differences(tmp_path):
    prompt = _prog("BENCH", 15)
    cfg = _bench(tmp_path, prompt)
    text = "      * a full-line comment\n" + prompt.replace("\n", " *> trailing\n")
    row = _row(text)
    [out], _ = decontaminate([row], {row.id: text}, cfg)
    assert out.decontam_hit


# --- syntax check

def _fake_execute(code, tests, **kw):
    """Stands in for the sandboxed cobc: programs containing BROKEN fail to compile."""
    if "BROKEN" in code:
        return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False, compile_stderr="e" * 600)
    return ExecResult(Status.OK, compiled=True, passed=True)


def test_syntax_check_tags_kept_files_and_leaves_the_rest(monkeypatch):
    monkeypatch.setattr(clean, "execute", _fake_execute)
    good, bad = _prog("GOOD"), _prog("BROKEN")
    cpy = _row(_prog("CPY"), path="c.cpy")
    cpy.ext = ".cpy"
    dropped = _row(_prog("DROP"), path="d.cbl")
    dropped.kept, dropped.filter_reason = False, "exact_dup"
    rows = [_row(good, path="g"), _row(bad, path="b"), cpy, dropped]
    records = {r.id: {"syntax_check": {"status": "skipped", "stderr": ""}} for r in rows}
    contents = {r.id: c for r, c in zip(rows, [good, bad, _prog("CPY"), _prog("DROP")])}
    r = _by_path(clean.syntax_check(rows, contents, records))
    assert (r["g"].syntax_check, r["g"].syntax_error) == ("ok", None)
    assert r["b"].syntax_check == "compile_error" and len(r["b"].syntax_error) == 500
    assert r["b"].kept  # tag, don't drop
    assert r["c.cpy"].syntax_check == "skipped" and r["d.cbl"].syntax_check == "skipped"
    assert records[r["b"].id]["syntax_check"] == {"status": "compile_error", "stderr": "e" * 500}
    assert records[r["g"].id]["syntax_check"]["status"] == "ok"


# --- clean.main end to end

def test_clean_main_filters_dedups_decontaminates_and_never_deletes(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    bench = tmp_path / "bench.jsonl"
    leak_prompt = _prog("BENCH", 30)
    bench.write_text(json.dumps({"task_id": "CobolEval/0", "prompt": leak_prompt, "tests": [{"test": "T"}]}) + "\n")
    dec = tmp_path / "configs/data/decontam.yaml"
    dec.write_text(yaml.safe_dump({**yaml.safe_load(dec.read_text()), "benchmark": str(bench)}))

    files = {  # name -> (source, content)
        "good": ("stack_v1", _prog("GOOD", 30)),
        "dup": ("xcobol", _prog("GOOD", 30)),
        "short": ("stack_v1", "ONE\n"),
        "leak": ("xcobol", leak_prompt),
    }
    raw = tmp_path / "raw"
    for src in {s for s, _ in files.values()}:
        (raw / src).mkdir(parents=True)
    rows = []
    for name, (src, content) in files.items():
        rec, h, n = raw_record(src, content, "MIT", {})
        with open(raw / src / "corpus.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        rows.append(ManifestRow(id=h, source=src, source_version="v", path=name, ext=".cbl", repo=None,
                                sha256=h, license="MIT", license_ok=True, n_bytes=n))
    write_manifest(rows, str(raw / "manifest.parquet"))

    out = tmp_path / "out"
    monkeypatch.setattr(clean, "execute", _fake_execute)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(clean, "RAW", raw)
    monkeypatch.setattr(clean, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["clean", "--no-wandb"])
    clean.main()

    result = _by_path(read_manifest(str(out / "manifest.parquet")))
    assert set(result) == set(files)  # flagged, never deleted
    assert {k: v.filter_reason for k, v in result.items()} == {
        "good": None, "dup": "exact_dup", "short": "too_short", "leak": "decontam"}
    kept = [json.loads(l) for l in open(out / "corpus.jsonl")]
    assert [d["content"] for d in kept] == [files["good"][1]]
    assert kept[0]["syntax_check"]["status"] == "ok" and result["good"].syntax_check == "ok"
    assert result["short"].syntax_check == "skipped"  # only kept files are checked
    stats = json.loads((out / "stats.json").read_text())
    assert stats["files_in"] == 4 and stats["files_kept"] == 1
    assert stats["by_syntax_check"] == {"ok": 1}
    assert stats["kept_by_license_ok"] == {"True": 1}  # license is a tag, not a filter
    assert stats["by_reason"] == {"kept": 1, "exact_dup": 1, "too_short": 1, "decontam": 1}


# --- W&B wiring

class _FakeRun:
    def __init__(self):
        self.used = []

    def use_artifact(self, name):
        self.used.append(name)
        return self

    def download(self, root):
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "manifest.parquet").write_text("downloaded")


def test_init_run_reads_project_and_vast_instance_from_env(monkeypatch):
    calls = {}
    fake = type(sys)("wandb")
    fake.init = lambda **kw: calls.update(kw) or "run"
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("CONTAINER_ID", "123")
    assert utils.init_run("clean", "n", {"a": 1}) == "run"
    assert calls["project"] == "proj" and calls["job_type"] == "clean"
    assert calls["config"]["a"] == 1 and calls["config"]["vast_instance"] == "123" and "git_commit" in calls["config"]


def test_ensure_raw_downloads_only_when_missing(tmp_path):
    run = _FakeRun()
    clean.ensure_raw(run, tmp_path / "raw")
    assert (tmp_path / "raw/manifest.parquet").read_text() == "downloaded"
    (tmp_path / "raw/manifest.parquet").write_text("local")
    clean.ensure_raw(run, tmp_path / "raw")
    assert (tmp_path / "raw/manifest.parquet").read_text() == "local"
    assert run.used == ["raw-corpus:latest"] * 2  # lineage is recorded either way
