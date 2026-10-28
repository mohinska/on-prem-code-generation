import json
import shutil
import sys
from pathlib import Path

import pytest

from onprem_codegen.data import clean, collect
from onprem_codegen import utils
from onprem_codegen.eval.benchmarks import Task
from onprem_codegen.eval.execute import ExecResult, Status
from onprem_codegen.data import decontaminate as decon
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


def _row(content: str, source="stack_v1_cobol", path="a.cbl") -> ManifestRow:
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
    rows = exact_dedup([_row(c, "xcobol", "x.cbl"), _row(c, "stack_v1_cobol", "s.cbl")])
    r = _by_path(rows)
    assert r["s.cbl"].kept and not r["x.cbl"].kept
    assert r["x.cbl"].filter_reason == "exact_dup"
    assert r["x.cbl"].exact_dup_of == "stack_v1_cobol:s.cbl"


def test_exact_dedup_keeps_distinct_files():
    rows = exact_dedup([_row(_prog("A"), path="a"), _row(_prog("B"), path="b")])
    assert all(r.kept for r in rows)


# --- near dedup

def test_near_dedup_drops_near_copy_across_sources_and_keeps_distinct():
    base = _prog("BASE", 40)
    near = base + "           DISPLAY WS-EXTRA.\n"  # one extra line: Jaccard well above 0.8
    rows = [_row(base, "stack_v1_cobol", "s.cbl"), _row(near, "xcobol", "x.cbl"), _row(_prog("OTHER", 40), "xcobol", "o.cbl")]
    contents = {r.id: c for r, c in zip(rows, [base, near, _prog("OTHER", 40)])}
    r = _by_path(near_dedup(rows, contents, DEDUP))
    assert r["s.cbl"].kept and r["o.cbl"].kept
    assert not r["x.cbl"].kept and r["x.cbl"].filter_reason == "near_dup"
    assert r["x.cbl"].dup_cluster_id == r["s.cbl"].dup_cluster_id


# --- decontamination

def _task(prompt, canonical=None, tests=(), benchmark="fake", mode="completion", tid="t0"):
    return Task(task_id=f"{benchmark}/{tid}", benchmark=benchmark, mode=mode, lang="cobol", prompt=prompt,
                entry_point="X", tests=list(tests), canonical_solution=canonical)


@pytest.fixture
def bench(monkeypatch):
    """Registers fake benchmarks for decontaminate.load; returns a cfg that uses them."""
    def register(**tasks):
        monkeypatch.setattr(decon, "load", lambda name: tasks[name])
        return {"benchmarks": list(tasks), "ngram": 5, "threshold": 0.5}
    return register


def test_decontaminate_drops_copy_of_benchmark_prompt(bench):
    prompt = _prog("BENCH", 15)
    cfg = bench(fake=[_task(prompt)])
    leak, clean_ = _row(prompt + "           STOP RUN.\n", path="leak"), _row(_prog("FINE", 15), path="fine")
    contents = {leak.id: prompt + "           STOP RUN.\n", clean_.id: _prog("FINE", 15)}
    rows, report = decontaminate([leak, clean_], contents, cfg)
    r = _by_path(rows)
    assert r["leak"].decontam_hit and not r["leak"].kept and r["leak"].filter_reason == "decontam"
    assert r["fine"].kept and not r["fine"].decontam_hit
    assert [e["path"] for e in report] == ["leak"]
    assert (report[0]["task_id"], report[0]["benchmark"], report[0]["part"]) == ("fake/t0", "fake", "prompt")


def test_decontaminate_ignores_comment_only_differences(bench):
    prompt = _prog("BENCH", 15)
    cfg = bench(fake=[_task(prompt)])
    text = "      * a full-line comment\n" + prompt.replace("\n", " *> trailing\n")
    row = _row(text)
    [out], _ = decontaminate([row], {row.id: text}, cfg)
    assert out.decontam_hit


def test_decontaminate_catches_canonical_solution_of_an_instruct_task_in_another_benchmark(bench):
    solution = _prog("SOLUTION", 15)
    cfg = bench(a=[_task(_prog("A", 15), benchmark="a")],
                b=[_task("Write a program that adds numbers and prints the sum, with natural language words only here.",
                         canonical=solution, benchmark="b", mode="instruct")])
    row = _row(solution)
    [out], report = decontaminate([row], {row.id: solution}, cfg)
    assert out.decontam_hit and (report[0]["benchmark"], report[0]["part"]) == ("b", "canonical_solution")


def test_decontaminate_checks_reference_files_of_file_io_tasks(bench):
    expected = "\n".join(f"ROW {i} TOTAL {i * 7}" for i in range(10))
    cfg = bench(fake=[_task("p " * 20, tests=[{"input_files": {"in": "x"}, "expected_outputs": {"out.txt": expected}}])])
    row = _row(expected)
    [out], report = decontaminate([row], {row.id: expected}, cfg)
    assert out.decontam_hit and report[0]["part"] == "expected_output"


def test_decontaminate_ignores_short_expected_values(bench):
    cfg = bench(fake=[_task(_prog("BENCH", 15), tests=[{"caller": "C", "expected": "True"}])])
    row = _row(_prog("OTHER", 15) + "           MOVE True TO X.\n")
    [out], report = decontaminate([row], {row.id: _prog("OTHER", 15) + "           MOVE True TO X.\n"}, cfg)
    assert not out.decontam_hit and report == []


def test_decontaminate_cli_reports_hits_in_any_jsonl_and_fails_on_them(bench, tmp_path):
    prompt = _prog("BENCH", 15)
    cfg = bench(fake=[_task(prompt)])
    src = tmp_path / "sft.jsonl"
    src.write_text("\n".join(json.dumps(d) for d in [
        {"prompt": "write something", "completion": "```cobol\n" + prompt + "```", "task_id": "x/1"},
        {"prompt": "write other", "completion": _prog("FINE", 15), "task_id": "x/2"}]) + "\n")
    report = tmp_path / "report.json"
    assert decon.run(src, report, cfg) == 1
    rep = json.loads(report.read_text())
    assert rep["checked"] == 2 and [h["line"] for h in rep["hits"]] == [1] and rep["by_benchmark"] == {"fake": 1}
    src.write_text(json.dumps({"prompt": "p", "completion": _prog("FINE", 15)}) + "\n")
    assert decon.run(src, report, cfg) == 0


class _FakeArtifact:
    def __init__(self, root, aliases=()):
        self.root, self.aliases, self.saved = root, list(aliases), False

    def download(self):
        return str(self.root)

    def save(self):
        self.saved = True


def test_check_artifact_sets_alias_only_when_clean(bench, tmp_path):
    cfg = bench(fake=[_task(_prog("BENCH", 15))])
    art = _FakeArtifact(tmp_path, aliases=["latest"])
    (tmp_path / "sft.jsonl").write_text(json.dumps({"prompt": "p", "completion": _prog("FINE", 15)}) + "\n")
    assert decon.check_artifact(art, cfg, tmp_path / "r.json") == 0
    assert art.aliases == ["latest", "decontaminated"] and art.saved

    dirty = _FakeArtifact(tmp_path, aliases=["latest"])
    (tmp_path / "sft.jsonl").write_text(json.dumps({"prompt": "p", "completion": _prog("BENCH", 15)}) + "\n")
    assert decon.check_artifact(dirty, cfg, tmp_path / "r.json") == 1
    assert dirty.aliases == ["latest"] and not dirty.saved


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
    leak_prompt = _prog("BENCH", 30)
    monkeypatch.setattr(decon, "load", lambda name: [_task(leak_prompt, benchmark=name)])

    files = {  # name -> (source, content)
        "good": ("stack_v1_cobol", _prog("GOOD", 30)),
        "dup": ("xcobol", _prog("GOOD", 30)),
        "short": ("stack_v1_cobol", "ONE\n"),
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
    assert sum(stats["by_benchmark"].values()) == 1
    assert stats["kept_by_license_ok"] == {"True": 1}  # license is a tag, not a filter
    assert stats["by_reason"] == {"kept": 1, "exact_dup": 1, "too_short": 1, "decontam": 1}


# --- collect

def _collect(tmp_path, monkeypatch, *args):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "xcobol").mkdir()
    one = lambda src: lambda *a, **k: [_row(_prog(src), src, src)]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(collect, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    for name, src in (("load_stack_v1", "stack_v1_cobol"), ("load_xcobol", "xcobol"), ("load_codenet", "codenet_cobol")):
        monkeypatch.setattr(collect, name, one(src))
    monkeypatch.setattr(sys, "argv", ["collect", "--no-wandb", "--xcobol-root", str(tmp_path / "xcobol"), *args])
    collect.main()
    return [r.source for r in read_manifest(str(tmp_path / "out/manifest.parquet"))]


def test_collect_fails_loudly_when_a_source_has_no_data(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="codenet_cobol"):
        _collect(tmp_path, monkeypatch, "--codenet-root", str(tmp_path / "missing"))


def test_collect_skip_leaves_the_source_out_and_sources_use_the_contract_names(tmp_path, monkeypatch):
    assert _collect(tmp_path, monkeypatch, "--skip", "codenet_cobol") == ["stack_v1_cobol", "xcobol"]


def test_collect_includes_codenet_when_its_data_exists(tmp_path, monkeypatch):
    (tmp_path / "cn").mkdir()
    assert _collect(tmp_path, monkeypatch, "--codenet-root", str(tmp_path / "cn")) == [
        "stack_v1_cobol", "xcobol", "codenet_cobol"]


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
