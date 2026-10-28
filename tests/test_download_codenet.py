import io
import json
import tarfile

from onprem_codegen.data.codenet import load_codenet
from onprem_codegen.data.download_codenet import build_task_pool, extract, html_to_text

CFG = {"min_tests": 1, "min_chars": 10, "max_chars": 6000, "max_non_ascii_ratio": 0.05, "drop_with_images": True}
PATTERNS = ["Project_CodeNet/metadata/*.csv", "Project_CodeNet/problem_descriptions/*.html",
            "Project_CodeNet/data/*/COBOL/*", "Project_CodeNet/derived/input_output/data/*/*"]
META = ("submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy\n"
        "s000000001,{p},u1,0,COBOL,COBOL,cbl,Accepted,0,0,10,\n"
        "s000000002,{p},u1,0,COBOL,COBOL,cbl,Wrong Answer,0,0,10,\n"
        "s000000003,{p},u1,0,C++,C++14,cpp,Accepted,0,0,10,\n")
FILES = {
    "Project_CodeNet/metadata/p00001.csv": META.format(p="p00001"),
    "Project_CodeNet/metadata/p00002.csv": META.format(p="p00002"),
    "Project_CodeNet/metadata/p00003.csv": META.format(p="p00003"),
    "Project_CodeNet/problem_descriptions/p00001.html":
        "<h1>Sum</h1><p>Print   A + B.</p><h2>Sample Input</h2><pre>3 5\n</pre><h2>Sample Output</h2><pre>8\n</pre>",
    "Project_CodeNet/problem_descriptions/p00002.html": "<p>See the figure for the long statement.</p><img src='x.png'>",
    "Project_CodeNet/data/p00001/COBOL/s000000001.cbl": "       PROGRAM-ID. A.\n",
    "Project_CodeNet/data/p00001/COBOL/s000000002.cbl": "       PROGRAM-ID. B.\n",
    "Project_CodeNet/data/p00001/C++/s000000003.cpp": "int main(){}",
    "Project_CodeNet/derived/input_output/data/p00001/input.txt": "3 5\n",
    "Project_CodeNet/derived/input_output/data/p00001/output.txt": "8\n",
    "Project_CodeNet/derived/input_output/data/p00002/input.txt": "1\n",
    "Project_CodeNet/derived/input_output/data/p00002/output.txt": "1\n",
    "Project_CodeNet/derived/duplicates/x.txt": "skip me",
}


def _archive() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in FILES.items():
            b = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    return buf.getvalue()


def test_extract_only_wanted_parts(tmp_path):
    data = _archive()
    info = extract(io.BytesIO(data), tmp_path, PATTERNS)
    root = tmp_path / "Project_CodeNet"
    assert (root / "data/p00001/COBOL/s000000001.cbl").exists()
    assert not (root / "data/p00001/C++").exists()
    assert not (root / "derived/duplicates").exists()
    import hashlib
    assert info["sha256"] == hashlib.sha256(data).hexdigest()


def test_task_pool_schema_and_filters(tmp_path):
    extract(io.BytesIO(_archive()), tmp_path, PATTERNS)
    tasks, stats = build_task_pool(tmp_path / "Project_CodeNet", CFG)
    assert [t["task_id"] for t in tasks] == ["codenet/p00001"]
    t = tasks[0]
    assert set(t) == {"task_id", "source", "prompt", "tests", "meta"}
    assert t["source"] == "codenet_cobol" and t["meta"] == {"problem_id": "p00001"}
    assert t["tests"] == [{"input": "3 5\n", "output": "8\n"}]
    assert "Print A + B." in t["prompt"]
    assert stats["by_reason"] == {"kept": 1, "has_image": 1, "no_tests": 1}


def test_html_keeps_english_only():
    html = '<span class="lang"><span class="lang-ja"><p>日本語 <b>x</b></p></span><span class="lang-en"><p>English</p></span></span>'
    text, has_image = html_to_text(html)
    assert text == "English" and not has_image


def test_codenet_records_follow_contract(tmp_path):
    extract(io.BytesIO(_archive()), tmp_path, PATTERNS)
    rows = load_codenet({"version": "1.0.0", "license": None}, tmp_path / "Project_CodeNet", tmp_path / "out")
    assert [r.path for r in rows] == ["data/p00001/COBOL/s000000001.cbl"]  # accepted COBOL only
    rec = json.loads((tmp_path / "out/corpus.jsonl").read_text())
    assert rec["id"] == rows[0].id == rec["content_sha256"]
    assert rec["source"] == rows[0].source == "codenet_cobol"
    assert rec["meta"] == {"problem_id": "p00001", "submission_id": "s000000001"}
    assert rec["syntax_check"] == {"status": "skipped", "stderr": ""}
