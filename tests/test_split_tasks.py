import hashlib
import json

from onprem_codegen.data.split_tasks import heldout_ids, main

IDS = [f"codenet/p{i:05d}" for i in range(100)]


def test_fraction_of_tasks_is_held_out():
    held = heldout_ids(IDS, 0.2, seed=7)
    assert len(held) == 20
    assert set(held) <= set(IDS)


def test_same_seed_same_ids_other_seed_differs():
    assert heldout_ids(IDS, 0.2, seed=7) == heldout_ids(IDS, 0.2, seed=7)
    assert heldout_ids(IDS, 0.2, seed=7) != heldout_ids(IDS, 0.2, seed=8)


def test_input_order_and_duplicates_do_not_change_the_choice():
    assert heldout_ids(IDS[::-1] + IDS[:10], 0.2, seed=7) == heldout_ids(IDS, 0.2, seed=7)


def test_rejects_fractions_that_hold_out_nothing_or_everything():
    for bad in (0, 1, -0.1, 1.5):
        try:
            heldout_ids(IDS, bad, seed=0)
        except ValueError:
            continue
        raise AssertionError(f"fraction {bad} was accepted")


def test_cli_writes_ids_seed_and_pool_hash(tmp_path):
    pool = tmp_path / "task_pool.jsonl"
    pool.write_text("".join(json.dumps({"task_id": t, "prompt": "p", "tests": []}) + "\n" for t in IDS))
    out = tmp_path / "splits" / "heldout_task_ids.json"

    main(["--task-pool", str(pool), "--seed", "7", "--fraction", "0.2", "--out", str(out)])

    data = json.loads(out.read_text())
    assert data["seed"] == 7 and data["fraction"] == 0.2
    assert data["n_tasks"] == 100 and len(data["task_ids"]) == 20
    assert data["task_pool_sha256"] == hashlib.sha256(pool.read_bytes()).hexdigest()
    assert data["task_ids"] == heldout_ids(IDS, 0.2, seed=7)
