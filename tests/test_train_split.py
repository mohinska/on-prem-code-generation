from datasets import Dataset

from onprem_codegen.train.sft import drop_overlength, split_indices

TASKS = [f"t{i}" for i in range(20) for _ in range(3)]  # 20 tasks x 3 rows


def test_no_task_in_both_splits():
    train, ev = split_indices(TASKS, 0.2, seed=0)
    assert ev and train
    assert {TASKS[i] for i in train}.isdisjoint({TASKS[i] for i in ev})
    assert sorted(train + ev) == list(range(len(TASKS)))
    assert len({TASKS[i] for i in ev}) == 4  # 20% of 20 tasks


def test_same_seed_same_split_other_seed_differs():
    assert split_indices(TASKS, 0.2, seed=1) == split_indices(TASKS, 0.2, seed=1)
    assert split_indices(TASKS, 0.2, seed=1) != split_indices(TASKS, 0.2, seed=2)


def test_rows_without_task_id_split_by_row():
    train, ev = split_indices([None] * 10, 0.3, seed=0)
    assert len(ev) == 3 and len(train) == 7


def test_zero_fraction_has_no_eval():
    assert split_indices(TASKS, 0, seed=0) == (list(range(len(TASKS))), [])


def test_eval_never_takes_everything():
    train, ev = split_indices(["a", "b"], 0.99, seed=0)
    assert len(train) == 1 and len(ev) == 1


class WordTok:
    """Whitespace tokenizer with the HF call signature."""

    def __call__(self, texts):
        return {"input_ids": [t.split() for t in texts]}


def test_overlength_row_dropped_not_truncated():
    ds = Dataset.from_list([
        {"prompt": "a b", "completion": "c d"},          # 4 tokens, kept
        {"prompt": "a b", "completion": "c d e f g"},    # 7 tokens, dropped
    ])
    kept, dropped = drop_overlength(ds, WordTok(), max_length=5)
    assert dropped == 1
    assert kept["completion"] == ["c d"]
