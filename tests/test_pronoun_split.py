"""Tests for scripts/pronoun_split.py: split construction (examples/names/
templates axes), the fold-based name holdout, and PronounSplitTask semantics.

Uses the real shipped name pool JSON for the fold splits, so these tests also
guard the packaged data files.
"""
import json

import pytest
import torch

from sparse_pretrain.paths import NAME_POOLS
from sparse_pretrain.scripts import pronoun_split as PS
from sparse_pretrain.src.pruning.tasks import DummyPronounTask

ALL_TEMPLATES = (DummyPronounTask.TRAIN_TEMPLATES
                 + DummyPronounTask.VAL_TEMPLATES
                 + DummyPronounTask.SUPERVAL_TEMPLATES)
ALL_NAMES = DummyPronounTask.MALE_NAMES + DummyPronounTask.FEMALE_NAMES


class TestPronounSplitTask:
    def test_empty_pairs_raises(self, fake_tokenizer):
        with pytest.raises(ValueError, match="empty pair list"):
            PS.PronounSplitTask(fake_tokenizer, [], label="x")

    def test_labels_follow_class_gender_lists_by_default(self, fake_tokenizer):
        pairs = [("when {name} ran,", "Leo"), ("when {name} ran,", "Mia")]
        task = PS.PronounSplitTask(fake_tokenizer, pairs, label="t")
        he = fake_tokenizer.encode(" he")[0]
        she = fake_tokenizer.encode(" she")[0]
        ex_leo = task._example("when {name} ran,", "Leo", "")
        ex_mia = task._example("when {name} ran,", "Mia", "")
        assert (ex_leo.correct_token, ex_leo.incorrect_token) == (he, she)
        assert (ex_mia.correct_token, ex_mia.incorrect_token) == (she, he)

    def test_explicit_male_names_override(self, fake_tokenizer):
        pairs = [("when {name} ran,", "zork")]
        task = PS.PronounSplitTask(fake_tokenizer, pairs, label="t",
                                   male_names={"zork"})
        ex = task._example("when {name} ran,", "zork", "")
        assert ex.correct_token == fake_tokenizer.encode(" he")[0]
        # and a name NOT in male_names is female
        ex2 = task._example("when {name} ran,", "Leo", "")
        assert ex2.correct_token == fake_tokenizer.encode(" she")[0]

    def test_generate_example_samples_only_own_pairs(self, fake_tokenizer):
        pairs = [("when {name} slept,", "Mia")]
        task = PS.PronounSplitTask(fake_tokenizer, pairs, label="t")
        for _ in range(5):
            ex = task.generate_example()
            assert fake_tokenizer.decode(ex.positive_ids.tolist()) == \
                "when Mia slept,"

    def test_name_property_uses_label(self, fake_tokenizer):
        task = PS.PronounSplitTask(fake_tokenizer, [("t {name},", "Mia")],
                                   label="prune")
        assert task.name == "dummy_pronoun_prune"

    def test_full_batch_covers_every_pair_deterministically(self, fake_tokenizer):
        pairs = [("when {name} ran to the beach,", "Leo"),
                 ("when {name} slept,", "Mia"),
                 ("when {name} sat,", "Alice")]
        task = PS.PronounSplitTask(fake_tokenizer, pairs, label="t")
        pos, neg, corr, inc, ep = task.full_batch()
        assert pos.shape[0] == 3
        assert torch.equal(pos, neg)
        # rows in pair order; eval position is each row's last real token
        for r, (tmpl, name) in enumerate(pairs):
            ids = fake_tokenizer.encode(tmpl.format(name=name))
            assert pos[r, :len(ids)].tolist() == ids
            assert ep[r] == len(ids) - 1
            assert (pos[r, len(ids):] == fake_tokenizer.pad_token_id).all()
        # deterministic: identical across calls
        again = task.full_batch()
        for a, b in zip((pos, neg, corr, inc, ep), again):
            assert torch.equal(a, b)


class TestMakePronounSplit:
    @pytest.mark.parametrize("axis", ["examples", "names", "templates"])
    def test_split_disjoint_and_sized(self, fake_tokenizer, axis):
        prune, test, info = PS.make_pronoun_split(
            fake_tokenizer, test_frac=0.2, split_over=axis, split_seed=0)
        ptr, pte = set(prune.pairs), set(test.pairs)
        assert not ptr & pte
        assert info["split_over"] == axis
        total = len(ptr) + len(pte)
        assert total == len(ALL_TEMPLATES) * len(ALL_NAMES)
        assert len(pte) / total == pytest.approx(0.2, abs=0.06)

    def test_examples_split_keeps_all_names_and_templates_in_prune(
            self, fake_tokenizer):
        prune, _, _ = PS.make_pronoun_split(fake_tokenizer, split_over="examples",
                                            split_seed=0)
        assert {n for _, n in prune.pairs} == set(ALL_NAMES)
        assert {t for t, _ in prune.pairs} == set(ALL_TEMPLATES)

    def test_names_split_holds_out_whole_names_stratified(self, fake_tokenizer):
        prune, test, info = PS.make_pronoun_split(
            fake_tokenizer, test_frac=0.2, split_over="names", split_seed=0)
        train_names = {n for _, n in prune.pairs}
        test_names = {n for _, n in test.pairs}
        assert not train_names & test_names
        # stratified: round(7*0.2)=1 male and round(8*0.2)=2 female held out
        assert sum(n in DummyPronounTask.MALE_NAMES for n in test_names) == 1
        assert sum(n in DummyPronounTask.FEMALE_NAMES for n in test_names) == 2

    def test_templates_split_holds_out_whole_templates(self, fake_tokenizer):
        prune, test, _ = PS.make_pronoun_split(
            fake_tokenizer, split_over="templates", split_seed=1)
        assert not {t for t, _ in prune.pairs} & {t for t, _ in test.pairs}
        # every held-out template is paired with every name
        n_test_templates = len({t for t, _ in test.pairs})
        assert len(test.pairs) == n_test_templates * len(ALL_NAMES)

    def test_split_seed_fixes_partition(self, fake_tokenizer):
        a = PS.make_pronoun_split(fake_tokenizer, split_over="examples",
                                  split_seed=3)[1]
        b = PS.make_pronoun_split(fake_tokenizer, split_over="examples",
                                  split_seed=3)[1]
        assert set(a.pairs) == set(b.pairs)
        c = PS.make_pronoun_split(fake_tokenizer, split_over="examples",
                                  split_seed=4)[1]
        assert set(a.pairs) != set(c.pairs)

    def test_bad_axis_raises(self, fake_tokenizer):
        with pytest.raises(ValueError, match="split_over must be"):
            PS.make_pronoun_split(fake_tokenizer, split_over="bogus")


class TestFoldSplitNames:
    def pool(self):
        return json.loads((NAME_POOLS / "name_pool_cast15.json").read_text())

    def test_shipped_pool_all_folds(self):
        pool = self.pool()
        for fold in range(len(pool["folds"])):
            train, test, male = PS.fold_split_names(pool, fold)
            assert not set(train) & set(test)
            assert sorted(test) == sorted(pool["folds"][fold])
            assert set(train) | set(test) == \
                set(pool["balanced_male"]) | set(pool["balanced_female"])
            assert male == set(pool["balanced_male"])

    def test_missing_keys_rejected(self):
        with pytest.raises(ValueError, match="missing keys"):
            PS.fold_split_names({"folds": []}, 0)

    def test_fold_out_of_range(self):
        pool = self.pool()
        with pytest.raises(ValueError, match="heldout_fold"):
            PS.fold_split_names(pool, len(pool["folds"]))

    def test_unknown_fold_name_rejected(self):
        pool = {"folds": [["ghost"]], "balanced_male": ["leo"],
                "balanced_female": ["mia"]}
        with pytest.raises(ValueError, match="absent from the balanced"):
            PS.fold_split_names(pool, 0)


class TestTasksFromSplitInfo:
    def make_info(self):
        return {
            "male_names": ["leo", "alex"],
            "train_names": ["leo", "mia", "alex"],
            "test_names": ["kim"],
            "train_templates": ["when {name} ran,", "when {name} slept,"],
            "val_templates": ["when {name} sang,"],
            "superval_templates": ["when {name} flew,"],
        }

    def test_task_composition(self, fake_tokenizer):
        train, val, test = PS.tasks_from_split_info(fake_tokenizer,
                                                    self.make_info())
        assert len(train.pairs) == 2 * 3
        assert len(val.pairs) == 1 * 3
        assert len(test.pairs) == 1 * 1
        assert (train.name, val.name, test.name) == \
            ("dummy_pronoun_train80", "dummy_pronoun_val80",
             "dummy_pronoun_superval20")
        # gender labels come from info["male_names"], not the class lists
        he = fake_tokenizer.encode(" he")[0]
        she = fake_tokenizer.encode(" she")[0]
        ex = test._example("when {name} flew,", "kim", "")
        assert ex.correct_token == she
        ex2 = train._example("when {name} ran,", "leo", "")
        assert ex2.correct_token == he


class TestMakePronounFoldSplit:
    def test_end_to_end_with_shipped_pool(self, fake_tokenizer):
        train, val, test, info = PS.make_pronoun_fold_split(
            fake_tokenizer, heldout_fold=0)
        assert info["split_over"] == "names_templates"
        assert not set(info["train_names"]) & set(info["test_names"])
        assert info["n_train_pairs"] == len(train.pairs)
        assert len(train.pairs) == \
            len(info["train_templates"]) * info["n_train_names"]
        assert len(val.pairs) == \
            len(info["val_templates"]) * info["n_train_names"]
        assert len(test.pairs) == \
            len(info["superval_templates"]) * info["n_test_names"]
        # every pair's label agrees with the pool's gender assignment
        he = fake_tokenizer.encode(" he")[0]
        she = fake_tokenizer.encode(" she")[0]
        male = set(info["male_names"])
        for t in (train, val, test):
            for tmpl, name in t.pairs:
                ex = t._example(tmpl, name, "")
                assert ex.correct_token == (he if name in male else she)

    def test_snapshot_info_is_json_serializable(self, fake_tokenizer):
        _, _, _, info = PS.make_pronoun_fold_split(fake_tokenizer,
                                                   heldout_fold=1)
        json.dumps(info)  # must not raise


def test_wiring_example_is_documentation_only():
    with pytest.raises(NotImplementedError):
        PS.wiring_example()
