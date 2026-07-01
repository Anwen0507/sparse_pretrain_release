#!/usr/bin/env python3
"""Random train/test split of the dummy_pronoun examples for held-out circuit testing.

The atomic "example" in DummyPronounTask is a (template, name) pair: the context
"<template with name filled>," whose target pronoun is fixed by the name's gender
(CONTINUATIONS is just [""], so there is no other factor). This module materializes that
example space and randomly partitions it into a PRUNING set and a held-out TEST set,
returning two task objects that each sample ONLY from their assigned subset.

split_over:
  "examples"  (default) -- assign each (template, name) pair to prune/test at random.
                Every template and every name is still seen during pruning, so this tests
                that the mask did not OVERFIT to specific example strings.
  "names"     -- hold out whole names (stratified by gender). Tests generalization to
                unseen names. With the 7M/8F cast and 20%, ~1M+2F names are held out.
  "templates" -- hold out whole templates. Tests generalization to unseen contexts
                (same idea as the existing train/val/superval split, but randomized).

Reproducible: split_seed fixes the PARTITION; task_seed fixes per-run sampling. For a
universality sweep, keep split_seed FIXED across all seeds (so "test" is the same examples
for every circuit) and set task_seed=seed for sampling diversity.

LEAKAGE WARNING: the held-out test set must not touch anything that *selects* the circuit --
not mask training, and not the target-loss success check / size bisection (which the
universality experiment runs on val_task). Prune AND select on the pruning set; touch the
test set only to report final generalization. See `wiring_example()` at the bottom.
"""
import sys
import random
import torch
from sparse_pretrain.src.pruning.tasks import DummyPronounTask, TaskExample, BinaryTask
from sparse_pretrain.paths import NAME_POOLS


class PronounSplitTask(DummyPronounTask):
    """DummyPronounTask that samples uniformly from an explicit list of (template, name) pairs."""

    def __init__(self, tokenizer, pairs, seed: int = 42, label: str = "split",
                 male_names=None):
        # Skip DummyPronounTask.__init__ (its train/val/superval template logic); we supply pairs.
        # male_names: explicit set of names labeled male (correct pronoun " he"); names not in
        # it are labeled female. Required for clean-pool names, which are lowercase and absent
        # from the class MALE_NAMES list. Default: class lists (legacy behavior).
        BinaryTask.__init__(self, tokenizer, seed)
        self.pairs = list(pairs)
        self.split = label
        self.male_names = set(male_names) if male_names is not None else set(self.MALE_NAMES)
        if not self.pairs:
            raise ValueError("PronounSplitTask received an empty pair list")

    @property
    def name(self) -> str:
        return f"dummy_pronoun_{self.split}"

    def _example(self, template: str, name: str, continuation: str) -> TaskExample:
        male = name in self.male_names
        correct, incorrect = (" he", " she") if male else (" she", " he")
        ctx_ids = self.tokenizer.encode(template.format(name=name) + continuation,
                                        add_special_tokens=False)
        c = self.tokenizer.encode(correct, add_special_tokens=False)
        i = self.tokenizer.encode(incorrect, add_special_tokens=False)
        unk = self.tokenizer.unk_token_id or 0
        t = torch.tensor(ctx_ids, dtype=torch.long)
        return TaskExample(positive_ids=t, negative_ids=t.clone(),
                           correct_token=c[0] if c else unk,
                           incorrect_token=i[0] if i else unk)

    def generate_example(self) -> TaskExample:
        template, name = self.rng.choice(self.pairs)
        return self._example(template, name, self.rng.choice(self.CONTINUATIONS))

    def full_batch(self):
        """One DETERMINISTIC batch over EVERY pair in this split (noise-free held-out eval).

        Returns the same 5-tuple as BinaryTask.generate_batch:
        (positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions).
        """
        exs = [self._example(t, n, self.CONTINUATIONS[0]) for t, n in self.pairs]
        L = max(len(e.positive_ids) for e in exs)
        pad = self.tokenizer.pad_token_id or 0
        pos = torch.full((len(exs), L), pad, dtype=torch.long)
        evalpos = torch.empty(len(exs), dtype=torch.long)
        corr = torch.tensor([e.correct_token for e in exs], dtype=torch.long)
        inc = torch.tensor([e.incorrect_token for e in exs], dtype=torch.long)
        for r, e in enumerate(exs):
            pos[r, :len(e.positive_ids)] = e.positive_ids
            evalpos[r] = len(e.positive_ids) - 1
        return pos, pos.clone(), corr, inc, evalpos


def make_pronoun_split(tokenizer, test_frac: float = 0.2, split_over: str = "examples",
                       split_seed: int = 0, task_seed: int = 42,
                       templates=None, names=None):
    """Build (prune_task, test_task, info) from a random `test_frac` split. See module docstring."""
    if templates is None:
        templates = (DummyPronounTask.TRAIN_TEMPLATES + DummyPronounTask.VAL_TEMPLATES
                     + DummyPronounTask.SUPERVAL_TEMPLATES)
    male = [n for n in DummyPronounTask.MALE_NAMES if names is None or n in names]
    female = [n for n in DummyPronounTask.FEMALE_NAMES if names is None or n in names]
    rng = random.Random(split_seed)

    def split_list(xs):
        xs = list(xs); rng.shuffle(xs)
        k = round(len(xs) * test_frac)
        return xs[k:], xs[:k]                      # (train, test)

    if split_over == "examples":
        pairs = [(t, n) for t in templates for n in male + female]
        rng.shuffle(pairs)
        k = round(len(pairs) * test_frac)
        test_pairs, train_pairs = pairs[:k], pairs[k:]
    elif split_over == "names":                    # stratified by gender
        trM, teM = split_list(male); trF, teF = split_list(female)
        train_pairs = [(t, n) for t in templates for n in trM + trF]
        test_pairs = [(t, n) for t in templates for n in teM + teF]
    elif split_over == "templates":
        trT, teT = split_list(templates)
        alln = male + female
        train_pairs = [(t, n) for t in trT for n in alln]
        test_pairs = [(t, n) for t in teT for n in alln]
    else:
        raise ValueError("split_over must be 'examples', 'names', or 'templates'")

    prune = PronounSplitTask(tokenizer, train_pairs, seed=task_seed, label="prune")
    test = PronounSplitTask(tokenizer, test_pairs, seed=task_seed, label="test")
    info = {"split_over": split_over, "test_frac": test_frac, "split_seed": split_seed,
            "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
            "n_templates": len(templates), "n_male": len(male), "n_female": len(female)}
    return prune, test, info


DEFAULT_NAME_POOL = str(NAME_POOLS / "name_pool_cast15.json")


def fold_split_names(pool: dict, heldout_fold: int):
    """80/20 name split from a name-pool JSON (scripts/build_cast_name_pool.py).

    The pool's gender-stratified folds ARE the split mechanism: fold `heldout_fold` is the
    held-out ~20%, the other folds are the train 80%. The cast names share no leading
    stems, so no leakage clustering is needed (the retired clean-pool builder pinned
    kim/kimmy-style variant clusters to one fold for that reason).
    Returns (train_names, test_names, male_set), names lowercase.
    """
    missing = [k for k in ("folds", "balanced_male", "balanced_female") if k not in pool]
    if missing:
        raise ValueError(f"pool file is missing keys {missing} -- not a (current) "
                         "build_cast_name_pool.py artifact?")
    folds = pool["folds"]
    if not (0 <= heldout_fold < len(folds)):
        raise ValueError(f"heldout_fold must be in [0, {len(folds)})), got {heldout_fold}")
    male, female = set(pool["balanced_male"]), set(pool["balanced_female"])
    unknown = {n for f in folds for n in f} - (male | female)
    if unknown:  # stale/inconsistent pool file would otherwise silently mislabel gender
        raise ValueError(f"pool folds contain names absent from the balanced gender lists "
                         f"(inconsistent pool file?): {sorted(unknown)[:10]}...")
    test_names = sorted(folds[heldout_fold])
    train_names = sorted(n for i, f in enumerate(folds) if i != heldout_fold for n in f)
    assert not set(train_names) & set(test_names)
    return train_names, test_names, male


def tasks_from_split_info(tokenizer, info: dict, task_seed: int = 42):
    """(train_task, val_task, test_task) from a split_info dict (see make_pronoun_fold_split).

    The universality experiment snapshots split_info.json at launch and workers rebuild
    tasks ONLY from that snapshot -- never from the live name-pool file, which may keep
    evolving while a run is in flight. All names/templates come from `info`.
    """
    male = set(info["male_names"])

    def task(templates, names, label):
        pairs = [(t, n) for t in templates for n in names]
        return PronounSplitTask(tokenizer, pairs, seed=task_seed, label=label, male_names=male)

    return (task(info["train_templates"], info["train_names"], "train80"),
            task(info["val_templates"], info["train_names"], "val80"),
            task(info["superval_templates"], info["test_names"], "superval20"))


def make_pronoun_fold_split(tokenizer, heldout_fold: int = 0, pool_path: str = DEFAULT_NAME_POOL,
                            task_seed: int = 42):
    """Name-holdout split CROSSED with the fixed train/val/superval TEMPLATE split.

    Design (universality pruning "names_templates" mode):
      train_task = TRAIN_TEMPLATES    x train names (80%)  -- mask training
      val_task   = VAL_TEMPLATES      x train names (80%)  -- success gate + size bisection
      test_task  = SUPERVAL_TEMPLATES x held-out names (20%) -- reporting ONLY (loss / 2AFC)

    So the held-out evaluation is doubly out-of-sample: templates the circuit was never
    selected on AND names it never saw. Names are read from `pool_path` at call time, a
    name-pool JSON (scripts/build_cast_name_pool.py) -- they are NOT hardcoded, so
    a regenerated pool flows through automatically; the pool is uncased and so is the
    tokenizer, so lowercase names are used as validated. `heldout_fold` picks which of the
    pool's gender-stratified folds is held out (5-fold CV, 3 names per fold).
    Returns (train_task, val_task, test_task, info).
    """
    import json
    with open(pool_path) as f:
        pool = json.load(f)
    train_names, test_names, male = fold_split_names(pool, heldout_fold)
    info = {"split_over": "names_templates", "pool_path": str(pool_path),
            "heldout_fold": heldout_fold, "n_folds": len(pool["folds"]),
            "train_names": train_names, "test_names": test_names,
            "male_names": sorted(male),
            "n_train_names": len(train_names), "n_test_names": len(test_names),
            "train_templates": list(DummyPronounTask.TRAIN_TEMPLATES),
            "val_templates": list(DummyPronounTask.VAL_TEMPLATES),
            "superval_templates": list(DummyPronounTask.SUPERVAL_TEMPLATES)}
    train_task, val_task, test_task = tasks_from_split_info(tokenizer, info, task_seed)
    info.update({"n_train_pairs": len(train_task.pairs), "n_val_pairs": len(val_task.pairs),
                 "n_test_pairs": len(test_task.pairs)})
    return train_task, val_task, test_task, info


def wiring_example():
    """How to use this in universality_pruning_experiment.run_single_seed (NOT executed).

        from pronoun_split import make_pronoun_split
        # split_seed FIXED across seeds; task_seed varies for sampling diversity:
        prune_task, test_task, _ = make_pronoun_split(
            tokenizer, test_frac=0.2, split_over="examples", split_seed=0, task_seed=seed)

        # prune AND select on the pruning set (this is what keeps test held out):
        trainer = PruningTrainer(masked_model=mm, task=prune_task, val_task=prune_task,
                                 config=pc, use_wandb=False)
        # ... train, then run the SAME success-check / bisection on prune_task (replace val_task) ...

        # touch the test set ONLY for the final generalization number, after k is chosen:
        from sparse_pretrain.src.pruning.discretize import evaluate_at_k
        test_loss = evaluate_at_k(mm, test_task, k_selected, pc, num_batches=eb)
        # ...or noise-free over the entire held-out set: test_task.full_batch().
    """
    raise NotImplementedError("documentation only")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for axis in ("examples", "names", "templates"):
        prune, test, info = make_pronoun_split(tok, test_frac=0.2, split_over=axis, split_seed=0)
        ptr, pte = set(prune.pairs), set(test.pairs)
        frac = len(pte) / (len(ptr) + len(pte))
        male_tr = sum(n in DummyPronounTask.MALE_NAMES for _, n in prune.pairs)
        male_te = sum(n in DummyPronounTask.MALE_NAMES for _, n in test.pairs)
        print(f"\nsplit_over={axis:9s}  prune={len(ptr):4d}  test={len(pte):4d}  "
              f"test_frac={frac:.3f}  overlap={len(ptr & pte)}")
        print(f"   gender balance  prune {male_tr}/{len(ptr)-male_tr} M/F   test {male_te}/{len(pte)-male_te} M/F")
        ex = prune.generate_example()
        print(f"   sample prune context tokens decode: {tok.decode(ex.positive_ids)!r} "
              f"-> correct id {ex.correct_token} ({tok.decode([ex.correct_token])!r})")
    print("\nOK (overlap must be 0 for all axes)")

    print("\n=== names_templates (cast-pool fold split) ===")
    import os
    if not os.path.exists(DEFAULT_NAME_POOL):
        print(f"   pool not found at {DEFAULT_NAME_POOL}; skipping")
    else:
        for fold in range(5):
            tr, va, te, info = make_pronoun_fold_split(tok, heldout_fold=fold)
            n_overlap = len(set(info["train_names"]) & set(info["test_names"]))
            m = info["male_names"]
            te_m = sum(n in m for n in info["test_names"])
            print(f" fold {fold}: train {info['n_train_names']} names x "
                  f"{len(info['train_templates'])}/{len(info['val_templates'])} train/val templates "
                  f"({info['n_train_pairs']}/{info['n_val_pairs']} pairs) | "
                  f"test {info['n_test_names']} names ({te_m}M/{info['n_test_names']-te_m}F) x "
                  f"{len(info['superval_templates'])} superval templates ({info['n_test_pairs']} pairs) | "
                  f"name overlap={n_overlap}")
        # label sanity: a known male + female pool name must get the right pronoun
        tr, va, te, info = make_pronoun_fold_split(tok, heldout_fold=0)
        he = tok.encode(" he", add_special_tokens=False)[0]
        she = tok.encode(" she", add_special_tokens=False)[0]
        for t in (tr, va, te):
            for tmpl, nm in t.pairs[:200]:
                ex = t._example(tmpl, nm, "")
                want = he if nm in info["male_names"] else she
                assert ex.correct_token == want, (t.split, nm)
        ex = te._example(te.pairs[0][0], te.pairs[0][1], "")
        print(f" sample test context: {tok.decode(ex.positive_ids)!r} -> "
              f"{tok.decode([ex.correct_token])!r} (vs {tok.decode([ex.incorrect_token])!r})")
        print(" OK (0 name overlap, gender labels from pool)")
