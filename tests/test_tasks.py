"""Correctness tests for every binary task in sparse_pretrain.src.pruning.tasks.

Each task's label semantics are checked against the tokenizer: e.g. a male
cast name must map to " he" as the correct token, an "a"-word context to " a",
a past-tense context to the past verb, IOI's name1 gender to him/her, etc.
Uses the whitespace FakeTokenizer, so single-word targets are single tokens.
"""
import pytest
import torch

from sparse_pretrain.src.pruning import tasks as T
from tests.conftest import FakeTokenizer


def one(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1
    return ids[0]


# ---------------------------------------------------------------------------
# Batch assembly (BinaryTask.generate_batch)
# ---------------------------------------------------------------------------
class TestGenerateBatch:
    def test_shapes_and_dynamic_padding(self, fake_tokenizer):
        task = T.DummyPronounTask(fake_tokenizer, seed=1)
        pos, neg, corr, inc, ep = task.generate_batch(6, max_length=0)
        assert pos.shape == neg.shape
        assert pos.shape[0] == 6
        # dynamic padding: width is the longest example in the batch
        lengths = [(row != fake_tokenizer.pad_token_id).sum().item()
                   for row in pos]
        assert pos.shape[1] == max(lengths)
        assert torch.equal(ep, torch.tensor(lengths) - 1)
        assert corr.shape == inc.shape == (6,)

    def test_fixed_max_length_pads(self, fake_tokenizer):
        task = T.DummyPronounTask(fake_tokenizer, seed=1)
        pos, _, _, _, ep = task.generate_batch(4, max_length=20)
        assert pos.shape[1] == 20
        assert (ep < 20).all()

    def test_truncation_when_max_length_short(self, fake_tokenizer):
        task = T.DummyPronounTask(fake_tokenizer, seed=1)
        pos, _, _, _, ep = task.generate_batch(4, max_length=3)
        assert pos.shape[1] == 3
        assert (ep == 2).all()  # last kept position

    def test_eval_position_clamped_to_actual_length(self, fake_tokenizer):
        class FixedEval(T.DummyPronounTask):
            def generate_example(self):
                ex = super().generate_example()
                ex.eval_position = 1000
                return ex

        task = FixedEval(fake_tokenizer, seed=0)
        pos, _, _, _, ep = task.generate_batch(3)
        lengths = [(row != fake_tokenizer.pad_token_id).sum().item()
                   for row in pos]
        assert torch.equal(ep, torch.tensor(lengths) - 1)

    def test_same_seed_reproducible(self, fake_tokenizer):
        a = T.DummyPronounTask(fake_tokenizer, seed=7).generate_batch(5)
        b = T.DummyPronounTask(fake_tokenizer, seed=7).generate_batch(5)
        for x, y in zip(a, b):
            assert torch.equal(x, y)
        c = T.DummyPronounTask(fake_tokenizer, seed=8).generate_batch(5)
        assert any(not torch.equal(x, y) for x, y in zip(a, c))


# ---------------------------------------------------------------------------
# Individual task label semantics
# ---------------------------------------------------------------------------
class TestDummyQuote:
    def test_correct_token_matches_opening_quote(self, fake_tokenizer):
        task = T.DummyQuoteTask(fake_tokenizer, seed=0)
        dq, sq = one(fake_tokenizer, '"'), one(fake_tokenizer, "'")
        for _ in range(30):
            ex = task.generate_example()
            text = fake_tokenizer.decode(ex.positive_ids.tolist())
            assert torch.equal(ex.positive_ids, ex.negative_ids)
            if '"' in text:
                assert (ex.correct_token, ex.incorrect_token) == (dq, sq)
            else:
                assert (ex.correct_token, ex.incorrect_token) == (sq, dq)
        assert task.name == "dummy_quote"


class TestDummyArticle:
    def test_a_for_consonant_an_for_vowel(self, fake_tokenizer):
        task = T.DummyArticleTask(fake_tokenizer, seed=0)
        a_id, an_id = one(fake_tokenizer, "a"), one(fake_tokenizer, "an")
        seen = set()
        for _ in range(40):
            ex = task.generate_example()
            seen.add(ex.correct_token)
            assert {ex.correct_token, ex.incorrect_token} == {a_id, an_id}
            assert ex.correct_token != ex.incorrect_token
        assert seen == {a_id, an_id}  # both label directions occur
        assert task.name == "dummy_article"


class TestDummyPronoun:
    def test_gender_to_pronoun_mapping(self, fake_tokenizer):
        task = T.DummyPronounTask(fake_tokenizer, seed=3)
        he, she = one(fake_tokenizer, "he"), one(fake_tokenizer, "she")
        for _ in range(60):
            ex = task.generate_example()
            text = fake_tokenizer.decode(ex.positive_ids.tolist())
            name = text.split()[1]  # "when {name} ..."
            if name in T.DummyPronounTask.MALE_NAMES:
                assert (ex.correct_token, ex.incorrect_token) == (he, she)
            else:
                assert name in T.DummyPronounTask.FEMALE_NAMES
                assert (ex.correct_token, ex.incorrect_token) == (she, he)

    def test_cast15_name_lists(self):
        assert len(T.DummyPronounTask.MALE_NAMES) == 7
        assert len(T.DummyPronounTask.FEMALE_NAMES) == 8
        assert not set(T.DummyPronounTask.MALE_NAMES) & \
            set(T.DummyPronounTask.FEMALE_NAMES)

    def test_splits_use_disjoint_templates(self, fake_tokenizer):
        train = T.DummyPronounTask(fake_tokenizer, split="train")
        val = T.DummyPronounTask(fake_tokenizer, split="val")
        sup = T.DummyPronounTask(fake_tokenizer, split="superval")
        assert train.templates == T.DummyPronounTask.TRAIN_TEMPLATES
        assert val.templates == T.DummyPronounTask.VAL_TEMPLATES
        assert sup.templates == T.DummyPronounTask.SUPERVAL_TEMPLATES
        assert not set(train.templates) & set(val.templates)
        assert not set(train.templates) & set(sup.templates)
        assert not set(val.templates) & set(sup.templates)
        assert train.name == "dummy_pronoun_train"

    def test_invalid_split_raises(self, fake_tokenizer):
        with pytest.raises(ValueError, match="split must be"):
            T.DummyPronounTask(fake_tokenizer, split="test")


class TestPronounVariants:
    def test_orig10_pinned_names(self, fake_tokenizer):
        task = T.DummyPronounOrig10Task(fake_tokenizer)
        assert task.MALE_NAMES == ["Leo", "Alex", "Samuel", "Jose", "Peter"]
        assert task.FEMALE_NAMES == ["Mia", "Kim", "Rita", "Lily", "Maria"]
        assert task.name == "dummy_pronoun_orig10_train"

    def test_wrong_task_swaps_labels(self, fake_tokenizer):
        seed = 5
        normal = T.DummyPronounTask(fake_tokenizer, seed=seed)
        wrong = T.DummyPronounWrongTask(fake_tokenizer, seed=seed)
        for _ in range(10):
            a, b = normal.generate_example(), wrong.generate_example()
            assert torch.equal(a.positive_ids, b.positive_ids)
            assert b.correct_token == a.incorrect_token
            assert b.incorrect_token == a.correct_token
        assert wrong.name == "dummy_pronoun_wrong_train"

    @pytest.mark.parametrize("cls,target,suffix", [
        (T.DummyPronounWhenTask, "when", "when"),
        (T.DummyPronounIsTask, "is", "is"),
        (T.DummyPronounEvilTask, "evil", "evil"),
        (T.DummyPronounWaterTask, "water", "water"),
    ])
    def test_constant_token_tasks(self, fake_tokenizer, cls, target, suffix):
        task = cls(fake_tokenizer, seed=0)
        target_id = one(fake_tokenizer, target)
        he_id = one(fake_tokenizer, "he")
        for _ in range(5):
            ex = task.generate_example()
            assert ex.correct_token == target_id
            assert ex.incorrect_token == he_id
        assert task.name == f"dummy_pronoun_{suffix}_train"

    def test_iswhen_gender_conditional_targets(self, fake_tokenizer):
        task = T.DummyPronounIsWhenTask(fake_tokenizer, seed=2)
        is_id, when_id = one(fake_tokenizer, "is"), one(fake_tokenizer, "when")
        for _ in range(40):
            ex = task.generate_example()
            name = fake_tokenizer.decode(ex.positive_ids.tolist()).split()[1]
            if name in task.MALE_NAMES:
                assert (ex.correct_token, ex.incorrect_token) == (is_id, when_id)
            else:
                assert (ex.correct_token, ex.incorrect_token) == (when_id, is_id)
        assert task.name == "dummy_pronoun_iswhen_train"


class TestGenderNameTokenIds:
    def test_single_token_names_returned_as_sets(self, fake_tokenizer):
        female, male = T.gender_name_token_ids(fake_tokenizer)
        assert isinstance(female, set) and isinstance(male, set)
        expected_female = {one(fake_tokenizer, n)
                           for n in T.DummyPronounTask.FEMALE_NAMES}
        assert female == expected_female
        assert len(male) == len(T.DummyPronounTask.MALE_NAMES)
        assert not female & male

    def test_list_mode_preserves_order(self, fake_tokenizer):
        female, male = T.gender_name_token_ids(fake_tokenizer, as_set=False)
        assert isinstance(female, list)
        assert female == [one(fake_tokenizer, n)
                          for n in T.DummyPronounTask.FEMALE_NAMES]

    def test_multi_token_names_skipped(self):
        class SplittingTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                out = []
                for w in text.split():
                    if w == "Emmanuel":  # force one name to be multi-token
                        out.extend([self._id("Emma"), self._id("##nuel")])
                    else:
                        out.append(self._id(w))
                return out

        tok = SplittingTokenizer()
        _, male = T.gender_name_token_ids(tok, as_set=False)
        assert len(male) == len(T.DummyPronounTask.MALE_NAMES) - 1


class TestDummyTense:
    def test_correct_verb_matches_context_tense(self, fake_tokenizer):
        task = T.DummyTenseTask(fake_tokenizer, seed=0)
        by_context = {}
        for tpl, pres, past, is_present in task.TRAIN_TEMPLATES:
            by_context.setdefault(tpl, []).append((pres, past, is_present))
        for _ in range(50):
            ex = task.generate_example()
            corr = fake_tokenizer.inv[ex.correct_token]
            inc = fake_tokenizer.inv[ex.incorrect_token]
            pairs = {(p, q) for _, p, q, _ in
                     [(None,) + t for t in
                      [(p, q, ip) for p, q, ip in
                       [tt[1:] for tt in task.TRAIN_TEMPLATES]]]}
            # correct/incorrect are one of the template's verb pairs, and the
            # pair orientation matches the template tense flag
            match = [t for t in task.TRAIN_TEMPLATES
                     if {t[1], t[2]} == {corr, inc}]
            assert match, (corr, inc)
            tpl, pres, past, is_present = match[0]
            if corr == pres:
                assert any(m[3] for m in match if m[1] == corr)
            else:
                assert any(not m[3] for m in match if m[2] == corr)

    def test_pronoun_agrees_with_name(self, fake_tokenizer):
        task = T.DummyTenseTask(fake_tokenizer, seed=1)
        for _ in range(30):
            ex = task.generate_example()
            words = fake_tokenizer.decode(ex.positive_ids.tolist()).split()
            names = [w for w in words if w in task.NAME_TO_PRONOUN]
            assert names
            expected = task.NAME_TO_PRONOUN[names[0]]
            assert words[-1].rstrip(",") == expected

    def test_get_all_candidate_examples(self):
        cands = T.DummyTenseTask.get_all_candidate_examples()
        n_templates = len(T.DummyTenseTask.TEMPLATE_STRUCTURES)
        n_verbs = len(T.DummyTenseTask.VERB_PAIRS)
        assert len(cands) == 2 * n_templates * n_verbs
        present = [c for c in cands if c[3]]
        assert len(present) == n_templates * n_verbs

    def test_split_selection_and_name(self, fake_tokenizer):
        assert T.DummyTenseTask(fake_tokenizer, split="val").templates == \
            T.DummyTenseTask.VAL_TEMPLATES
        assert T.DummyTenseTask(fake_tokenizer, split="superval").name == \
            "dummy_tense_superval"
        with pytest.raises(ValueError):
            T.DummyTenseTask(fake_tokenizer, split="bogus")


class TestIOITasks:
    @pytest.mark.parametrize("cls,prefix", [
        (T.IOIStrictTask, "ioi"),
        (T.IOIRelaxedTask, "ioi_relaxed"),
    ])
    def test_pronoun_refers_to_name1(self, fake_tokenizer, cls, prefix):
        task = cls(fake_tokenizer, seed=0)
        him, her = one(fake_tokenizer, "him"), one(fake_tokenizer, "her")
        for _ in range(40):
            ex = task.generate_example()
            words = fake_tokenizer.decode(ex.positive_ids.tolist()).split()
            name1 = words[1]  # "when {name1} ..."
            if name1 in task.MALE_NAMES:
                assert (ex.correct_token, ex.incorrect_token) == (him, her)
            else:
                assert name1 in task.FEMALE_NAMES
                assert (ex.correct_token, ex.incorrect_token) == (her, him)
            # name2 is the opposite gender for strict/relaxed
            name2 = [w for w in words[2:] if w in
                     task.MALE_NAMES + task.FEMALE_NAMES]
            assert name2
            assert (name1 in task.MALE_NAMES) != (name2[-1] in task.MALE_NAMES)
        assert task.name.startswith(prefix)

    def test_mixed_allows_same_gender_but_distinct_names(self, fake_tokenizer):
        task = T.IOIMixedTask(fake_tokenizer, seed=0)
        him, her = one(fake_tokenizer, "him"), one(fake_tokenizer, "her")
        saw_same_gender = False
        for _ in range(80):
            ex = task.generate_example()
            words = fake_tokenizer.decode(ex.positive_ids.tolist()).split()
            name1 = words[1]
            others = [w for w in words[2:]
                      if w in task.MALE_NAMES + task.FEMALE_NAMES]
            name2 = others[-1]
            assert name1 != name2  # same-gender pairs must use distinct names
            same = (name1 in task.MALE_NAMES) == (name2 in task.MALE_NAMES)
            saw_same_gender |= same
            expected = him if name1 in task.MALE_NAMES else her
            assert ex.correct_token == expected
        assert saw_same_gender
        assert task.TEMPLATES is T.IOIRelaxedTask.TEMPLATES

    def test_relaxed_template_count(self):
        assert len(T.IOIRelaxedTask.TEMPLATES) == 1394


class TestPronounDistractor:
    def test_possessive_matches_subject_not_distractor(self, fake_tokenizer):
        task = T.PronounDistractorTask(fake_tokenizer, seed=0)
        his, her = one(fake_tokenizer, "his"), one(fake_tokenizer, "her")
        for _ in range(40):
            ex = task.generate_example()
            words = fake_tokenizer.decode(ex.positive_ids.tolist()).split()
            names = [w for w in words
                     if w in task.MALE_NAMES + task.FEMALE_NAMES]
            assert len(names) == 2
            distractor, subject = names
            # distractor and subject always opposite gender
            assert (distractor in task.MALE_NAMES) != (subject in task.MALE_NAMES)
            expected = his if subject in task.MALE_NAMES else her
            assert ex.correct_token == expected
        assert task.name == "pronoun_distractor_train"


# ---------------------------------------------------------------------------
# Registry / factory / dataset wrapper
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_all_tasks_registered(self):
        assert set(T.TASK_REGISTRY) == {
            "dummy_quote", "dummy_article", "dummy_pronoun",
            "dummy_pronoun_orig10", "dummy_pronoun_wrong",
            "dummy_pronoun_when", "dummy_pronoun_is", "dummy_pronoun_evil",
            "dummy_pronoun_water", "dummy_pronoun_iswhen", "dummy_tense",
            "ioi_strict", "ioi_relaxed", "ioi_mixed", "pronoun_distractor",
        }

    def test_get_task_passes_split_when_supported(self, fake_tokenizer):
        task = T.get_task("dummy_pronoun", fake_tokenizer, split="superval")
        assert task.split == "superval"
        quote = T.get_task("dummy_quote", fake_tokenizer, split="superval")
        assert isinstance(quote, T.DummyQuoteTask)

    def test_get_task_unknown_raises(self, fake_tokenizer):
        with pytest.raises(ValueError, match="Unknown task"):
            T.get_task("nope", fake_tokenizer)

    def test_every_registered_task_generates_valid_batches(self, fake_tokenizer):
        for name in T.TASK_REGISTRY:
            task = T.get_task(name, fake_tokenizer, seed=0)
            pos, neg, corr, inc, ep = task.generate_batch(2)
            assert pos.dtype == torch.long
            assert (corr != inc).all(), name
            assert (ep >= 0).all() and (ep < pos.shape[1]).all()


class TestTaskDataset:
    def test_iteration_yields_n_samples(self, fake_tokenizer):
        task = T.DummyPronounTask(fake_tokenizer)
        ds = T.TaskDataset(task, n_samples=10, max_length=16, batch_size=4)
        batches = list(ds)
        assert [b.shape[0] for b in batches] == [4, 4, 2]
        assert all(b.shape[1] == 16 for b in batches)
        assert len(ds) == 3

    def test_get_all_texts_strips_padding(self, fake_tokenizer):
        task = T.DummyPronounTask(fake_tokenizer)
        ds = T.TaskDataset(task, n_samples=5, max_length=32, batch_size=3)
        texts = ds.get_all_texts()
        assert len(texts) == 5
        assert all("<pad>" not in t for t in texts)
        assert all(t.startswith("when") for t in texts)
