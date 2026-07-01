#!/usr/bin/env python3
"""Build the CAST name pool artifact: the SimpleStories generation-prompt cast as the
dummy_pronoun name pool (the lists committed in DummyPronounTask).

The generation prompt (github.com/simple-stories/simple_stories_generate,
generate_stories.py) instructs: "Either don't give characters a name, or select from
Mia, Alex, Jean, Samuel, Lily, Leo, Jose, Kim, Alice, Lena, Rita, Emmanuel, Anne, Peter,
Maria or Luis" -- the dataset's ONLY recurring human names (other proper names are
"made from space-separated common words"). Pool = the 15 gender-consistent cast names;
Jean (mixed-gender corpus usage) is recorded as a frequency-matched no-gender control.

This script only JOINS existing measurements (no model run): corpus freq / pronoun skew
from mined_multitoken_names.json, fp32 she-he gap stats from name_pool_gap1.json. Both
are FROZEN artifacts -- their generator scripts were deleted 2026-06-10: the pool is the
prompt cast regardless of any score, so the stats here are provenance, not criteria.

Run: python3 scripts/build_cast_name_pool.py
"""
from sparse_pretrain.paths import OUTPUTS
import json
from pathlib import Path

EXP = OUTPUTS
PROMPT_CAST = ["Mia", "Alex", "Jean", "Samuel", "Lily", "Leo", "Jose", "Kim",
               "Alice", "Lena", "Rita", "Emmanuel", "Anne", "Peter", "Maria", "Luis"]
FEMALE = ["mia", "kim", "rita", "lily", "alice", "maria", "lena", "anne"]   # freq-ordered
MALE = ["leo", "alex", "samuel", "jose", "emmanuel", "peter", "luis"]       # freq-ordered
CONTROL = "jean"  # in the prompt cast, but corpus usage is mixed-gender -> no valid label

mined = {r["word"]: r for r in
         json.load(open(EXP / "ss_d128_f1_pronoun/mined_multitoken_names.json"))}
gap = json.load(open(EXP / "name_pool_gap1.json"))
gap_by_name = {**gap["per_name"], **{r["name"]: r for r in gap["below_threshold"]}}

# ---- gender-stratified folds for the names_templates held-out split ----
# Schema consumed by scripts/pronoun_split.py fold_split_names: "folds" plus the
# "balanced_male"/"balanced_female" rosters (key names kept for compatibility; they hold
# the FULL 8F/7M rosters -- the split task labels by male-set membership and samples
# pairs uniformly, so the 8v7 imbalance doesn't skew anything). No cast names share a
# leading stem, so no leakage clustering is needed. Females round-robin in freq order
# (spreads strong names), males fill every fold to 3.
N_FOLDS = 5
fold_f = [[] for _ in range(N_FOLDS)]
for i, n in enumerate(FEMALE):
    fold_f[i % N_FOLDS].append(n)
fold_m, males = [[] for _ in range(N_FOLDS)], iter(MALE)
for i in range(N_FOLDS):
    for _ in range(3 - len(fold_f[i])):
        fold_m[i].append(next(males))
FOLDS = [sorted(fold_f[i] + fold_m[i]) for i in range(N_FOLDS)]

def stats(n):
    m, g = mined[n], gap_by_name.get(n, {})
    return {"name": n, "corpus_freq": m["freq"], "corpus_female_share": m["female_share"],
            "ntok": m["ntok"], "gap_mean": g.get("gap_mean"),
            "consistency": g.get("consistency"), "binary_ce": g.get("binary_ce")}

out = {
    "rule": "the SimpleStories generation-prompt cast (16 names), minus Jean (mixed-gender "
            "corpus usage, F-share 0.64, model gap +0.59 -> no valid task label)",
    "provenance": {
        "prompt": "Either don't give characters a name, or select from Mia, Alex, Jean, "
                  "Samuel, Lily, Leo, Jose, Kim, Alice, Lena, Rita, Emmanuel, Anne, Peter, "
                  "Maria or Luis",
        "source": "github.com/simple-stories/simple_stories_generate generate_stories.py",
        "paper": "arXiv:2504.09184",
        "prompt_cast": PROMPT_CAST,
    },
    "gap_stats_from": {"model": gap["model"], "n_templates": gap["n_templates"],
                       "artifact": str(EXP / "name_pool_gap1.json")},
    "female": FEMALE, "male": MALE,
    "balanced_female": FEMALE, "balanced_male": MALE,  # fold_split_names schema (full rosters)
    "folds": FOLDS,
    "per_name": {n: stats(n) for n in FEMALE + MALE},
    "control_no_gender_signal": stats(CONTROL),
}
json.dump(out, open(EXP / "name_pool_cast15.json", "w"), indent=2)

print(f"{'name':>10} {'freq':>11} {'F-share':>8} {'gap':>7} {'cons':>5}")
for n in FEMALE + MALE + [CONTROL]:
    s = out["per_name"].get(n) or out["control_no_gender_signal"]
    cons = f"{s['consistency']:.2f}" if s["consistency"] is not None else "  n/a"
    tag = "  [CONTROL: no valid label]" if n == CONTROL else ""
    print(f"{n:>10} {s['corpus_freq']:>11,} {s['corpus_female_share']:>8.2f} "
          f"{s['gap_mean']:>+7.2f} {cons:>5}{tag}")
fset = set(FEMALE)
for i, f in enumerate(FOLDS):
    print(f"fold {i}: {f}  ({sum(n in fset for n in f)}F/{sum(n not in fset for n in f)}M)")
print(f"\nwrote {EXP / 'name_pool_cast15.json'}  ({len(FEMALE)}F / {len(MALE)}M + control, "
      f"{len(FOLDS)} folds)")
