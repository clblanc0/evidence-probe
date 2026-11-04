# Does the model know when it doesn't know?

**Question.** When an LLM reads clinical text, do its activations encode whether the
evidence is enough to answer? If so, does its behavior follow?

**Status: no model has been run yet.** The Results section stays empty until one has.

## Design in one example

The same context produces all three labels, depending on which drug the question names:

```
Chart notes, all entered during today's visit:
- Dr. Okafor: discontinue furosemide now.
- Dr. Haddad: discontinue pantoprazole now.
- Dr. Lindqvist: keep furosemide unchanged.
- Dr. Mensah: metformin on med list.
- Dr. Castillo: pantoprazole on med list.
- Dr. Tanaka: metformin on med list.
```

| Question: should the patient keep taking... | Label |
|---|---|
| pantoprazole? | sufficient (answer: no) |
| furosemide? | contradictory |
| metformin? | insufficient |

Length, vocabulary and mention count match by construction, so the label exists only in
how the question relates to the context. See [docs/design_notes.md](docs/design_notes.md)
for the reasoning behind each choice.

## Text-only controls (measured)

Accuracy on 600 rows, chance = 0.333. Reproduce with `python -m evidence_probe.baselines`.

| Baseline | Matched, new template | Matched, new context | Naive, new template | Naive, new context |
|---|---|---|---|---|
| Bag of words, whole prompt | 0.313 | 0.312 | 0.503 | 0.807 |
| Position of the drug's sentences | 0.333 | 0.347 | 0.323 | 0.292 |
| Only the sentences naming the drug | 0.543 | 0.963 | 0.580 | 0.955 |

The naive design is the obvious one, with a single drug per context. Its word counts
match across labels (33.1 to 33.7), yet bag of words reaches 0.81, so length matching
alone would have missed the leak.

## Run it

On Colab or Cheaha, with a GPU for steps 1 and 2:

```bash
pip install -e ".[model,dev]"
pytest
```

**Step 0, the pilot. Do this first.** It takes minutes and decides whether the rest is
worth running.

```bash
python -m evidence_probe.behavior data/matched.jsonl --limit-contexts 40 --out results/pilot.jsonl
```

Go/no-go checks are in [the design notes](docs/design_notes.md#go-or-no-go-after-the-pilot).

**Then the full run.**

```bash
python -m evidence_probe.extract data/matched.jsonl --out activations/matched_neutral.npz
```

```bash
python -m evidence_probe.behavior data/matched.jsonl --out results/behavior.jsonl --export-blind results/hand_code.csv
```

```bash
python -m evidence_probe.probes data/matched.jsonl activations/matched_neutral.npz --out-dir results
```

```bash
python -m evidence_probe.gap results/behavior.jsonl results/probe_oof.npz --layer <peak layer>
```

Add `--variant permission` to `extract` and `behavior` to allow abstention explicitly.
Use the same variant for both.

## Layout

| File | Purpose |
|---|---|
| `src/evidence_probe/templates.py` | Wording: 8 held-out template families, 24 drugs |
| `src/evidence_probe/generate.py` | Matched and naive datasets |
| `src/evidence_probe/baselines.py` | Text-only controls |
| `src/evidence_probe/splits.py` | Held-out splits and within-context label shuffling |
| `src/evidence_probe/prompting.py` | The one place prompts are built |
| `src/evidence_probe/extract.py` | Final-token activations per layer |
| `src/evidence_probe/behavior.py` | Generation and abstention scoring |
| `src/evidence_probe/probes.py` | Accuracy by layer, shuffled-label null, figure |
| `src/evidence_probe/gap.py` | Probe against behavior, including the within-label test |

## Results

*To be filled in: accuracy by layer, the probe-against-behavior table, and one paragraph.*

## Limitations

- Small model, synthetic data, one question type. Nothing here is clinical evidence.
- Decoding shows that information is present, not that the model uses it. Only an
  intervention such as steering can speak to use.
- The probe is trained on the input's label, so "probe says insufficient" means the
  activations resemble those of insufficient inputs. It doesn't show that the model
  holds a belief about its own evidence.
- *Add what the results reveal, especially scorer agreement and any empty table cells.*
