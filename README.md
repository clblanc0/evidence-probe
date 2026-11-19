# Do models register patient mix-ups internally?

**When an LLM files a visit into the wrong patient's chart, do its activations register
the mismatch even though it says nothing?**

*A weekend pilot, not a paper: one small open model, synthetic charts.*

## Why

<!-- Check these numbers against the paper before publishing. -->
Klang et al. (2026) found that clinical agents usually write straight through tampered
charts. The best model caught 17.4% of full patient swaps and almost none of the
one-digit medical record number (MRN) changes. That study measured behavior only. This
pilot asks the inside question, using the methods of neural decoding: train a decoder on
the model's activations, with the controls that keep a decoder honest.

## Method

- **Data:** 540 synthetic charts in 6 layouts. Each base chart (a patient, two prior
  visits, one new visit to file) comes in 6 versions that differ only in the new visit's
  identity line:
  - **Untampered:** clean, name reformatted ("GOMEZ, MARIA"), MRN dashed ("004-8213")
  - **Tampered:** patient swap, one-digit MRN change, age shift
- **Task:** the model is asked to file the new visit as a one-line chart entry. With
  the `check` prompt, it's also asked to confirm the visit belongs to the patient.
- **Activations:** every layer, recorded at the last token of the new visit and at the
  last prompt token.
- **Probe:** logistic regression per layer, always tested on chart layouts it never
  trained on. A second probe is tested on a tampering type it never trained on.
- **Controls:** shuffled labels (within each base chart), two text-only baselines, and
  the same reformatting on tampered and untampered rows, so formatting can't give the
  label away.

## Results

*To be filled in after the Colab run.*

**Figure 1.** *Probe AUC by layer, with shuffled-label and text-only baselines.*

**Figure 2.** *For each condition: how often the probe caught it, how often the model
flagged it, and how often a name-or-MRN string rule would have.*

**Findings.** *Three sentences.*

## Controls measured so far

These don't need the model. AUC on held-out layouts, where chance is 0.5:

| Baseline | AUC |
|---|---|
| Bag of words, whole prompt | 0.55 |
| New visit only, character n-grams | 0.48 |

The new-visit-only baseline caught a leak in the first version of the generator. Only
untampered rows had reformatted names and MRNs, so formatting alone meant "not
tampered", and that baseline scored 0.64. Giving tampered rows the same formatting mix
brought it down to chance.

A name-or-MRN string rule flags every swap and every MRN change. It also raises a false
alarm on every reformatted name or dashed MRN, and it can't see an age shift at all.

## Limitations

- One small model, synthetic charts, a single task. Nothing here is clinical evidence.
- A careful rule (normalize formatting, compute age from date of birth) would solve this
  synthetic task perfectly. The point is not to beat rules but to compare what the model
  registers with what it says.
- Decoding shows that information is present, not that the model uses it. Only an
  intervention, such as steering along the probe direction, could show that.
- The probe is trained on the input's label, so "the probe caught it" means the
  activations separate tampered from untampered charts. It doesn't show that the model
  believes something is wrong.
- *Add what the results reveal: scorer agreement with hand-coding, and any condition
  where the model almost never flags.*

## Reproduce

Open [`notebooks/run_pipeline.ipynb`](notebooks/run_pipeline.ipynb) in Colab with a T4
GPU and run the cells in order. It takes about an hour. To work locally instead:

```bash
pip install -e ".[model,dev]"
pytest
```

| Step | Module |
|---|---|
| Generate charts | `evidence_probe.generate` |
| Text controls and rule baseline | `evidence_probe.baselines` |
| Record activations | `evidence_probe.extract` |
| Model behavior and flag scoring | `evidence_probe.behavior` |
| Probes, null and figure 1 | `evidence_probe.probes` |
| Inside vs behavior and figure 2 | `evidence_probe.gap` |

The reasoning behind each design choice is in [docs/design_notes.md](docs/design_notes.md).

## Reference

Klang E, Glicksberg BS, Gorenshtein A, Gavin N, Freeman R, Stump L, Charney AW, Ting DSW,
Omar M, Nadkarni GN. Clinical agents fail silently on patient identity. *International
Journal of Medical Informatics*. 2026;218:106514.
[doi:10.1016/j.ijmedinf.2026.106514](https://doi.org/10.1016/j.ijmedinf.2026.106514).
Preprint: [medRxiv](https://www.medrxiv.org/content/10.1101/2025.10.17.25338226).
