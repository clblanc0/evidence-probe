# Design notes

The choices you'll be asked to defend, and why each was made.

## The data

**Six versions of every chart.** Each base chart is identical except for the new
visit's identity line. Any difference a probe picks up has to come from that line and
how it relates to the rest of the chart. It's a minimal-pair design, like contrasting two
stimuli that differ in one feature.

**Benign variants as decoys.** Real charts write the same person in different ways.
Without benign variants, "tampered" and "looks different" would be the same thing, and
a probe could score well by detecting any change. The research plan calls these
"legitimate same-patient differences".

**Formatting balanced across labels.** In each base, the three tampered rows get plain,
name-reformatted and MRN-dashed formatting, rotated across bases. The first version
skipped this, and the new-visit-only baseline caught the leak (AUC 0.64). Keep that
story: it's a real example of a control doing its job.

**Tampering types.** These follow the research plan and the Klang et al. study:
- **Patient swap:** another patient's name, MRN and age. The easiest to see.
- **One-digit MRN change:** the subtle case the study found agents almost always miss.
- **Age shift:** of at least 4 years, so it's a real inconsistency with the date of
  birth and not a birthday-boundary rounding.

**Six chart layouts, held out in turn.** A probe is always tested on a layout it never
saw, the way a decoder is tested on a held-out participant.

## The controls

| Control | What chance would mean |
|---|---|
| Bag of words, whole prompt | Wording alone doesn't give tampering away. It scores 0.55, likely because a tampered row's identity tokens repeat less across the chart. That's a faint trace of the mismatch itself, so it's the floor to beat. |
| New visit only | No single visit looks tampered without comparing it to the chart. |
| Shuffled labels within each base chart | The size of an effect that noise alone can produce, with a max statistic across layers so choosing the peak layer is already accounted for. |
| Held-out tampering type | Tests the shortcut concern from the research plan. The unit test shows the logic: a planted signal shared across types transfers, and a type-specific one doesn't. |

## The model side

**Model.** Qwen2.5-1.5B-Instruct: ungated and runs on a free Colab T4.

**Two recording positions.**
- **`visit_end`:** the last token of the new visit, right after the model reads the
  identity line. Nothing later in the prompt, the task instruction included, can affect
  it. It's closest to "as it reads each visit" in the research plan.
- **`final`:** the last prompt token, just before the model answers. It's the same token
  in every prompt, so layer 0 must be identical everywhere, which is a free pipeline check.

**Two prompts.**
- **`neutral`:** asks only for a chart entry, the way an agent is asked to file.
- **`check`:** explicitly asks for an identity check.

The neutral prompt matches the study's setting. The check prompt shows whether the model
can catch tampering at all when asked.

**What counts as a flag.** Any mention of an identity problem anywhere in the response,
because a clinician reading "Filed. Note: MRN differs." would be warned. The regex has
not seen real output yet, so read the pilot responses and fix it, then freeze it.
Hand-code 100 responses blind and report agreement.

## Reading the results

**The gap figure.** For each tampering type, compare the probe's catch rate with the
model's flag rate. On the untampered conditions, the bars are false alarms, so a probe
that flags everything is not a win.

**What "the probe caught it" means.** The probe learned to separate tampered from
untampered inputs. That shows the information is present in the activations. It doesn't
show the model "noticed" in any stronger sense. The within-type test in `gap.py` gets
closer: among rows of the same tampering type, does the probe's score predict which ones
the model flags? It needs at least 8 rows of each outcome, and with a model that rarely
flags it will report that it can't be tested. That's a result to report.

**Choose the layer before looking at behavior.** The notebook makes you set it after
figure 1 and before figure 2.

## Not built

Steering ("stimulate the model along that direction" in the research plan). Add it
only if the probe works and time allows. It's the only analysis here that could show
the model uses this signal.
