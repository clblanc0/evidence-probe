# Design notes

The choices you'll be asked to defend, and where this differs from the original plan.

## Differences from the original plan

### 1. Triplets instead of separately written scenarios

The plan generates a scenario per label, then checks a bag-of-words baseline and length.
That leaves the baseline as a test the design can fail. Here it can't leak by
construction, because all three labels share one context.

The cost: a design this clean needs a positive control, or a chance baseline proves
nothing. That's why the naive dataset exists. It shows the baseline does catch a leak
when there is one.

### 2. "Insufficient" means the drug is mentioned but no instruction is given

The natural version of "the needed fact is missing" leaves the drug out entirely. Then
the mention count alone gives the label (0, 1 or 2 mentions), and a model could look
informed just by noticing whether a string appeared. Here every target drug appears
exactly twice. Absent-drug items could be added later as a separate, easier condition.

### 3. A third text baseline as a reference, not a floor

The local-window baseline reads only the sentences naming the drug. It scores 0.96 when
it has seen the template and 0.54 when it hasn't. So:

- Beating 0.33 means the probe isn't reading whole-prompt word counts.
- Beating about 0.54 on new templates means it transfers across wording better than a
  simple cue-word reader does. This is the more interesting bar.

### 4. The behavior gap needs a within-label test

This is the biggest change. The plan's key cell is "probe says insufficient, model
answers anyway". But the probe is trained on the input's label. If it's accurate, that
cell is almost the same as "the input is insufficient and the model answers", which is
a plain behavioral error rate that needs no activations.

The test that uses the activations: **among items with the same true label, does the
probe's score predict whether the model abstains?** That's `within_label_auc` in
`gap.py`. It may come back null or be untestable if the model rarely abstains. Report
that as it stands.

Be careful with the claim "the model represents its uncertainty but doesn't say it".
What this design can support is narrower: "information that distinguishes these evidence
states is linearly decodable at layer N, and the model answers anyway."

### 5. Also holding out drugs

Models have priors about specific drugs. Holding out templates alone would let a probe
learn drug identity, so test rows use both unseen templates and unseen target drugs.
Drug-to-role assignment is random, and this split makes that protection explicit. It
costs training data: each fold trains on about 44% of rows.

## Choices worth stating aloud

**Model.** Qwen2.5-1.5B-Instruct is the default because it's ungated. Llama and Gemma
need a license acceptance on Hugging Face first. Gemma remains the pick if you want
Gemma Scope for the stretch goal.

**Final token.** It's the same token in every prompt, so layer 0 is identical across
prompts and a probe there must land near chance. That's a free check on the pipeline.
Expect it at or slightly below 0.33: tiny class imbalances in training folds push a
constant-input classifier toward the class that's scarcer in the test fold.

**Same-time headers, same-rank clinicians.** Otherwise "the later note wins" or "the
specialist wins" would resolve a contradiction, and the label would be wrong. This is a
real departure from MedDelta-style timelines, where order carries meaning.

**Shuffled labels within context.** This preserves the data's structure, like a
within-participant surrogate. The threshold is a max statistic across layers, so choosing
the peak layer doesn't inflate significance.

**Layer selection for the gap.** Choose the layer from the accuracy curve, before
looking at behavior. Accuracy doesn't use behavior, so this isn't double dipping for the
within-label test, though the peak accuracy itself is mildly optimistic.

**Scoring on the first sentence.** "Yes, keep taking it. However, the records conflict"
counts as an answer, because the first sentence is what a reader acts on. That's a
judgment call. Hand-code the sample from `--export-blind` without looking at labels and
report Cohen's kappa. The regexes are written for expected phrasing and have never seen
real model output, so expect to revise them after the pilot. Do that on the pilot, then
freeze them before the full run.

**Greedy decoding.** Behavior is reproducible and maps to one activation. Abstention
under sampling is a possible extension.

## Go or no-go after the pilot

Check these on the pilot before running anything else:

| Check | If it fails |
|---|---|
| Correct on answered sufficient items is clearly above 0.5 | The model isn't resolving which drug is which. Its probes may still decode something, but "it knows" is hard to claim. Try a larger model. |
| Abstains on at least 10-15% of contradictory and insufficient items | Empty cells. Try `--variant permission`. If it now abstains, that contrast is a finding in itself. |
| Doesn't abstain on nearly everything, including sufficient items | Over-hedging, and common in medical prompts. Abstention would then be unrelated to evidence, which is also a finding. |
| "other" is under about 10% | Read those responses and extend the scorer. |

A chance-level probe on a small model is a legitimate outcome. The triplet task is
harder than a naive one, and a 1.5B model may not resolve it. Have a fallback: a larger
model such as Qwen2.5-7B-Instruct, or fewer drugs per context.

## Not built yet

Steering (step 6). Add it only after the gap analysis. It's also the only thing here
that could show the model uses this information.
