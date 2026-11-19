"""Write notebooks/run_pipeline.ipynb. Edit this file, not the notebook, then rerun it."""

from pathlib import Path

import nbformat as nbf

CELLS = [
    ("md", """# Do models register patient mix-ups internally?

A weekend pilot. Runs the whole study on a free Colab GPU in about an hour.

**Before you start:** Runtime → Change runtime type → **T4 GPU**.

The cells run the steps in order. Read the pilot output (step 2) before running anything
after it. If the model never flags anything, or flags everything, the later steps can't
tell you much, and it's better to know that now."""),

    ("md", "## 0. Get the code\n\nIf the repo is private, add a GitHub token with read access as a Colab secret named `GITHUB_TOKEN` (key icon on the left). If it's public, the token isn't needed."),
    ("code", """import os, subprocess
REPO = "clblanc0/evidence-probe"
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
    url = f"https://{token}@github.com/{REPO}.git"
except Exception:
    url = f"https://github.com/{REPO}.git"
if not os.path.exists("evidence-probe"):
    subprocess.run(["git", "clone", "-q", url], check=True)
%cd evidence-probe
!pip install -q -e ".[model,dev]"
!git log --oneline -1"""),

    ("md", "## 1. Check the data and the text-only controls\n\nRuns in seconds and doesn't touch the model. The two text baselines should sit near 0.5. They're the floor the probe has to beat."),
    ("code", """!python -m evidence_probe.generate --out data/charts.jsonl
!python -m evidence_probe.baselines data/charts.jsonl
!python -m pytest -q"""),

    ("md", "## 2. Pilot: does the model ever flag anything?\n\nAbout 5 minutes. Two prompts: `neutral` just asks for a chart entry, the way an agent would be asked to file. `check` explicitly asks the model to confirm identity first."),
    ("code", """MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
for variant in ["neutral", "check"]:
    !python -m evidence_probe.behavior data/charts.jsonl --model {MODEL} --variant {variant} --limit-bases 12 --out results/pilot_{variant}.jsonl"""),
    ("code", """# Read real responses before trusting any rate. Edit the scorer in behavior.py if it misreads them.
import json
for line in open("results/pilot_neutral.jsonl").readlines()[:12]:
    r = json.loads(line)
    print(f"[{r['condition']:11}] flagged={r['flagged']!s:5} | {r['response'][:160]!r}")"""),

    ("md", """**Go or no-go.** Continue if, under at least one prompt:
- the model flags some tampered rows, but not nearly all rows, clean ones included, and
- reading the responses, the scorer's `flagged` matches what you would say.

If the model flags nothing under either prompt, that's still a finding and the probe is still worth running: it asks whether the activations register what the behavior misses."""),

    ("md", "## 3. Record activations\n\nAbout 10 minutes. One forward pass per chart, saving every layer at two points: the end of the new visit, and the final prompt token."),
    ("code", """VARIANT = "neutral"  # Use the prompt you'll report behavior for.
!python -m evidence_probe.extract data/charts.jsonl --model {MODEL} --variant {VARIANT} --out activations/{VARIANT}.npz"""),

    ("md", "## 4. Probes: accuracy by layer, controls, held-out tampering types\n\nAbout 10–20 minutes, most of it the shuffled-label null. This makes figure 1."),
    ("code", """for position in ["visit_end", "final"]:
    !python -m evidence_probe.probes data/charts.jsonl activations/{VARIANT}.npz --position {position} --out-dir results --n-perm 100"""),
    ("code", """from IPython.display import Image, display
display(Image("results/fig1_auc_by_layer_visit_end.png"))
display(Image("results/fig1_auc_by_layer_final.png"))"""),

    ("md", "**Choose the layer now**, from figure 1, before looking at behavior. Write down why. The gap analysis uses it."),
    ("code", """LAYER = None  # Set from figure 1, e.g. the peak of the blue line at visit_end.
assert LAYER is not None, "Choose the layer from figure 1 first." """),

    ("md", "## 5. Behavior on all 540 charts\n\nAbout 15–25 minutes per prompt."),
    ("code", """!python -m evidence_probe.behavior data/charts.jsonl --model {MODEL} --variant {VARIANT} --out results/behavior_{VARIANT}.jsonl --export-blind results/hand_code_{VARIANT}.csv"""),

    ("md", "## 6. Inside vs behavior\n\nThis makes figure 2: for each condition, how often the probe caught it, how often the model flagged it, and how often a simple string rule would have."),
    ("code", """!python -m evidence_probe.gap results/behavior_{VARIANT}.jsonl results/probe_visit_end_oof.npz --layer {LAYER} --out-dir results
display(Image(f"results/fig2_inside_vs_behavior_{VARIANT}.png"))"""),

    ("md", "## 7. Save the results\n\nActivations are large and can be regenerated, so only `results/` is kept. Download it, or copy it to Drive."),
    ("code", """!zip -qr results.zip results
from google.colab import files
files.download("results.zip")"""),

    ("md", """## Then, by hand

1. Open `results/hand_code_neutral.csv`. Code each response 1 (flagged an identity problem) or 0 **without looking up its condition**. Compare with the scorer and report agreement.
2. Fill in the README's results and limitations."""),
]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"}}
    notebook.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                      for kind, src in CELLS]
    out = Path(__file__).resolve().parents[1] / "notebooks" / "run_pipeline.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(notebook, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
