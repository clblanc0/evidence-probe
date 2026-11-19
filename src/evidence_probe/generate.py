"""Generate synthetic patient charts where the visit to be filed may be tampered.

Each base chart (one patient, two prior visits, one new visit) produces six rows. All six
share every character except the identity line of the new visit:

    clean        the patient's own name, MRN and correct age          label 0
    benign_name  same person, name written another way ("GOMEZ, MARIA")  label 0
    benign_mrn   same MRN digits, formatted with a dash ("004-8213")      label 0
    swap         another patient's name, MRN and age                      label 1
    mrn_digit    one digit of the MRN changed                             label 1
    age_shift    age that doesn't fit the date of birth                   label 1

The benign rows are the decoy. A probe that flags "anything looks different" fails on
them, so they separate detecting a mismatch from detecting a change in wording.

The tampered rows get the same formatting mix: in each base, one tampered row is plain,
one has its name reformatted and one has its MRN dashed, rotating across bases. Without
that, formatting alone would mean "not tampered". The first version of this generator
had exactly that leak, and the new-visit-only baseline caught it (AUC 0.64).

Usage:
    python -m evidence_probe.generate --per-format 15 --out data/charts.jsonl
"""

import argparse
import json
import random
from datetime import date
from pathlib import Path

from evidence_probe.patients import FORMATS, ChartFormat, Patient, age_on, random_note, random_patient

CONDITIONS = ("clean", "benign_name", "benign_mrn", "swap", "mrn_digit", "age_shift")
TAMPERED = ("swap", "mrn_digit", "age_shift")
NOT_TAMPERED = ("clean", "benign_name", "benign_mrn")


def change_one_digit(mrn: str, rng: random.Random) -> str:
    position = rng.randrange(len(mrn))
    new_digit = rng.choice([d for d in "0123456789" if d != mrn[position]])
    return mrn[:position] + new_digit + mrn[position + 1:]


def benign_name(patient: Patient, rng: random.Random) -> str:
    return rng.choice((
        f"{patient.last.upper()}, {patient.first.upper()}",
        f"{patient.last}, {patient.first}",
        f"{patient.first[0]}. {patient.last}",
    ))


def shifted_age(true_age: int, rng: random.Random) -> int:
    shift = rng.choice([s for s in range(-12, 13) if abs(s) >= 4])
    return true_age + shift if true_age + shift >= 18 else true_age + abs(shift)


SURFACES = ("plain", "name_format", "mrn_format")
UNTAMPERED_SURFACE = {"clean": "plain", "benign_name": "name_format", "benign_mrn": "mrn_format"}


def identity_for(condition: str, surface: str, patient: Patient, other: Patient,
                 visit_date: date, rng: random.Random) -> dict:
    """Name, MRN and age shown on the new visit for one condition and formatting."""
    person = other if condition == "swap" else patient
    mrn = person.mrn
    if condition == "mrn_digit":
        mrn = change_one_digit(mrn, rng)
    age = age_on(person.dob, visit_date)
    if condition == "age_shift":
        age = shifted_age(age, rng)
    name = benign_name(person, rng) if surface == "name_format" else person.name
    if surface == "mrn_format":
        mrn = f"{mrn[:3]}-{mrn[3:]}"
    return {"name": name, "mrn": mrn, "age": age}


def build_base(fmt: ChartFormat, index: int, rng: random.Random) -> list[dict]:
    patient = random_patient(rng)
    other = random_patient(rng, exclude=patient)
    visit_dates = sorted(date(year, rng.randint(1, 12), rng.randint(1, 28)) for year in (2023, 2024))
    new_date = date(2026, rng.randint(1, 10), rng.randint(1, 28))

    header = fmt.chart_header.format(
        name=patient.name, mrn=patient.mrn, dob=patient.dob.strftime(fmt.dob_format), sex=patient.sex,
    )
    prior = [
        fmt.visit.format(date=d.isoformat(), name=patient.name, mrn=patient.mrn,
                         age=age_on(patient.dob, d), note=random_note(fmt, rng))
        for d in visit_dates
    ]
    new_note = random_note(fmt, rng)
    chart_before = f"{header}\n\n{fmt.prior_label}\n" + "\n\n".join(prior) + f"\n\n{fmt.new_label}\n"

    base_id = f"{fmt.name}_{index:03d}"
    surfaces = dict(UNTAMPERED_SURFACE)
    for j, condition in enumerate(TAMPERED):
        surfaces[condition] = SURFACES[(index + j) % len(SURFACES)]

    rows = []
    for condition in CONDITIONS:
        identity = identity_for(condition, surfaces[condition], patient, other, new_date, rng)
        new_visit = fmt.visit.format(date=new_date.isoformat(), note=new_note, **identity)
        rows.append({
            "id": f"{base_id}__{condition}",
            "base_id": base_id,
            "template": fmt.name,
            "condition": condition,
            "label": int(condition in TAMPERED),
            "surface": surfaces[condition],
            "chart": chart_before + new_visit,
            "new_visit": new_visit,
            # Ground truth for the rule baseline and error analysis. Never given to a probe.
            "true_name": patient.name,
            "true_mrn": patient.mrn,
            "true_age": age_on(patient.dob, new_date),
            "shown_name": identity["name"],
            "shown_mrn": identity["mrn"],
            "shown_age": identity["age"],
        })
    return rows


def generate(per_format: int = 15, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [row for fmt in FORMATS for i in range(per_format) for row in build_base(fmt, i, rng)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-format", type=int, default=15, help="Base charts per format; 6 rows each.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = generate(args.per_format, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
