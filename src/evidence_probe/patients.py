"""Building blocks for synthetic patient charts. Every name, number and note is invented.

Each ChartFormat is one unit of held-out validation, like a participant in a decoding
study: a probe has to work on chart layouts it never saw during training.
"""

import random
from dataclasses import dataclass
from datetime import date

FIRST_NAMES = {
    "F": ("Maria", "Aisha", "Elena", "Grace", "Priya", "Hannah", "Lucia", "Naomi", "Sofia",
          "Yuki", "Fatima", "Claire", "Ines", "Ruth", "Mei", "Olivia"),
    "M": ("James", "Omar", "Daniel", "Kenji", "Mateo", "Samuel", "Arjun", "Thomas", "Luca",
          "Kwame", "David", "Hassan", "Pablo", "Ethan", "Victor", "Noah"),
}
LAST_NAMES = (
    "Gomez", "Okafor", "Lindqvist", "Tanaka", "Haddad", "Novak", "Mensah", "Castillo",
    "Bennett", "Kowalski", "Nguyen", "Abara", "Fischer", "Moreau", "Patel", "Sullivan",
    "Rossi", "Andersen", "Delgado", "Kim", "Osei", "Brennan", "Varga", "Ibrahim",
)
REASONS = (
    "hypertension follow-up", "type 2 diabetes check", "persistent cough", "knee pain",
    "medication review", "chest tightness on exertion", "low back pain", "fatigue",
    "annual physical", "headaches", "shortness of breath", "dizziness",
)
PLANS = (
    "Continue current medications.", "Order basic metabolic panel.", "Refer to physical therapy.",
    "Recheck in 3 months.", "Start home blood pressure log.", "Order chest X-ray.",
    "Adjust dose and recheck in 6 weeks.", "Order HbA1c.",
)


@dataclass(frozen=True)
class Patient:
    first: str
    last: str
    sex: str
    dob: date
    mrn: str

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


@dataclass(frozen=True)
class ChartFormat:
    name: str
    chart_header: str  # Uses {name} {mrn} {dob} {sex}.
    visit: str  # Uses {date} {name} {mrn} {age} {note}.
    note: str  # Uses {reason} {sbp} {dbp} {hr} {plan}.
    prior_label: str
    new_label: str
    dob_format: str = "%Y-%m-%d"


FORMATS = (
    ChartFormat(
        name="labeled",
        chart_header="PATIENT: {name}\nMRN: {mrn}\nDOB: {dob}\nSEX: {sex}",
        visit="Visit {date}\nPatient: {name} | MRN: {mrn} | Age: {age}\n{note}",
        note="Reason: {reason}. BP {sbp}/{dbp}, HR {hr}. Plan: {plan}",
        prior_label="PRIOR VISITS", new_label="NEW VISIT TO FILE",
    ),
    ChartFormat(
        name="key_value",
        chart_header="patient_name={name}\nmrn={mrn}\ndob={dob}\nsex={sex}",
        visit="[visit date={date}]\nname={name}; mrn={mrn}; age={age}\n{note}",
        note="reason={reason}; bp={sbp}/{dbp}; hr={hr}; plan={plan}",
        prior_label="## history", new_label="## incoming visit",
    ),
    ChartFormat(
        name="narrative",
        chart_header="This is the chart of {name}, medical record number {mrn}, born {dob}, sex {sex}.",
        visit="On {date}, {name} (MRN {mrn}), age {age}, was seen. {note}",
        note="The visit was for {reason}. Blood pressure was {sbp}/{dbp} and heart rate {hr}. {plan}",
        prior_label="Earlier visits:", new_label="The visit below has just arrived and needs filing:",
        dob_format="%B %d, %Y",
    ),
    ChartFormat(
        name="soap",
        chart_header="Name: {name}   MRN#: {mrn}   DOB: {dob}   Sex: {sex}",
        visit="DATE OF SERVICE: {date}\nPt: {name}, {age}yo, MRN# {mrn}\n{note}",
        note="S: {reason}.\nO: BP {sbp}/{dbp}, HR {hr}.\nP: {plan}",
        prior_label="=== PREVIOUS NOTES ===", new_label="=== NOTE PENDING FILING ===",
        dob_format="%m/%d/%Y",
    ),
    ChartFormat(
        name="table",
        chart_header="| Name | MRN | DOB | Sex |\n| {name} | {mrn} | {dob} | {sex} |",
        visit="| {date} | {name} | {mrn} | age {age} |\n{note}",
        note="{reason}; BP {sbp}/{dbp}; HR {hr}; {plan}",
        prior_label="Encounter log:", new_label="Encounter to add:",
    ),
    ChartFormat(
        name="discharge",
        chart_header="PATIENT IDENTIFICATION\nName: {name}\nRecord no.: {mrn}\nDate of birth: {dob}\nSex: {sex}",
        visit="Encounter {date}: {name}, record no. {mrn}, {age} years old\n{note}",
        note="Presented with {reason}. Vitals: BP {sbp}/{dbp}, HR {hr}. Disposition: {plan}",
        prior_label="PAST ENCOUNTERS", new_label="ENCOUNTER AWAITING SIGN-OFF",
        dob_format="%d %b %Y",
    ),
)


def age_on(dob: date, when: date) -> int:
    return when.year - dob.year - ((when.month, when.day) < (dob.month, dob.day))


def random_patient(rng: random.Random, exclude: Patient | None = None) -> Patient:
    while True:
        sex = rng.choice("FM")
        patient = Patient(
            first=rng.choice(FIRST_NAMES[sex]),
            last=rng.choice(LAST_NAMES),
            sex=sex,
            dob=date(rng.randint(1945, 2000), rng.randint(1, 12), rng.randint(1, 28)),
            mrn="".join(rng.choice("0123456789") for _ in range(7)),
        )
        if exclude is None or (patient.first != exclude.first and patient.last != exclude.last
                               and patient.mrn != exclude.mrn):
            return patient


def random_note(fmt: ChartFormat, rng: random.Random) -> str:
    return fmt.note.format(
        reason=rng.choice(REASONS), sbp=rng.randint(108, 162), dbp=rng.randint(64, 98),
        hr=rng.randint(56, 102), plan=rng.choice(PLANS),
    )
