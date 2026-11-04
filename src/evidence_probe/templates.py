"""Text building blocks for the synthetic scenarios.

Each TemplateFamily is one unit of held-out validation, the way a participant is
in a neural decoding study. Families differ in format and wording, so a probe
that generalises across them is not just memorising one family's phrasing.

Rules these frames have to follow, because the labels depend on them:
  * Stop and continue frames are unambiguous directives about the named drug.
  * Neutral frames mention the drug but say nothing about whether to keep taking it.
    Avoid wording such as "tolerating well", which implies an answer.
  * Headers put every note at the same time, and sources are clinicians of similar
    standing. Otherwise "the later note wins" or "the specialist wins" would resolve
    a contradiction and the label would be wrong.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateFamily:
    name: str
    header: str
    stop: tuple[str, ...]
    cont: tuple[str, ...]
    neutral: tuple[str, ...]
    questions: tuple[str, ...]
    sources: tuple[str, ...]


# Chronic oral medications with no strong default answer to "keep taking it?".
# Roles are assigned to drugs at random, and drugs are held out in validation, so a
# probe cannot score by learning "people stay on metformin".
DRUGS = (
    "lisinopril", "metformin", "atorvastatin", "amlodipine", "sertraline", "levothyroxine",
    "losartan", "omeprazole", "gabapentin", "metoprolol", "simvastatin", "escitalopram",
    "pantoprazole", "furosemide", "bupropion", "rosuvastatin", "carvedilol", "duloxetine",
    "spironolactone", "mirtazapine", "pravastatin", "trazodone", "venlafaxine", "valsartan",
)

_CLINICIANS = (
    "Dr. Reyes", "Dr. Okafor", "Dr. Lindqvist", "Dr. Tanaka", "Dr. Haddad",
    "Dr. Novak", "Dr. Mensah", "Dr. Castillo",
)

FAMILIES = (
    TemplateFamily(
        name="chart_bullets",
        header="Chart notes, all entered during today's visit:",
        stop=("- {source}: stop {drug}.", "- {source}: discontinue {drug} now."),
        cont=("- {source}: continue {drug}.", "- {source}: keep {drug} unchanged."),
        neutral=("- {source}: {drug} reviewed.", "- {source}: {drug} on med list."),
        questions=("Should the patient keep taking {drug}?", "Should the patient stay on {drug}?"),
        sources=_CLINICIANS,
    ),
    TemplateFamily(
        name="narrative",
        header="The patient saw several clinicians this morning. Their advice was as follows.",
        stop=(
            "{source} told the patient to stop taking {drug}.",
            "{source} advised the patient to come off {drug}.",
        ),
        cont=(
            "{source} told the patient to keep taking {drug}.",
            "{source} advised the patient to remain on {drug}.",
        ),
        neutral=(
            "{source} asked the patient about {drug}.",
            "{source} wrote down the patient's {drug} dose.",
        ),
        questions=(
            "Given all of this, should the patient keep taking {drug}?",
            "What should the patient do: keep taking {drug} or not?",
        ),
        sources=_CLINICIANS,
    ),
    TemplateFamily(
        name="portal_messages",
        header="Patient portal messages, all sent at 9:00 AM today:",
        stop=(
            'Message from {source}: "Please stop your {drug}."',
            'Message from {source}: "You should no longer take {drug}."',
        ),
        cont=(
            'Message from {source}: "Please continue your {drug}."',
            'Message from {source}: "You should still take {drug}."',
        ),
        neutral=(
            'Message from {source}: "I see {drug} in your file."',
            'Message from {source}: "Your question about {drug} was received."',
        ),
        questions=(
            "Should I keep taking {drug}?",
            "Am I supposed to stay on {drug}?",
        ),
        sources=_CLINICIANS,
    ),
    TemplateFamily(
        name="discharge_orders",
        header="Discharge medication orders, signed together at time of discharge:",
        stop=(
            "ORDER ({source}): DISCONTINUE {drug}.",
            "ORDER ({source}): HOLD {drug} indefinitely.",
        ),
        cont=(
            "ORDER ({source}): CONTINUE {drug}.",
            "ORDER ({source}): RESUME {drug} at home.",
        ),
        neutral=(
            "ENTRY ({source}): {drug} verified in record.",
            "ENTRY ({source}): {drug} home supply noted.",
        ),
        questions=(
            "After discharge, should the patient keep taking {drug}?",
            "Should {drug} be taken after discharge?",
        ),
        sources=_CLINICIANS,
    ),
    TemplateFamily(
        name="handoff",
        header="Shift handoff. Everything below was said at the same 7 PM handoff.",
        stop=(
            "{source} says to take the patient off {drug}.",
            "{source} wants {drug} stopped.",
        ),
        cont=(
            "{source} says to leave the patient on {drug}.",
            "{source} wants {drug} continued.",
        ),
        neutral=(
            "{source} says the patient brought in {drug}.",
            "{source} mentioned {drug} during rounds.",
        ),
        questions=(
            "Based on this handoff, should the patient keep taking {drug}?",
            "Does the patient stay on {drug}?",
        ),
        sources=_CLINICIANS,
    ),
    TemplateFamily(
        name="med_reconciliation",
        header="Medication reconciliation worksheet, completed in a single session:",
        stop=(
            "{drug} | reviewer: {source} | action: stop",
            "{drug} | reviewer: {source} | action: discontinue",
        ),
        cont=(
            "{drug} | reviewer: {source} | action: continue",
            "{drug} | reviewer: {source} | action: no change",
        ),
        neutral=(
            "{drug} | reviewer: {source} | action: none recorded",
            "{drug} | reviewer: {source} | action: blank",
        ),
        questions=(
            "According to the worksheet, should the patient keep taking {drug}?",
            "Is the patient meant to continue {drug}?",
        ),
        sources=_CLINICIANS,
    ),
    TemplateFamily(
        name="phone_log",
        header="Clinic phone log. All calls took place within the same hour.",
        stop=(
            "Call with {source}: patient instructed to quit {drug}.",
            "Call with {source}: patient told {drug} should end.",
        ),
        cont=(
            "Call with {source}: patient instructed to stick with {drug}.",
            "Call with {source}: patient told {drug} should carry on.",
        ),
        neutral=(
            "Call with {source}: patient asked where to buy {drug}.",
            "Call with {source}: patient spelled {drug} for the record.",
        ),
        questions=(
            "From this log, should the patient keep taking {drug}?",
            "Should the patient carry on with {drug}?",
        ),
        sources=_CLINICIANS,
    ),
    TemplateFamily(
        name="visit_summary",
        header="After-visit summary. All recommendations were made at today's joint appointment.",
        stop=(
            "Recommendation from {source}: patient should stop {drug}.",
            "Recommendation from {source}: {drug} is no longer needed.",
        ),
        cont=(
            "Recommendation from {source}: patient should continue {drug}.",
            "Recommendation from {source}: {drug} is still needed.",
        ),
        neutral=(
            "Comment from {source}: patient's chart includes {drug}.",
            "Comment from {source}: pharmacy has {drug} on file.",
        ),
        questions=(
            "According to this summary, should the patient keep taking {drug}?",
            "Is the patient supposed to keep using {drug}?",
        ),
        sources=_CLINICIANS,
    ),
)
