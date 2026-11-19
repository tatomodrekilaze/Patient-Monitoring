"""HL7 FHIR R4 export of vital signs, so hospital record systems can import them.

Codes: LOINC 9279-1 respiratory rate, 8867-4 heart rate; units in UCUM.
The patient is referenced by bed id only - no names leave this system.
"""
import time

VITALS = {
    "rr": ("9279-1", "Respiratory rate", "/min"),
    "hr": ("8867-4", "Heart rate", "/min"),
}


def observation(bed_id, ts, key, value, idx):
    code, display, unit = VITALS[key]
    return {
        "resourceType": "Observation",
        "id": f"{bed_id}-{key}-{int(ts)}-{idx}",
        "status": "final",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}], "text": display},
        "subject": {"reference": f"Patient/{bed_id}"},
        "effectiveDateTime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts)),
        "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
        "device": {"display": "Silent Early Warning camera monitor"},
    }


def bundle(vitals_entries, bed_id=None, count=100):
    """vitals_entries: event-log entries of kind 'vitals'."""
    entries = [e for e in vitals_entries if not bed_id or e["bed_id"] == bed_id][-count:]
    obs = []
    for i, e in enumerate(entries):
        for key in VITALS:
            if e.get(key) is not None:
                obs.append(observation(e["bed_id"], e["ts"], key, e[key], i))
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(obs),
        "entry": [{"resource": o} for o in obs],
    }
