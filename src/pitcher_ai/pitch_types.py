from __future__ import annotations

from collections import OrderedDict
from typing import Any

PITCH_CLASS = "pitch_class"
PITCH_TYPE_NAME = "pitch_type_name"

PITCH_CLASS_TO_CODES: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("Fastball", ["FF", "SI", "FC", "FT", "FA"]),
        ("Breaking", ["SL", "ST", "CU", "KC", "SV"]),
        ("Offspeed", ["CH", "FS", "FO"]),
        ("Other/Rare", ["KN", "EP", "SC", "GY", "PO", "IN", "NP", "AB", "AS", "UN"]),
    ]
)

PITCH_CLASS_ORDER = list(PITCH_CLASS_TO_CODES.keys())

PITCH_TYPE_TO_CLASS = {
    code: pitch_class
    for pitch_class, codes in PITCH_CLASS_TO_CODES.items()
    for code in codes
}

PITCH_TYPE_TO_NAME = {
    "FF": "Four-seam fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "FT": "Two-seam fastball",
    "FA": "Fastball",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle-curve",
    "SV": "Slurve",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "SC": "Screwball",
    "GY": "Gyroball",
    "PO": "Pitchout",
    "IN": "Intentional ball",
    "NP": "No pitch",
    "AB": "Automatic ball",
    "AS": "Automatic strike",
    "UN": "Unknown",
}


def normalize_pitch_type(value: Any) -> str:
    if value is None:
        return "UN"
    text = str(value).strip().upper()
    if not text or text in {"<NA>", "NAN", "NONE", "NULL"}:
        return "UN"
    return text


def pitch_type_to_class(value: Any) -> str:
    return PITCH_TYPE_TO_CLASS.get(normalize_pitch_type(value), "Other/Rare")


def pitch_type_to_name(value: Any) -> str:
    code = normalize_pitch_type(value)
    return PITCH_TYPE_TO_NAME.get(code, f"Unknown/other pitch type ({code})")


def pitch_type_label(value: Any) -> str:
    code = normalize_pitch_type(value)
    return f"{code} - {pitch_type_to_name(code)}"


def pitch_type_key_rows() -> list[dict[str, str]]:
    codes = sorted(PITCH_TYPE_TO_NAME)
    return [
        {
            "pitch_type": code,
            "pitch_name": pitch_type_to_name(code),
            "pitch_class": pitch_type_to_class(code),
        }
        for code in codes
    ]
