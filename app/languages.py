import unicodedata


def detect_language(text: str) -> str:
    devanagari = 0
    tamil = 0
    for character in text:
        name = unicodedata.name(character, "")
        if "DEVANAGARI" in name:
            devanagari += 1
        elif "TAMIL" in name:
            tamil += 1
    if tamil > devanagari:
        return "ta"
    if devanagari:
        return "hi"
    return "en"
