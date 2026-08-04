"""Small, deterministic helpers for scientific material-name matching."""

import re


_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def material_aliases(material: str) -> tuple[str, ...]:
    """Return bounded aliases for phase and polytype-qualified names."""

    canonical = material.strip()
    if not canonical:
        raise ValueError("Material name must not be blank.")
    aliases = [canonical]
    phase_free = re.sub(r"^(?:alpha|beta|[αβ])-", "", canonical, flags=re.I)
    polytype_free = re.sub(r"^\d+[A-Za-z]-", "", phase_free)
    for alias in (phase_free, polytype_free):
        if alias and alias not in aliases:
            aliases.append(alias)
    return tuple(aliases)


def text_mentions_material(text: str, material: str) -> bool:
    """Match aliases with alphanumeric boundaries after subscript normalization."""

    normalized_text = text.translate(_SUBSCRIPT_TRANSLATION).replace("β-", "beta-")
    return any(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(alias.translate(_SUBSCRIPT_TRANSLATION))}"
            r"(?![A-Za-z0-9])",
            normalized_text,
            flags=re.I,
        )
        is not None
        for alias in material_aliases(material)
    )
