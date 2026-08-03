"""Transparent default domain policies for conventional device screening."""

# Elements with no conventionally stable isotopes that are excluded from the
# default, non-nuclear semiconductor-device workflow. The policy is explicit so
# it can be audited and made configurable instead of being hidden in ranking.
DEFAULT_RADIOACTIVE_ELEMENT_EXCLUSIONS = (
    "Tc",
    "Pm",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

DEFAULT_MAX_ENERGY_ABOVE_HULL_EV_ATOM = 0.1
