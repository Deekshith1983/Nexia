"""Universal Question-Understanding Engine: Extracts Structured Intent, Required Constraints, and Context Rules."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.schemas.retrieval import (
    StructuredQueryRepresentation,
    QueryConstraints,
    GeographicConstraint,
)

# =============================================================================
# Countries
# =============================================================================
_COUNTRY_NAMES_RAW = """
afghanistan, albania, algeria, andorra, angola, antigua and barbuda, argentina, armenia,
australia, austria, azerbaijan,
bahamas, bahrain, bangladesh, barbados, belarus, belgium, belize, benin, bhutan, bolivia,
bosnia and herzegovina, botswana, brazil, brunei, bulgaria, burkina faso, burundi, burma,
cabo verde, cape verde, cambodia, cameroon, canada, central african republic, chad, chile,
china, colombia, comoros, congo, republic of the congo, democratic republic of the congo,
drc, costa rica, cote d'ivoire, côte d'ivoire, ivory coast, croatia, cuba, cyprus,
czech republic, czechia,
denmark, djibouti, dominica, dominican republic,
east timor, timor-leste, ecuador, egypt, el salvador, equatorial guinea, eritrea, estonia,
eswatini, swaziland, ethiopia,
fiji, finland, france,
gabon, gambia, georgia, germany, ghana, greece, grenada, guatemala, guinea, guinea-bissau,
guyana,
haiti, honduras, hungary,
iceland, india, indonesia, iran, iraq, ireland, israel, italy,
jamaica, japan, jordan,
kazakhstan, kenya, kiribati, korea, north korea, south korea, kosovo, kuwait, kyrgyzstan,
laos, latvia, lebanon, lesotho, liberia, libya, liechtenstein, lithuania, luxembourg,
madagascar, malawi, malaysia, maldives, mali, malta, marshall islands, mauritania,
mauritius, mexico, micronesia, moldova, monaco, mongolia, montenegro, morocco, mozambique,
myanmar,
namibia, nauru, nepal, netherlands, holland, new zealand, nicaragua, niger, nigeria,
north macedonia, macedonia, norway,
oman,
pakistan, palau, palestine, panama, papua new guinea, paraguay, peru, philippines, poland,
portugal,
qatar,
romania, russia, rwanda,
saint kitts and nevis, saint lucia, saint vincent and the grenadines, samoa, san marino,
sao tome and principe, saudi arabia, senegal, serbia, seychelles, sierra leone, singapore,
slovakia, slovenia, solomon islands, somalia, south africa, south sudan, spain, sri lanka,
sudan, suriname, sweden, switzerland, syria,
taiwan, tajikistan, tanzania, thailand, togo, tonga, trinidad and tobago, tunisia, turkey,
turkiye, turkmenistan, tuvalu,
uganda, ukraine, united arab emirates, uae, united kingdom, uk, great britain, britain,
england, scotland, wales, united states, united states of america, usa, america, uruguay,
uzbekistan,
vanuatu, vatican city, holy see, venezuela, vietnam,
yemen,
zambia, zimbabwe
"""

# NOTE: "us" is deliberately NOT in this set (it matches the pronoun "us").
# It is detected separately, case-sensitively, as "US" / "U.S." / "USA".
KNOWN_COUNTRIES: Set[str] = {
    c.strip() for c in _COUNTRY_NAMES_RAW.replace("\n", " ").split(",") if c.strip()
}

# lowercase alias -> canonical display name (anything not listed is title-cased)
COUNTRY_ALIASES: Dict[str, str] = {
    "usa": "United States",
    "us": "United States",
    "america": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "uae": "United Arab Emirates",
    "drc": "Democratic Republic of the Congo",
    "holland": "Netherlands",
    "burma": "Myanmar",
    "swaziland": "Eswatini",
    "turkiye": "Turkey",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d'ivoire": "Côte d'Ivoire",
    "cape verde": "Cabo Verde",
    "czech republic": "Czechia",
    "macedonia": "North Macedonia",
    "east timor": "Timor-Leste",
    "korea": "South Korea",
    "holy see": "Vatican City",
}

_SMALL_WORDS = {"of", "the", "and", "in", "on", "for", "de", "la"}


def _smart_title(s: str) -> str:
    """Title-case lowercase text, but never touch text that already has capitals
    (NASA, iPhone, McDonald's) and never produce "Einstein'S"."""
    s = s.strip()
    if not s or any(ch.isupper() for ch in s):
        return s
    out = []
    for i, w in enumerate(s.split()):
        if i > 0 and w in _SMALL_WORDS:
            out.append(w)
        else:
            out.append("-".join(p[:1].upper() + p[1:] for p in w.split("-")))
    return " ".join(out)


def _canon_country(raw: str) -> str:
    key = raw.lower().strip()
    return COUNTRY_ALIASES.get(key) or _smart_title(key)


# Longest names first so "south korea" / "papua new guinea" win over "korea" / "guinea".
_COUNTRY_RE = re.compile(
    r"(?<![\w])("
    + "|".join(re.escape(c) for c in sorted(KNOWN_COUNTRIES, key=len, reverse=True))
    + r")(?![\w])",
    re.IGNORECASE,
)
_US_RE = re.compile(r"(?<![\w.])(?:U\.S\.A\.|U\.S\.|USA|US)(?![\w])")  # case-sensitive

# canonical country -> every lowercase surface form (used to expand retrieval cues)
COUNTRY_VARIANTS: Dict[str, List[str]] = {}
for _name in KNOWN_COUNTRIES | {"us", "u.s."}:
    COUNTRY_VARIANTS.setdefault(_canon_country(_name), []).append(_name)
for _k in COUNTRY_VARIANTS:
    COUNTRY_VARIANTS[_k] = sorted(set(COUNTRY_VARIANTS[_k]) | {_k.lower()})


# =============================================================================
# Relational synonym cues
# =============================================================================
RELATION_SYNONYM_MAP: Dict[str, List[str]] = {
    # 1. BIOGRAPHICAL & IDENTITY
    "birth_date": [
        "born", "birth", "birth date", "date of birth", "dob", "born on",
        "day of birth", "year of birth", "how old",
    ],
    "birth_place": [
        "born in", "birthplace", "place of birth", "native of", "hometown",
        "originated from", "raised in", "childhood home", "early life in",
    ],
    "death_date": [
        "died", "death", "date of death", "dod", "died on", "passed away on",
        "deceased on", "killed on", "demise",
    ],
    "death_place": [
        "died in", "place of death", "passed away in", "killed in", "resting place",
        "buried in", "interred at", "cemetery", "grave", "tomb", "mausoleum",
    ],
    "cause_of_death": [
        "cause of death", "died of", "killed by", "succumbed to", "assassinated by",
        "fatal injury", "manner of death", "autopsy findings",
    ],
    "parent": [
        "parent", "father", "mother", "paternal", "maternal", "raised by",
        "son of", "daughter of", "child of", "parentage", "lineage",
    ],
    "child": [
        "child", "children", "son", "daughter", "offspring", "descendant",
        "issue", "heir",
    ],
    "spouse": [
        "spouse", "husband", "wife", "married to", "partner", "consort",
        "wedding", "marriage", "divorced from", "widow", "widower",
    ],
    "nationality": [
        "nationality", "citizenship", "citizen of", "national of", "country of origin",
        "naturalized", "passport holder",
    ],
    "honorific_title": [
        "title", "honorific", "father of the nation", "founding father", "national title",
        "epithet", "nickname", "sobriquet", "alias", "pater patriae", "monumental title",
    ],

    # 2. EDUCATION & ACADEMIC CREDENTIALS
    "educated_at": [
        "educated", "education", "study", "studied", "school", "university",
        "college", "polytechnic", "academy", "institute", "alma mater",
        "matriculated", "enrolled at", "schooled at", "trained at", "campus",
    ],
    "degree_earned": [
        "degree", "graduated with", "graduated", "bachelor", "master",
        "doctorate", "phd", "diploma", "alumnus", "alumna", "qualification",
        "academic credential", "major in", "dissertation", "thesis",
    ],

    # 3. GOVERNANCE, LEADERSHIP & OCCUPATION
    "head_of_state": [
        "president", "head of state", "monarch", "king", "queen", "emperor",
        "sovereign", "supreme leader", "presidency", "chief executive",
    ],
    "head_of_government": [
        "prime minister", "pm", "premier", "chancellor", "head of government",
        "chief minister", "cm", "first minister", "cabinet leader",
    ],
    "organization_leader": [
        "ceo", "chief executive officer", "chairman", "chairperson", "managing director",
        "head of the company", "company head",
    ],
    "leadership_tenure": [
        "incumbent", "officeholder", "assumed office", "took office", "appointed",
        "sworn in", "elected", "term of office", "administration", "tenure",
        "inaugurated", "reign", "stepped down", "resigned",
    ],
    "occupation": [
        "occupation", "profession", "vocation", "career", "job", "role",
        "line of work", "field of work", "employment", "practitioner",
    ],

    # 4. ORGANIZATIONS, CREATION & FOUNDING
    "founded_by": [
        "founded by", "founder", "founding", "co-founder", "established by",
        "instituted by", "created by", "chartered by", "incorporated by",
        "originated by", "incepted by", "launched by", "father of",
    ],
    "established_date": [
        "founded in", "established in", "founding date", "date of establishment",
        "chartered on", "incorporated in", "inception", "genesis", "launched on",
        "instituted on", "opened in", "formation date",
    ],
    "headquarters": [
        "headquarters", "hq", "head office", "headquartered in", "based in",
        "corporate office", "central office", "main branch", "seat", "registered office",
    ],
    "subsidiary_or_parent_company": [
        "subsidiary", "parent company", "holding company", "owned by", "acquired by",
        "division of", "branch of", "conglomerate", "merged with",
    ],

    # 5. GEOGRAPHY, SPATIAL & DEMOGRAPHICS
    "capital": [
        "capital", "capital city", "seat of government", "administrative capital",
        "national capital", "legislative capital", "judicial capital", "seat of power",
        "county seat", "state capital",
    ],
    "located_in": [
        "located in", "situated in", "found in", "lies in", "geographic location",
        "coordinates", "region", "province", "territory", "district", "part of",
    ],
    "bordering": [
        "borders", "bordered by", "neighboring", "adjacent to", "surrounded by",
        "shares border with", "adjoins", "abutting", "flanked by",
    ],
    "population": [
        "population", "inhabitants", "residents", "demographics", "headcount",
        "citizenry", "census count", "densely populated", "how many people",
    ],
    "area_size": [
        "area", "total area", "surface area", "square kilometers", "sq km",
        "square miles", "sq mi", "acreage", "footprint", "geographic extent",
    ],
    "currency": [
        "currency", "legal tender", "monetary unit", "official currency",
        "coinage", "banknotes", "denominated in",
    ],
    "national_symbol": [
        "national animal", "national anthem", "national bird", "national flower",
        "national flag", "national emblem", "national song", "national language",
    ],

    # 6. SCIENCE, METRICS & EXTREMES
    "largest_extreme": [
        "largest", "biggest", "greatest", "massive", "most massive", "broadest",
        "widest", "heaviest", "highest volume", "maximum diameter", "tallest",
        "longest", "deepest", "maximum capacity", "bulk", "dimensions",
    ],
    "smallest_extreme": [
        "smallest", "tiniest", "minimal", "least massive", "narrowest", "shortest",
        "shallowest", "lowest", "microscopic", "compact",
    ],
    "age_extreme": [
        "oldest", "youngest", "earliest", "newest", "most ancient",
    ],
    "spatial_proximity": [
        "nearest", "closest", "proximity", "distance to", "shortest distance",
        "adjacent", "immediate vicinity", "neighboring", "light-years", "parsecs",
        "astronomical units", "au", "perihelion", "perigee", "aphelion", "apogee",
    ],
    "invented_by": [
        "invented by", "inventor", "invention", "patented by", "patent",
        "devised by", "engineered by", "designed by", "conceived by",
        "formulated by", "developed by", "pioneered by",
    ],
    "discovered_by": [
        "discovered by", "discoverer", "discovery", "first detected by",
        "first observed by", "identified by", "isolated by", "unearthed by",
        "uncovered by", "sighted by", "first documented by",
    ],
    "invention_or_discovery_date": [
        "invented in", "discovered in", "invention date", "discovery date",
        "year of invention", "year of discovery",
    ],

    # 7. CAUSALITY, MECHANISMS & PROCEDURES
    "causes_and_reasons": [
        "cause", "causes", "caused by", "reason for", "reasons", "led to",
        "result of", "factors behind", "origin of", "etiology", "triggered by",
        "catalyst for", "induced by", "precipitated by", "stems from", "source of",
    ],
    "effects_and_outcomes": [
        "effect", "effects", "consequence", "consequences", "outcome", "impact",
        "result", "aftermath", "side effects", "manifestations", "ramifications",
    ],
    "mechanism_of_action": [
        "mechanism", "process", "operation", "function", "how it works",
        "mode of action", "principle of operation", "operational principle",
        "inner workings", "mechanics", "pathway", "reaction sequence",
    ],
    "procedure_steps": [
        "steps", "procedure", "method", "methodology", "protocol", "workflow",
        "stages", "instructions", "algorithm", "technique", "guidelines",
    ],

    # 8. INTELLECTUAL PROPERTY & CREATIVE WORKS
    "author_or_creator": [
        "author", "writer", "written by", "penned by", "composed by", "composer",
        "artist", "sculptor", "architect", "directed by", "director", "produced by",
        "creator", "lyricist", "playwright", "novelist",
    ],
    "release_or_publication_date": [
        "published in", "publication date", "release date", "released on",
        "premiered on", "issued on", "first edition", "air date", "debuted on",
    ],
    "genre_or_classification": [
        "genre", "category", "classification", "type of", "form of", "style",
        "taxonomic rank", "phylum", "genus", "species", "order", "family",
    ],
}

# Verbs that signal a relation even when no "...by" cue is present
# ("who invented the telephone", "who wrote hamlet").
PATTERN_VERB_REL: Dict[str, str] = {
    "invented": "invented_by", "designed": "invented_by", "developed": "invented_by",
    "discovered": "discovered_by",
    "founded": "founded_by", "established": "founded_by", "created": "founded_by",
    "started": "founded_by", "launched": "founded_by", "formed": "founded_by",
    "built": "author_or_creator", "wrote": "author_or_creator", "written": "author_or_creator",
    "directed": "author_or_creator", "composed": "author_or_creator",
    "painted": "author_or_creator", "authored": "author_or_creator",
    "produced": "author_or_creator",
    "married": "spouse",
    "ruled": "leadership_tenure",
}
_DETECT_VERBS = {"invented", "discovered", "founded", "wrote", "directed",
                 "composed", "painted", "authored", "married"}

# Cues too generic to trigger a relation by substring match on their own.
NOISY_CUES: Set[str] = {
    "order", "family", "issue", "role", "master", "au", "result", "style",
    "form of", "type of", "region", "part of", "bulk", "compact", "minimal",
    "massive", "grave", "seat", "found in",
}

# Relation -> expected answer type
REL_ANSWER_TYPE: Dict[str, str] = {
    "birth_date": "DATE_TIME", "birth_place": "LOCATION",
    "death_date": "DATE_TIME", "death_place": "LOCATION",
    "cause_of_death": "CAUSAL_FACTORS",
    "parent": "PERSON", "child": "PERSON", "spouse": "PERSON",
    "nationality": "LOCATION", "honorific_title": "PERSON",
    "educated_at": "ORGANIZATION", "degree_earned": "ENTITY",
    "head_of_state": "PERSON", "head_of_government": "PERSON",
    "organization_leader": "PERSON", "leadership_tenure": "DATE_TIME",
    "occupation": "ENTITY",
    "founded_by": "PERSON_OR_GROUP", "established_date": "DATE_TIME",
    "headquarters": "LOCATION", "subsidiary_or_parent_company": "ORGANIZATION",
    "capital": "LOCATION", "located_in": "LOCATION", "bordering": "LOCATION",
    "population": "NUMBER", "area_size": "NUMBER", "currency": "ENTITY",
    "national_symbol": "ENTITY",
    "largest_extreme": "ENTITY", "smallest_extreme": "ENTITY",
    "age_extreme": "ENTITY", "spatial_proximity": "ENTITY",
    "invented_by": "PERSON_OR_GROUP", "discovered_by": "PERSON_OR_GROUP",
    "invention_or_discovery_date": "DATE_TIME",
    "causes_and_reasons": "CAUSAL_FACTORS", "effects_and_outcomes": "CAUSAL_FACTORS",
    "mechanism_of_action": "PROCESS_STEPS", "procedure_steps": "PROCESS_STEPS",
    "author_or_creator": "PERSON_OR_GROUP", "release_or_publication_date": "DATE_TIME",
    "genre_or_classification": "ENTITY",
}

# (question-word category, detected relation) -> better relation
WH_REFINE: Dict[Tuple[str, str], str] = {
    ("when", "birth_place"): "birth_date",
    ("where", "birth_date"): "birth_place",
    ("when", "death_place"): "death_date",
    ("where", "death_date"): "death_place",
    ("when", "founded_by"): "established_date",
    ("who", "established_date"): "founded_by",
    ("when", "invented_by"): "invention_or_discovery_date",
    ("when", "discovered_by"): "invention_or_discovery_date",
    ("when", "author_or_creator"): "release_or_publication_date",
}

# Superlatives: (surface words, canonical, property, direction, relation key)
SUPERLATIVES: List[Tuple[Tuple[str, ...], str, str, str, str]] = [
    (("most populous",), "most_populous", "population", "MAX", "largest_extreme"),
    (("nearest", "closest"), "nearest", "distance", "MIN", "spatial_proximity"),
    (("farthest", "furthest"), "farthest", "distance", "MAX", "spatial_proximity"),
    (("largest", "biggest", "greatest"), "largest", "size", "MAX", "largest_extreme"),
    (("smallest", "tiniest"), "smallest", "size", "MIN", "smallest_extreme"),
    (("tallest", "highest"), "tallest", "height", "MAX", "largest_extreme"),
    (("lowest",), "lowest", "height", "MIN", "smallest_extreme"),
    (("longest",), "longest", "length", "MAX", "largest_extreme"),
    (("shortest",), "shortest", "length", "MIN", "smallest_extreme"),
    (("deepest",), "deepest", "depth", "MAX", "largest_extreme"),
    (("shallowest",), "shallowest", "depth", "MIN", "smallest_extreme"),
    (("heaviest",), "heaviest", "weight", "MAX", "largest_extreme"),
    (("lightest",), "lightest", "weight", "MIN", "smallest_extreme"),
    (("fastest",), "fastest", "speed", "MAX", "largest_extreme"),
    (("slowest",), "slowest", "speed", "MIN", "smallest_extreme"),
    (("oldest",), "oldest", "age", "MAX", "age_extreme"),       # oldest = MAX age
    (("youngest", "newest"), "youngest", "age", "MIN", "age_extreme"),
    (("earliest",), "earliest", "date", "MIN", "age_extreme"),  # earliest = MIN date
]
_SUP_BY_WORD = {w: e for e in SUPERLATIVES for w in e[0]}
_SUP_RE = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(w) for w in sorted(_SUP_BY_WORD, key=len, reverse=True)) + r")(?![\w])",
    re.IGNORECASE,
)


# =============================================================================
# Cue indexes (built once at import time)
# =============================================================================
def _build_cue_indexes() -> Tuple[Dict[str, str], Dict[str, str], "re.Pattern[str]"]:
    role_cues: Dict[str, str] = {}
    for rel, cues in RELATION_SYNONYM_MAP.items():
        for c in cues:
            role_cues.setdefault(c.lower(), rel)
    detect: Dict[str, str] = {c: r for c, r in role_cues.items() if c not in NOISY_CUES}
    for verb, rel in PATTERN_VERB_REL.items():
        if verb in _DETECT_VERBS:
            detect.setdefault(verb, rel)
    pattern = re.compile(
        r"(?<![\w])(?:" + "|".join(re.escape(c) for c in sorted(detect, key=len, reverse=True)) + r")(?![\w])",
        re.IGNORECASE,
    )
    return role_cues, detect, pattern


_ROLE_CUE_TO_REL, _DETECT_CUE_TO_REL, _DETECT_RE = _build_cue_indexes()


# =============================================================================
# Misc regexes
# =============================================================================
_QUANT_WORDS = "many|much|long|old|far|tall|big|large|high|deep|fast|often|wide|heavy"
_AUX = "is|are|was|were|did|does|do|can|could|has|have|had|will|would|should"
_FILLER = r"(?:(?:and|so|ok|okay|please)[,\s]+)*"
_WH_RE = re.compile(
    r"^\s*" + _FILLER + r"(?P<wh>how\s+(?:" + _QUANT_WORDS + r")|whose|whom|who|where|when|why|how|which|what|"
    + _AUX + r")\b",
    re.IGNORECASE,
)
_WHAT_IS_RE = re.compile(r"^\s*" + _FILLER + r"what\s+(?:is|are|was|were)\b", re.IGNORECASE)
_AUX_SET = set(_AUX.split("|"))

_WH_AND_AUX = {
    "who", "whom", "whose", "what", "when", "where", "why", "how", "which",
    "tell", "please", "name", "give", "list", "show", "explain", "describe", "define",
} | _AUX_SET

_LEAD_STRIP = re.compile(
    r"^\s*" + _FILLER
    + r"(?:how\s+(?:" + _QUANT_WORDS + r")|which|whose|whom|who|what|when|where|why|how)?\s*"
    r"(?:(?:" + _AUX + r")\s+)?(?:the\s+|a\s+|an\s+)?",
    re.IGNORECASE,
)
_TAIL_TEMPORAL = re.compile(
    r"\s+(?:(?:in|on|during|since|before|after|by|as\s+of|from)\s+(?:the\s+)?(?:year\s+)?\d.*"
    r"|right\s+now|today|currently|now|at\s+present)$",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")

_ORD_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
_ORD_RE = re.compile(
    r"(?<![\w])(?:(\d+)(?:st|nd|rd|th)|(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|initial))"
    r"(?![\w])(?!\s+(?:century|world|aid|lady|minister|amendment|birthday|anniversary|millennium))",
    re.IGNORECASE,
)
_CURRENT_WORDS = {"current", "currently", "now", "present", "incumbent", "today", "latest"}
_FORMER_WORDS = {"former", "previous", "past", "erstwhile"}

_ROLE_HEAD = re.compile(r"^(?:(?:who|what|which)\s+(?:is|was|are|were)\s+)?(?:the\s+)?(?P<body>.+)$", re.IGNORECASE)
_LEADING_MODS = re.compile(
    r"^(?:(?:current|present|former|previous|incumbent|first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+)+",
    re.IGNORECASE,
)
_PREP = re.compile(r"\s+(?:of|in|for|at)\s+", re.IGNORECASE)
_POSSESSIVE = re.compile(r"^(?P<ent>.+?)['’]s\s+(?P<rel>.+)$", re.IGNORECASE)

_CMP_1 = re.compile(
    r"^(?:who|which|what)\s+(?:is|was|are|were)\s+(?P<cmp>\w{3,}er|more\s+\w+|less\s+\w+)\s*,?\s+(?:the\s+)?"
    r"(?P<a>.+?)\s+(?:or|vs\.?|versus)\s+(?:the\s+)?(?P<b>.+)$",
    re.IGNORECASE,
)
_CMP_2 = re.compile(
    r"^(?:compare|differences?\s+between)\s+(?:the\s+)?(?P<a>.+?)\s+(?:and|vs\.?|versus|with|to)\s+(?:the\s+)?(?P<b>.+)$",
    re.IGNORECASE,
)

_WHO_VERB = re.compile(
    r"^who\s+(?:(?:was|is)\s+)?(?P<verb>" + "|".join(PATTERN_VERB_REL) + r")\s+(?:by\s+)?"
    r"(?:the\s+|a\s+|an\s+)?(?P<ent>.+)$",
    re.IGNORECASE,
)

WHEN_VERB_REL: Dict[str, str] = {
    "born": "birth_date", "die": "death_date", "died": "death_date",
    "founded": "established_date", "established": "established_date", "formed": "established_date",
    "started": "established_date", "launched": "established_date", "incorporated": "established_date",
    "opened": "established_date", "created": "established_date", "began": "established_date",
    "built": "established_date", "completed": "established_date",
    "invented": "invention_or_discovery_date", "discovered": "invention_or_discovery_date",
    "published": "release_or_publication_date", "released": "release_or_publication_date",
    "written": "release_or_publication_date", "wrote": "release_or_publication_date",
}
_WHEN_VERB = re.compile(
    r"^when\s+(?:was|did|is|were|does|do|will)\s+(?:the\s+)?(?P<ent>.+?)\s+"
    r"(?P<verb>born|die|died|founded|established|invented|discovered|published|released|written|wrote|built|"
    r"created|launched|formed|started|signed|passed|incorporated|completed|opened|ended|began|happen|occur|held)\b",
    re.IGNORECASE,
)

WHERE_VERB_REL: Dict[str, str] = {
    "born": "birth_place", "die": "death_place", "died": "death_place", "buried": "death_place",
    "study": "educated_at", "studied": "educated_at", "educated": "educated_at",
    "located": "located_in", "situated": "located_in", "held": "located_in", "built": "located_in",
    "filmed": "located_in", "made": "located_in", "produced": "located_in", "grown": "located_in",
    "based": "headquarters", "headquartered": "headquarters",
    "found": "founded_by", "founded": "founded_by",
}
_WHERE_VERB = re.compile(
    r"^where\s+(?:was|did|is|are|were|does|do)\s+(?:the\s+)?(?P<ent>.+?)\s+"
    r"(?P<verb>born|study|studied|educated|die|died|buried|located|situated|based|headquartered|"
    r"found|founded|held|built|filmed|made|produced|grown)\b",
    re.IGNORECASE,
)
_WHERE_IS = re.compile(r"^where\s+(?:is|are)\s+(?:the\s+)?(?P<ent>.+)$", re.IGNORECASE)

_NEG_RE = re.compile(
    r"\b(?:except|excluding|without|other\s+than|not\s+including|apart\s+from|besides)\s+(?:the\s+)?"
    r"(?P<x>[a-z][\w'-]*(?:\s+[a-z][\w'-]*){0,3})",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_CONNECTORS = {"of", "the", "and", "de", "la", "for", "von", "van", "bin"}


def _clean_entity(ent: str) -> str:
    ent = ent.strip(" ,.;:?!\"'“”")
    ent = re.sub(r"^(?:the|a|an)\s+", "", ent, flags=re.IGNORECASE)
    ent = re.sub(r"['’]s$", "", ent)
    if not ent:
        return ent
    key = ent.lower()
    if key in KNOWN_COUNTRIES or key in COUNTRY_ALIASES:
        return _canon_country(key)
    return _smart_title(ent)


@dataclass
class _Semantics:
    subject: str
    secondary: List[str] = field(default_factory=list)
    relation: Optional[str] = None
    question_type: Optional[str] = None
    answer_type: Optional[str] = None
    pattern: str = "fallback"
    confidence: float = 0.3
    entity_source: str = "remainder"  # pattern | quoted | caps | geo | remainder


class QueryAnalyzer:
    """Universal Question-Understanding Engine extracting required constraints, answer types, and context rules."""

    # "before/after/between/during/more/most/not/no/nor/above/below" are intentionally absent:
    # they carry meaning for temporal, comparative and negated queries.
    STOPWORDS: Set[str] = {
        "a", "about", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "being",
        "both", "but", "by", "could", "did", "do", "does", "doing", "down",
        "each", "few", "for", "from", "further", "had", "has", "have",
        "having", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into",
        "is", "it", "its", "me", "my", "of",
        "off", "on", "once", "only", "or", "other", "our", "out", "over", "own",
        "same", "she", "should", "so", "some", "such", "than", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "we",
        "were", "what", "when", "where", "which", "while", "who", "whom", "why",
        "with", "you", "your",
    }

    KNOWN_COUNTRIES = KNOWN_COUNTRIES
    RELATION_SYNONYM_MAP = RELATION_SYNONYM_MAP

    # ------------------------------------------------------------------ public

    @classmethod
    def analyze(cls, question: str) -> StructuredQueryRepresentation:
        """Convert a natural language question into a structured query representation."""
        return cls.analyze_with_trace(question)[0]

    @classmethod
    def find_countries(cls, text: str) -> List[str]:
        """Canonical country names found in text, in order of appearance.
        Use this at retrieval time to reject chunks about a *different* country."""
        found: List[Tuple[int, str]] = []
        for m in _COUNTRY_RE.finditer(text):
            name = m.group(1).lower()
            if name == "america" and text[: m.start()].lower().rstrip().endswith(
                ("north", "south", "central", "latin")
            ):
                continue
            found.append((m.start(), _canon_country(name)))
        for m in _US_RE.finditer(text):
            found.append((m.start(), "United States"))
        found.sort(key=lambda x: x[0])
        out: List[str] = []
        for _, c in found:
            if c not in out:
                out.append(c)
        return out

    @classmethod
    def analyze_with_trace(cls, question: str) -> Tuple[StructuredQueryRepresentation, Dict[str, object]]:
        """Same as analyze(), plus a trace dict (matched pattern, confidence, ...) for debugging
        and for letting downstream retrieval decide how much to trust the parse."""
        cleaned_q = " ".join(question.split())
        q_lower = cleaned_q.lower()
        q_clean = re.sub(r"[?!]+\s*$", "", cleaned_q)
        q_clean = re.sub(r"(?<=[A-Za-z]{2})\.\s*$", "", q_clean).strip()

        q_plain = re.sub(r"['’]s\b", "", q_lower)
        q_plain = re.sub(r"[^\w\s'-]", " ", q_plain)
        words = [w.strip("'-") for w in q_plain.split()]
        words = [w for w in words if w]

        # 1. Geography
        countries = cls.find_countries(cleaned_q)
        geo_country: Optional[str] = countries[0] if countries else None

        # 2. Superlative
        sup = None
        sm = _SUP_RE.search(q_lower)
        if sm:
            entry = _SUP_BY_WORD[sm.group(1).lower()]
            sup = (sm.group(1), *entry[1:])  # (matched_word, canonical, prop, dir, rel)

        # 3. Temporal
        years = _YEAR_RE.findall(q_lower)
        temporal = cls._extract_temporal(q_lower, words, years)

        # 4. Question word + relation cues
        wh_cat = cls._wh_category(q_lower)
        detected = cls._detect_relations(q_lower)

        # 5. Structure (subject / secondary / explicit relation)
        sem = cls._extract_semantics(q_clean, q_lower, wh_cat, geo_country, sup)

        # 6. Resolve relation, answer type, question type
        relation = sem.relation or (detected[0] if detected else None)
        if relation:
            relation = WH_REFINE.get((wh_cat, relation), relation)
        if relation is None and wh_cat == "where":
            relation = "located_in"
        relation_name = relation or "general"

        ans_type = sem.answer_type or cls._answer_type(wh_cat, relation_name)
        q_type = sem.question_type or cls._question_type(ans_type, relation_name, wh_cat)

        confidence = sem.confidence
        if sem.pattern == "fallback":
            confidence = 0.3 + (0.2 if detected else 0.0) + (0.2 if sem.entity_source != "remainder" else 0.0)

        # 7. Constraints
        required: List[str] = []
        if geo_country:
            required.append(geo_country)
        if sem.entity_source != "remainder" and sem.subject:
            sk = sem.subject.lower()
            same_as_geo = bool(geo_country) and (sk == geo_country.lower() or COUNTRY_ALIASES.get(sk) == geo_country)
            if not same_as_geo and sem.subject not in required:
                required.append(sem.subject)

        secondary = list(sem.secondary)
        for extra in countries[1:]:
            if extra not in secondary and extra not in required:
                secondary.append(extra)

        required_tokens: Set[str] = set()
        for e in required:
            required_tokens.update(_TOKEN_RE.findall(e.lower()))

        core_tokens = [
            w for w in _TOKEN_RE.findall(q_plain)
            if w not in cls.STOPWORDS and (len(w) > 1 or w.isdigit())
        ]

        constraints = QueryConstraints(
            required_entities=required,
            optional_terms=[t for t in core_tokens if t not in required_tokens],
            excluded_contexts=cls._extract_exclusions(q_clean),
            geographic=GeographicConstraint(country=geo_country) if geo_country else None,
            temporal=temporal,
            superlative=sup[1] if sup else None,
            superlative_property=sup[2] if sup else None,
            superlative_direction=sup[3] if sup else None,
            expected_answer_type=ans_type,
        )

        # 8. Expanded retrieval cues
        expanded: Set[str] = set(core_tokens)
        for rel in [relation_name] + [r for r in detected if r != relation_name][:1]:
            expanded.update(cls.RELATION_SYNONYM_MAP.get(rel, []))
        for tok in core_tokens:
            if tok in cls.RELATION_SYNONYM_MAP:
                expanded.update(cls.RELATION_SYNONYM_MAP[tok])
        if geo_country:
            expanded.update(COUNTRY_VARIANTS.get(geo_country, []))

        rep = StructuredQueryRepresentation(
            raw_question=cleaned_q,
            question_type=q_type,
            subject_entity=sem.subject,
            secondary_entities=secondary,
            relation=relation_name,
            constraints=constraints,
            retrieval_terms=core_tokens,
            expanded_semantic_cues=sorted(expanded),
        )
        trace: Dict[str, object] = {
            "pattern": sem.pattern,
            "confidence": round(confidence, 2),
            "wh": wh_cat,
            "entity_source": sem.entity_source,
            "detected_relations": detected,
        }
        return rep, trace

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _wh_category(q_lower: str) -> str:
        m = _WH_RE.match(q_lower)
        if not m:
            return "none"
        wh = re.sub(r"\s+", " ", m.group("wh").lower())
        if wh.startswith("how "):
            return "quantity"
        if wh in ("who", "whom"):
            return "who"
        if wh == "what":
            return "what_is" if _WHAT_IS_RE.match(q_lower) else "what"
        if wh in _AUX_SET:
            return "aux"
        return wh  # whose, where, when, why, how, which

    @staticmethod
    def _detect_relations(q_lower: str) -> List[str]:
        """Relation keys whose cues appear in the question, longest matched cue first.
        Matches are non-overlapping, so 'father of the nation' does not also fire 'father'."""
        hits: List[Tuple[int, int, str]] = []
        for i, m in enumerate(_DETECT_RE.finditer(q_lower)):
            rel = _DETECT_CUE_TO_REL.get(m.group(0).lower())
            if rel:
                hits.append((-len(m.group(0)), i, rel))
        hits.sort()
        out: List[str] = []
        for _, _, rel in hits:
            if rel not in out:
                out.append(rel)
        return out

    @staticmethod
    def _extract_temporal(q_lower: str, words: List[str], years: List[str]) -> Optional[str]:
        ws = set(words)
        if ws & _CURRENT_WORDS:
            return "current"
        m = _ORD_RE.search(q_lower)
        if m:
            if m.group(1):
                n = int(m.group(1))
                return _ORD_WORDS[n - 1] if 1 <= n <= 10 else m.group(0).lower()
            w = m.group(2).lower()
            return "first" if w == "initial" else w
        if ws & _FORMER_WORDS:
            return "former"
        if years:
            return years[0]
        return None

    @staticmethod
    def _answer_type(wh_cat: str, relation: str) -> str:
        rt = REL_ANSWER_TYPE.get(relation)
        if wh_cat == "when":
            return "DATE_TIME"
        if wh_cat == "where":
            return "LOCATION"
        if wh_cat == "why":
            return "CAUSAL_FACTORS"
        if wh_cat == "quantity":
            return "NUMBER"
        if wh_cat == "aux":
            return "BOOLEAN"
        if wh_cat in ("who", "whose"):
            return rt if rt in ("PERSON", "PERSON_OR_GROUP") else "PERSON_OR_GROUP"
        if wh_cat == "how":
            return "PROCESS_STEPS"
        if rt:
            return rt
        if wh_cat == "what_is":
            return "DEFINITION"
        return "ENTITY"

    @staticmethod
    def _question_type(ans_type: str, relation: str, wh_cat: str) -> str:
        if relation == "honorific_title":
            return "ATTRIBUTE_OR_TITLE_LOOKUP"
        return {
            "PERSON": "PERSON_IDENTIFICATION",
            "PERSON_OR_GROUP": "PERSON_IDENTIFICATION",
            "LOCATION": "LOCATION",
            "DATE_TIME": "DATE_TIME",
            "CAUSAL_FACTORS": "CAUSE_REASON",
            "PROCESS_STEPS": "PROCESS_MECHANISM",
            "NUMBER": "QUANTITY",
            "BOOLEAN": "YES_NO",
            "DEFINITION": "DEFINITION",
        }.get(ans_type, "SELECTION" if wh_cat == "which" else "GENERAL")

    @classmethod
    def _extract_exclusions(cls, q_clean: str) -> List[str]:
        """Phrases the user explicitly rules out ("except France", "excluding the 1990s")."""
        out: List[str] = []
        for m in _NEG_RE.finditer(q_clean):
            phrase = m.group("x")
            found = cls.find_countries(phrase)
            if found:
                items = found
            else:
                toks: List[str] = []
                for t in phrase.split():
                    if t.lower() in cls.STOPWORDS:
                        break
                    toks.append(t)
                items = [_smart_title(" ".join(toks))] if toks else []
            for it in items:
                if it and it not in out:
                    out.append(it)
        return out

    @staticmethod
    def _capitalized_spans(text: str) -> List[str]:
        tokens = [t.strip(",;:()[]\"“”") for t in text.split()]
        skip0 = bool(tokens) and tokens[0].lower() in _WH_AND_AUX
        spans: List[str] = []
        cur: List[str] = []
        pending: List[str] = []
        for i, t in enumerate(tokens):
            if not t:
                continue
            is_cap = t[0].isupper() and not (i == 0 and skip0)
            if is_cap:
                if cur and pending:
                    cur.extend(pending)
                pending = []
                cur.append(t)
            elif cur and t.lower() in _CONNECTORS:
                pending.append(t)
            else:
                if cur:
                    spans.append(" ".join(cur))
                cur, pending = [], []
        if cur:
            spans.append(" ".join(cur))
        return spans

    @classmethod
    def _entity_candidate(cls, q_clean: str, geo: Optional[str]) -> Tuple[str, str, List[str]]:
        quoted = re.findall(r'"([^"]+)"|“([^”]+)”', q_clean)
        if quoted:
            q = next(a or b for a, b in quoted)
            return _clean_entity(q), "quoted", []
        spans = [s for s in (_clean_entity(x) for x in cls._capitalized_spans(q_clean)) if s]
        if spans:
            return spans[0], "caps", spans[1:]
        if geo:
            return geo, "geo", []
        remainder = _TAIL_TEMPORAL.sub("", _LEAD_STRIP.sub("", q_clean)).strip()
        return _clean_entity(remainder) if remainder else _smart_title(q_clean), "remainder", []

    @staticmethod
    def _split_role(q_clean: str) -> Optional[Tuple[str, str, str]]:
        """'(who|what) is the <role> of <entity>' -> (relation_key, role_text, entity).
        The longest role phrase that is an exact synonym cue wins, so 'father of the nation of india'
        resolves to honorific_title, not parent."""
        m = _ROLE_HEAD.match(q_clean)
        if not m:
            return None
        body = m.group("body")
        bodies = [body]
        stripped = _LEADING_MODS.sub("", body)
        if stripped != body and stripped:
            bodies.append(stripped)

        best: Optional[Tuple[int, str, str, str]] = None

        def consider(rel_txt: str, ent_txt: str) -> None:
            nonlocal best
            rel_norm = re.sub(r"^the\s+", "", rel_txt.strip().lower())
            rel_key = _ROLE_CUE_TO_REL.get(rel_norm)
            ent = _TAIL_TEMPORAL.sub("", ent_txt).strip()
            ent = re.sub(r"^the\s+", "", ent, flags=re.IGNORECASE)
            if rel_key and ent and (best is None or len(rel_norm) > best[0]):
                best = (len(rel_norm), rel_key, rel_norm, ent)

        for b in bodies:
            pm = _POSSESSIVE.match(b)
            if pm:
                consider(pm.group("rel"), pm.group("ent"))
            for sm in _PREP.finditer(b):
                consider(b[: sm.start()], b[sm.end():])
        if best is None:
            return None
        return best[1], best[2], best[3]

    # ------------------------------------------------------------- structure

    @classmethod
    def _extract_semantics(
        cls,
        q_clean: str,
        q_lower: str,
        wh_cat: str,
        geo_country: Optional[str],
        sup: Optional[Tuple[str, str, str, str, str]],
    ) -> _Semantics:
        first_word = q_lower.split()[0] if q_lower.split() else ""

        # Pattern A: comparisons ("which is older, X or Y", "compare X and Y")
        cm = _CMP_1.match(q_clean) or _CMP_2.match(q_clean)
        if cm:
            return _Semantics(
                subject=_clean_entity(cm.group("a")),
                secondary=[_clean_entity(cm.group("b"))],
                relation="comparison",
                question_type="COMPARISON",
                answer_type="ENTITY",
                pattern="comparison",
                confidence=0.8,
                entity_source="pattern",
            )

        # Pattern B: "<role> of <entity>" / "<entity>'s <role>" where role is a known relation cue
        if first_word not in ("when", "where", "why", "how"):
            role = cls._split_role(q_clean)
            if role:
                rel_key, role_txt, ent = role
                return _Semantics(
                    subject=_clean_entity(ent),
                    secondary=[_smart_title(role_txt)],
                    relation=rel_key,
                    pattern="role_of_entity",
                    confidence=0.9,
                    entity_source="pattern",
                )

        # Pattern C: superlatives ("closest galaxy to the milky way")
        if sup:
            word, canonical, _prop, _dir, rel = sup
            sm = re.search(
                re.escape(word) + r"\s+(?P<type>[a-z][\w\s-]*?)\s+(?:to|of|in|from|on|around|among|within)\s+"
                r"(?:the\s+)?(?P<ent>.+)$",
                q_clean,
                re.IGNORECASE,
            )
            if sm:
                ent = _TAIL_TEMPORAL.sub("", sm.group("ent")).strip()
                return _Semantics(
                    subject=_clean_entity(ent),
                    secondary=[_smart_title(sm.group("type").strip())],
                    relation=rel,
                    question_type="RELATIONSHIP_SUPERLATIVE",
                    answer_type="ENTITY",
                    pattern="superlative_of_entity",
                    confidence=0.85,
                    entity_source="pattern",
                )
            tail = re.split(re.escape(word) + r"\s+", q_clean, maxsplit=1, flags=re.IGNORECASE)
            subject = tail[1].strip() if len(tail) > 1 else q_clean
            return _Semantics(
                subject=_clean_entity(subject),
                relation=rel,
                question_type="RELATIONSHIP_SUPERLATIVE",
                answer_type="ENTITY",
                pattern="superlative_bare",
                confidence=0.6,
                entity_source="pattern",
            )

        # Pattern D: "who <verb> <entity>" (who invented / discovered / founded / wrote ...)
        wm = _WHO_VERB.match(q_clean)
        if wm:
            ent = _TAIL_TEMPORAL.sub("", wm.group("ent")).strip()
            return _Semantics(
                subject=_clean_entity(ent),
                relation=PATTERN_VERB_REL[wm.group("verb").lower()],
                pattern="who_verb_entity",
                confidence=0.85,
                entity_source="pattern",
            )

        # Pattern E: "when was <entity> born/founded/published ..."
        nm = _WHEN_VERB.match(q_clean)
        if nm:
            return _Semantics(
                subject=_clean_entity(nm.group("ent")),
                relation=WHEN_VERB_REL.get(nm.group("verb").lower(), "general"),
                pattern="when_entity_verb",
                confidence=0.85,
                entity_source="pattern",
            )

        # Pattern F: "where was/is <entity> born/located/studied ..."
        rm = _WHERE_VERB.match(q_clean)
        if rm:
            return _Semantics(
                subject=_clean_entity(rm.group("ent")),
                relation=WHERE_VERB_REL.get(rm.group("verb").lower(), "located_in"),
                pattern="where_entity_verb",
                confidence=0.85,
                entity_source="pattern",
            )
        im = _WHERE_IS.match(q_clean)
        if im:
            return _Semantics(
                subject=_clean_entity(_TAIL_TEMPORAL.sub("", im.group("ent"))),
                relation="located_in",
                pattern="where_is_entity",
                confidence=0.75,
                entity_source="pattern",
            )

        # Fallback: best-effort entity (quoted > capitalised span > country > cleaned remainder)
        subject, source, extras = cls._entity_candidate(q_clean, geo_country)
        return _Semantics(
            subject=subject,
            secondary=extras,
            pattern="fallback",
            entity_source=source,
        )