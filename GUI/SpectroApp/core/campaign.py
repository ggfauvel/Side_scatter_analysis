"""
core/campaign.py — Metadonnees de campagne : groupes de shots.

Dans le notebook, les listes de shots par condition (Si%, profil de pulse)
etaient codees en dur dans chaque cellule. Ici elles deviennent des groupes
EDITABLES dans l'interface, persistes en JSON dans le workspace, avec les
valeurs de votre campagne actuelle pre-chargees par defaut (cellule 12-MAP,
qui est la version la plus complete des 9 configurations).
"""
from __future__ import annotations

import json
from pathlib import Path

# Couleurs : palette qualitative distincte (chaque groupe identifiable
# meme superpose) — modifiable groupe par groupe dans l'interface.
DEFAULT_GROUPS = [
    {"name": "0% Si - 10/90", "si_pct": 0, "profile": "10/90", "color": "#e6194B",
     "shots": ["shot322", "shot324", "shot325", "shot326", "shot327",
               "shot328", "shot329", "shot330", "shot331", "shot341", "shot342"]},
    {"name": "5% Si - 10/90", "si_pct": 5, "profile": "10/90", "color": "#3cb44b",
     "shots": ["shot376", "shot377", "shot378", "shot379", "shot380",
               "shot381", "shot382", "shot383", "shot384", "shot385"]},
    {"name": "15% Si - 10/90", "si_pct": 15, "profile": "10/90", "color": "#4363d8",
     "shots": ["shot319", "shot320", "shot321", "shot332", "shot333",
               "shot335", "shot336", "shot337", "shot338", "shot339", "shot340"]},
    {"name": "0% Si - 30/70", "si_pct": 0, "profile": "30/70", "color": "#f58231",
     "shots": ["shot428", "shot429", "shot430", "shot431", "shot432",
               "shot433", "shot434", "shot435", "shot436"]},
    {"name": "5% Si - 30/70", "si_pct": 5, "profile": "30/70", "color": "#17becf",
     "shots": ["shot440", "shot441", "shot442", "shot443", "shot445",
               "shot446", "shot447", "shot448", "shot449", "shot450"]},
    {"name": "15% Si - 30/70", "si_pct": 15, "profile": "30/70", "color": "#911eb4",
     "shots": ["shot451", "shot452", "shot453", "shot454", "shot455",
               "shot456", "shot457", "shot458", "shot459", "shot460"]},
    {"name": "0% Si - 2ns", "si_pct": 0, "profile": "2ns", "color": "#9A6324",
     "shots": ["shot490", "shot494", "shot495", "shot496", "shot497", "shot498",
               "shot499", "shot515", "shot516", "shot517", "shot518", "shot519",
               "shot520", "shot521"]},
    {"name": "5% Si - 2ns", "si_pct": 5, "profile": "2ns", "color": "#f032e6",
     "shots": ["shot500", "shot501", "shot502", "shot503", "shot504", "shot505",
               "shot506", "shot507", "shot508", "shot510", "shot511", "shot512",
               "shot513"]},
    # (15% Si, 2ns) : pas de donnees dans la campagne - volontairement absent.
]

PROFILE_ORDER = ["10/90", "30/70", "2ns"]


def groups_path(workspace: Path) -> Path:
    return Path(workspace) / "groups.json"


def load_groups(workspace) -> list[dict]:
    p = groups_path(Path(workspace))
    if p.exists():
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(g, list) and g:
                return g
        except Exception:
            pass
    return [dict(g) for g in DEFAULT_GROUPS]


def save_groups(workspace, groups: list[dict]):
    p = groups_path(Path(workspace))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(groups, indent=2, ensure_ascii=False),
                 encoding="utf-8")
