"""
core/uistate.py — Memoire d'interface (plots + reglages) entre les pages.

Probleme resolu
---------------
Dash reconstruit la page a chaque changement d'onglet (`main.route` appelle
`mod.layout()`), donc tout ce qui avait ete calcule disparaissait : il fallait
re-cliquer sur « Show » / « Compare » a chaque aller-retour.

Solution
--------
Un cache serveur (mono-utilisateur, comme SESSION) qui memorise :
  * les SORTIES des callbacks (children / figure / data),
  * les VALEURS des controles (value) passees en Input/State.

Deux points de branchement seulement, donc aucune logique metier a toucher :

1. `from core.uistate import callback` a la place de `from dash import callback`
   dans les pages : le decorateur enregistre automatiquement entrees et sorties.
2. `uistate.restore(mod.layout())` dans le routeur : on reinjecte les valeurs
   memorisees dans l'arbre de composants fraichement construit.

Le layout reste donc reconstruit a chaque visite (les options des menus
deroulants restent a jour), mais l'etat visible est restaure par-dessus.
"""
from __future__ import annotations

import threading

import dash
from dash import callback as _dash_callback
from dash.dependencies import Input, Output, State

# {component_id: {property: value}}
_STORE: dict[str, dict] = {}
_LOCK = threading.Lock()

# Proprietes restaurees dans le layout. On ne touche volontairement pas a
# `options` (recalculees par layout(), donc toujours a jour) ni a `n_clicks`.
_RESTORABLE = ("children", "figure", "data", "value")


def _is_no_update(v) -> bool:
    return isinstance(v, type(dash.no_update)) or v is dash.no_update


def remember(cid: str, prop: str, value):
    if prop not in _RESTORABLE or _is_no_update(value):
        return
    with _LOCK:
        _STORE.setdefault(str(cid), {})[prop] = value


def recall(cid: str, prop: str, default=None):
    return _STORE.get(str(cid), {}).get(prop, default)


def forget(cid: str | None = None):
    """Oublie un composant, ou tout le cache si cid est None."""
    with _LOCK:
        if cid is None:
            _STORE.clear()
        else:
            _STORE.pop(str(cid), None)


def clear_outputs():
    """Efface les resultats affiches (plots, tableaux) mais garde les reglages.
    Utilise apres un changement de calibration : les plots deviennent faux,
    les reglages restent valables."""
    with _LOCK:
        for cid, props in list(_STORE.items()):
            for p in ("children", "figure", "data"):
                props.pop(p, None)
            if not props:
                _STORE.pop(cid, None)


def stats() -> dict:
    return {"components": len(_STORE),
            "values": sum(len(v) for v in _STORE.values())}


# ── Decorateur de callback ──────────────────────────────────────────────────
def callback(*args, **kwargs):
    """Remplacant de `dash.callback` qui memorise entrees et sorties.

    Compatible avec la syntaxe utilisee dans les pages : Output/Input/State
    positionnels, `prevent_initial_call`, `allow_duplicate`, sorties multiples.
    """
    outputs = [a for a in args if isinstance(a, Output)]
    # L'ordre des arguments recus par la fonction : Inputs puis States.
    in_state = [a for a in args if isinstance(a, (Input, State))]
    inputs = [a for a in in_state if isinstance(a, Input)]
    states = [a for a in in_state if isinstance(a, State)]
    arg_specs = inputs + states

    def decorator(func):
        def wrapper(*fargs, **fkwargs):
            # 1. memoriser les reglages (uniquement la propriete `value`)
            for spec, val in zip(arg_specs, fargs):
                try:
                    cid, prop = spec.component_id, spec.component_property
                except AttributeError:
                    continue
                if prop == "value" and isinstance(cid, str):
                    remember(cid, prop, val)
            # 2. executer
            result = func(*fargs, **fkwargs)
            # 3. memoriser les sorties
            try:
                if len(outputs) == 1:
                    vals = [result]
                elif isinstance(result, (list, tuple)):
                    vals = list(result)
                else:
                    vals = []
                for spec, val in zip(outputs, vals):
                    if isinstance(spec.component_id, str):
                        remember(spec.component_id,
                                 spec.component_property, val)
            except Exception:      # jamais casser un callback pour le cache
                pass
            return result

        wrapper.__name__ = getattr(func, "__name__", "callback")
        wrapper.__doc__ = func.__doc__
        return _dash_callback(*args, **kwargs)(wrapper)

    return decorator


# ── Restauration dans un arbre de composants ────────────────────────────────
def _options_values(comp):
    """Valeurs acceptables d'un composant a options, ou None si non concerne."""
    opts = getattr(comp, "options", None)
    if opts is None:
        return None
    vals = set()
    for o in opts:
        if isinstance(o, dict):
            if "value" in o:
                vals.add(o["value"])
        else:
            vals.add(o)
    return vals


def _restore_one(comp):
    cid = getattr(comp, "id", None)
    if not isinstance(cid, str):
        return
    saved = _STORE.get(cid)
    if not saved:
        return
    valid = getattr(comp, "_prop_names", ()) or ()
    for prop, val in saved.items():
        # `hasattr` est insuffisant : un composant Dash n'expose que les
        # proprietes explicitement passees a sa construction. On teste donc
        # la liste des proprietes valides de la CLASSE.
        if prop not in valid and prop != "children":
            continue
        if prop == "value":
            allowed = _options_values(comp)
            if allowed is not None:
                # menu deroulant : ne pas restaurer une valeur disparue
                if isinstance(val, (list, tuple)):
                    val = [v for v in val if v in allowed]
                    if not val:
                        continue
                elif val not in allowed:
                    continue
        try:
            setattr(comp, prop, val)
        except Exception:
            pass


def restore(tree):
    """Reinjecte l'etat memorise dans un arbre de composants Dash (en place).
    Retourne l'arbre, pour un usage direct : `return restore(mod.layout())`."""
    _walk(tree)
    return tree


def _walk(node, depth=0):
    if node is None or depth > 60:
        return
    if isinstance(node, (list, tuple)):
        for n in node:
            _walk(n, depth + 1)
        return
    if not hasattr(node, "_prop_names"):
        return                      # str, nombre, figure brute…
    _restore_one(node)
    for prop in ("children", "body", "label", "title", "tab_id"):
        child = getattr(node, prop, None)
        if child is not None and prop in ("children", "body"):
            _walk(child, depth + 1)
