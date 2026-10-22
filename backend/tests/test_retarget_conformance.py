"""Conformance between the two avatar retargeting implementations.

The web viewport (frontend/src/components/VrmAvatar.jsx, three.js) and the
Unity client (ISL-3D Avatar/Assets/Scripts/HandPoseMapper.cs, UniVRM)
implement the same hand/torso retargeting independently - neither can import
the other. They had already drifted once: the same namaste source gave a
measured hand separation of 0.0547 m on the web and 0.0481 m in Unity.

Docs/Description/10_Avatar_Retargeting_Spec.md is the normative table. These
tests parse that table and both sources and fail if the three disagree, so a
tuned constant cannot be changed on one side only.

These files live outside backend/, so this runs against a full checkout (CI
checks the whole repo out); the Docker image ships neither the tests nor the
Unity project.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "Docs" / "Description" / "10_Avatar_Retargeting_Spec.md"
JS = REPO / "frontend" / "src" / "components" / "VrmAvatar.jsx"
CS = REPO / "ISL-3D Avatar" / "Assets" / "Scripts" / "HandPoseMapper.cs"

# Spelled out rather than taken from the spec table, so that deleting a row
# from the document cannot silently shrink the coverage of this check.
SHARED = (
    "TORSO_PROFILE_BANDS",
    "TORSO_MIDLINE_HALF",
    "HAND_CLEARANCE",
    "HAND_SPAN_UP",
    "HAND_SPAN_DOWN",
    "HAND_SPAN_SAMPLES",
    "HAND_BACK_MAX",
    "FINGER_TIP_MARGIN",
    "CLASP_THRESHOLD",
    "CLASP_FULL",
    "CLASP_MIN_SCALE",
    "HAND_MERGE_DIST",
    "SIDE_SWITCH_HYST",
    "IDLE_LATERAL",
    "IDLE_DROP",
    "IDLE_FWD",
)

_SPEC_ROW = re.compile(r"\|\s*`([A-Z0-9_]+)`\s*\|\s*([0-9.]+)\s*\|")
_JS_CONST = re.compile(r"^const ([A-Z][A-Z0-9_]*) = ([0-9.]+);", re.MULTILINE)
_CS_CONST = re.compile(
    r"^\s*const (?:float|int|double) ([A-Z][A-Z0-9_]*) = ([0-9.]+)f?;", re.MULTILINE
)


def _spec_table() -> dict:
    text = SPEC.read_text(encoding="utf-8")
    return {m.group(1): float(m.group(2)) for m in _SPEC_ROW.finditer(text)}


def _js_constants() -> dict:
    text = JS.read_text(encoding="utf-8")
    return {m.group(1): float(m.group(2)) for m in _JS_CONST.finditer(text)}


def _cs_constants() -> dict:
    text = CS.read_text(encoding="utf-8")
    return {m.group(1): float(m.group(2)) for m in _CS_CONST.finditer(text)}


@pytest.mark.parametrize("name", SHARED)
def test_declared_in_both_implementations(name):
    assert name in _js_constants(), f"{name} is missing from VrmAvatar.jsx"
    assert name in _cs_constants(), f"{name} is missing from HandPoseMapper.cs"


@pytest.mark.parametrize("name", SHARED)
def test_both_implementations_match_the_spec(name):
    spec, js, cs = _spec_table(), _js_constants(), _cs_constants()
    assert name in spec, f"{name} is missing from the spec table"
    assert js[name] == pytest.approx(spec[name]), (
        f"web has {name} = {js[name]}, spec says {spec[name]}"
    )
    assert cs[name] == pytest.approx(spec[name]), (
        f"Unity has {name} = {cs[name]}, spec says {spec[name]}"
    )


def test_spec_table_lists_exactly_the_shared_constants():
    assert set(_spec_table()) == set(SHARED)


def test_the_two_implementations_agree_with_each_other():
    js, cs = _js_constants(), _cs_constants()
    drifted = {
        name: (js[name], cs[name]) for name in SHARED if js[name] != pytest.approx(cs[name])
    }
    assert not drifted, f"web/Unity drift: {drifted}"


@pytest.mark.parametrize("path", [JS, CS], ids=["web", "unity"])
@pytest.mark.parametrize("name", SHARED)
def test_each_constant_is_used_not_just_declared(name, path):
    """A declaration plus no use is dead code that still passes the checks
    above. Every shared constant should appear at least twice."""
    occurrences = path.read_text(encoding="utf-8").count(name)
    assert occurrences >= 2, f"{name} declared but never used in {path.name}"


@pytest.mark.parametrize("path", [JS, CS], ids=["web", "unity"])
def test_the_guard_constrains_the_hand_span_not_the_wrist_point(path):
    """The regression that motivated the measured profile: a guard that only
    checks the wrist point lets the rest of the palm sink into the chest."""
    text = path.read_text(encoding="utf-8")
    for required in ("HAND_SPAN_UP", "HAND_SPAN_DOWN", "HAND_SPAN_SAMPLES"):
        assert required in text, f"{required} missing from {path.name}"
