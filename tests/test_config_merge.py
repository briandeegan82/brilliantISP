from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from util.config_merge import deep_merge, load_merged_yaml, load_yaml_mapping


def _write(path: Path, mapping: dict) -> Path:
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


def test_deep_merge_returns_independent_copy() -> None:
    base = {"a": {"b": 1, "list": [1, 2]}, "keep": {"nested": [3]}}
    overlay = {"a": {"b": 2}}
    base_snapshot = copy.deepcopy(base)
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"b": 2, "list": [1, 2]}, "keep": {"nested": [3]}}
    merged["a"]["list"].append(9)
    merged["a"]["b"] = 99
    merged["keep"]["nested"].append(4)
    assert base == base_snapshot
    overlay["a"]["b"] = 7
    assert merged["a"]["b"] == 99
    again = deep_merge(base, overlay)
    assert again["a"]["list"] == [1, 2]


def test_deep_merge_does_not_share_overlay_values() -> None:
    overlay = {"x": {"items": [1, 2, 3]}}
    merged = deep_merge({}, overlay)
    overlay["x"]["items"].append(4)
    assert merged["x"]["items"] == [1, 2, 3]
    merged["x"]["items"].append(5)
    assert overlay["x"]["items"] == [1, 2, 3, 4]


def test_load_yaml_mapping_is_raw_loader(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yml", {"sec": {"v": 1}})
    child = _write(
        tmp_path / "child.yml", {"extends": "base.yml", "sec": {"w": 2}}
    )
    raw = load_yaml_mapping(child)
    assert raw["extends"] == "base.yml"
    assert base.is_file()


def test_extends_single_and_removal(tmp_path: Path) -> None:
    _write(tmp_path / "base.yml", {"a": 1, "sec": {"x": 1, "y": 2}})
    child = _write(
        tmp_path / "child.yml",
        {"extends": "base.yml", "a": 2, "sec": {"y": 3}},
    )
    merged = load_merged_yaml([child])
    assert merged == {"a": 2, "sec": {"x": 1, "y": 3}}
    assert "extends" not in merged


def test_extends_list_order(tmp_path: Path) -> None:
    _write(tmp_path / "first.yml", {"v": 1, "only_first": True})
    _write(tmp_path / "second.yml", {"v": 2, "only_second": True})
    child = _write(
        tmp_path / "child.yml",
        {"extends": ["first.yml", "second.yml"], "extra": 5},
    )
    merged = load_merged_yaml([child])
    assert merged["v"] == 2
    assert merged["only_first"] is True
    assert merged["only_second"] is True
    assert merged["extra"] == 5


def test_extends_nested_and_relative(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(tmp_path / "root.yml", {"depth": 0, "root_only": 1})
    _write(sub / "mid.yml", {"extends": "../root.yml", "depth": 1})
    _write(sub / "leaf.yml", {"extends": "mid.yml", "depth": 2})
    merged = load_merged_yaml([sub / "leaf.yml"])
    assert merged == {"depth": 2, "root_only": 1}


def test_extends_diamond_no_false_cycle(tmp_path: Path) -> None:
    _write(tmp_path / "common.yml", {"shared": "yes", "v": 0})
    _write(tmp_path / "a.yml", {"extends": "common.yml", "a": 1, "v": 1})
    _write(tmp_path / "b.yml", {"extends": "common.yml", "b": 1, "v": 2})
    _write(
        tmp_path / "top.yml", {"extends": ["a.yml", "b.yml"], "top": True}
    )
    merged = load_merged_yaml([tmp_path / "top.yml"])
    assert merged["shared"] == "yes"
    assert merged["a"] == 1 and merged["b"] == 1
    assert merged["v"] == 2


def test_extends_cycle_reports_chain(tmp_path: Path) -> None:
    _write(tmp_path / "a.yml", {"extends": "b.yml"})
    _write(tmp_path / "b.yml", {"extends": "a.yml"})
    with pytest.raises(ValueError, match="cycle"):
        load_merged_yaml([tmp_path / "a.yml"])
    _write(tmp_path / "self.yml", {"extends": "self.yml"})
    with pytest.raises(ValueError, match="self.yml"):
        load_merged_yaml([tmp_path / "self.yml"])


def test_extends_missing_base_shows_chain(tmp_path: Path) -> None:
    _write(tmp_path / "child.yml", {"extends": "missing_base.yml"})
    with pytest.raises(FileNotFoundError) as excinfo:
        load_merged_yaml([tmp_path / "child.yml"])
    assert "child.yml" in str(excinfo.value)
    assert "missing_base.yml" in str(excinfo.value)


def test_extends_nested_missing_leaf_shows_full_chain(tmp_path: Path) -> None:
    _write(tmp_path / "mid.yml", {"extends": "leaf_missing.yml"})
    _write(tmp_path / "top.yml", {"extends": "mid.yml"})
    with pytest.raises(FileNotFoundError) as excinfo:
        load_merged_yaml([tmp_path / "top.yml"])
    message = str(excinfo.value)
    assert "top.yml" in message
    assert "mid.yml" in message
    assert "leaf_missing.yml" in message


def test_extends_null_rejected(tmp_path: Path) -> None:
    (tmp_path / "null_extends.yml").write_text("extends:\nkey: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extends"):
        load_merged_yaml([tmp_path / "null_extends.yml"])


def test_extends_bad_shapes(tmp_path: Path) -> None:
    for bad in ([], "", 5, {"a": 1}, ["ok.yml", 3]):
        _write(tmp_path / "bad.yml", {"extends": bad})
        with pytest.raises(ValueError, match="extends"):
            load_merged_yaml([tmp_path / "bad.yml"])


def test_extends_absolute_path(tmp_path: Path) -> None:
    base = _write(tmp_path / "abs_base.yml", {"v": 4})
    child = _write(
        tmp_path / "child.yml", {"extends": str(base), "w": 5}
    )
    merged = load_merged_yaml([child])
    assert merged == {"v": 4, "w": 5}


def test_extends_inside_listed_documents(tmp_path: Path) -> None:
    _write(tmp_path / "base.yml", {"sec": {"a": 1}})
    _write(tmp_path / "child.yml", {"extends": "base.yml", "sec": {"b": 2}})
    overlay = _write(tmp_path / "overlay.yml", {"sec": {"c": 3}})
    merged = load_merged_yaml([tmp_path / "child.yml", overlay])
    assert merged["sec"] == {"a": 1, "b": 2, "c": 3}


def test_extends_malformed_yaml_base_shows_chain(tmp_path: Path) -> None:
    (tmp_path / "broken.yml").write_text("key: [unclosed\n  bad: : :\n", encoding="utf-8")
    _write(tmp_path / "child.yml", {"extends": "broken.yml"})
    with pytest.raises(ValueError) as excinfo:
        load_merged_yaml([tmp_path / "child.yml"])
    assert "child.yml" in str(excinfo.value)
    assert "broken.yml" in str(excinfo.value)


def test_extends_nonmapping_base_root_shows_chain(tmp_path: Path) -> None:
    (tmp_path / "list_root.yml").write_text("- 1\n- 2\n", encoding="utf-8")
    _write(tmp_path / "child.yml", {"extends": "list_root.yml"})
    with pytest.raises(ValueError) as excinfo:
        load_merged_yaml([tmp_path / "child.yml"])
    assert "child.yml" in str(excinfo.value)
    assert "list_root.yml" in str(excinfo.value)


def test_extends_nested_invalid_shape_shows_full_chain(tmp_path: Path) -> None:
    _write(tmp_path / "mid.yml", {"extends": {"not": "a path"}})
    _write(tmp_path / "top.yml", {"extends": "mid.yml"})
    with pytest.raises(ValueError, match="extends") as excinfo:
        load_merged_yaml([tmp_path / "top.yml"])
    message = str(excinfo.value)
    assert "top.yml" in message
    assert "mid.yml" in message
