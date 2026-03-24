"""config/settings.yaml のロードテスト"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings():
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(settings_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def settings_path():
    return PROJECT_ROOT / "config" / "settings.yaml"


def test_settings_file_exists(settings_path):
    assert settings_path.exists(), "config/settings.yaml が存在しません"


def test_settings_has_required_sections(settings):
    required_sections = ["database", "paths", "logging", "feature_engine", "late_money", "submodel"]
    for section in required_sections:
        assert section in settings, f"settings.yaml に '{section}' セクションがありません"


def test_settings_database_fields(settings):
    db = settings["database"]
    required_fields = ["host", "port", "dbname", "user"]
    for field in required_fields:
        assert field in db, f"database セクションに '{field}' がありません"


def test_settings_submodel_surfaces(settings):
    surfaces = settings["submodel"]["surfaces"]
    assert "turf" in surfaces
    assert "dirt" in surfaces
    assert len(surfaces) == 2  # 設計書§6: 2分割のみ
