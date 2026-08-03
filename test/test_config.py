"""
Tests for app/config.py validation and settings parsing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.config import Settings


def test_settings_default_values():
    s = Settings(gemini_api_key="test_key")
    assert s.chunk_size_words == 300
    assert s.chunk_overlap_words == 50
    assert s.top_k_results == 5
    assert s.max_file_size_bytes == 20 * 1024 * 1024


def test_settings_cors_origins_parsing():
    s = Settings(cors_origins="http://localhost:3000, https://app.example.com")
    assert s.cors_origin_list == ["http://localhost:3000", "https://app.example.com"]


def test_settings_overlap_validation():
    with pytest.raises(ValueError):
        Settings(chunk_size_words=100, chunk_overlap_words=100)


def test_startup_validation_detects_placeholder_key():
    s = Settings(gemini_api_key="your_gemini_api_key_here")
    issues = s.validate_startup()
    assert any("GEMINI_API_KEY is not set" in issue for issue in issues)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
