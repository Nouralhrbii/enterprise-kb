"""
test_safety.py
--------------
Unit tests for content safety screening.
Mocks Azure AI Content Safety — no real API calls.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.safety.content_safety import SafetyDecision, SafetyResult, check_text


def _category(name, severity):
    item = MagicMock()
    item.category = name
    item.severity = severity
    return item


def _response(scores: dict):
    mock = MagicMock()
    mock.categories_analysis = [_category(k, v) for k, v in scores.items()]
    return mock


@patch("src.safety.content_safety.get_safety_client")
def test_safe_input_passes(mock_client_factory):
    client = MagicMock()
    client.analyze_text.return_value = _response({"Hate": 0, "Violence": 0, "Sexual": 0, "SelfHarm": 0})
    mock_client_factory.return_value = client
    result = check_text("How do I reset my password?")
    assert result.is_safe is True
    assert result.flagged_categories == []
    assert result.max_severity == 0


@patch("src.safety.content_safety.get_safety_client")
def test_high_severity_blocks(mock_client_factory):
    client = MagicMock()
    client.analyze_text.return_value = _response({"Hate": 6, "Violence": 0, "Sexual": 0, "SelfHarm": 0})
    mock_client_factory.return_value = client
    result = check_text("harmful content")
    assert result.is_safe is False
    assert result.decision == SafetyDecision.BLOCKED
    assert "Hate" in result.flagged_categories


@patch("src.safety.content_safety.get_safety_client")
def test_multiple_categories_flagged(mock_client_factory):
    client = MagicMock()
    client.analyze_text.return_value = _response({"Hate": 4, "Violence": 4, "Sexual": 0, "SelfHarm": 2})
    mock_client_factory.return_value = client
    result = check_text("bad content")
    assert set(result.flagged_categories) == {"Hate", "Violence", "SelfHarm"}


@patch("src.safety.content_safety.get_safety_client")
def test_raw_scores_returned(mock_client_factory):
    client = MagicMock()
    client.analyze_text.return_value = _response({"Hate": 0, "Violence": 2, "Sexual": 0, "SelfHarm": 0})
    mock_client_factory.return_value = client
    result = check_text("neutral text")
    assert result.raw_scores["Violence"] == 2
    assert result.raw_scores["Hate"] == 0


@patch("src.safety.content_safety.get_safety_client")
def test_explicit_client_not_factory(mock_factory):
    explicit = MagicMock()
    explicit.analyze_text.return_value = _response({"Hate": 0, "Violence": 0, "Sexual": 0, "SelfHarm": 0})
    check_text("test", client=explicit)
    mock_factory.assert_not_called()
    explicit.analyze_text.assert_called_once()