"""
test_metrics.py
---------------
Unit tests for the lightweight local metric helpers.
Pure Python — no Azure, no mocks needed.
"""

from src.evaluation.metrics import (
    answer_length_ok,
    compute_local_metrics,
    has_citation,
    has_version_citation,
    no_refusal,
    source_type_present,
)


class TestHasCitation:
    def test_detects_source_filename(self):
        assert has_citation("See user-guide.md for details.", ["user-guide.md"])

    def test_false_when_no_citation(self):
        assert not has_citation("The answer is 42.", ["user-guide.md"])

    def test_case_insensitive(self):
        assert has_citation("Refer to USER-GUIDE.MD.", ["user-guide.md"])

    def test_matches_any_source(self):
        assert has_citation("See faq.pdf.", ["guide.md", "faq.pdf"])

    def test_empty_sources_returns_false(self):
        assert not has_citation("Good answer.", [])


class TestHasVersionCitation:
    def test_detects_v_prefix_version(self):
        assert has_version_citation("This was added in v2.1.0.")

    def test_detects_plain_version(self):
        assert has_version_citation("Released in 2.0.0.")

    def test_false_when_no_version(self):
        assert not has_version_citation("No version mentioned here.")

    def test_detects_multi_part_version(self):
        assert has_version_citation("Fixed in v1.2.3.4.")


class TestAnswerLengthOk:
    def test_normal_answer_passes(self):
        assert answer_length_ok("Go to Settings to reset your password.")

    def test_too_short_fails(self):
        assert not answer_length_ok("No.", min_chars=20)

    def test_too_long_fails(self):
        assert not answer_length_ok("x" * 3000, max_chars=2000)

    def test_empty_string_fails(self):
        assert not answer_length_ok("")


class TestNoRefusal:
    def test_normal_answer_passes(self):
        assert no_refusal("Go to Settings > Security > Reset Password.")

    def test_detects_cannot_find(self):
        assert not no_refusal("I cannot find that information in the documents.")

    def test_detects_not_enough_information(self):
        assert not no_refusal("There is not enough information in the context.")

    def test_detects_knowledge_base_variation(self):
        assert not no_refusal("The knowledge base does not contain an answer.")

    def test_partial_word_does_not_trigger(self):
        assert no_refusal("You can find the answer in Settings.")


class TestSourceTypePresent:
    def test_detects_ticket_mention(self):
        assert source_type_present("Based on the resolved ticket...", "ticket")

    def test_detects_changelog_mention(self):
        assert source_type_present("According to the changelog entry...", "changelog")

    def test_false_when_absent(self):
        assert not source_type_present("Go to Settings.", "changelog")

    def test_case_insensitive(self):
        assert source_type_present("From the TICKET record...", "ticket")


class TestComputeLocalMetrics:
    def test_returns_all_base_keys(self):
        metrics = compute_local_metrics("Answer here (Source: guide.md)", ["guide.md"])
        assert "has_citation" in metrics
        assert "answer_length_ok" in metrics
        assert "no_refusal" in metrics
        assert "char_count" in metrics
        assert "has_version_citation" in metrics

    def test_source_type_key_added_when_provided(self):
        metrics = compute_local_metrics("Answer.", [], source_type="ticket")
        assert "source_type_present" in metrics

    def test_source_type_key_absent_when_not_provided(self):
        metrics = compute_local_metrics("Answer.", [])
        assert "source_type_present" not in metrics

    def test_char_count_matches_stripped_length(self):
        answer = "  Hello world.  "
        metrics = compute_local_metrics(answer, [])
        assert metrics["char_count"] == len(answer.strip())

    def test_all_pass_for_good_changelog_answer(self):
        answer = "In v2.1.0 (changelog.md), the feature was added."
        metrics = compute_local_metrics(answer, ["changelog.md"], source_type="changelog")
        assert metrics["has_citation"] is True
        assert metrics["has_version_citation"] is True
        assert metrics["answer_length_ok"] is True
        assert metrics["no_refusal"] is True