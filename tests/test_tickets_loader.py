"""
test_tickets_loader.py
----------------------
Unit tests for the CSV ticket loader and sentence chunker.
"""

import csv
import pytest
from src.ingestion.sources.tickets import TicketChunk, load_tickets, _sentence_chunk, _process_csv


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id","subject","body","resolution","status","created_at"])
        writer.writeheader()
        writer.writerows(rows)


class TestSentenceChunk:
    def test_produces_at_least_one_chunk(self):
        chunks = _sentence_chunk("Hello world.", "tickets.csv", "1", "Subject", "resolved", "2025-01-01", max_tokens=256)
        assert len(chunks) >= 1

    def test_source_type_is_ticket(self):
        chunks = _sentence_chunk("Hello.", "t.csv", "1", "Sub", "open", "2025-01-01", max_tokens=256)
        assert chunks[0].source_type == "ticket"

    def test_ticket_id_preserved(self):
        chunks = _sentence_chunk("Content.", "t.csv", "TKT-42", "Sub", "open", "2025-01-01", max_tokens=256)
        assert chunks[0].ticket_id == "TKT-42"

    def test_status_preserved(self):
        chunks = _sentence_chunk("Content.", "t.csv", "1", "Sub", "resolved", "2025-01-01", max_tokens=256)
        assert chunks[0].status == "resolved"

    def test_long_text_splits_into_multiple_chunks(self):
        long_text = "This is a sentence. " * 200
        chunks = _sentence_chunk(long_text, "t.csv", "1", "Sub", "open", "2025-01-01", max_tokens=50)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        long_text = "This is a sentence. " * 200
        chunks = _sentence_chunk(long_text, "t.csv", "1", "Sub", "open", "2025-01-01", max_tokens=50)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestLoadTickets:
    def test_loads_csv_file(self, tmp_path):
        _write_csv(tmp_path / "tickets.csv", [
            {"ticket_id": "1", "subject": "Login issue", "body": "Cannot log in.",
             "resolution": "Reset password.", "status": "resolved", "created_at": "2025-01-01"},
        ])
        chunks = load_tickets(str(tmp_path))
        assert len(chunks) >= 1
        assert any("Login issue" in c.content for c in chunks)

    def test_skips_non_csv_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a ticket")
        chunks = load_tickets(str(tmp_path))
        assert chunks == []

    def test_multiple_tickets(self, tmp_path):
        _write_csv(tmp_path / "tickets.csv", [
            {"ticket_id": "1", "subject": "A", "body": "Body A.", "resolution": "", "status": "open", "created_at": "2025-01-01"},
            {"ticket_id": "2", "subject": "B", "body": "Body B.", "resolution": "", "status": "open", "created_at": "2025-01-02"},
        ])
        chunks = load_tickets(str(tmp_path))
        ticket_ids = {c.ticket_id for c in chunks}
        assert "1" in ticket_ids
        assert "2" in ticket_ids

    def test_resolution_included_in_content(self, tmp_path):
        _write_csv(tmp_path / "tickets.csv", [
            {"ticket_id": "1", "subject": "Issue", "body": "Problem here.",
             "resolution": "Fixed by restarting.", "status": "resolved", "created_at": "2025-01-01"},
        ])
        chunks = load_tickets(str(tmp_path))
        full_content = " ".join(c.content for c in chunks)
        assert "Fixed by restarting" in full_content

    def test_empty_folder(self, tmp_path):
        assert load_tickets(str(tmp_path)) == []