"""
Knowledge Base — Mutable document store for the Helpdesk environment.

Agents can:
  - search(query) → find relevant past solutions
  - add(entry)    → document a new solution (self-improvement!)
  - get_all()     → retrieve the full KB (for metrics/debugging)

The KB persists across episodes within a session, enabling measurable
self-improvement: agents that document solutions help future agents
resolve similar tickets faster.
"""

from typing import Optional
from pydantic import BaseModel, Field


class KBEntry(BaseModel):
    """A single Knowledge Base article."""
    entry_id: str
    ticket_category: str
    title: str
    problem_description: str
    solution: str
    keywords: list[str] = Field(default_factory=list)
    created_by: str = "l3"
    times_used: int = 0


class KnowledgeBase:
    """
    In-memory Knowledge Base that grows across episodes.

    Self-improvement mechanism:
    - Episode 1: KB is seeded with basic entries
    - Episode N: KB contains solutions from all previous episodes
    - Measurable: track KB size and retrieval hit rate over time
    """

    def __init__(self):
        self._entries: dict[str, KBEntry] = {}
        self._seed_kb()

    def _seed_kb(self):
        """Pre-populate with basic IT helpdesk knowledge."""
        seeds = [
            KBEntry(
                entry_id="kb_001",
                ticket_category="password_reset",
                title="Standard Password Reset Procedure",
                problem_description="User locked out after failed login attempts.",
                solution=(
                    "1. Verify employee ID in HR system\n"
                    "2. Open Admin Portal > User Management\n"
                    "3. Search user > Click 'Reset Password'\n"
                    "4. Generate temporary password and email to user\n"
                    "5. Instruct user to change password on first login"
                ),
                keywords=["password", "locked", "account", "reset", "login"],
            ),
            KBEntry(
                entry_id="kb_002",
                ticket_category="software_install",
                title="Standard Software Installation via SCCM",
                problem_description="User requires software installed on corporate device.",
                solution=(
                    "1. Check license availability in license portal\n"
                    "2. Verify user's VLAN/network segment allows the software\n"
                    "3. Push install via SCCM > Applications > Deploy\n"
                    "4. Verify installation completed via SCCM monitoring\n"
                    "5. Notify user and confirm functionality"
                ),
                keywords=["install", "software", "SCCM", "license", "deploy"],
            ),
        ]
        for entry in seeds:
            self._entries[entry.entry_id] = entry

    def search(self, query: str, top_k: int = 3) -> list[KBEntry]:
        query_words = set(query.lower().split())
        scored = []
        for entry in self._entries.values():
            entry_text = f"{entry.title} {entry.problem_description} {entry.solution}".lower()
            keyword_hits = sum(1 for kw in entry.keywords if kw in query.lower())
            word_overlap = len(query_words.intersection(set(entry_text.split())))
            score = keyword_hits * 2 + word_overlap
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, entry in scored[:top_k]:
            entry.times_used += 1
            results.append(entry)
        return results

    def add(self, entry: KBEntry) -> bool:
        if entry.entry_id in self._entries:
            return False
        self._entries[entry.entry_id] = entry
        return True

    def get_all(self) -> list[KBEntry]:
        return list(self._entries.values())

    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        entries = self.get_all()
        return {
            "total_entries": len(entries),
            "seeded_entries": sum(1 for e in entries if e.entry_id.startswith("kb_")),
            "agent_authored": sum(1 for e in entries if not e.entry_id.startswith("kb_")),
            "total_retrievals": sum(e.times_used for e in entries),
        }
