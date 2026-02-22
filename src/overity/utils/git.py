"""
Git utilities
=============

**February 2026**

- Florian Dupeyron (florian.dupeyron@mugcat.fr)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations
from typing import Self

import subprocess

from dataclasses import dataclass
from enum import Enum


class GitStatusEntryKind(Enum):
    """Represents the type of modification in git status"""

    Added = "A"
    Modified = "M"
    Deleted = "D"
    Renamed = "R"
    Copied = "C"
    Untracked = "?"
    Ignored = "!"
    UpdatedButUnmerged = "U"

    @classmethod
    def try_from_str(cls, char: str) -> Self | None:
        try:
            return cls(char)
        except ValueError:
            return None


@dataclass
class GitStatusEntry:
    """Represents a single git status entry"""

    from_path: str
    to_path: str | None = None

    staged_kind: GitStatusEntryKind | None = None
    unstaged_kind: GitStatusEntryKind | None = None


def git_status(repo_path: str) -> list[GitStatusEntry]:
    """
    Run git status and parse the output into GitStatusEntry objects

    Args:
        repo_path: Path to the git repository

    Returns:
        List of GitStatusEntry objects representing the repository status
    """

    # Run git status command
    cmd = ["git", "status", "--porcelain", "-z", "-uall"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo_path,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        raise RuntimeError("git command not found. Is git installed?")

    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr}")

    # Get the entries
    entries = []
    output = result.stdout

    if not output:
        return []

    parts = filter(lambda x: isinstance(x, str) and (len(x) >= 3), output.split("\x00"))

    for part in parts:
        # Parse the status line
        staged = part[0]
        unstaged = part[1]

        # Check for both staged and unstaged changes (e.g., MM, AM, MD)
        staged_kind = GitStatusEntryKind.try_from_str(staged)
        unstaged_kind = GitStatusEntryKind.try_from_str(unstaged)

        # Get the filename(s)
        if staged_kind in (GitStatusEntryKind.Renamed, GitStatusEntryKind.Copied):
            # For renames and copies, git outputs: XY <new_path>\0<old_path>\0,
            # so the first path is the destination (new), second is source (old)
            to_path = part[3:]
            part = next(parts)
            from_path = part

            # Add the entry
            entries.append(
                GitStatusEntry(
                    staged_kind=staged_kind,
                    unstaged_kind=unstaged_kind,
                    from_path=from_path,
                    to_path=to_path,
                )
            )

        else:
            # Regular entry with a single filename
            from_path = part[3:]

            # Note: avoid double entries for untracked and ignored entries ("??" and "!!")

            entries.append(
                GitStatusEntry(
                    staged_kind=(
                        staged_kind
                        if (
                            unstaged_kind
                            not in (
                                GitStatusEntryKind.Untracked,
                                GitStatusEntryKind.Ignored,
                            )
                        )
                        else None
                    ),
                    unstaged_kind=unstaged_kind,
                    from_path=from_path,
                    to_path=None,
                )
            )

    return entries
