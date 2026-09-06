from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .workspace import Workspace


@dataclass
class GitResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Git:
    """
    Git operations scoped to a single workspace.

    This class deliberately does not expose arbitrary Git commands.
    The agent receives structured operations for common repository
    inspection tasks instead.
    """

    MAX_OUTPUT_BYTES = 128 * 1024
    DEFAULT_TIMEOUT = 10.0

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def _run(
        self,
        args: list[str],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> GitResult:
        if not args:
            raise ValueError("Git arguments cannot be empty.")

        command = ["git", *args]

        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace.root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            return GitResult(
                returncode=-1,
                stdout=self._limit_output(stdout),
                stderr=f"Git command timed out after {timeout:.1f}s.\n"
                + self._limit_output(stderr),
            )
        except OSError as exc:
            return GitResult(
                returncode=-1,
                stdout="",
                stderr=f"Failed to execute Git: {exc}",
            )

        return GitResult(
            returncode=completed.returncode,
            stdout=self._limit_output(completed.stdout),
            stderr=self._limit_output(completed.stderr),
        )

    @classmethod
    def _limit_output(cls, output: str) -> str:
        encoded = output.encode("utf-8")

        if len(encoded) <= cls.MAX_OUTPUT_BYTES:
            return output

        truncated = encoded[: cls.MAX_OUTPUT_BYTES].decode(
            "utf-8",
            errors="ignore",
        )

        return (
            truncated
            + "\n\n[output truncated at "
            + f"{cls.MAX_OUTPUT_BYTES} bytes]"
        )

    def is_repository(self) -> bool:
        result = self._run(
            ["rev-parse", "--is-inside-work-tree"],
        )
        return result.ok and result.stdout.strip() == "true"

    def current_branch(self) -> str | None:
        result = self._run(
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
        )

        if not result.ok:
            return None

        branch = result.stdout.strip()
        return branch or None

    def status(self) -> dict:
        result = self._run(
            [
                "status",
                "--porcelain=v2",
                "--branch",
            ],
        )

        if not result.ok:
            return {
                "ok": False,
                "repository": False,
                "branch": None,
                "clean": None,
                "staged": [],
                "unstaged": [],
                "untracked": [],
                "conflicted": [],
                "error": result.stderr.strip() or "Git status failed.",
            }

        branch = None
        upstream = None
        staged = []
        unstaged = []
        untracked = []
        conflicted = []

        for line in result.stdout.splitlines():
            if line.startswith("# branch.head "):
                branch = line[len("# branch.head "):]

                if branch == "(detached)":
                    branch = None

            elif line.startswith("# branch.upstream "):
                upstream = line[len("# branch.upstream "):]

            elif line.startswith("1 "):
                entry = self._parse_changed_entry(line)

                if entry["index_status"] != ".":
                    staged.append(entry)

                if entry["worktree_status"] != ".":
                    unstaged.append(entry)

            elif line.startswith("2 "):
                entry = self._parse_rename_entry(line)

                if entry["index_status"] != ".":
                    staged.append(entry)

                if entry["worktree_status"] != ".":
                    unstaged.append(entry)

            elif line.startswith("u "):
                entry = self._parse_unmerged_entry(line)
                conflicted.append(entry)

            elif line.startswith("? "):
                path = line[2:]

                untracked.append(
                    {
                        "path": path,
                    }
                )

        clean = not staged and not unstaged and not untracked and not conflicted

        return {
            "ok": True,
            "repository": True,
            "branch": branch,
            "upstream": upstream,
            "clean": clean,
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "conflicted": conflicted,
        }

    def diff(
        self,
        staged: bool = False,
        path: str | None = None,
    ) -> dict:
        args = ["diff"]

        if staged:
            args.append("--cached")

        if path is not None:
            safe_path = self.workspace.relative(
                self.workspace.resolve(path)
            )
            args.extend(["--", safe_path])

        result = self._run(args)

        return {
            "ok": result.ok,
            "staged": staged,
            "path": path,
            "diff": result.stdout,
            "error": None if result.ok else result.stderr.strip(),
        }

    def log(self, max_count: int = 20) -> dict:
        if max_count < 1:
            raise ValueError("max_count must be >= 1")

        if max_count > 100:
            raise ValueError("max_count must be <= 100")

        result = self._run(
            [
                "log",
                f"--max-count={max_count}",
                "--format=%H%x00%h%x00%an%x00%aI%x00%s",
            ]
        )

        if not result.ok:
            return {
                "ok": False,
                "commits": [],
                "error": result.stderr.strip() or "Git log failed.",
            }

        commits = []

        for line in result.stdout.splitlines():
            fields = line.split("\x00", 4)

            if len(fields) != 5:
                continue

            full_hash, short_hash, author, timestamp, subject = fields

            commits.append(
                {
                    "hash": full_hash,
                    "short_hash": short_hash,
                    "author": author,
                    "date": timestamp,
                    "subject": subject,
                }
            )

        return {
            "ok": True,
            "commits": commits,
        }

    def show(
        self,
        revision: str,
        path: str | None = None,
    ) -> dict:
        if not revision.strip():
            raise ValueError("revision cannot be empty.")

        args = [
            "show",
            "--no-ext-diff",
            "--format=fuller",
            revision,
        ]

        if path is not None:
            safe_path = self.workspace.relative(
                self.workspace.resolve(path)
            )
            args.extend(["--", safe_path])

        result = self._run(args)

        return {
            "ok": result.ok,
            "revision": revision,
            "path": path,
            "content": result.stdout,
            "error": None if result.ok else result.stderr.strip(),
        }

    @staticmethod
    def _parse_changed_entry(line: str) -> dict:
        fields = line.split(" ", 8)

        if len(fields) < 9:
            return {
                "path": "",
                "index_status": "?",
                "worktree_status": "?",
            }

        status = fields[1]
        path = fields[8]

        return {
            "path": path,
            "index_status": status[0],
            "worktree_status": status[1],
        }

    @staticmethod
    def _parse_rename_entry(line: str) -> dict:
        fields = line.split(" ", 9)

        if len(fields) < 10:
            return {
                "path": "",
                "index_status": "?",
                "worktree_status": "?",
            }

        status = fields[1]
        path_data = fields[9]

        if "\t" in path_data:
            old_path, new_path = path_data.split("\t", 1)
        else:
            old_path = path_data
            new_path = path_data

        return {
            "path": new_path,
            "old_path": old_path,
            "index_status": status[0],
            "worktree_status": status[1],
        }

    @staticmethod
    def _parse_unmerged_entry(line: str) -> dict:
        fields = line.split(" ", 9)

        if len(fields) < 10:
            return {
                "path": "",
                "conflicted": True,
            }

        return {
            "path": fields[9],
            "conflicted": True,
        }
