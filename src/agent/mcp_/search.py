from __future__ import annotations
from pathlib import Path
from agent.mcp_.workspace import Workspace


class Search:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def search_in_file(
        self,
        path: str,
        pattern: str,
        case_sensitive: bool = True,
    ) -> list[dict]:
        target = self.workspace.resolve(path)

        if not target.is_file():
            raise FileNotFoundError(path)

        if not pattern:
            raise ValueError("pattern cannot be empty")

        results = []

        with target.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                haystack = line if case_sensitive else line.lower()
                needle = pattern if case_sensitive else pattern.lower()

                if needle in haystack:
                    results.append(
                        {
                            "path": self.workspace.relative(target),
                            "line": line_number,
                            "text": line.rstrip("\n"),
                        }
                    )

        return results

    def search_files(
        self,
        pattern: str,
        path: str = "",
        case_sensitive: bool = True,
        max_results: int = 100,
    ) -> list[dict]:
        root = self.workspace.resolve(path)

        if not root.exists():
            raise FileNotFoundError(path)

        if not pattern:
            raise ValueError("pattern cannot be empty")

        ignored = {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "build",
            "dist",
        }

        needle = pattern if case_sensitive else pattern.lower()
        results = []

        for file_path in self._iter_files(root, ignored):
            if len(results) >= max_results:
                break

            try:
                with file_path.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    for line_number, line in enumerate(f, start=1):
                        haystack = (
                            line
                            if case_sensitive
                            else line.lower()
                        )

                        if needle in haystack:
                            results.append(
                                {
                                    "path": self.workspace.relative(
                                        file_path
                                    ),
                                    "line": line_number,
                                    "text": line.rstrip("\n"),
                                }
                            )

                            if len(results) >= max_results:
                                break

            except (OSError, UnicodeError):
                continue

        return results

    def _iter_files(
        self,
        root: Path,
        ignored: set[str],
    ):
        if root.is_file():
            yield root
            return

        for current, dirs, files in root.walk():
            dirs[:] = [
                d for d in dirs
                if d not in ignored
            ]

            for filename in files:
                yield current / filename
