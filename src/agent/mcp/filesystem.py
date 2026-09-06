from __future__ import annotations

import os
from pathlib import Path

from .workspace import Workspace


class Filesystem:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict:
        target = self.workspace.resolve(path)

        if not target.is_file():
            raise FileNotFoundError(path)

        with target.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)

        if start_line is None:
            start_line = 1

        if end_line is None:
            end_line = total

        if start_line < 1:
            raise ValueError("start_line must be >= 1")

        if end_line < start_line:
            raise ValueError("end_line must be >= start_line")

        start = start_line - 1
        end = min(end_line, total)

        return {
            i + 1: lines[i].rstrip("\n")
            for i in range(start, end)
        }

    def write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> dict:
        target = self.workspace.resolve(path)

        target.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"

        with target.open(mode, encoding="utf-8") as f:
            f.write(content)

        return {
            "path": self.workspace.relative(target),
            "bytes": len(content.encode("utf-8")),
            "mode": "append" if append else "write",
        }

    def create_file(self, path: str) -> dict:
        target = self.workspace.resolve(path)

        if target.exists():
            raise FileExistsError(path)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

        return {
            "path": self.workspace.relative(target),
            "created": True,
        }

    def edit_file_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        content: str,
    ) -> dict:
        target = self.workspace.resolve(path)

        if not target.is_file():
            raise FileNotFoundError(path)

        if start_line < -1:
            raise ValueError("start_line must be >= 1 or -1")

        if start_line != -1 and start_line < 1:
            raise ValueError("start_line must be >= 1 or -1")

        if end_line < 0:
            raise ValueError("end_line must be >= 0")

        with target.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line == -1:
            if end_line > len(lines):
                raise ValueError(
                    f"Cannot insert after line {end_line}; "
                    f"file has {len(lines)} lines"
                )

            insert_at = end_line
            replacement = content.splitlines(keepends=True)

            lines[insert_at:insert_at] = replacement

        else:
            if start_line > len(lines) + 1:
                raise ValueError(
                    f"start_line {start_line} is beyond EOF"
                )

            if end_line < start_line:
                raise ValueError(
                    "end_line must be >= start_line"
                )

            start = start_line - 1
            end = min(end_line, len(lines))

            replacement = content.splitlines(keepends=True)

            lines[start:end] = replacement

        with target.open("w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "path": self.workspace.relative(target),
            "start_line": start_line,
            "end_line": end_line,
            "lines": len(lines),
        }

    def create_folder(self, path: str) -> dict:
        target = self.workspace.resolve(path)

        target.mkdir(parents=True, exist_ok=False)

        return {
            "path": self.workspace.relative(target),
            "created": True,
        }

    def move_item(
        self,
        old_path: str,
        new_path: str,
    ) -> dict:
        source = self.workspace.resolve(old_path)
        destination = self.workspace.resolve(new_path)

        if not source.exists():
            raise FileNotFoundError(old_path)

        if destination.exists():
            raise FileExistsError(new_path)

        destination.parent.mkdir(parents=True, exist_ok=True)

        source.rename(destination)

        return {
            "old_path": self.workspace.relative(source),
            "new_path": self.workspace.relative(destination),
        }

    def delete_item(self, path: str) -> dict:
        target = self.workspace.resolve(path)

        if not target.exists():
            raise FileNotFoundError(path)

        if target.is_dir():
            for child in target.iterdir():
                if child.is_dir():
                    self._remove_directory(child)
                else:
                    child.unlink()

            target.rmdir()
        else:
            target.unlink()

        return {
            "path": self.workspace.relative(target),
            "deleted": True,
        }

    def list_dir_contents(self, path: str = "") -> list[dict]:
        target = self.workspace.resolve(path)

        if not target.is_dir():
            raise NotADirectoryError(path)

        result = []

        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            result.append(
                {
                    "name": entry.name,
                    "path": self.workspace.relative(entry),
                    "type": (
                        "directory"
                        if entry.is_dir()
                        else "file"
                    ),
                }
            )

        return result

    def list_tree(
        self,
        path: str = "",
        max_depth: int = 8,
        max_entries: int = 5000,
    ) -> list[dict]:
        root = self.workspace.resolve(path)

        if not root.is_dir():
            raise NotADirectoryError(path)

        result = []
        ignored = {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
        }

        def walk(directory: Path, depth: int) -> None:
            if len(result) >= max_entries:
                return

            if depth > max_depth:
                return

            try:
                entries = sorted(
                    directory.iterdir(),
                    key=lambda p: p.name.lower(),
                )
            except PermissionError:
                return

            for entry in entries:
                if len(result) >= max_entries:
                    return

                if entry.name in ignored:
                    continue

                relative = self.workspace.relative(entry)

                result.append(
                    {
                        "path": relative,
                        "type": (
                            "directory"
                            if entry.is_dir()
                            else "file"
                        ),
                    }
                )

                if entry.is_dir():
                    walk(entry, depth + 1)

        walk(root, 0)

        return result

    @staticmethod
    def _remove_directory(path: Path) -> None:
        for child in path.iterdir():
            if child.is_dir():
                Filesystem._remove_directory(child)
            else:
                child.unlink()

        path.rmdir()
