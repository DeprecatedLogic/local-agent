from pathlib import Path


class Workspace:
    """Provides secure access to the agent's project workspace."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

        if not self.root.exists():
            self.root.mkdir(parents=True)

        if not self.root.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.root}")

    def resolve(self, relative_path: str | Path = "") -> Path:
        """
        Resolve a path inside the workspace.

        Raises ValueError if the resulting path escapes the workspace.
        """

        relative = Path(relative_path)

        if relative.is_absolute():
            raise ValueError("Absolute paths are not allowed")

        target = (self.root / relative).resolve()

        try:
            target.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"Path escapes workspace: {relative_path}"
            )

        return target

    def relative(self, path: str | Path) -> str:
        """Convert an absolute workspace path back to a relative path."""

        target = Path(path).resolve()

        try:
            return str(target.relative_to(self.root))
        except ValueError:
            raise ValueError(f"Path is outside workspace: {path}")
