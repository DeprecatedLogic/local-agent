from __future__ import annotations

from fastmcp import FastMCP

from agent.mcp.filesystem import Filesystem
from agent.mcp.search import Search
from agent.mcp.workspace import Workspace
from agent.state import StateStore
from agent.mcp.git import Git

class AgentServer:
    def __init__(self, workspace_path: str):
        self.workspace = Workspace(workspace_path)

        self.state = StateStore(
            self.workspace.resolve(".agent/state.json")
        )

        self.filesystem = Filesystem(self.workspace)
        self.search = Search(self.workspace)
        self.git = Git(self.workspace)

        self.mcp = FastMCP("local-agent")

        self._register_tools()

    def _register_tools(self) -> None:

        @self.mcp.tool()
        def project_info() -> dict:
            """
            Return basic information about the currently active project workspace.

            Returns:
                A dictionary containing:
                - workspace: absolute filesystem path of the project workspace.

            Notes:
                This operation is read-only.
                The returned workspace path identifies the root directory that all
                filesystem tools are restricted to.
            """
            return {
                "workspace": str(self.workspace.root),
            }

        @self.mcp.tool()
        def get_task_state() -> dict:
            """
            Return the persistent state of the current agent task.

            Returns:
                A dictionary containing the current task state, including:
                - task: description of the overall task.
                - status: current task status.
                - current: description of the work currently being performed.
                - completed: completed task items.
                - blocked: blocked task items and reasons.
                - files: files relevant to the task.

            Notes:
                Task state is persistent and stored inside the project workspace.
                This operation is read-only.
            """
            return self.state.load().to_dict()

        @self.mcp.tool()
        def set_task_state(
            task: str | None = None,
            status: str | None = None,
            current: str | None = None,
        ) -> dict:
            """
            Update one or more fields of the persistent agent task state.

            Args:
                task:
                    Optional description of the overall task.
                    If omitted, the existing task description is preserved.

                status:
                    Optional task status.
                    If omitted, the existing status is preserved.

                current:
                    Optional description of the work currently being performed.
                    If omitted, the existing value is preserved.

            Returns:
                The complete updated task state.

            Notes:
                Only arguments explicitly provided are changed.
                Existing completed, blocked, and files fields are preserved.
                This operation modifies persistent task state.
            """
            state = self.state.load()

            if task is not None:
                state.task = task

            if status is not None:
                state.status = status

            if current is not None:
                state.current = current

            self.state.save(state)

            return state.to_dict()

        @self.mcp.tool()
        def read_file(
            path: str,
            start_line: int | None = None,
            end_line: int | None = None,
        ) -> dict:
            """
            Read text from a file inside the project workspace.

            Args:
                path:
                    Workspace-relative path to the file.
                    Absolute paths are not allowed.

                start_line:
                    Optional 1-based first line to read.
                    If omitted, reading starts at line 1.

                end_line:
                    Optional 1-based last line to read, inclusive.
                    If omitted, reading continues to the end of the file.

            Returns:
                A dictionary mapping 1-based line numbers to their contents.

            Notes:
                Only files inside the configured workspace can be accessed.
                This operation does not modify the file.
            """
            return self.filesystem.read_file(
                path,
                start_line,
                end_line,
            )

        @self.mcp.tool()
        def write_file(
            path: str,
            content: str,
            append: bool = False,
        ) -> dict:
            """
            Write text to a file inside the project workspace.

            Args:
                path:
                    Workspace-relative path of the target file.
                    Absolute paths are not allowed.

                content:
                    Exact text to write to the file.

                append:
                    If false, replace the existing file contents.
                    If true, append content to the end of the file.

            Returns:
                Information about the modified file, including its workspace-relative
                path, number of bytes written, and write mode.

            Warning:
                When append is false, existing file contents are replaced.
            """
            return self.filesystem.write_file(
                path,
                content,
                append,
            )

        @self.mcp.tool()
        def edit_file_lines(
            path: str,
            start_line: int,
            end_line: int,
            content: str,
        ) -> dict:
            """
            Replace a range of lines in a file, or insert text at a specific position.

            Args:
                path:
                    Workspace-relative path of the file to modify.

                start_line:
                    1-based first line to replace.
                    Use -1 to insert instead of replacing.

                end_line:
                    1-based last line to replace, inclusive.
                    When start_line is -1, content is inserted immediately after
                    this line. Use 0 to insert at the beginning of the file.

                content:
                    Exact replacement or insertion text.

            Returns:
                Information about the modified file and resulting line count.

            Examples:
                start_line=10, end_line=15:
                    Replace lines 10 through 15.

                start_line=-1, end_line=20:
                    Insert content immediately after line 20.

                start_line=-1, end_line=0:
                    Insert content at the beginning of the file.

            Notes:
                This operation modifies the file in place.
            """
            return self.filesystem.edit_file_lines(
                path,
                start_line,
                end_line,
                content,
            )

        @self.mcp.tool()
        def create_file(path: str) -> dict:
            """
            Create a new empty file inside the project workspace.

            Args:
                path:
                    Workspace-relative path of the file to create.
                    Absolute paths are not allowed.
                    Parent directories are created automatically if necessary.

            Returns:
                A dictionary containing:
                - path: workspace-relative path of the created file.
                - created: true when creation succeeds.

            Raises:
                FileExistsError:
                    If a file or directory already exists at the specified path.

            Notes:
                This operation creates an empty file and does not modify existing files.
            """
            return self.filesystem.create_file(path)

        @self.mcp.tool()
        def create_folder(path: str) -> dict:
            """
            Create a directory inside the project workspace.

            Args:
                path:
                    Workspace-relative path of the directory to create.
                    Absolute paths are not allowed.
                    Parent directories are created automatically if necessary.

            Returns:
                A dictionary containing:
                - path: workspace-relative path of the created directory.
                - created: true when creation succeeds.

            Raises:
                FileExistsError:
                    If the target path already exists.

            Notes:
                This operation modifies the project filesystem.
            """
            return self.filesystem.create_folder(path)

        @self.mcp.tool()
        def list_dir_contents(path: str = "") -> list[dict]:
            """
            List the immediate contents of a directory in the project workspace.

            Args:
                path:
                    Workspace-relative path of the directory to inspect.
                    An empty string refers to the workspace root.
                    Absolute paths are not allowed.

            Returns:
                A list of entries. Each entry contains:
                - name: name of the file or directory.
                - path: workspace-relative path.
                - type: either "file" or "directory".

            Notes:
                Only the immediate contents are returned.
                Subdirectories are not recursively traversed.
                This operation is read-only.
            """
            return self.filesystem.list_dir_contents(path)

        @self.mcp.tool()
        def list_tree(
            path: str = "",
            max_depth: int = 8,
            max_entries: int = 5000,
        ) -> list[dict]:
            """
            List files and directories recursively inside the workspace.

            Args:
                path:
                    Optional workspace-relative directory to inspect.
                    An empty string refers to the workspace root.

                max_depth:
                    Maximum directory depth to traverse.

                max_entries:
                    Maximum number of entries to return.

            Returns:
                A list containing workspace-relative paths and their entry types.

            Notes:
                Version-control, dependency, cache, and common generated directories
                are skipped to avoid returning unnecessarily large results.
                This operation does not modify the workspace.
            """
            return self.filesystem.list_tree(
                path,
                max_depth,
                max_entries,
            )

        @self.mcp.tool()
        def search_in_file(
            path: str,
            pattern: str,
            case_sensitive: bool = True,
        ) -> list[dict]:
            """
            Search for literal text inside a single file.

            Args:
                path:
                    Workspace-relative path of the file to search.
                    Absolute paths are not allowed.

                pattern:
                    Literal text to search for.
                    This is not a regular expression.

                case_sensitive:
                    If true, uppercase and lowercase characters must match exactly.
                    If false, matching ignores character case.

            Returns:
                A list of matching lines. Each match contains:
                - path: workspace-relative file path.
                - line: 1-based line number.
                - text: complete contents of the matching line.

            Notes:
                Multiple matches on the same line produce only one result.
                This operation is read-only.
            """
            return self.search.search_in_file(
                path,
                pattern,
                case_sensitive,
            )

        @self.mcp.tool()
        def search_files(
            pattern: str,
            path: str = "",
            case_sensitive: bool = True,
            max_results: int = 100,
        ) -> list[dict]:
            """
            Search text across files in the project workspace.

            Args:
                pattern:
                    Literal text to search for.
                    This is not a regular expression.

                path:
                    Optional workspace-relative directory or file to search.
                    An empty string searches the entire workspace.

                case_sensitive:
                    Whether matching should distinguish uppercase and lowercase
                    characters.

                max_results:
                    Maximum number of matching lines to return.

            Returns:
                A list of matches. Each match contains:
                - path: workspace-relative file path
                - line: 1-based line number
                - text: complete matching line

            Notes:
                Common generated and dependency directories such as .git,
                node_modules, build, dist, and virtual environments are ignored.
            """
            return self.search.search_files(
                pattern,
                path,
                case_sensitive,
                max_results,
            )

        @self.mcp.tool()
        def git_status() -> dict:
            """
            Return the current Git repository status.

            Returns:
                A dictionary containing:
                - ok: whether the Git operation succeeded.
                - repository: whether the workspace is a Git repository.
                - branch: current branch name, or null when detached.
                - upstream: configured upstream branch, when available.
                - clean: whether the working tree has no changes.
                - staged: files with staged changes.
                - unstaged: files with unstaged changes.
                - untracked: untracked files.
                - conflicted: files with merge conflicts.

            Notes:
                This operation is read-only.
                The repository is always scoped to the current workspace.
            """
            return self.git.status()

        @self.mcp.tool()
        def git_diff(
            staged: bool = False,
            path: str | None = None,
        ) -> dict:
            """
            Return changes in the current Git working tree.

            Args:
                staged:
                    If true, return changes currently staged in the Git index.
                    If false, return unstaged working-tree changes.

                path:
                    Optional workspace-relative path to restrict the diff.
                    Absolute paths are not allowed.

            Returns:
                A dictionary containing:
                - ok: whether the Git operation succeeded.
                - staged: whether the staged diff was requested.
                - path: the requested path, if any.
                - diff: Git's unified diff output.
                - error: error text when the operation fails.

            Notes:
                Diff output is bounded to prevent excessively large results.
                This operation is read-only.
            """
            return self.git.diff(
                staged=staged,
                path=path,
            )

        @self.mcp.tool()
        def git_log(max_count: int = 20) -> dict:
            """
            Return recent commits from the current Git repository.

            Args:
                max_count:
                    Maximum number of commits to return.
                    Must be between 1 and 100.

            Returns:
                A dictionary containing:
                - ok: whether the Git operation succeeded.
                - commits: list of commits, each containing:
                  - hash: full commit hash.
                  - short_hash: abbreviated commit hash.
                  - author: commit author.
                  - date: ISO-formatted commit timestamp.
                  - subject: commit subject.
                - error: error text when the operation fails.

            Notes:
                This operation is read-only.
            """
            return self.git.log(max_count=max_count)

        @self.mcp.tool()
        def git_show(
            revision: str,
            path: str | None = None,
        ) -> dict:
            """
            Show the contents and metadata of a Git revision.

            Args:
                revision:
                    Git revision to inspect, such as a commit hash,
                    branch name, or HEAD.

                path:
                    Optional workspace-relative path to restrict the output.
                    Absolute paths are not allowed.

            Returns:
                A dictionary containing:
                - ok: whether the Git operation succeeded.
                - revision: the requested revision.
                - path: the requested path, if any.
                - content: Git's formatted revision output.
                - error: error text when the operation fails.

            Notes:
                Output is bounded to prevent excessively large results.
                This operation is read-only.
            """
            return self.git.show(
                revision=revision,
                path=path,
            )

    def run(self) -> None:
        self.mcp.run()
