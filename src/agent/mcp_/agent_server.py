from __future__ import annotations
import asyncio
from fastmcp import FastMCP
from agent.mcp_.filesystem import Filesystem
from agent.mcp_.search import Search
from agent.mcp_.workspace import Workspace
from agent.context import AgentContextStore
from agent.state import ActiveTaskStateStore, TaskState, TaskStatus, TaskStore
from agent.mcp_.git import Git
from agent.command import CommandExecutor
from typing import Literal


class AgentServer:
    def __init__(
        self,
        workspace_path: str,
        *,
        config_dir: str | None = None,
    ):
        self.workspace = Workspace(workspace_path)
        self.context = AgentContextStore(config_dir)
        self.tasks = TaskStore(
            self.workspace.resolve(".agent/task_history.db")
        )
        self.active_task_id: str | None = None

        # Transitional compatibility for code/tests that still access server.state.
        self.state = ActiveTaskStateStore(
            self.tasks,
            lambda: self.active_task_id,
        )

        self.filesystem = Filesystem(self.workspace)
        self.search = Search(self.workspace)
        self.git = Git(self.workspace)

        self.mcp = FastMCP("local-agent")

        self._register_tools()

    def start_task(self, title: str, goal: str) -> TaskState:
        if self.active_task_id is not None:
            active = self.tasks.get_state(self.active_task_id)
            if active.status == "in_progress":
                raise RuntimeError(
                    "A task is already active. Finish or cancel it before "
                    "starting another task."
                )

        state = self.tasks.create_task(title, goal)
        self.active_task_id = state.task_id
        return state

    def resume_task(self, task_id: str) -> TaskState:
        if self.active_task_id is not None:
            active = self.tasks.get_state(self.active_task_id)
            if active.status == "in_progress" and active.task_id != task_id:
                raise RuntimeError(
                    "A different task is already active. Finish or cancel it first."
                )

        state = self.tasks.get_state(task_id)
        if state.status in {"completed", "cancelled"}:
            raise RuntimeError(
                f"Task {task_id} is {state.status} and cannot be resumed."
            )
        if state.status == "blocked":
            state.status = "in_progress"
            state = self.tasks.save_state(state)

        self.active_task_id = task_id
        return state

    def get_active_task_state(self) -> TaskState:
        if self.active_task_id is None:
            return TaskState()
        return self.tasks.get_state(self.active_task_id)

    def save_active_task_state(self, state: TaskState) -> TaskState:
        if self.active_task_id is None:
            raise RuntimeError("No active task.")
        if state.task_id != self.active_task_id:
            raise RuntimeError("Task state does not belong to the active task.")
        return self.tasks.save_state(state)

    def finish_active_task(
        self,
        status: Literal["completed", "blocked", "cancelled"] = "completed",
    ) -> TaskState:
        if self.active_task_id is None:
            raise RuntimeError("No active task.")

        state = self.tasks.get_state(self.active_task_id)
        state.status = status
        return self.tasks.save_state(state)

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
        def get_agent_context(topic: str | None = None) -> dict:
            """
            Return agent identity information or one lazily loaded guidance topic.

            Call with no topic to retrieve agent identity and list the available topics.
            Retrieve a specific topic only when its detailed guidance is relevant to the current work.
            """
            return self.context.get(topic)

        @self.mcp.tool()
        def start_task(title: str, goal: str) -> dict:
            """
            Start persistent tracking for substantive work.

            Do not use this for greetings, casual conversation, or trivial factual
            questions. Use it when the work benefits from progress history or later
            resumption. Only one task may be active in this agent session at a time.
            """
            return self.start_task(title, goal).to_dict()

        @self.mcp.tool()
        def resume_task(task_id: str) -> dict:
            """Resume an unfinished persistent task by its stable task ID."""
            return self.resume_task(task_id).to_dict()

        @self.mcp.tool()
        def get_task_state(task_id: str | None = None) -> dict:
            """
            Return a task's latest state.

            If task_id is omitted, return the active task state. When no task is
            active, the result has active=false rather than inventing a task from the
            current chat message.
            """
            if task_id is None:
                return self.get_active_task_state().to_dict()
            return self.tasks.get_state(task_id).to_dict()

        @self.mcp.tool()
        def set_task_state(
            current: str | None = None,
            completed: list[str] | None = None,
            blocked: list[str] | None = None,
        ) -> dict:
            """
            Update model-managed progress fields for the active persistent task.

            completed and blocked are replacement-based lists. Task ID, title, goal,
            status, revision metadata, and modified files are runtime-owned.
            """
            state = self.get_active_task_state()
            if not state.active:
                raise RuntimeError(
                    "No active task. Call start_task only if this work warrants "
                    "persistent task tracking."
                )

            if current is not None:
                state.current = current
            if completed is not None:
                state.completed = completed
            if blocked is not None:
                state.blocked = blocked

            return self.save_active_task_state(state).to_dict()

        @self.mcp.tool()
        def finish_task(
            status: Literal["completed", "blocked", "cancelled"] = "completed",
        ) -> dict:
            """Finish the active persistent task with an explicit lifecycle status."""
            return self.finish_active_task(status).to_dict()

        @self.mcp.tool()
        def list_tasks(
            limit: int = 10,
            status: TaskStatus | None = None,
            before: str | None = None,
        ) -> list[dict]:
            """
            List bounded task summaries, newest first.

            Use before with the oldest returned updated_at timestamp to paginate
            without loading the entire history.
            """
            return self.tasks.list_tasks(
                limit=limit,
                status=status,
                before=before,
            )

        @self.mcp.tool()
        def search_tasks(query: str, limit: int = 10) -> list[dict]:
            """Search task titles, goals, and latest current-work text."""
            return self.tasks.search_tasks(query, limit=limit)

        @self.mcp.tool()
        def get_task_history(
            task_id: str,
            limit: int = 5,
            before_revision: int | None = None,
        ) -> dict:
            """
            Return a compact, bounded page of task-state revisions, newest first.

            Task metadata is returned once rather than repeated in every revision.
            Use next_before_revision to request an older page only when needed.
            """
            task = self.tasks.get_task(task_id)
            states = self.tasks.get_history(
                task_id,
                limit=limit,
                before_revision=before_revision,
            )
            revisions = [
                {
                    "revision": state.revision,
                    "saved_at": state.saved_at,
                    "status": state.status,
                    "current": state.current,
                    "completed": state.completed,
                    "blocked": state.blocked,
                    "files": state.files,
                }
                for state in states
            ]
            next_before_revision = None
            if len(states) == limit and states[-1].revision > 1:
                next_before_revision = states[-1].revision

            return {
                "task": task.to_dict(),
                "revisions": revisions,
                "next_before_revision": next_before_revision,
            }

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
                An empty file returns an empty dictionary.
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
        
        @self.mcp.tool()
        def move_item(old_path: str, new_path: str) -> dict:
            """Move or rename a file/folder inside the workspace."""
            return self.filesystem.move_item(old_path, new_path)

        @self.mcp.tool()
        def delete_item(path: str) -> dict:
            """Delete a file or empty directory inside the workspace."""
            return self.filesystem.delete_item(path)

        @self.mcp.tool()
        async def run_command(command: str, timeout: float = 30.0) -> dict:
            """Execute a shell command inside the workspace with bounded output."""
            executor = CommandExecutor(workspace=self.workspace.root, timeout=timeout)
            return await executor.run(command)

    def run(self) -> None:
        self.mcp.run()
