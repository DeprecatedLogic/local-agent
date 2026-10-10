from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable
from fastmcp import FastMCP
from agent.mcp_.filesystem import Filesystem
from agent.mcp_.search import Search
from agent.mcp_.workspace import Workspace
from agent.context import AgentContextStore
from agent.embeddings import EmbeddingProvider
from agent.session import (
    ChatHistorySearch,
    ChatSessionAccess,
    ChatSessionStore,
)
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
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.workspace = Workspace(workspace_path)
        self.context = AgentContextStore(config_dir)
        self.tasks = TaskStore(
            self.workspace.resolve(".agent/task_history.db")
        )
        self.chat_sessions = ChatSessionStore(
            self.workspace.resolve(".agent/chat_history.db")
        )
        self.chat_access = ChatSessionAccess()
        self.current_chat_session_id: str | None = None
        self.chat_search = (
            ChatHistorySearch(
                self.chat_sessions,
                embedding_provider,
            )
            if embedding_provider is not None
            else None
        )
        self.active_task_id: str | None = None
        self._delegation_handler: (
            Callable[[str, str, str | None], Awaitable[dict]] | None
        ) = None
        self._agent_catalog_handler: Callable[[], list[dict]] | None = None

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

    def set_delegation_handler(
        self,
        handler: Callable[[str, str, str | None], Awaitable[dict]] | None,
    ) -> None:
        self._delegation_handler = handler

    def set_agent_catalog_handler(
        self,
        handler: Callable[[], list[dict]] | None,
    ) -> None:
        self._agent_catalog_handler = handler

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

    def attach_task(self, task_id: str | None) -> TaskState:
        """Attach an existing unfinished task to the current runtime session.

        This restores only the active-task pointer. It never changes task status,
        so resuming a chat cannot implicitly unblock or otherwise mutate a task.
        """
        if task_id is None:
            self.active_task_id = None
            return TaskState()

        state = self.tasks.get_state(task_id)
        if state.status in {"completed", "cancelled"}:
            self.active_task_id = None
            return TaskState()

        self.active_task_id = task_id
        return state

    def readable_chat_session_ids(self) -> list[str]:
        """Return saved session IDs visible through the current chat capability."""
        return [
            session_id
            for session_id in self.chat_sessions.all_session_ids()
            if self.chat_access.can_read(session_id)
        ]

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
        def list_chat_sessions(
            limit: int = 10,
            before: str | None = None,
            query: str | None = None,
        ) -> list[dict]:
            """
            List compact metadata for saved chat sessions.

            This reveals titles, IDs, timestamps, message counts, and task links,
            but never conversation contents. Use query to filter session titles.
            """
            return [
                session.to_dict()
                for session in self.chat_sessions.list_sessions(
                    limit=limit,
                    before=before,
                    query=query,
                )
            ]

        @self.mcp.tool()
        def read_chat_session(
            session_id: str,
            limit: int = 20,
            before_sequence: int | None = None,
            around_sequence: int | None = None,
            role: Literal["user", "agent"] | None = None,
        ) -> dict:
            """
            Read a bounded page from an explicitly authorized saved chat session.

            role="user" returns only user messages, role="agent" returns only
            assistant messages, and null returns both. Tool/system messages are not
            returned. The user grants temporary cross-session access with
            /allow-sessions; this tool cannot grant access to itself.
            """
            resolved = self.chat_sessions.resolve_session_id(session_id)
            if not self.chat_access.can_read(resolved):
                raise PermissionError(
                    "Chat session is not authorized for cross-session reading. "
                    "The user can grant access with /allow-sessions <session-id> "
                    "or /allow-sessions all."
                )
            return self.chat_sessions.read_messages(
                resolved,
                limit=limit,
                before_sequence=before_sequence,
                around_sequence=around_sequence,
                role=role,
            )

        @self.mcp.tool()
        def search_chat_history(
            query: str,
            role: Literal["user", "agent"] | None = None,
            limit: int = 5,
        ) -> list[dict]:
            """
            Semantically search authorized saved chat history using chunk embeddings.

            Results are bounded and contain session IDs, chunk ranges, similarity
            scores, and short previews. role selects the user-only, agent-only, or
            combined chunk view. Use read_chat_session around a returned sequence
            range only when more exact context is needed.
            """
            session_ids = self.readable_chat_session_ids()
            if not session_ids:
                raise PermissionError(
                    "No saved chat sessions are authorized for cross-session search. "
                    "The user can grant access with /allow-sessions <session-id> "
                    "or /allow-sessions all."
                )
            if self.chat_search is None:
                raise RuntimeError(
                    "Semantic chat search is unavailable because no embedding "
                    "provider is configured."
                )
            return self.chat_search.search(
                query,
                session_ids,
                role=role,
                limit=limit,
            )

        @self.mcp.tool()
        def delete_chat_session(session_id: str) -> dict:
            """
            Permanently delete one saved chat session when the user explicitly asks.

            This never deletes task history. The currently active chat cannot be
            deleted through MCP because doing so mid-turn would invalidate the live
            conversation; use /delete-current for that case. There is intentionally
            no MCP tool for deleting all sessions.
            """
            resolved = self.chat_sessions.resolve_session_id(session_id)
            if resolved == self.current_chat_session_id:
                raise RuntimeError(
                    "Cannot delete the current live chat through MCP. "
                    "Use /delete-current instead."
                )
            self.chat_sessions.delete_session(resolved)
            self.chat_access.forget_session(resolved)
            return {
                "deleted": True,
                "session_id": resolved,
            }

        @self.mcp.tool()
        def list_agents() -> list[dict]:
            """
            List specialist agents configured for bounded delegation.

            Agent definitions are loaded from the user's agents.toml file at startup.
            The catalog includes each enabled agent's ID, name, description, backend,
            and configured tools.
            """
            if self._agent_catalog_handler is None:
                raise RuntimeError(
                    "Agent catalog is unavailable because no orchestrator is attached."
                )
            return self._agent_catalog_handler()

        @self.mcp.tool()
        async def delegate_task(
            agent: str,
            task: str,
            context: str | None = None,
        ) -> dict:
            """
            Delegate one bounded, self-contained subtask to a configured specialist.

            Use list_agents to inspect the currently enabled specialists. Agent IDs are
            user-configurable and are not hardcoded into this tool schema. Workers are
            depth-one and cannot recursively delegate or own the parent task lifecycle.
            The primary agent remains responsible for integrating and validating the
            specialist result.
            """
            if self._delegation_handler is None:
                raise RuntimeError(
                    "Delegation is unavailable because no orchestrator is attached."
                )
            return await self._delegation_handler(agent, task, context)

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
        async def run_command(command: str, timeout: float = 180.0) -> dict:
            """
            Execute a shell command inside the workspace with bounded output.
            
            Args:
                command:
                    Shell command to execute.

                timeout:
                    By default 180 seconds. 
                    Increase it for commands that might take a long time to finish.
                    
            Notes:
                A short timeout can stop a command mid-way, but a very long timeout 
                can waste a lot of time waiting for a command that might be stuck (e.g.: looping).
            """
            executor = CommandExecutor(workspace=self.workspace.root, timeout=timeout)
            return await executor.run(command)

    def run(self) -> None:
        self.mcp.run()
