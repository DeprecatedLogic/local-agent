# Persistent tasks

A chat message and a persistent task are different things. Greetings, casual discussion,
small factual questions, and other lightweight turns should remain ordinary conversation.
Create a persistent task only when work is substantive enough to benefit from progress
tracking, history, or later resumption.

A task has stable metadata (ID, title, goal, lifecycle status) and versioned progress
state (current work, completed items, blockers, and modified files). Task state is
descriptive rather than authoritative evidence of what happened.

Use start_task to begin substantive tracked work. Use set_task_state to maintain current,
completed, and blocked fields. completed and blocked are replacement-based lists, so
provide the complete current list when changing them. Files are runtime-owned and are
updated from successful file-mutating tool calls. Use finish_task when tracked work is
completed, blocked, or cancelled.

Use list_tasks/search_tasks to locate relevant prior work and get_task_history with a
small limit to retrieve only the revisions needed. Do not retrieve large amounts of
history when a narrower query is sufficient.
