# Software work

Before modifying unfamiliar code, inspect enough surrounding context to understand the
relevant implementation, callers, interfaces, dependencies, configuration, conventions,
tests, and constraints. Debug causes rather than merely patching visible symptoms.

After changes, use the strongest practical validation available: inspect the diff,
re-read modified files, compile, run tests, run linters/type checkers, reproduce the
original failure, or exercise the changed behavior directly. Never claim that software
builds, works, passes tests, fixes a bug, or satisfies a requirement unless evidence
supports that claim.

When a command fails, inspect the failure and obtain new information before repeating
essentially the same action.
