# Implementation status

This directory is the live status board for the build defined in
`specs/implementation-plan.md`.

- One file per task: `<TASK-ID>.md`, created from `TEMPLATE.md`.
- One file per phase gate: `phase-<N>.md`, created from
  `PHASE-TEMPLATE.md` by the phase integrator.
- Task files are owned by the task's implementer and reviewer. No other task
  edits them, so parallel work does not conflict here.
- The phase integrator is the only writer of `phase-<N>.md`.

Status values: `pending`, `in-progress`, `review`, `changes-requested`,
`merged`, `blocked`. Phase values: `pending`, `in-progress`, `passed`,
`failed`.

Do not put implementation notes here. Put them in the task's PR and in the
architecture document (only the human owner edits the architecture).
