"""Run mode exit codes (D-17)."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Run mode exit codes (D-17).

    Severity ordering is defined in EXIT_SEVERITY (D-18).
    """

    SUCCESS = 0
    GENERAL_ERROR = 1
    PENDING_REMAIN = 2
    DB_FETCH_ERROR = 3
    DATA_INTEGRITY_ERROR = 4
    MODEL_VALIDATION_ERROR = 5
    REPORT_ERROR = 6
    SIGINT = 130


# D-18: severity ordering (higher = more severe)
EXIT_SEVERITY: dict[ExitCode, int] = {
    ExitCode.SUCCESS: 0,
    ExitCode.PENDING_REMAIN: 1,
    ExitCode.GENERAL_ERROR: 2,
    ExitCode.REPORT_ERROR: 3,
    ExitCode.DB_FETCH_ERROR: 4,
    ExitCode.DATA_INTEGRITY_ERROR: 5,
    ExitCode.MODEL_VALIDATION_ERROR: 6,
    ExitCode.SIGINT: 7,
}


def determine_final_exit_code(errors: list[ExitCode]) -> ExitCode:
    """Return the highest-severity exit code from the list, or SUCCESS if empty."""
    if not errors:
        return ExitCode.SUCCESS
    return max(errors, key=lambda e: EXIT_SEVERITY[e])
