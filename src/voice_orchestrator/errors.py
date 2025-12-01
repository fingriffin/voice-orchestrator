"""Custom exceptions for the VOICE orchestration module."""

class ErrorMessages:
    """Error messages for Pod operations."""

    POD_INTERRUPTED_WAITING = "Wait for pod spin-up interrupted by user."
    POD_INTERRUPTED_EXECUTION = "Execution interrupted by user."

class PodError(Exception):
    """Base exception for Pod-related errors."""

    pass

class PodCommandError(PodError):
    """Exception for errors during Pod command execution."""

    def __init__(self, message: str):
        """
        Initialize PodCommandError with a message.

        :param message: Error message describing the command failure.
        """
        super().__init__(message)
        self.message = message

class PodInterrupted(PodError):
    """Exception for user interruptions during Pod operations."""

    pass
