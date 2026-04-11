import sys
import traceback
from typing import Optional, Any, Dict


class DocumentPortalException(Exception):
    MAX_CONTEXT_LEN = 500  # truncate context values

    def __init__(
        self,
        error_message: Any,
        error_details: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        raise_from_original: bool = True,  # chain original exception
    ):
        self.error_message = str(error_message)
        self.context = self._safe_context(context or {})

        # Extract exception info
        exc_type, exc_value, exc_tb = self._extract_exc_info(error_details)

        # Get last traceback frame
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = (
            last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        )
        self.lineno = last_tb.tb_lineno if last_tb else -1

        # Store full traceback (for logs)
        self.traceback_str = (
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            if exc_type and exc_tb
            else ""
        )

        # Store original exception
        self.original_exception = exc_value

        # Initialize Exception with user-friendly message
        super().__init__(self._user_message())

        # Optionally raise from original exception for chaining
        if raise_from_original and self.original_exception:
            self.__cause__ = self.original_exception

    # -----------------------------
    # Helpers
    # -----------------------------
    def _extract_exc_info(self, error_details):
        if error_details is None:
            return sys.exc_info()
        if hasattr(error_details, "exc_info"):
            return error_details.exc_info()
        if isinstance(error_details, BaseException):
            return (
                type(error_details),
                error_details,
                error_details.__traceback__,
            )
        return sys.exc_info()

    def _safe_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        safe_ctx = {}
        for k, v in context.items():
            try:
                val = str(v)
                safe_ctx[k] = val[: self.MAX_CONTEXT_LEN]
            except Exception:
                safe_ctx[k] = "<unserializable>"
        return safe_ctx

    def _user_message(self) -> str:
        base = f"{self.error_message}"
        if self.context:
            ctx_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            base += f" | Context: {ctx_str}"
        return base

    # -----------------------------
    # Representations
    # -----------------------------
    def __str__(self):
        return self._user_message()

    def debug(self) -> str:
        """Full debug string (use in logs only)"""
        base = (
            f"Error in [{self.file_name}] at line [{self.lineno}] | "
            f"Message: {self.error_message}"
        )
        if self.context:
            ctx_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            base += f" | Context: {ctx_str}"

        if self.traceback_str:
            base += f"\nTraceback:\n{self.traceback_str}"

        return base

    def __repr__(self):
        return (
            f"DocumentPortalException("
            f"file={self.file_name!r}, "
            f"line={self.lineno}, "
            f"message={self.error_message!r}, "
            f"context={self.context!r})"
        )

    # -----------------------------
    # Optional helper
    # -----------------------------
    @staticmethod
    def log_exception(e: "DocumentPortalException", logger) -> None:
        """Use this in production to log full debug info"""
        logger.error(e.debug())
