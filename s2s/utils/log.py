"""
Logging helper for the S2S framework.

Uses m00nny_utils.system.log.Log (multiprocess-safe, mp.Queue-backed
QueueListener) when available; falls back to stdlib logging with the
same format string so output is identical in both cases.

Usage:
    from s2s.utils.log import get_logger
    _log = get_logger("s2s.omni2")
    _log.info("Loading weights...")
    _log.warning("Missing keys: ...")
"""
import logging
import sys
import os

# Inject the project root so m00nny_utils is importable without install
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from m00nny_utils.system.log import Log as _M00nnyLog
    _HAVE_M00NNY = True
except ImportError:
    _HAVE_M00NNY = False

_FMT = "[ %(asctime)s | %(name)s | %(levelname)s ]: %(message)s"


def get_logger(name: str, level: str = "INFO"):
    """Return a named logger.

    The returned object exposes:  .debug(), .info(), .warning(), .error(), .critical()

    When m00nny_utils is available the logger is backed by a single
    multiprocess mp.Queue so it is safe across DDP worker processes.
    When it is not available a stdlib StreamHandler to stdout is used.

    Args:
        name:  Logger name (shown in every log line, e.g. "s2s.omni2").
        level: Minimum level for this logger ("DEBUG" / "INFO" / "WARNING" / …).
    """
    if _HAVE_M00NNY:
        return _M00nnyLog(
            name=name,
            level=level,
            global_level="INFO",
            use_STDOUT=True,
            use_STDERR=False,
        )

    # ── stdlib fallback ─────────────────────────────────────────────────
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
