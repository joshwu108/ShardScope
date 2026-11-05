"""Stage complete runs before exposing them; never replace a competing run."""

import ctypes
import errno
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path

from shardscope.errors import ConfigError


def publish_exclusive(source, destination):
    """Atomically rename a directory only if the destination is absent."""
    if sys.platform == "win32":
        # Windows rename refuses existing destinations.
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(source), os.fsencode(destination), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise OSError(errno.ENOTSUP, "exclusive directory publication is unsupported")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


@contextmanager
def staged_run(destination):
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (
        destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise ConfigError(f"refusing nonempty or non-directory output: {destination}")
    # Reserve the public name exclusively. It stays empty until publication.
    # rmdir only removes an empty existing destination; a competing writer wins.
    if destination.exists():
        destination.rmdir()
    destination.mkdir()
    identity = destination.stat()
    stage = None
    owns_reservation = True
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        yield stage
        current = destination.lstat()
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise ConfigError(f"output destination changed during execution: {destination}")
        # Remove only our empty reservation. The exclusive rename also refuses
        # an empty directory created by another writer after this rmdir.
        destination.rmdir()
        owns_reservation = False
        publish_exclusive(stage, destination)
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        if owns_reservation and destination.exists():
            current = destination.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                with suppress(OSError):
                    destination.rmdir()
