from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock, RLock

# Expression declarations, conversion declarations, and deferred rules are
# lazily initialized and shared by all e-graphs. The metadata lock also protects
# the canonical CallDecl pool so recursive literal hashing cannot invert locks.
# Registration is setup-time activity and must not race execution, and
# initialization callbacks must not wait for another thread that may access DSL
# metadata (see docs/reference/usage.md).
INITIALIZE_LOCK = RLock()
INITIALIZATION_ACTIVE = Lock()


@contextmanager
def initialize() -> Generator[None, None, None]:
    """Serialize initialization and publish its transitive state atomically."""
    with INITIALIZE_LOCK:
        if INITIALIZATION_ACTIVE.locked():
            # The RLock means only this outer initializer can hold the marker.
            yield
        else:
            with INITIALIZATION_ACTIVE:
                yield
