"""A small generic plug-in registry.

Every extension point in cmtool -- flexure types, materials, solvers, design
strategies, linkage kinematics -- is a :class:`Registry`. This is the mechanism
that keeps the library general: adding a five-bar solver or a notch flexure means
registering a new object, never editing a dispatch table.

See ``docs/extending.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised when a registry lookup or registration fails."""


class Registry(Generic[T]):
    """A name -> object registry with decorator-based registration.

    Parameters
    ----------
    kind
        Human-readable name of what is being registered, used in error messages
        (for example ``"flexure type"``).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator registering the decorated object under ``name``."""

        def decorator(obj: T) -> T:
            self.add(name, obj)
            return obj

        return decorator

    def add(self, name: str, obj: T) -> None:
        """Register ``obj`` under ``name``.

        Raises
        ------
        RegistryError
            If ``name`` is already taken. Silent overwrites make plug-in bugs
            very hard to find, so they are refused.
        """
        if name in self._items:
            raise RegistryError(f"{self.kind} {name!r} is already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        """Look up a registered object by name."""
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none registered>"
            raise RegistryError(f"unknown {self.kind} {name!r}; available: {known}") from None

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[tuple[str, T]]:
        yield from sorted(self._items.items())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"
