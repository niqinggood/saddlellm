"""Lazy, deterministic plugin registries for SaddleLLM extension points.

The registry intentionally depends only on the Python standard library.  A
base ``saddlellm`` install can therefore discover plugin metadata without
importing Torch, Transformers, serving runtimes, or the plugin implementation
itself.  Entry points are loaded only when a configured plugin is resolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import re
import threading
from typing import Any, Generic, Iterable, Optional, Tuple, TypeVar


PluginT = TypeVar("PluginT")
_PLUGIN_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class PluginRegistryError(RuntimeError):
    """Base error for plugin registration, discovery, and loading."""


class DuplicatePluginError(PluginRegistryError):
    """Raised when two providers claim the same normalized plugin name."""


class PluginNotFoundError(PluginRegistryError, LookupError):
    """Raised when a requested plugin has not been registered."""


class PluginLoadError(PluginRegistryError):
    """Raised when a lazy entry point cannot be imported."""


@dataclass(frozen=True)
class PluginInfo:
    """Read-only registry metadata suitable for diagnostics and CLIs."""

    name: str
    source: str
    loaded: bool
    entry_point: Optional[str] = None


@dataclass
class _Registration(Generic[PluginT]):
    provider: Optional[PluginT]
    source: str
    entry_point: Any = None
    loaded: bool = True


class PluginRegistry(Generic[PluginT]):
    """Register runtime providers and lazily discover packaging entry points.

    Built-in or explicitly registered providers take precedence over
    discovered entry points.  Duplicate entry points are recorded as discovery
    errors by default so an unrelated third-party package cannot prevent the
    framework from starting.  Pass ``strict=True`` to :meth:`discover` in
    conformance tests to fail on those conflicts.
    """

    def __init__(self, *, kind: str, entry_point_group: Optional[str] = None):
        self.kind = str(kind).strip() or "plugin"
        self.entry_point_group = entry_point_group
        self._registrations: dict[str, _Registration[PluginT]] = {}
        self._discovered = False
        self._discovery_errors: list[str] = []
        self._lock = threading.RLock()

    @staticmethod
    def normalize_name(name: str) -> str:
        normalized = str(name or "").strip().lower()
        if not normalized or not _PLUGIN_NAME.fullmatch(normalized):
            raise ValueError(
                "plugin name must match [a-z0-9][a-z0-9_.-]*; "
                f"got {name!r}"
            )
        return normalized

    def register(
        self,
        name: str,
        provider: PluginT,
        *,
        source: str = "runtime",
        replace: bool = False,
    ) -> PluginT:
        """Register an already available provider under a stable name."""

        normalized = self.normalize_name(name)
        if provider is None:
            raise TypeError(f"{self.kind} provider for {normalized!r} cannot be None")
        with self._lock:
            if normalized in self._registrations and not replace:
                existing = self._registrations[normalized]
                raise DuplicatePluginError(
                    f"{self.kind} {normalized!r} is already registered from "
                    f"{existing.source}"
                )
            self._registrations[normalized] = _Registration(
                provider=provider,
                source=str(source or "runtime"),
                loaded=True,
            )
        return provider

    def unregister(self, name: str) -> bool:
        """Remove one provider.  Intended for tests and explicit teardown."""

        normalized = self.normalize_name(name)
        with self._lock:
            return self._registrations.pop(normalized, None) is not None

    def discover(self, *, strict: bool = False, refresh: bool = False) -> Tuple[str, ...]:
        """Discover entry points without importing their target modules."""

        if not self.entry_point_group:
            return ()
        with self._lock:
            if self._discovered and not refresh:
                if strict and self._discovery_errors:
                    raise PluginRegistryError("; ".join(self._discovery_errors))
                return ()
            if refresh:
                for name in [
                    key
                    for key, registration in self._registrations.items()
                    if registration.entry_point is not None
                ]:
                    self._registrations.pop(name, None)
                self._discovery_errors.clear()

            discovered: list[str] = []
            try:
                available = metadata.entry_points()
                if hasattr(available, "select"):
                    selected: Iterable[Any] = available.select(
                        group=self.entry_point_group
                    )
                else:  # pragma: no cover - Python/importlib compatibility path
                    selected = available.get(self.entry_point_group, ())
            except Exception as exc:  # pragma: no cover - platform metadata failure
                message = (
                    f"could not discover {self.kind} entry points from "
                    f"{self.entry_point_group!r}: {exc}"
                )
                self._discovery_errors.append(message)
                self._discovered = True
                if strict:
                    raise PluginRegistryError(message) from exc
                return ()

            for entry_point in sorted(
                selected,
                key=lambda item: (str(item.name).lower(), str(item.value)),
            ):
                try:
                    normalized = self.normalize_name(entry_point.name)
                    if normalized in self._registrations:
                        existing = self._registrations[normalized]
                        raise DuplicatePluginError(
                            f"{self.kind} {normalized!r} from entry point "
                            f"{entry_point.value!r} conflicts with {existing.source}"
                        )
                    self._registrations[normalized] = _Registration(
                        provider=None,
                        source=f"entry-point:{entry_point.value}",
                        entry_point=entry_point,
                        loaded=False,
                    )
                    discovered.append(normalized)
                except (DuplicatePluginError, ValueError) as exc:
                    self._discovery_errors.append(str(exc))
                    if strict:
                        raise
            self._discovered = True
            return tuple(discovered)

    def resolve(self, name: str) -> PluginT:
        """Resolve one provider, importing a lazy entry point at most once."""

        normalized = self.normalize_name(name)
        self.discover()
        with self._lock:
            registration = self._registrations.get(normalized)
            if registration is None:
                available = ", ".join(sorted(self._registrations)) or "none"
                raise PluginNotFoundError(
                    f"Unknown {self.kind} {normalized!r}; available: {available}"
                )
            if not registration.loaded:
                try:
                    registration.provider = registration.entry_point.load()
                    registration.loaded = True
                except Exception as exc:
                    raise PluginLoadError(
                        f"Failed to load {self.kind} {normalized!r} from "
                        f"{registration.source}: {exc}"
                    ) from exc
            if registration.provider is None:  # defensive invariant
                raise PluginLoadError(
                    f"{self.kind} {normalized!r} resolved without a provider"
                )
            return registration.provider

    def contains(self, name: str, *, discover: bool = True) -> bool:
        try:
            normalized = self.normalize_name(name)
        except ValueError:
            return False
        if discover:
            self.discover()
        with self._lock:
            return normalized in self._registrations

    def names(self, *, discover: bool = True) -> Tuple[str, ...]:
        if discover:
            self.discover()
        with self._lock:
            return tuple(sorted(self._registrations))

    def info(self, *, discover: bool = True) -> Tuple[PluginInfo, ...]:
        if discover:
            self.discover()
        with self._lock:
            return tuple(
                PluginInfo(
                    name=name,
                    source=registration.source,
                    loaded=registration.loaded,
                    entry_point=(
                        str(registration.entry_point.value)
                        if registration.entry_point is not None
                        else None
                    ),
                )
                for name, registration in sorted(self._registrations.items())
            )

    @property
    def discovery_errors(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._discovery_errors)


__all__ = [
    "DuplicatePluginError",
    "PluginInfo",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginRegistry",
    "PluginRegistryError",
]
