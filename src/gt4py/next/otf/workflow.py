# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import functools
import typing
from collections.abc import Callable, Hashable, Sequence
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
    overload,
    TYPE_CHECKING,
)

from typing_extensions import Self
from zmq import TYPE

from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import OpaqueMutableMapping
from gt4py.next import utils


InT = TypeVar("InT")
InT_contra = TypeVar("InT_contra", contravariant=True)
OutT = TypeVar("OutT")
OutT_co = TypeVar("OutT_co", covariant=True)
NewOutT = TypeVar("NewOutT")
NewOutT_co = TypeVar("NewOutT_co", covariant=True)
IntermediateT = TypeVar("IntermediateT")
HashT = TypeVar("HashT", bound=collections.abc.Hashable)
DataT = TypeVar("DataT")
ArgT = TypeVar("ArgT")


class _WorkflowCall(Protocol[InT_contra, OutT_co]):
    def __call__(self, inp: InT_contra, /) -> OutT_co: ...


class _FingerprintedWorkflow(
    _WorkflowCall[InT_contra, OutT_co], utils.Fingerprinted, Protocol[InT_contra, OutT_co]
): ...


class _HashableWorkflow(
    _WorkflowCall[InT_contra, OutT_co], Hashable, Protocol[InT_contra, OutT_co]
): ...


Workflow: TypeAlias = (
    _FingerprintedWorkflow[InT_contra, OutT_co] | _HashableWorkflow[InT_contra, OutT_co]
)


class _MultiWorkflowSteps(Protocol):
    @property
    def steps(self) -> Sequence[Workflow]: ...


class _FingerprintedMultiWorkflow(
    _MultiWorkflowSteps, _FingerprintedWorkflow[InT_contra, OutT_co], Protocol[InT_contra, OutT_co]
): ...


class _HashableMultiWorkflow(
    _MultiWorkflowSteps, _HashableWorkflow[InT_contra, OutT_co], Protocol[InT_contra, OutT_co]
): ...


MultiWorkflow: TypeAlias = (
    _FingerprintedMultiWorkflow[InT_contra, OutT_co] | _HashableMultiWorkflow[InT_contra, OutT_co]
)


@dataclasses.dataclass(frozen=True)
class WorkflowABC(
    utils.CachedFingerprintedMixin, utils.ModelPicklerMixin, Generic[InT_contra, OutT_co]
):
    """Base implementation of a workflow step, with caching and optimized fingerprinting."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if (
            cls is Workflow
            and issubclass(subclass, (Hashable, utils.FingerprintedABC))
            and issubclass(subclass, collections.abc.Callable)
        ):
            return True

        return NotImplemented

    @abc.abstractmethod
    def __call__(self, inp: InT_contra, /) -> OutT_co: ...

    @abc.abstractmethod
    def chain(self, next_step: Workflow[OutT_co, NewOutT]) -> WorkflowABC[InT, NewOutT]: ...


@dataclasses.dataclass(frozen=True)
class Step(WorkflowABC[InT, OutT]):
    """Convenience base class for workflow steps, with chaining and replacement support."""

    wf: Workflow[InT, OutT]
    name: str = dataclasses.field(default="<unnamed>", kw_only=True)

    def __call__(self, inp: InT) -> OutT:
        return self.wf(inp)

    def chain(self, next_step: Workflow[OutT_co, NewOutT]) -> StepSequence[InT, NewOutT]:
        return StepSequence.from_steps(self, make_step(next_step))


if TYPE_CHECKING:
    _WF: type[Workflow] = Step


def make_step(wf: Workflow[InT, OutT], *, cached: bool = False) -> Step[InT, OutT]:
    """
    Wrap any workflow as a Step instance.

    Examples:
        >>> @Step.from_workflow
        ... def times_two(x: int) -> int:
        ...     return x * 2
        ...
        ...
        ... times_two(3)
        6

        >>> times_two.chain(lambda x: x * 3)(3)
        18
    """
    if isinstance(wf, WorkflowABC):
        wf = Step(wf)

    if isinstance(wf, Step):
        if cached and not isinstance(wf, CachedStep):
            wf = CachedStep.from_workflow(wf)

        return wf

    # assert False, (wf, callable(wf), isinstance(wf, WorkflowABC), isinstance(wf, WorkflowABC))

    raise TypeError(f"Expected a Workflow, got type '{type(wf)}' with value '{wf!r}'")


@dataclasses.dataclass(frozen=True)
class CachedStep(Step[InT, OutT]):
    """
    Concrete implementation of a cached workflow.

    Examples:
        >>> def heavy_computation(x: int) -> int:
        ...     print("This might take a while...")
        ...     return x

        >>> cached_step = CachedStep(heavy_computation)

        >>> cached_step(42)
        This might take a while...
        42

        >>> cached_step(42)  # result is cached, so no print statement
        42

        >>> cached_step(1)
        This might take a while...
        1
    """

    hash_function: Callable[[InT], Hashable] = dataclasses.field(
        default=eve_utils.content_hash, kw_only=True
    )
    cache: OpaqueMutableMapping[Hashable, OutT] = dataclasses.field(
        kw_only=True,
        repr=False,
        default_factory=dict[Hashable, OutT],
        metadata=utils.gt4py_metadata(pickle=False),
    )

    @classmethod
    def from_workflow(
        cls, wf: Workflow[InT, OutT], hash_function: Callable[[InT], Hashable] | None = None
    ) -> CachedStep[InT, OutT]:
        """Wrap any workflow as a CachedStep instance."""
        if isinstance(wf, CachedStep):
            if hash_function is not None and wf.hash_function != hash_function:
                raise ValueError("Cannot change hash function of an existing CachedStep")
            return wf
        if isinstance(wf, WorkflowABC):
            return cls(wf, hash_function=hash_function or eve_utils.content_hash)

    def __call__(self, inp: InT) -> OutT:
        """Run the step only if the input is not cached, else return from cache."""
        key = self.cache_key(inp)
        try:
            result = self.cache[key]
        except KeyError:
            result = self.cache[key] = self.wf(inp)
        return result

    def cache_key(self, inp: InT) -> str:
        return eve_utils.content_hash(self.fingerprint, self.hash_function(inp))


@dataclasses.dataclass(frozen=True)
class StepSequence(WorkflowABC[InT, OutT], Sequence[Workflow]):
    """
    Composable workflow of single input callables.

    Examples:
        >>> def plus_one(x: int) -> int:
        ...     return x + 1

        >>> def plus_half(x: int) -> float:
        ...     return x + 0.5

        >>> StepSequence.from_steps(plus_one, plus_half, str)(73)
        '74.5'

    """

    steps: tuple[Step[InT, OutT]] | tuple[Step, ...] = dataclasses.field()

    def __call__(self, inp: InT) -> OutT:
        step_result: Any = inp
        for step in self.steps:
            step_result = step(step_result)
        return step_result

    def chain(self, next_step: Workflow[OutT, NewOutT]) -> StepSequence[InT, NewOutT]:
        return StepSequence((*self.steps, make_step(next_step)))

    @overload
    def __getitem__(self, index: int) -> Workflow: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> Workflow | Self:
        return self.steps[index]

    def __len__(self) -> int:
        return len(self.steps)

    @classmethod
    def from_steps(cls, *steps: Workflow) -> StepSequence:
        return cls(tuple(make_step(step) for step in steps))


@dataclasses.dataclass(frozen=True)
class NamedStepSequence(WorkflowABC[InT, OutT]):
    """
    Dataclass definining steps as dataclass fields.

    Examples:
        >>> import dataclasses

        >>> def parse_fn(x: str) -> int:
        ...     return int(x)

        >>> def plus_half_fn(x: int) -> float:
        ...     return x + 0.5

        >>> def stringify_fn(x: float) -> str:
        ...     return str(x)

        >>> @dataclasses.dataclass(frozen=True)
        ... class ParseOpPrint(NamedStepSequence[str, str]):
        ...     parse: Workflow[str, int]
        ...     op: Workflow[int, float]
        ...     print: Workflow[float, str]

        >>> composite = ParseOpPrint(parse=parse_fn, op=plus_half_fn, print=stringify_fn)

        >>> composite.steps == (parse_fn, plus_half_fn, stringify_fn)
        True

        >>> composite(73)
        '73.5'

        >>> def plus_tenth_fn(x: int) -> float:
        ...     return x + 0.1


        >>> composite.replace(op=plus_tenth_fn)(73)
        '73.1'
    """

    @functools.cached_property
    def steps(self) -> tuple[Workflow, ...]:
        """
        Read step order from class definition by default.

        Only attributes who are type hinted to be of a type that
        conforms to the Workflow protocol are considered steps.
        """
        cls = type(self)
        if not hasattr(cls, "_cached_step_names_"):
            result: list[str] = []
            annotations = typing.get_type_hints(self.__class__)
            for field in dataclasses.fields(self):
                field_type = annotations[field.name]
                field_type = typing.get_origin(field_type) or field_type
                if issubclass(field_type, WorkflowABC):
                    result.append(field.name)
            cls._cached_step_names_ = tuple(result)  # type: ignore[attr-defined]

        return tuple(getattr(self, name) for name in cls._cached_step_names_)  # type: ignore[attr-defined]

    def replace(self, **kwargs: Any) -> Self:
        """Build a new instance with replaced steps."""
        return dataclasses.replace(self, **kwargs)
