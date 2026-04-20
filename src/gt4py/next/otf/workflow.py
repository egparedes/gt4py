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
import inspect
import types
import typing
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from typing_extensions import Self

from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import OpaqueMutableMapping
from gt4py.next import utils


StartT = TypeVar("StartT")
StartT_contra = TypeVar("StartT_contra", contravariant=True)
EndT = TypeVar("EndT")
EndT_co = TypeVar("EndT_co", covariant=True)
NewEndT = TypeVar("NewEndT")
IntermediateT = TypeVar("IntermediateT")
HashT = TypeVar("HashT", bound=collections.abc.Hashable)
DataT = TypeVar("DataT")
ArgT = TypeVar("ArgT")


@runtime_checkable
class Transform(utils.FingerprintableProtocol, Protocol[StartT_contra, EndT_co]):
    """
    Transform protocol.

    Anything that implements this interface can be a transform of one or more steps.
    - callable
    - take a single input argument
    """

    __slots__ = ()

    def __call__(self, inp: StartT_contra) -> EndT_co: ...


@dataclasses.dataclass(frozen=True)
class Workflow(
    utils.CachedFingerprintableMixin, utils.ModelPicklerMixin, Generic[StartT_contra, EndT_co]
):
    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, C: type) -> bool | types.NotImplementedType:
        if issubclass(C, utils.CachedFingerprintableMixin) and (
            cls is Workflow
            and callable(getattr(C, "__call__", None))
            and inspect.signature(C.__call__).parameters.keys() == ["inp"]
        ):
            return True

        return NotImplemented

    @abc.abstractmethod
    def __call__(self, inp: StartT_contra) -> EndT_co: ...


class ChainableWorkflowMixin(Workflow[StartT, EndT_co]):
    __slots__ = ()

    def chain(self, next_step: Transform[EndT_co, NewEndT]) -> StepSequence[StartT, NewEndT]:
        return make_step(self).chain(next_step)


def make_step(function: Transform[StartT, EndT]) -> StepSequence[StartT, EndT]:
    """
    Wrap a function in the workflow step convenience wrapper.

    Examples:
    ---------
    >>> @make_step
    ... def times_two(x: int) -> int:
    ...     return x * 2

    >>> def stringify(x: int) -> str:
    ...     return str(x)

    >>> # create a workflow int -> int -> str
    >>> times_two.chain(stringify)(3)
    '6'
    """
    return StepSequence.start(function)


class ReplaceEnabledWorkflowMixin(Workflow[StartT_contra, EndT_co]):
    """
    Subworkflow replacement mixin.

    Any subclass MUST be a dataclass for `.replace` to work
    """

    __slots__ = ()

    def replace(self, **kwargs: Any) -> Self:
        """Build a new instance with replaced substeps."""
        if not dataclasses.is_dataclass(self):
            raise TypeError(
                f"'{self.__class__.__name__}' must be a dataclass to use ReplaceEnabledWorkflowMixin."
            )
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass(frozen=True)
class CachedStep(
    ReplaceEnabledWorkflowMixin[StartT, EndT],
    ChainableWorkflowMixin[StartT, EndT],
    Workflow[StartT, EndT],
):
    """
    Cached workflow of single input callables.

    Examples:
    ---------
    >>> def heavy_computation(x: int) -> int:
    ...     print("This might take a while...")
    ...     return x

    >>> cached_step = CachedStep(step=heavy_computation)

    >>> cached_step(42)
    This might take a while...
    42

    The next invocation for the same argument will be cached:
    >>> cached_step(42)
    42

    >>> cached_step(1)
    This might take a while...
    1
    """

    step: Workflow[StartT, EndT]
    hash_function: Callable[[StartT], Hashable] = dataclasses.field(default=eve_utils.content_hash)
    cache: OpaqueMutableMapping[Hashable, EndT] = dataclasses.field(
        repr=False, default_factory=dict, metadata=utils.gt4py_metadata(pickle=False)
    )

    def __call__(self, inp: StartT) -> EndT:
        """Run the step only if the input is not cached, else return from cache."""
        key = self.cache_key(inp)
        try:
            result = self.cache[key]
        except KeyError:
            result = self.cache[key] = self.step(inp)
        return result

    def cache_key(self, inp: StartT) -> str:
        return eve_utils.content_hash(self.fingerprint, self.hash_function(inp))


@dataclasses.dataclass(frozen=True)
class DispatchingWorkflow(Workflow[StartT, EndT]):
    """
    Workflow that dispatches on the input type.
    """

    dispatcher: eve_utils.TypeMapping[Workflow[StartT, EndT]]

    def __call__(self, inp: StartT) -> EndT:
        """Dispatch the input to the correct sequence of steps based on its type."""
        workflow = self.dispatcher[type(inp)]
        return workflow(inp)


@dataclasses.dataclass(frozen=True)
class MultiStepWorkflow(Workflow[StartT, EndT]):
    """A flexible workflow, where the sequence of steps depends on the input type."""

    def __call__(self, inp: StartT) -> EndT:
        """Compose the steps in the order defined in the `.step_order` class attribute."""
        result: Any = inp
        for step in self.steps:
            result = step(result)
        return result

    @property
    @abc.abstractmethod
    def steps(self) -> Sequence[Transform]: ...

    @property
    def fingerprinter(self) -> Callable[[utils.FingerprintableProtocol], str]:
        return lambda x: utils.fingerprinter(getattr(x, "steps", x))


@dataclasses.dataclass(frozen=True)
class StepSequence(MultiStepWorkflow[StartT, EndT]):
    """
    Composable workflow of single input callables.

    Examples:
    ---------
    >>> def plus_one(x: int) -> int:
    ...     return x + 1

    >>> def plus_half(x: int) -> float:
    ...     return x + 0.5

    >>> def stringify(x: float) -> str:
    ...     return str(x)

    >>> StepSequence.start(plus_one).chain(plus_half).chain(stringify)(73)
    '74.5'

    """

    steps: tuple[Transform[StartT, EndT]] | tuple[Transform[Any, Any], ...] = dataclasses.field()

    def __call__(self, inp: StartT) -> EndT:
        step_result: Any = inp
        for step in self.steps:
            step_result = step(step_result)
        return step_result

    def chain(self, next_step: Transform[EndT, NewEndT]) -> StepSequence[StartT, NewEndT]:
        return cast(StepSequence[StartT, NewEndT], self.__class__((*self.steps, next_step)))

    @classmethod
    def start(cls, step: Transform[StartT, EndT]) -> StepSequence[StartT, EndT]:
        return cls((step,))

    @classmethod
    def from_steps(cls, *steps: Transform) -> StepSequence:
        return cls(steps)


@dataclasses.dataclass(frozen=True)
class NamedStepSequence(MultiStepWorkflow[StartT, EndT]):
    """
    Workflow with linear succession of named steps.

    Examples:
    ---------
    >>> import dataclasses

    >>> def parse(x: str) -> int:
    ...     return int(x)

    >>> def plus_half(x: int) -> float:
    ...     return x + 0.5

    >>> def stringify(x: float) -> str:
    ...     return str(x)

    >>> @dataclasses.dataclass(frozen=True)
    ... class ParseOpPrint(NamedStepSequence[str, str]):
    ...     parse: Workflow[str, int]
    ...     op: Workflow[int, float]
    ...     print: Workflow[float, str]

    >>> pop = ParseOpPrint(parse=parse, op=plus_half, print=stringify)

    >>> pop.ordered_step_names
    ['parse', 'op', 'print']

    >>> pop(73)
    '73.5'

    >>> def plus_tenth(x: int) -> float:
    ...     return x + 0.1


    >>> pop.replace(op=plus_tenth)(73)
    '73.1'
    """

    @functools.cached_property
    def steps(self) -> tuple[Workflow, ...]:
        """
        Read step order from class definition by default.

        Only attributes who are type hinted to be of a type that
        conforms to the Workflow protocol are considered steps.
        """
        result: list[Workflow] = []
        annotations = typing.get_type_hints(self.__class__)
        for field in dataclasses.fields(self):
            field_type = annotations[field.name]
            field_type = typing.get_origin(field_type) or field_type
            if issubclass(field_type, Workflow):
                result.append(getattr(self, field.name))
        return tuple(result)
