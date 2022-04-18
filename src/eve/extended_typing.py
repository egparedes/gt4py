# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Typing definitions working across different Python versions (via `typing_extensions`)."""


import dataclasses as _dataclasses
import pprint as _pprint
import sys as _sys
import types as _types
import typing as _typing

# Definitions in 'typing_extensions' take priority over those in 'typing'
from typing import *

from typing_extensions import *

from .python_info import IS_PYTHON_AT_LEAST_3_9


if IS_PYTHON_AT_LEAST_3_9:
    # Standard library already supports PEP 585 (Type Hinting Generics In Standard Collections)
    from builtins import (
        tuple as Tuple,
        list as List,
        dict as Dict,
        set as Set,
        frozenset as FrozenSet,
        type as Type,
    )

    from collections import (
        ChainMap as ChainMap,
        Counter as Counter,
        OrderedDict as OrderedDict,
        defaultdict as defaultdict,
        deque as deque,
    )
    from collections.abc import (
        AsyncGenerator as AsyncGenerator,
        AsyncIterable as AsyncIterable,
        AsyncIterator as AsyncIterator,
        Awaitable as Awaitable,
        ByteString as ByteString,
        Callable as Callable,
        Collection as Collection,
        Container as Container,
        Coroutine as Coroutine,
        Generator as Generator,
        ItemsView as ItemsView,
        Iterable as Iterable,
        Iterator as Iterator,
        KeysView as KeysView,
        Mapping as Mapping,
        MappingView as MappingView,
        MutableMapping as MutableMapping,
        MutableSequence as MutableSequence,
        MutableSet as MutableSet,
        Reversible as Reversible,
        Sequence as Sequence,
    )
    from collections.abc import Set as AbstractSet
    from collections.abc import ValuesView as ValuesView
    from contextlib import AbstractAsyncContextManager as AsyncContextManager
    from contextlib import AbstractContextManager as ContextManager
    from re import Match as Match, Pattern as Pattern


# These fallbacks are useful for public symbols not exported by default.
# Again, definitions in 'typing_extensions' take priority over those in 'typing'
def __getattr__(name: str) -> Any:
    import sys

    import typing_extensions

    result = SENTINEL = object()
    if not (name.startswith("__") and name.endswith("__")):
        result = getattr(typing_extensions, name, SENTINEL)
        if result is SENTINEL:
            import typing

            result = getattr(typing, name, SENTINEL)

    if result is SENTINEL:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    setattr(sys.modules[__name__], name, result)  # cache result

    return result


def __dir__() -> List[str]:
    if not hasattr(self_func := (globals()["__dir__"]), "__cached_dir"):
        import typing

        import typing_extensions

        orig_dir = typing.__dir__()
        self_func.__cached_dir = orig_dir + [
            name for name in typing_extensions.__dir__() if name not in orig_dir
        ]

    return self_func.__cached_dir


# Common type aliases
_T_co = TypeVar("_T_co", covariant=True)

FrozenList: TypeAlias = Tuple[_T_co, ...]
NoArgsCallable = Callable[[], Any]

# Typing of annotations
_TypingGenericAliasType: TypeAlias = (
    Union[_types.GenericAlias, _typing._BaseGenericAlias]
    if IS_PYTHON_AT_LEAST_3_9
    else _typing._GenericAlias
)

_TypingSpecialFormType = _typing._SpecialForm

TypingAnnotation = Union[Type, ForwardRef, _TypingGenericAliasType, _TypingSpecialFormType]
SourceTypingAnnotation = Union[str, TypingAnnotation]

# Third party protocols
class DevToolsPrettyPrintable(Protocol):
    """Used by python-devtools: https://python-devtools.helpmanual.io/"""

    def __pretty__(self, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
        ...


# Extra functionality
def is_protocol(tp: type) -> bool:
    """Check if a type is a Protocol definition."""
    this_module = _sys.modules[is_protocol.__module__]
    return isinstance(tp, this_module._ProtocolMeta) and tp.__bases__[-1] is this_module.Protocol


def is_namedtuple(tp: type) -> bool:
    """Check if a type is a NamedTuple class."""
    this_module = _sys.modules[is_namedtuple.__module__]
    return isinstance(tp, this_module.NamedTupleMeta)


def get_partial_type_hints(
    obj: Union[
        object,
        Callable,
        _types.FunctionType,
        _types.BuiltinFunctionType,
        _types.MethodType,
        _types.ModuleType,
        _types.WrapperDescriptorType,
        _types.MethodWrapperType,
        _types.MethodDescriptorType,
    ],
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
    include_extras: bool = False,
) -> Dict[str, Union[Type, ForwardRef]]:
    """Return a dictionary with type hints (with undefined names as forward references) for a function, method, module or class object.

    For each member type hint in the object a :class:`typing.ForwarRef` instance will be
    returned if some names in the string annotation have not been found.
    """
    if getattr(obj, "__no_type_check__", None):
        return {}
    if not hasattr(obj, "__annotations__"):
        return get_type_hints(
            obj, globalns=globalns, localns=localns, include_extras=include_extras
        )

    hints: Dict[str, Union[Type, ForwardRef]] = {}
    annotations = getattr(obj, "__annotations__", {})
    for name, hint in annotations.items():
        obj.__annotations__ = {name: hint}
        try:
            resolved_hints = get_type_hints(
                obj, globalns=globalns, localns=localns, include_extras=include_extras
            )
            hints.update(resolved_hints)
        except NameError as error:
            if isinstance(hint, str):
                hints[name] = ForwardRef(hint)
            elif isinstance(hint, (ForwardRef, _typing.ForwardRef)):
                hints[name] = hint
            else:
                raise error

    obj.__annotations__ = annotations

    return hints


def eval_forward_ref(
    ref: Union[str, ForwardRef],
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
    *,
    include_extras: bool = False,
) -> Type:
    """Resolve forward references in type annotations.

    Arguments:
        globalns: globals dict used in the evaluation of the annotations.
        localns: locals dict used in the evaluation of the annotations.

    Keyword Arguments:
        allow_partial: if ``True``, the resolution is allowed to fail and
            a :class:`typing.ForwardRef` will be returned.

    Examples:
        >>> import typing
        >>> resolve_type(
        ...     typing.Dict[typing.ForwardRef('str'), 'typing.Tuple["int", typing.ForwardRef("float")]']
        ... )
        typing.Dict[str, typing.Tuple[int, float]]

    """
    actual_type = ForwardRef(ref) if isinstance(ref, str) else ref

    def _f():
        ...

    _f.__annotations__ = {"ref": actual_type}

    if localns:
        safe_local_ns = {**localns}
        safe_local_ns.setdefault("typing", _sys.modules[__name__])
        safe_local_ns.setdefault("NoneType", type(None))
    else:
        safe_local_ns = {"typing": _sys.modules[__name__], "NoneType": type(None)}

    actual_type = get_type_hints(_f, globalns, safe_local_ns, include_extras=include_extras)["ref"]

    return actual_type


@_dataclasses.dataclass(frozen=True)
class TypingForm:
    annotation: TypingAnnotation
    form: Any
    args: Tuple[TypingAnnotation, ...]

    def __iter__(self) -> Iterable:
        yield self.annotation
        yield self.form
        yield self.args

    def __str__(self) -> str:
        return _pprint.pformat(self, compact=True)


@_dataclasses.dataclass(frozen=True)
class TypingConstruct(TypingForm):
    ...


@_dataclasses.dataclass(frozen=True)
class TypingPlaceholder(TypingForm):
    ...


@_dataclasses.dataclass(frozen=True)
class TypingQualifier(TypingForm):
    ...


@_dataclasses.dataclass(frozen=True)
class TypingMetaType(TypingForm):
    ...


@_dataclasses.dataclass(frozen=True)
class RegularTypeForm(TypingForm):
    ...


_TYPING_CONSTRUCTS: Final = (
    Any,
    None,
    NoReturn,
    Annotated,
    Concatenate,  # type: ignore
    Literal,
    Type,
    TypeAlias,
    TypeGuard,
    Union,
)
_TYPING_PLACEHOLDERS: Final = (ForwardRef, NewType, ParamSpec, TypeVar)
_TYPING_QUALIFIERS: Final = (ClassVar, Final)
_TYPING_META_TYPES: Final = (NamedTuple, Protocol, TypedDict)

TYPING_FORMS: Final = {
    TypingConstruct: _TYPING_CONSTRUCTS,
    TypingPlaceholder: _TYPING_PLACEHOLDERS,
    TypingQualifier: _TYPING_QUALIFIERS,
    TypingMetaType: _TYPING_META_TYPES,
    RegularTypeForm: (),
}


def get_typing_form(tp_annotation: TypingAnnotation, *, recurse: bool = True) -> TypingForm:
    args = get_args(tp_annotation)
    if args and recurse:
        args = tuple(get_typing_form(arg, recurse=recurse) for arg in args)

    if type(tp_annotation) in _TYPING_PLACEHOLDERS:
        return TypingPlaceholder(tp_annotation, type(tp_annotation), args)

    if (form := get_origin(tp_annotation)) is None:
        form = tp_annotation

    for typing_form, members in TYPING_FORMS.items():
        if form in members:
            return typing_form(tp_annotation, form, args)
    else:
        return RegularTypeForm(tp_annotation, form, args)
