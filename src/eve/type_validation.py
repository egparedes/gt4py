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

"""Definitions of useful field and general types."""


from __future__ import annotations

import abc
import collections.abc
import functools

from . import extended_typing as xtyping, type_definitions
from .extended_typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    ForwardRef,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    TypingAnnotation,
    Union,
    runtime_checkable,
)


@runtime_checkable
class TypeValidator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        value: Any,
        type_annotation: TypingAnnotation,
        name: Optional[str] = None,
        *,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Protocol defining the interface for type validation functions ensuring that ``value`` matches ``expected_type``.

        Arguments:
            value: value to be checked against the typing annotation.
            type_annotation: a valid typing annotation.

        Keyword Arguments:
            name: the name of the value to check (used for error messages).
            globalns: globals dict used in the evaluation of the annotations.
            localns: locals dict used in the evaluation of the annotations.
            **kwargs: arbitrary implementation-defined arguments (e.g. for memoization).

        Raises:
            TypeError: if there is a type mismatch.
            ValueError: if there is a type mismatch.

        """
        ...


TypeValidatorResult = type_definitions.Result[bool, Union[TypeError, ValueError]]


class SafeTypeValidator(Protocol):
    def __call__(
        self,
        value: Any,
        type_annotation: TypingAnnotation,
        name: Optional[str] = None,
        *,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TypeValidatorResult:
        """Protocol defining the interface for type validation functions ensuring that ``value`` matches ``expected_type``.

        Arguments:
            value: value to be checked against the typing annotation.
            type_annotation: a valid typing annotation.

        Keyword Arguments:
            name: the name of the value to check (used for error messages).
            globalns: globals dict used in the evaluation of the annotations.
            localns: locals dict used in the evaluation of the annotations.
            **kwargs: arbitrary implementation-defined arguments (e.g. for memoization).

        """
        ...


def as_safe_type_validator(type_validator: TypeValidator) -> SafeTypeValidator:
    @functools.wraps(type_validator)
    def safe_type_validator(*args, **kwargs) -> TypeValidatorResult:
        return TypeValidatorResult.from_try(
            type_validator, *args, **kwargs, __errors=(TypeError, ValueError)
        )

    return safe_type_validator


class FixedTypeValidator(Protocol):
    expected_annotation: ClassVar[TypingAnnotation]

    @abc.abstractmethod
    def __call__(
        self,
        value: Any,
        **kwargs: Any,
    ) -> None:
        """Protocol defining the interface for type validation functions ensuring that ``value`` matches ``expected_type``."""
        ...


@runtime_checkable
class TypeValidatorFactory(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        type_annotation: TypingAnnotation,
        name: Optional[str] = None,
        *,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        ...


# Implementation
class _SimpleTypeValidatorFactory:
    @classmethod
    def make_validator(  # noqa: C901  # too complex but well organized
        cls,
        type_annotation: TypingAnnotation,
        name: Optional[str] = None,
        *,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        """Make a simple :class:`TypeValidator` for the given annotation.

        Check :class:`FixedTypeValidator` and :class:`TypeValidatorFactory` for details.

        Keyword Arguments:
            strict_int: do not accept ``bool`` values as ``int`` (default: ``True``).

        """
        if name is None:
            name = "<value>"

        make_recursive = functools.partial(
            cls.make_validator, name=name, globalns=globalns, localns=localns, **kwargs
        )

        # Non-generic types
        if isinstance(
            type_annotation, type
        ) and type_annotation is not type(  # noqa: E721  # use isinstance
            None
        ):
            assert not xtyping.get_args(type_annotation)
            if type_annotation is int and kwargs.get("strict_int", True):
                return cls.make_is_instance_of_int(name)
            else:
                return cls.make_is_instance_of(name, type_annotation)

        if isinstance(type_annotation, TypeVar):
            if type_annotation.__bound__:
                return cls.make_is_instance_of(name, type_annotation.__bound__)
            else:
                return cls._make_is_any(name)

        if isinstance(type_annotation, ForwardRef):
            return xtyping.eval_forward_ref(type_annotation, globalns=globalns, localns=localns)

        if type_annotation is Any:
            return cls._make_is_any(name)

        # Generic and parametrized type hints
        origin_type = xtyping.get_origin(type_annotation)
        type_args = xtyping.get_args(type_annotation)

        if origin_type is xtyping.Literal:
            if len(type_args) == 1:
                return cls.make_is_literal(name, type_args[0])
            else:
                return cls.combine_validators_as_or(
                    name, *(cls.make_is_literal(name, a) for a in type_args), error_type=ValueError
                )

        if origin_type is xtyping.Union:
            has_none = False
            validators = []
            for t in type_args:
                if t in (type(None), None):
                    has_none = True
                else:
                    validators.append(make_recursive(t))

            validator = (
                cls.combine_validators_as_or(name, *validators)
                if len(validators) > 1
                else validators[0]
            )
            return cls.combine_optional(name, validator) if has_none else validator

        if isinstance(origin_type, type):
            # Deal with generic collections
            if issubclass(origin_type, tuple):

                if len(type_args) == 2 and (type_args[1] is Ellipsis):
                    # Tuple as an immutable sequence type (e.g. Tuple[int, ...])
                    member_type_hint = type_args[0]
                    return cls.make_is_iterable_of(
                        name,
                        make_recursive(member_type_hint),
                        iterable_validator=cls.make_is_instance_of(name, origin_type),
                    )

                else:
                    # Tuple as a heterogeneous container (e.g. Tuple[int, float])
                    return cls.make_is_tuple_of(
                        name, tuple(make_recursive(t) for t in type_args), origin_type
                    )

            if issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
                assert len(type_args) == 1
                member_type_hint = type_args[0]
                return cls.make_is_iterable_of(
                    name,
                    make_recursive(member_type_hint),
                    iterable_validator=cls.make_is_instance_of(name, origin_type),
                )

            if issubclass(origin_type, collections.abc.Mapping):
                assert len(type_args) == 2
                key_type_hint, value_type_hint = type_args
                return cls.make_is_mapping_of(
                    name,
                    make_recursive(key_type_hint),
                    make_recursive(value_type_hint),
                    make_recursive(origin_type),
                )

        return None

    @staticmethod
    def _make_is_any(name: str) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for any type."""

        def _is_any(value: Any, **kwargs: Any) -> None:
            pass

        return _is_any

    @staticmethod
    def make_is_instance_of(name: str, type_: type) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for a specific type."""

        def _is_instance_of(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, type_):
                raise TypeError(
                    f"'{name}' must be {type_} (got '{value}' that is a {type(value)})."
                )

        return _is_instance_of

    @staticmethod
    def make_is_instance_of_int(name: str) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for ``int`` values which fails with ``bool`` values."""

        def _is_instance_of_int(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"'{name}' must be {int} (got '{value}' that is a {type(value)}).")

        return _is_instance_of_int

    @staticmethod
    def make_is_literal(name: str, literal_value) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for a literal value."""
        if isinstance(literal_value, bool):

            def _is_literal(value: Any, **kwargs: Any) -> None:
                if value is not literal_value:
                    raise ValueError(
                        f"Provided value '{value}' for '{name}' does not match {literal_value}."
                    )

        else:

            def _is_literal(value: Any, **kwargs: Any) -> None:
                if value != literal_value:
                    raise ValueError(
                        f"Provided value '{value}' for '{name}' does not match {literal_value}."
                    )

        return _is_literal

    @staticmethod
    def make_is_tuple_of(
        name: str, item_validators: Sequence[FixedTypeValidator], tuple_type: type
    ) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for tuple types."""

        def _is_tuple_of(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, tuple_type):
                raise TypeError(
                    f"In '{name}' validation, got '{value}' that is a {type(value)} instead of {tuple_type}."
                )
            if len(value) != len(item_validators):
                raise TypeError(
                    f"In '{name}' validation, got '{value}' tuple which contains {len(value)} elements instead of {len(item_validators)}."
                )

            _i = None
            item_value = ""
            try:
                for _i, (item_value, item_validator) in enumerate(zip(value, item_validators)):
                    item_validator(item_value)
            except Exception as e:
                raise TypeError(
                    f"In '{name}' validation, tuple '{value}' contains invalid value '{item_value}' at position {_i}."
                ) from e

        return _is_tuple_of

    @staticmethod
    def make_is_iterable_of(
        name: str,
        member_validator: FixedTypeValidator,
        iterable_validator: Optional[FixedTypeValidator] = None,
    ) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for deep checks of typed iterables."""

        def _is_iterable_of(value: Any, **kwargs: Any) -> None:
            if iterable_validator is not None:
                iterable_validator(value, **kwargs)

            for member in value:
                member_validator(member, **kwargs)

        return _is_iterable_of

    @staticmethod
    def make_is_mapping_of(
        name: str,
        key_validator: FixedTypeValidator,
        value_validator: FixedTypeValidator,
        mapping_validator: Optional[FixedTypeValidator] = None,
    ) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for deep checks of typed iterables."""

        def _is_mapping_of(value: Any, **kwargs: Any) -> None:
            if mapping_validator is not None:
                mapping_validator(value, **kwargs)

            for k in value:
                key_validator(k, **kwargs)
                value_validator(value[k], **kwargs)

        return _is_mapping_of

    @staticmethod
    def combine_optional(name: str, actual_validator: FixedTypeValidator) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for an optional constraint."""

        def _is_optional(value: Any, **kwargs: Any) -> None:
            if value is not None:
                actual_validator(value, **kwargs)

        return _is_optional

    @staticmethod
    def combine_validators_as_or(
        name: str, *validators, error_type: Type[Exception] = TypeError
    ) -> FixedTypeValidator:
        def _combined_validator(value: Any, **kwargs: Any) -> Any:
            for v in validators:
                try:
                    v(value, **kwargs)
                    break
                except Exception:
                    pass
            else:
                raise error_type(
                    f"In '{name}' validation, provided value '{value}' fails for all the possible validators."
                )

        return _combined_validator


simple_type_validator_factory = _SimpleTypeValidatorFactory.make_validator


def simple_type_validator(
    value: Any,
    type_annotation: TypingAnnotation,
    name: Optional[str] = None,
    *,
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    simple_type_validator_factory(
        type_annotation, name=name, globalns=globalns, localns=localns, **kwargs
    )(value, **kwargs)


safe_simple_type_validator: Final[SafeTypeValidator] = as_safe_type_validator(simple_type_validator)


# def typeguard_type_validator_factory(
#     type_annotation: SourceTypingAnnotation,
# ) -> Optional[AttrsValidatorType]:
#     import typeguard

#     def _validator(instance: _C, attribute_info: _A, value: _V) -> None:
#         typeguard.check_type(f"'{attribute_info.name}'", value, expected_type=annotation)

#     return _validator
