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
import dataclasses
import functools
import sys

import attr

from . import extended_typing as xtyping, type_definitions, utils
from .extended_typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    ForwardRef,
    Generator,
    Optional,
    Protocol,
    SourceTypingAnnotation,
    Tuple,
    Type,
    TypeVar,
    TypingAnnotation,
    Union,
    runtime_checkable,
)


_A = TypeVar("_A")
_C = TypeVar("_C")
_V = TypeVar("_V")


_ValidatorT = TypeVar("_ValidatorT")


class GenericTypeValidatorFactory(Protocol[_ValidatorT]):
    def __call__(self, annotation: SourceTypingAnnotation) -> Optional[_ValidatorT]:
        ...


class ClassAttributeValidatorType(Protocol[_C, _A, _V]):
    def __call__(self, instance: _C, attribute_info: _A, value: _V) -> None:
        ...


_T = TypeVar("_T")

if xtyping.TYPE_CHECKING:
    AttrsValidatorType = ClassAttributeValidatorType[Any, attr.Attribute[_T], _T]
else:
    AttrsValidatorType = ClassAttributeValidatorType[Any, attr.Attribute, _T]


AttrsTypeValidatorFactory = GenericTypeValidatorFactory[AttrsValidatorType]

# _ClsAttribValT = TypeVar("_ClsAttribValT", bound=ClassAttributeValidatorType)
# TypeValidationResult = type_definitions.Result[bool]


if sys.version_info >= (3, 10):
    _frozen_dataclass: Final = functools.partial(dataclasses.dataclass, frozen=True, slots=True)
else:
    _frozen_dataclass: Final = functools.partial(dataclasses.dataclass, frozen=True)


@runtime_checkable
class TypeValidator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        value: Any,
        type_annotation: SourceTypingAnnotation,
        *,
        name: Optional[str] = None,
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

        """
        ...


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
        type_annotation: SourceTypingAnnotation,
        *,
        name: Optional[str] = None,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        ...


class SimpleTypeValidatorFactory(TypeValidatorFactory):
    @utils.optional_lru_cache
    def __call__(
        self,
        type_annotation: SourceTypingAnnotation,
        *,
        name: Optional[str] = None,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        """Simple :class:`TypeValidatorFactory` implementation.

        Check :class:`FixedTypeValidator` and :class:`TypeValidatorFactory` for details.

        Keyword Arguments:
            strict_int: do not accept ``bool`` values as ``int`` (default: ``True``).

        """
        # Non-generic types
        if isinstance(type_annotation, type) and type_annotation is not type(
            None
        ):  # noqa: E721  # use isinstance
            assert not xtyping.get_args(type_annotation)
            if type_annotation is int and kwargs.get("strict_int", True):
                return self.make_is_instance_of_int(name)
            else:
                return self.make_is_instance_of(type_annotation, name)
        if isinstance(type_annotation, xtyping.TypeVar):
            if type_annotation.__bound__:
                return self.make_is_instance_of(type_annotation.__bound__)
            else:
                return self.make_any()
        if type_annotation is Any:
            return self.make_any()

        # Generic and parametrized type hints
        origin_type = xtyping.get_origin(type_annotation)
        type_args = xtyping.get_args(type_annotation)

        if origin_type is xtyping.Literal:
            return SimpleTypeValidatorFactory.literal_type(*type_args)
        if origin_type is xtyping.Union:
            return SimpleTypeValidatorFactory.union_type(*type_args)

        if isinstance(origin_type, type):
            # Deal with generic collections
            if issubclass(origin_type, tuple):
                return SimpleTypeValidatorFactory.tuple_type(*type_args, tuple_type=origin_type)
            if issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
                assert len(type_args) == 1
                member_type_hint = type_args[0]
                return SimpleTypeValidatorFactory.deep_iterable(
                    member_validator=attrs_type_validator_factory(member_type_hint),
                    iterable_validator=SimpleTypeValidatorFactory.instance_of(origin_type),
                )
            if issubclass(origin_type, collections.abc.Mapping):
                assert len(type_args) == 2
                key_type_hint, value_type_hint = type_args
                return SimpleTypeValidatorFactory.deep_mapping(
                    key_validator=attrs_type_validator_factory(key_type_hint),
                    value_validator=attrs_type_validator_factory(value_type_hint),
                    mapping_validator=SimpleTypeValidatorFactory.instance_of(origin_type),
                )

        raise TypeError(f"Type description '{type_annotation}' is not supported.")

    @staticmethod
    def make_is_instance_of_int(name: str) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for ``int`` values which fails with ``bool`` values."""

        def _is_instance_of_int(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"'{name}' must be {int} (got '{value}' that is a {type(value)}).")

        return _is_instance_of_int

    @staticmethod
    def make_is_instance_of(type_: type, name: str) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for a specific type."""

        def _is_instance_of(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, type_):
                raise TypeError(
                    f"'{name}' must be {type_} (got '{value}' that is a {type(value)})."
                )

        return _is_instance_of

    @staticmethod
    def make_any(name: str) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for any type."""

        def _is_any(value: Any, **kwargs: Any) -> None:
            pass

        return _is_any

    @_frozen_dataclass(frozen=True)
    class _TupleValidator:
        """Implementation of ``attr.s`` type validator for ``Tuple`` typings."""

        #: Collection of validators.
        validators: Tuple[AttrsValidatorType, ...]
        #: Class used in the container ``isintance()`` check.
        tuple_type: Type[Tuple]

        def __call__(self, instance: Any, attribute: attr.Attribute, value: Any) -> None:
            if not isinstance(value, self.tuple_type):
                raise TypeError(
                    f"In '{attribute.name}' validation, got '{value}' that is a {type(value)} instead of {self.tuple_type}."
                )
            if len(value) != len(self.validators):
                raise TypeError(
                    f"In '{attribute.name}' validation, got '{value}' tuple which contains {len(value)} elements instead of {len(self.validators)}."
                )

            _i = None
            item_value = ""
            try:
                for _i, (item_value, item_validator) in enumerate(zip(value, self.validators)):
                    item_validator(instance, attribute, item_value)
            except Exception as e:
                raise TypeError(
                    f"In '{attribute.name}' validation, tuple '{value}' contains invalid value '{item_value}' at position {_i}."
                ) from e

    @_frozen_dataclass(frozen=True)
    class _OrValidator:
        """Implementation of ``attr.s`` validator composing multiple validators together using OR."""

        #: Collection of validators.
        validators: Tuple[AttrsValidatorType, ...]
        #: Exception class for validation errors.
        error_type: Type[Exception]

        def __call__(self, instance: Any, attribute: attr.Attribute, value: Any) -> None:
            passed = False
            for v in self.validators:
                try:
                    v(instance, attribute, value)
                    passed = True
                    break
                except Exception:
                    pass

            if not passed:
                raise self.error_type(
                    f"In '{attribute.name}' validation, provided value '{value}' fails for all the possible validators."
                )

    @_frozen_dataclass(frozen=True)
    class _LiteralValidator:
        """Implementation of ``attr.s`` type validator for ``Literal`` typings."""

        literal: Any

        def __call__(self, instance: Any, attribute: attr.Attribute, value: Any) -> None:
            if isinstance(self.literal, bool):
                valid = value is self.literal
            else:
                valid = value == self.literal
            if not valid:
                raise ValueError(
                    f"Provided value '{value}' field does not match {self.literal} during '{attribute.name}' validation."
                )

    def or_combinator(
        *validators: AttrsValidatorType, error_type: Type[Exception]
    ) -> AttrsValidatorType:
        """Create an ``attr.s`` validator combinator where only one of the validators needs to pass."""
        if len(validators) == 1:
            return validators[0]
        else:
            return SimpleTypeValidatorFactory._OrValidator(validators, error_type=error_type)

    instance_of = staticmethod(attr.validators.instance_of)
    deep_iterable = staticmethod(attr.validators.deep_iterable)
    deep_mapping = staticmethod(attr.validators.deep_mapping)
    optional = staticmethod(attr.validators.optional)

    @staticmethod
    def literal_type(*type_args: Type) -> AttrsValidatorType:
        """Create an ``attr.s`` strict type validator for ``Literal`` typings."""
        return SimpleTypeValidatorFactory.or_combinator(
            *(SimpleTypeValidatorFactory._LiteralValidator(t) for t in type_args),
            error_type=ValueError,
        )

    @staticmethod
    def tuple_type(*type_args: Type, tuple_type: Type = tuple) -> AttrsValidatorType:
        """Create an ``attr.s`` strict type validator for ``Tuple`` typings."""
        if len(type_args) == 2 and (type_args[1] is Ellipsis):
            # Tuple as an immutable sequence type: Tuple[int, ...]
            if not issubclass(tuple_type, tuple):
                raise TypeError(f"Invalid 'tuple' subclass '{tuple_type}'.")
            member_type_hint = type_args[0]
            return SimpleTypeValidatorFactory.deep_iterable(
                member_validator=attrs_type_validator_factory(member_type_hint),
                iterable_validator=SimpleTypeValidatorFactory.instance_of(tuple_type),
            )
        else:
            # Tuple as a heterogeneous container: Tuple[int, float]
            return SimpleTypeValidatorFactory._TupleValidator(
                tuple(attrs_type_validator_factory(t) for t in type_args),
                tuple_type,
            )

    @staticmethod
    def union_type(*type_args: Type) -> AttrsValidatorType:
        """Create an ``attr.s`` strict type validator for Union typings."""
        if len(type_args) == 2 and (type_args[1] is type(None)):  # noqa: E721  # use isinstance()
            non_optional_validator = attrs_type_validator_factory(type_args[0])
            return SimpleTypeValidatorFactory.optional(non_optional_validator)
        else:
            return SimpleTypeValidatorFactory.or_combinator(
                *(attrs_type_validator_factory(t) for t in type_args),
                error_type=TypeError,
            )


class AttrsValidators:
    @utils.optional_lru_cache
    @staticmethod
    def make_validator(annotation: SourceTypingAnnotation) -> Optional[AttrsValidatorType]:
        # Non-generic types
        if isinstance(annotation, type) and annotation is not type(
            None
        ):  # noqa: E721  # use isinstance
            assert not xtyping.get_args(annotation)
            if annotation is int:
                return AttrsValidators.instance_of_int()
            else:
                return AttrsValidators.instance_of(annotation)
        if isinstance(annotation, xtyping.TypeVar):
            if annotation.__bound__:
                return AttrsValidators.instance_of(annotation.__bound__)
            else:
                return AttrsValidators.any_type()
        if annotation is Any:
            return AttrsValidators.any_type()

        # Generic and parametrized type hints
        origin_type = xtyping.get_origin(annotation)
        type_args = xtyping.get_args(annotation)

        if origin_type is xtyping.Literal:
            return AttrsValidators.literal_type(*type_args)
        if origin_type is xtyping.Union:
            return AttrsValidators.union_type(*type_args)

        if isinstance(origin_type, type):
            # Deal with generic collections
            if issubclass(origin_type, tuple):
                return AttrsValidators.tuple_type(*type_args, tuple_type=origin_type)
            if issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
                assert len(type_args) == 1
                member_type_hint = type_args[0]
                return AttrsValidators.deep_iterable(
                    member_validator=attrs_type_validator_factory(member_type_hint),
                    iterable_validator=AttrsValidators.instance_of(origin_type),
                )
            if issubclass(origin_type, collections.abc.Mapping):
                assert len(type_args) == 2
                key_type_hint, value_type_hint = type_args
                return AttrsValidators.deep_mapping(
                    key_validator=attrs_type_validator_factory(key_type_hint),
                    value_validator=attrs_type_validator_factory(value_type_hint),
                    mapping_validator=AttrsValidators.instance_of(origin_type),
                )

        raise TypeError(f"Type description '{annotation}' is not supported.")

    @_frozen_dataclass(frozen=True)
    class _TupleValidator:
        """Implementation of ``attr.s`` type validator for ``Tuple`` typings."""

        #: Collection of validators.
        validators: Tuple[AttrsValidatorType, ...]
        #: Class used in the container ``isintance()`` check.
        tuple_type: Type[Tuple]

        def __call__(self, instance: Any, attribute: attr.Attribute, value: Any) -> None:
            if not isinstance(value, self.tuple_type):
                raise TypeError(
                    f"In '{attribute.name}' validation, got '{value}' that is a {type(value)} instead of {self.tuple_type}."
                )
            if len(value) != len(self.validators):
                raise TypeError(
                    f"In '{attribute.name}' validation, got '{value}' tuple which contains {len(value)} elements instead of {len(self.validators)}."
                )

            _i = None
            item_value = ""
            try:
                for _i, (item_value, item_validator) in enumerate(zip(value, self.validators)):
                    item_validator(instance, attribute, item_value)
            except Exception as e:
                raise TypeError(
                    f"In '{attribute.name}' validation, tuple '{value}' contains invalid value '{item_value}' at position {_i}."
                ) from e

    @_frozen_dataclass(frozen=True)
    class _OrValidator:
        """Implementation of ``attr.s`` validator composing multiple validators together using OR."""

        #: Collection of validators.
        validators: Tuple[AttrsValidatorType, ...]
        #: Exception class for validation errors.
        error_type: Type[Exception]

        def __call__(self, instance: Any, attribute: attr.Attribute, value: Any) -> None:
            passed = False
            for v in self.validators:
                try:
                    v(instance, attribute, value)
                    passed = True
                    break
                except Exception:
                    pass

            if not passed:
                raise self.error_type(
                    f"In '{attribute.name}' validation, provided value '{value}' fails for all the possible validators."
                )

    @_frozen_dataclass(frozen=True)
    class _LiteralValidator:
        """Implementation of ``attr.s`` type validator for ``Literal`` typings."""

        literal: Any

        def __call__(self, instance: Any, attribute: attr.Attribute, value: Any) -> None:
            if isinstance(self.literal, bool):
                valid = value is self.literal
            else:
                valid = value == self.literal
            if not valid:
                raise ValueError(
                    f"Provided value '{value}' field does not match {self.literal} during '{attribute.name}' validation."
                )

    def or_combinator(
        *validators: AttrsValidatorType, error_type: Type[Exception]
    ) -> AttrsValidatorType:
        """Create an ``attr.s`` validator combinator where only one of the validators needs to pass."""
        if len(validators) == 1:
            return validators[0]
        else:
            return AttrsValidators._OrValidator(validators, error_type=error_type)

    instance_of = staticmethod(attr.validators.instance_of)
    deep_iterable = staticmethod(attr.validators.deep_iterable)
    deep_mapping = staticmethod(attr.validators.deep_mapping)
    optional = staticmethod(attr.validators.optional)

    @staticmethod
    def instance_of_int() -> AttrsValidatorType:
        """Create an ``attr.s`` validator for ``int`` values which fails with ``bool`` values."""

        def _int_validator(instance: Any, attribute: attr.Attribute, value: Any) -> None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"'{attribute.name}' must be {int} (got '{value}' that is a {type(value)})."
                )

        return _int_validator

    @staticmethod
    def any_type() -> AttrsValidatorType:
        """Create an ``attr.s`` empty validator which always succeeds."""

        def _empty_validator(instance: Any, attribute: attr.Attribute, value: Any) -> None:
            pass

        return _empty_validator

    @staticmethod
    def literal_type(*type_args: Type) -> AttrsValidatorType:
        """Create an ``attr.s`` strict type validator for ``Literal`` typings."""
        return AttrsValidators.or_combinator(
            *(AttrsValidators._LiteralValidator(t) for t in type_args), error_type=ValueError
        )

    @staticmethod
    def tuple_type(*type_args: Type, tuple_type: Type = tuple) -> AttrsValidatorType:
        """Create an ``attr.s`` strict type validator for ``Tuple`` typings."""
        if len(type_args) == 2 and (type_args[1] is Ellipsis):
            # Tuple as an immutable sequence type: Tuple[int, ...]
            if not issubclass(tuple_type, tuple):
                raise TypeError(f"Invalid 'tuple' subclass '{tuple_type}'.")
            member_type_hint = type_args[0]
            return AttrsValidators.deep_iterable(
                member_validator=attrs_type_validator_factory(member_type_hint),
                iterable_validator=AttrsValidators.instance_of(tuple_type),
            )
        else:
            # Tuple as a heterogeneous container: Tuple[int, float]
            return AttrsValidators._TupleValidator(
                tuple(attrs_type_validator_factory(t) for t in type_args),
                tuple_type,
            )

    @staticmethod
    def union_type(*type_args: Type) -> AttrsValidatorType:
        """Create an ``attr.s`` strict type validator for Union typings."""
        if len(type_args) == 2 and (type_args[1] is type(None)):  # noqa: E721  # use isinstance()
            non_optional_validator = attrs_type_validator_factory(type_args[0])
            return AttrsValidators.optional(non_optional_validator)
        else:
            return AttrsValidators.or_combinator(
                *(attrs_type_validator_factory(t) for t in type_args),
                error_type=TypeError,
            )


# attrs_type_validator_factory: AttrsTypeValidatorFactory = AttrsValidators.make_validator


# def typeguard_type_validator_factory(
#     type_annotation: SourceTypingAnnotation,
# ) -> Optional[AttrsValidatorType]:
#     import typeguard

#     def _validator(instance: _C, attribute_info: _A, value: _V) -> None:
#         typeguard.check_type(f"'{attribute_info.name}'", value, expected_type=annotation)

#     return _validator
