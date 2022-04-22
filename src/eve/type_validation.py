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

import dataclasses

from . import extended_typing as xtyping
from .extended_typing import (
    Any,
    Callable,
    Generator,
    Optional,
    SourceTypingAnnotation,
    Tuple,
    Type,
    Union,
)


@dataclasses.dataclass
class _ForwardRefValidator:
    """Implementation of ``attr.s`` type validator for ``ForwardRef`` typings."""

    #: Actual type validators created after resolving the forward references.
    validator: Optional[ValidatorType] = None

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        if self.validator is None:
            model_cls = instance.__class__
            update_forward_refs(model_cls)
            self.validator = strict_type_attrs_validator(
                getattr(getattr(model_cls, _MODEL_FIELDS), attribute.name).type
            )

        self.validator(instance, attribute, value)


@dataclasses.dataclass(frozen=True)
class _TupleValidator:
    """Implementation of ``attr.s`` type validator for ``Tuple`` typings."""

    #: Collection of validators.
    validators: Tuple[ValidatorType, ...]
    #: Class used in the container ``isintance()`` check.
    tuple_type: Type[Tuple]

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
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


@dataclasses.dataclass(frozen=True)
class _OrValidator:
    """Implementation of ``attr.s`` validator composing multiple validators together using OR."""

    #: Collection of validators.
    validators: Tuple[ValidatorType, ...]
    #: Exception class for validation errors.
    error_type: Type[Exception]

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
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


@dataclasses.dataclass(frozen=True)
class _LiteralValidator:
    """Implementation of ``attr.s`` type validator for ``Literal`` typings."""

    literal: Any

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        if isinstance(self.literal, bool):
            valid = value is self.literal
        else:
            valid = value == self.literal
        if not valid:
            raise ValueError(
                f"Provided value '{value}' field does not match {self.literal} during '{attribute.name}' validation."
            )


def empty_attrs_validator() -> ValidatorType:
    """Create an ``attr.s`` empty validator which always succeeds."""

    def _empty_validator(instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        pass

    return _empty_validator


def forward_ref_type_attrs_validator() -> ValidatorType:
    """Create an ``attr.s`` strict type validator for ``ForwardRef`` typings.

    The generated validator will resolve the field type to an actual type
    the first time is called.
    """
    return _ForwardRefValidator()


def instance_of_int_attrs_validator() -> ValidatorType:
    """Create an ``attr.s`` validator for ``int`` values which fails with ``bool`` values."""

    def _int_validator(instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"'{attribute.name}' must be {int} (got '{value}' that is a {type(value)})."
            )

    return _int_validator


def or_attrs_validator(*validators: ValidatorType, error_type: Type[Exception]) -> ValidatorType:
    """Create an ``attr.s`` validator combinator where only one of the validators needs to pass."""
    if len(validators) == 1:
        return validators[0]
    else:
        return _OrValidator(validators, error_type=error_type)


def literal_type_attrs_validator(*type_args: Type) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for ``Literal`` typings."""
    return or_attrs_validator(*(_LiteralValidator(t) for t in type_args), error_type=ValueError)


def tuple_type_attrs_validator(*type_args: Type, tuple_type: Type = tuple) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for ``Tuple`` typings."""
    if len(type_args) == 2 and (type_args[1] is Ellipsis):
        # Tuple as an immutable sequence type: Tuple[int, ...]
        if not issubclass(tuple_type, tuple):
            raise TypeError(f"Invalid 'tuple' subclass '{tuple_type}'.")
        member_type_hint = type_args[0]
        return attr.validators.deep_iterable(
            member_validator=strict_type_attrs_validator(member_type_hint),
            iterable_validator=attr.validators.instance_of(tuple_type),
        )
    else:
        # Tuple as a heterogeneous container: Tuple[int, float]
        return _TupleValidator(
            tuple(strict_type_attrs_validator(t) for t in type_args),
            tuple_type,
        )


def union_type_attrs_validator(*type_args: Type) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for Union typings."""
    if len(type_args) == 2 and (type_args[1] is type(None)):  # noqa: E721  # use isinstance()
        non_optional_validator = strict_type_attrs_validator(type_args[0])
        return attr.validators.optional(non_optional_validator)
    else:
        return or_attrs_validator(
            *(strict_type_attrs_validator(t) for t in type_args),
            error_type=TypeError,
        )


def strict_type_attrs_validator(
    type_hint: Any, *, forward_eval_module: Optional[str] = None
) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for a specific typing hint."""
    type_args = typing.get_args(type_hint)

    # Custom type validator
    if isinstance(type_hint, TypeWithAttrValidatorTp):
        return type_hint.__type_validator__()

    # Non-generic types
    if isinstance(type_hint, type) and type_hint is not type(None):  # noqa: E721  # use isinstance
        assert not type_args
        if type_hint is int:
            return instance_of_int_attrs_validator()
        else:
            return attr.validators.instance_of(type_hint)
    if isinstance(type_hint, typing.TypeVar):
        if type_hint.__bound__:
            return attr.validators.instance_of(type_hint.__bound__)
        else:
            return empty_attrs_validator()
    if isinstance(type_hint, ForwardRef):
        return forward_ref_type_attrs_validator()
    if type_hint is Any:
        return empty_attrs_validator()

    # Generic and parametrized type hints
    origin_type = typing.get_origin(type_hint)

    if origin_type is typing.Literal:
        return literal_type_attrs_validator(*type_args)
    if origin_type is typing.Union:
        return union_type_attrs_validator(*type_args)
    if isinstance(origin_type, type):
        # Deal with generic collections
        if issubclass(origin_type, tuple):
            return tuple_type_attrs_validator(*type_args, tuple_type=origin_type)
        if issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
            assert len(type_args) == 1
            member_type_hint = type_args[0]
            return attr.validators.deep_iterable(
                member_validator=strict_type_attrs_validator(member_type_hint),
                iterable_validator=attr.validators.instance_of(origin_type),
            )
        if issubclass(origin_type, collections.abc.Mapping):
            assert len(type_args) == 2
            key_type_hint, value_type_hint = type_args
            return attr.validators.deep_mapping(
                key_validator=strict_type_attrs_validator(key_type_hint),
                value_validator=strict_type_attrs_validator(value_type_hint),
                mapping_validator=attr.validators.instance_of(origin_type),
            )

    raise TypeError(f"Type description '{type_hint}' is not supported.")


# TypeValidationFactory = Callable[[xtyping.SourceTypingAnnotation], ValidatorType]
ValidatorType = Callable[[DataModelTp, Attribute, T], None]
ValidatorType = Callable[[Any], bool]


@dataclasses.dataclass(frozen=True)
class Result:
    sucess: bool
    error: Optional[Exception] = None

    def __bool__():


def type_validator_factory(type_annotation: SourceTypingAnnotation) -> ValidatorType:
    ...
