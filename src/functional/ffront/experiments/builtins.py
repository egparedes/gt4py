from __future__ import annotations

import abc
import collections.abc
import numbers
import types
from typing import Any, Generic, Protocol, TypeVar
from _collections_abc import _check_methods


class _FieldGenericAlias(types.GenericAlias):
    """Represent `Callable[argtypes, resulttype]`.
    This sets ``__args__`` to a tuple containing the flattened ``dimtypes``
    followed by ``resulttype``.
    Example: ``Field[[I, J], float]`` sets ``__args__`` to
    ``(I, J, float)``.
    """

    __slots__ = ()

    def __new__(cls, origin, args):
        if not (isinstance(args, tuple) and len(args) == 2):
            raise TypeError("Field must be used as Field[[dimension, ...], dtype].")
        t_args, t_result = args
        if isinstance(t_args, list):
            args = (*t_args, t_result)
        else:
            raise TypeError(
                f"Expected a list of types or an ellipsis. Got {t_args}"
            )
        return super().__new__(cls, origin, args)

    @property
    def __parameters__(self):
        params = []
        for arg in self.__args__:
            # Looks like a genericalias
            if hasattr(arg, "__parameters__") and isinstance(arg.__parameters__, tuple):
                params.extend(arg.__parameters__)
            else:
                if _is_typevarlike(arg):
                    params.append(arg)
        return tuple(dict.fromkeys(params))

    def __repr__(self):
        if len(self.__args__) == 2 and _is_param_expr(self.__args__[0]):
            return super().__repr__()
        return (
            f"collections.abc.Callable"
            f'[[{", ".join([_type_repr(a) for a in self.__args__[:-1]])}], '
            f"{_type_repr(self.__args__[-1])}]"
        )

    def __reduce__(self):
        args = self.__args__
        if not (len(args) == 2 and _is_param_expr(args[0])):
            args = list(args[:-1]), args[-1]
        return _CallableGenericAlias, (Callable, args)

    def __getitem__(self, item):
        # Called during TypeVar substitution, returns the custom subclass
        # rather than the default types.GenericAlias object.  Most of the
        # code is copied from typing's _GenericAlias and the builtin
        # types.GenericAlias.

        # A special case in PEP 612 where if X = Callable[P, int],
        # then X[int, str] == X[[int, str]].
        param_len = len(self.__parameters__)
        if param_len == 0:
            raise TypeError(f"{self} is not a generic class")
        if not isinstance(item, tuple):
            item = (item,)
        if (
            param_len == 1
            and _is_param_expr(self.__parameters__[0])
            and item
            and not _is_param_expr(item[0])
        ):
            item = (list(item),)
        item_len = len(item)
        if item_len != param_len:
            raise TypeError(
                f'Too {"many" if item_len > param_len else "few"}'
                f" arguments for {self};"
                f" actual {item_len}, expected {param_len}"
            )
        subst = dict(zip(self.__parameters__, item))
        new_args = []
        for arg in self.__args__:
            if _is_typevarlike(arg):
                if _is_param_expr(arg):
                    arg = subst[arg]
                    if not _is_param_expr(arg):
                        raise TypeError(
                            f"Expected a list of types, an ellipsis, "
                            f"ParamSpec, or Concatenate. Got {arg}"
                        )
                else:
                    arg = subst[arg]
            # Looks like a GenericAlias
            elif hasattr(arg, "__parameters__") and isinstance(arg.__parameters__, tuple):
                subparams = arg.__parameters__
                if subparams:
                    subargs = tuple(subst[x] for x in subparams)
                    arg = arg[subargs]
            new_args.append(arg)

        # args[0] occurs due to things like Z[[int, str, bool]] from PEP 612
        if not isinstance(new_args[0], list):
            t_result = new_args[-1]
            t_args = new_args[:-1]
            new_args = (t_args, t_result)
        return _CallableGenericAlias(Callable, tuple(new_args))


class Field(metaclass=abc.ABCMeta):

    __slots__ = ()

    @abc.abstractmethod
    def __getitem__(self, *args):
        return NotImplemented

    @abc.abstractmethod
    def __gt_data_interface__(self) -> dict[str, Any]:
        return NotImplemented

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Field:
            return _check_methods(C, "__getitem__", "__gt_data_interface__")
        return NotImplemented

    __class_getitem__ = classmethod(_FieldGenericAlias)
