# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

"""Visitor classes to work with IR trees."""


from __future__ import annotations

import collections.abc
import contextlib
import copy
import functools
import operator

from . import concepts, iterators, utils
from .concepts import NOTHING
from .typingx import (
    Any,
    Callable,
    ClassVar,
    Collection,
    ContextManager,
    Dict,
    Iterable,
    MutableSequence,
    MutableSet,
    Optional,
    Tuple,
    Union,
)


ContextCallable = Callable[["NodeVisitor", concepts.TreeNode, Dict[str, Any]], ContextManager[None]]


@functools.lru_cache(maxsize=None, typed=False)
def _get_visitor_method_name(visitor_class, node_class) -> str:
    method_name = "visit_" + node_class.__name__
    if hasattr(visitor_class, method_name):
        return method_name

    elif concepts.BaseNode in (mro := node_class.__mro__):
        for base_node_class in mro[1:]:
            if base_node_class is concepts.BaseNode:
                return "generic_visit"

            method_name = "visit_" + base_node_class.__name__
            if hasattr(visitor_class, method_name):
                return method_name

    else:
        return "generic_visit"


class NodeVisitor:
    """Simple node visitor class based on :class:`ast.NodeVisitor`.

    A NodeVisitor instance walks a node tree and calls a visitor
    function for every item found. This class is meant to be subclassed,
    with the subclass adding visitor methods.

    Visitor functions for tree elements are named with a standard
    pattern: ``visit_`` + class name of the node. Thus, the visitor
    function for a ``BinOpExpr`` node class should be called ``visit_BinOpExpr``.
    If no visitor function exists for a specific node, the dispatcher
    mechanism looks for the visitor of each one of its parent classes
    in the order define by the class' ``__mro__`` attribute. Finally,
    if no visitor function has been found, the ``generic_visit`` visitor
    is used instead. A concise summary of the steps followed to find an
    appropriate visitor function is:

        1. A ``self.visit_NODE_CLASS_NAME()`` method where `NODE_CLASS_NAME`
           matches ``type(node).__name__``.
        2. A ``self.visit_NODE_BASE_CLASS_NAME()`` method where
           `NODE_BASE_CLASS_NAME` matches ``base.__name__``, and `base` is
           one of the node base classes (evaluated following the order
           given in ``type(node).__mro__``).
        3. ``self.generic_visit()``.

    This dispatching mechanism is implemented in the main :meth:`visit`
    method and can be overriden in subclasses. Additionally, a class can
    define a list of context handlers to be applied before the actual visit
    to customize the context. Each context receives the visitor instance,
    the node instance, and the keywords arguments of the call.

    Note that return values are not forwarded to the caller in the default
    :meth:`generic_visit` implementation. If you want to return a value from
    a nested node in the tree, make sure all the intermediate nodes explicitly
    return children values.

    The recommended idiom to use and define NodeVisitors can be summarized as:

        * Inside visitor functions:

            + call ``self.visit(child_node)`` to visit children.
            + call ``self.generic_visit(node)`` to continue the tree
              traversal in the usual way.

        * Define an ``apply()`` `classmethod` as a shortcut to create an
          instance and start the visit::

                class Visitor(NodeVisitor)
                    @classmethod
                    def apply(cls, tree, init_var, foo, bar=5, **kwargs):
                        instance = cls(init_var)
                        return instance(tree, foo=foo, bar=bar, **kwargs)

                    ...

                result = Visitor.apply(...)

        * If the visitor has internal state, make sure visitor instances
          are never reused or clean up the state at the end.

    Notes:
        If you want to apply changes to nodes during the traversal,
        use the :class:`NodeMutator` subclass, which handles correctly
        structural modifications of the visited tree.

    """

    contexts: ClassVar[Optional[Tuple[ContextCallable, ...]]] = None

    def visit(self, node: concepts.TreeNode, **kwargs: Any) -> Any:
        visitor = getattr(self, _get_visitor_method_name(self.__class__, node.__class__))

        if ctxs := type(self).contexts:
            with contextlib.ExitStack() as stack:
                for ctx in ctxs:
                    stack.enter_context(ctx(self, node, kwargs))
                return visitor(node, **kwargs)
        else:
            return visitor(node, **kwargs)

    def generic_visit(self, node: concepts.TreeNode, **kwargs: Any) -> Any:
        for child in iterators.generic_iter_children(node):
            self.visit(child, **kwargs)


@functools.singledispatch
def _NodeTranslator_generic_visit(
    node: concepts.TreeNode, translator: NodeTranslator, **kwargs: Any
) -> Any:
    if not hasattr(translator, "_memo_dict_"):
        translator._memo_dict_ = {}

    return copy.deepcopy(node, memo=translator._memo_dict_)


@_NodeTranslator_generic_visit.register(concepts.BaseNode)
def _translate_node(node, translator, **kwargs):
    return node.__class__(  # type: ignore
        **{key: value for key, value in node.iter_impl_fields()},
        **{
            key: processed_value
            for key, value in node.iter_children()
            if (processed_value := translator.visit(value, **kwargs)) is not NOTHING
        },
    )


@_NodeTranslator_generic_visit.register(str)
@_NodeTranslator_generic_visit.register(bytes)
def _translate_string(node, translator, **kwargs):
    return node


@_NodeTranslator_generic_visit.register(list)
@_NodeTranslator_generic_visit.register(tuple)
@_NodeTranslator_generic_visit.register(set)
@_NodeTranslator_generic_visit.register(frozenset)
@_NodeTranslator_generic_visit.register(collections.abc.Sequence)
@_NodeTranslator_generic_visit.register(collections.abc.Set)
def _translate_sequence(node, translator, **kwargs):
    return node.__class__(  # type: ignore
        processed_value
        for value in node
        if (processed_value := translator.visit(value, **kwargs)) is not NOTHING  # type: ignore[no-redef]
    )


@_NodeTranslator_generic_visit.register(dict)
@_NodeTranslator_generic_visit.register(collections.abc.Mapping)
def _translate_mapping(node, translator, **kwargs):
    return node.__class__(  # type: ignore[call-arg]
        {
            key: processed_value
            for key, value in node.items()
            if (processed_value := translator.visit(value, **kwargs)) is not NOTHING  # type: ignore[no-redef]
        }
    )


class NodeTranslator(NodeVisitor):
    """Special `NodeVisitor` to translate nodes and trees.

    A NodeTranslator instance will walk the tree exactly as a regular
    :class:`NodeVisitor` while building an output tree using the return
    values of the visitor methods. If the return value is :obj:`eve.NOTHING`,
    the node will be removed from its location in the output tree,
    otherwise it will be replaced with this new value. The default visitor
    method (:meth:`generic_visit`) returns a `deepcopy` of the original
    node.

    Keep in mind that if the node you're operating on has child nodes
    you must either transform the child nodes yourself or call the
    :meth:`generic_visit` method for the node first.

    Usually you use a NodeTranslator like this::

       output_node = YourTranslator.apply(input_node)

    Notes:
        Check :class:`NodeVisitor` documentation for more details.

    """

    _memo_dict_: Dict[int, Any]

    def generic_visit(self, node: concepts.TreeNode, **kwargs: Any) -> Any:
        return _NodeTranslator_generic_visit(node, self, **kwargs)


class NodeMutator(NodeVisitor):
    """Special `NodeVisitor` to modify nodes in place.

    A NodeMutator instance will walk the tree exactly as a regular
    :class:`NodeVisitor` and use the return value of the visitor
    methods to replace or remove the old node. If the return value
    is :data:`eve.NOTHING`, the node will be removed from its location,
    otherwise it is replaced with the return value. The return value
    may also be the original node, in which case no replacement takes place.

    Keep in mind that if the node you're operating on has child nodes
    you must either transform the child nodes yourself or call the
    :meth:`generic_visit` method for the node first. In case a child node
    is a mutable collection of elements, and one of this element is meant
    to be deleted, it will deleted in place. If the collection is immutable,
    a new immutable collection instance will be created without the removed
    element.

    Usually you use a NodeMutator like this::

       YourMutator.apply(node)

    Notes:
        Check :class:`NodeVisitor` documentation for more details.

    """

    def generic_visit(self, node: concepts.TreeNode, **kwargs: Any) -> Any:
        result: Any = node
        if isinstance(
            node, (concepts.BaseNode, collections.abc.Collection)
        ) and utils.is_collection(node):
            items: Iterable[Tuple[Any, Any]] = []
            tmp_items: Collection[concepts.TreeNode] = []
            set_op: Union[Callable[[Any, str, Any], None], Callable[[Any, int, Any], None]]
            del_op: Union[Callable[[Any, str], None], Callable[[Any, int], None]]

            if isinstance(node, concepts.Node):
                items = list(node.iter_children())
                set_op = setattr
                del_op = delattr
            elif isinstance(node, collections.abc.MutableSequence):
                items = enumerate(node)
                index_shift = 0

                def set_op(container: MutableSequence, idx: int, value: concepts.TreeNode) -> None:
                    container[idx - index_shift] = value

                def del_op(container: MutableSequence, idx: int) -> None:
                    nonlocal index_shift
                    del container[idx - index_shift]
                    index_shift += 1

            elif isinstance(node, collections.abc.MutableSet):
                items = list(enumerate(node))

                def set_op(container: MutableSet, idx: Any, value: concepts.TreeNode) -> None:
                    container.add(value)

                def del_op(container: MutableSet, idx: int) -> None:
                    container.remove(items[idx])  # type: ignore

            elif isinstance(node, collections.abc.MutableMapping):
                items = node.items()
                set_op = operator.setitem
                del_op = operator.delitem

            elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)):
                # Inmutable sequence or set: create a new container instance with the new values
                tmp_items = [self.visit(value, **kwargs) for value in node]
                result = node.__class__(  # type: ignore
                    [value for value in tmp_items if value is not concepts.NOTHING]
                )

            elif isinstance(node, collections.abc.Mapping):
                # Inmutable mapping: create a new mapping instance with the new values
                tmp_items = {key: self.visit(value, **kwargs) for key, value in node.items()}
                result = node.__class__(  # type: ignore
                    {
                        key: value
                        for key, value in tmp_items.items()
                        if value is not concepts.NOTHING
                    }
                )

            # Finally, in case current node object is mutable, process selected items (if any)
            for key, value in items:
                new_value = self.visit(value, **kwargs)
                if new_value is concepts.NOTHING:
                    del_op(result, key)
                elif new_value != value:
                    set_op(result, key, new_value)

        return result


NodeMutator = None
