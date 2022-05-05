# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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


import pydantic
import pytest

import eve


class TestSourceLocation:
    def test_valid_position(self):
        eve.type_definitions.SourceLocation(line=1, column=1, source="source.py")

    def test_invalid_position(self):
        with pytest.raises(TypeError, match="column"):
            eve.type_definitions.SourceLocation(line=1, column=-1, source="source.py")

    def test_str(self):
        loc = eve.type_definitions.SourceLocation(line=1, column=1, source="dir/source.py")
        assert str(loc) == "<'dir/source.py': Line 1, Col 1>"

        loc = eve.type_definitions.SourceLocation(
            line=1, column=1, source="dir/source.py", end_line=2
        )
        assert str(loc) == "<'dir/source.py': Line 1, Col 1 to Line 2>"

        loc = eve.type_definitions.SourceLocation(
            line=1, column=1, source="dir/source.py", end_line=2, end_column=2
        )
        assert str(loc) == "<'dir/source.py': Line 1, Col 1 to Line 2, Col 2>"

    def test_construction_from_ast(self):
        import ast

        ast_node = ast.parse("a = b + 1").body[0]
        loc = eve.type_definitions.SourceLocation.from_AST(ast_node, "source.py")

        assert loc.line == ast_node.lineno
        assert loc.column == ast_node.col_offset + 1
        assert loc.source == "source.py"
        assert loc.end_line == ast_node.end_lineno
        assert loc.end_column == ast_node.end_col_offset + 1

        loc = eve.type_definitions.SourceLocation.from_AST(ast_node)

        assert loc.line == ast_node.lineno
        assert loc.column == ast_node.col_offset + 1
        assert loc.source == f"<ast.Assign at 0x{id(ast_node):x}>"
        assert loc.end_line == ast_node.end_lineno
        assert loc.end_column == ast_node.end_col_offset + 1


class TestSourceLocationGroup:
    def test_valid_locations(self):
        loc1 = eve.type_definitions.SourceLocation(line=1, column=1, source="source1.py")
        loc2 = eve.type_definitions.SourceLocation(line=2, column=2, source="source2.py")
        eve.type_definitions.SourceLocationGroup(loc1)
        eve.type_definitions.SourceLocationGroup(loc1, loc2)
        eve.type_definitions.SourceLocationGroup(loc1, loc1, loc2, loc2, context="test context")

    def test_invalid_locations(self):
        with pytest.raises(pydantic.ValidationError):
            eve.type_definitions.SourceLocationGroup()
        loc1 = eve.type_definitions.SourceLocation(line=1, column=1, source="source.py")
        with pytest.raises(pydantic.ValidationError):
            eve.type_definitions.SourceLocationGroup(loc1, "loc2")

    def test_str(self):
        loc1 = eve.type_definitions.SourceLocation(line=1, column=1, source="source1.py")
        loc2 = eve.type_definitions.SourceLocation(line=2, column=2, source="source2.py")
        loc = eve.type_definitions.SourceLocationGroup(loc1, loc2, context="some context")
        assert (
            str(loc)
            == "<#some context#[<'source1.py': Line 1, Col 1>, <'source2.py': Line 2, Col 2>]>"
        )


class TestNode:
    def test_validation(self, invalid_sample_node_maker):
        with pytest.raises(pydantic.ValidationError):
            invalid_sample_node_maker()

    def test_unique_id(self, sample_node_maker):
        node_a = sample_node_maker()
        node_b = sample_node_maker()
        node_c = sample_node_maker()

        assert id(node_a) != id(node_b) != id(node_c)

    def test_impl_fields(self, sample_node):
        impl_names = set(name for name, _ in sample_node.iter_impl_fields())

        assert all(name.endswith("_") and not name.endswith("__") for name in impl_names)
        assert (
            set(
                name
                for name in sample_node.__fields__.keys()
                if name.endswith("_") and not name.endswith("__")
            )
            == impl_names
        )

    def test_children(self, sample_node):
        impl_field_names = set(name for name, _ in sample_node.iter_impl_fields())
        children_names = set(name for name, _ in sample_node.iter_children())
        public_names = impl_field_names | children_names
        field_names = set(sample_node.__fields__.keys())

        assert not any(name.endswith("__") for name in children_names)
        assert not any(name.endswith("_") for name in children_names)

        assert public_names <= field_names
        assert all(name.endswith("_") for name in field_names - public_names)

        assert all(
            node1 is node2
            for (name, node1), node2 in zip(
                sample_node.iter_children(), sample_node.iter_children_values()
            )
        )

    def test_node_metadata(self, sample_node):
        assert all(
            name in sample_node.__node_impl_fields__ for name, _ in sample_node.iter_impl_fields()
        )
        assert all(
            isinstance(metadata, dict)
            and isinstance(metadata["definition"], pydantic.fields.ModelField)
            for metadata in sample_node.__node_impl_fields__.values()
        )

        assert all(name in sample_node.__node_children__ for name, _ in sample_node.iter_children())
        assert all(
            isinstance(metadata, dict)
            and isinstance(metadata["definition"], pydantic.fields.ModelField)
            for metadata in sample_node.__node_children__.values()
        )
