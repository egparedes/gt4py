# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Union

from lark import (
    exceptions as lark_exceptions,
    lark,
    lexer as lark_lexer,
    tree as lark_tree,
    visitors as lark_visitors,
)

from gt4py.next import common
from gt4py.next.iterator import ir, pretty_printer
from gt4py.next.type_system import type_specifications as ts


_AXIS_KIND_FROM_SUFFIX = {v: k for k, v in pretty_printer._AXIS_KIND_SUFFIX.items()}
_AXIS_CATEGORY_FROM_SUFFIX = {v: k for k, v in pretty_printer._AXIS_CATEGORY_SUFFIX.items() if v}


GRAMMAR = r"""
    start: dim_decl* _root
    _root: fencil_definition
        | function_definition
        | declaration
        | stencil_closure
        | set_at
        | if_stmt
        | program
        | prec0

    // Header entry giving a short dimension name its full spelling (see `pretty_printer`).
    dim_decl: "dim" TAG "=" AXIS_LITERAL ";"

    SYM: CNAME
    SYM_REF: CNAME
    TYPE_LITERAL: CNAME
    INT_LITERAL: SIGNED_INT
    FLOAT_LITERAL: SIGNED_FLOAT
    // A dimension tag is `module:qualname` (ADR 0029), and -- for a parametrized dimension
    // such as `Staggered[pkg:K]` -- with one bracketed tag. A short dimension name (declared
    // in the header) and an offset tag are the same without the module part.
    // Unambiguous here: a tag starts with a letter (a float does not), and the literal's
    // suffix terminates it.
    TAG: /[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?(?:\[[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?\])?/
    OFFSET_LITERAL: ( INT_LITERAL | TAG ) "ₒ"
    AXIS_LITERAL: TAG ("ᵥ" | "ₕ" | "ₗ") ("ᵃ" | "ˡ")?
    INFINITY_LITERAL: "∞" | "-∞"
    _literal: INT_LITERAL | FLOAT_LITERAL | OFFSET_LITERAL | AXIS_LITERAL | INFINITY_LITERAL
    ID_NAME: CNAME

    ?prec0: prec1
        | "λ(" ( SYM "," )* SYM? ")" "→" prec0 -> lam

    ?prec1: prec2
        | "if" prec1 "then" prec1 "else" prec1 -> ifthenelse

    ?prec2: prec3
        | prec2 "∨" prec3 -> bool_or  

    ?prec3: prec4
        | prec3 "∧" prec4 -> bool_and

    ?prec4: prec5
        | prec4 "==" prec5 -> eq
        | prec4 "<" prec5 -> less
        | prec4 ">" prec5 -> greater
        | prec4 "<=" prec5 -> less_equal
        | prec4 ">=" prec5 -> greater_equal

    ?prec5: prec6
        | prec5 "+" prec6 -> plus
        | prec5 "-" prec6 -> minus

    ?prec6: prec7
        | prec6 "×" prec7 -> multiplies
        | prec6 "/" prec7 -> divides

    ?prec7: prec8
        | "·" prec7 -> deref
        | "¬" prec7 -> bool_not
        | "↑" prec7 -> lift
        | "⇑" prec7 -> as_fieldop

    ?prec8: prec9
        | prec8 "[" prec0 "]" -> tuple_get
        | prec8 "(" ( prec0 "," )* prec0? ")" -> call
        | "{" ( prec0 "," )* prec0? "}" -> make_tuple
        | "⟪" ( prec0 "," )* prec0? "⟫" -> shift
        | "u⟨" ( prec0 "," )* prec0? "⟩" -> unstructured_domain
        | "c⟨" ( prec0 "," )* prec0? "⟩" -> cartesian_domain

    ?prec9: _literal
        | typed_literal
        | SYM_REF
        | named_range
        | cartesian_offset
        | "(" prec0 ")"

    ?stmt: set_at | if_stmt
    set_at: prec0 "@" prec0 "←" prec1 ";"
    else_branch_seperator: "else"
    if_stmt: "if" "(" prec0 ")" "{" ( stmt )* "}" else_branch_seperator "{" ( stmt )* "}"

    typed_literal: ( INT_LITERAL | FLOAT_LITERAL | SYM_REF ) ":" TYPE_LITERAL
    ?type_expr: TYPE_LITERAL
        | TYPE_LITERAL "[" INT_LITERAL ("," INT_LITERAL)* "]" -> shaped_scalar_type
        | "{" type_expr ("," type_expr)* "}" -> tuple_type

    named_range: AXIS_LITERAL ":" "[" prec0 "," prec0 "["
    cartesian_offset: AXIS_LITERAL "→" AXIS_LITERAL
    function_definition: ID_NAME "=" "λ(" ( SYM "," )* SYM? ")" "→" prec0 ";"
    declaration: ID_NAME "=" "temporary(" "domain=" prec0 "," "dtype=" type_expr ")" ";"
    stencil_closure: prec0 "←" "(" prec0 ")" "(" ( SYM_REF ", " )* SYM_REF ")" "@" prec0 ";"
    fencil_definition: ID_NAME "(" ( SYM "," )* SYM ")" "{" ( function_definition )* ( stencil_closure )+ "}"
    program: ID_NAME "(" ( SYM "," )* SYM ")" "{" ( function_definition )* ( declaration )* ( stmt )+ "}"

    %import common (CNAME, SIGNED_FLOAT, SIGNED_INT, WS)
    %ignore WS
"""  # noqa: RUF001 [ambiguous-unicode-character-string]


def _split_axis_literal(
    text: str,
) -> tuple[str, common.DimensionKind, common.DimensionCategory | None]:
    """Split an axis literal into its name, kind and category (`None` if not spelled)."""
    category = _AXIS_CATEGORY_FROM_SUFFIX.get(text[-1])
    if category is not None:
        text = text[:-1]
    return text[:-1], _AXIS_KIND_FROM_SUFFIX[text[-1]], category


def _is_full_tag(name: str) -> bool:
    # Only a full tag has a module part; a short name is a (dotted) qualname.
    return ":" in name


def _full_dimension_ref(text: str) -> common.DimensionRef:
    # NOTE: the suffixes are the kind and the category (if absent: the default); with the tag
    # they rebuild the `DimensionRef` without importing the declaring module.
    tag, kind, category = _split_axis_literal(text)
    return common.DimensionRef.from_tag(tag, kind, category or common.DimensionCategory.OTHER)


def _bare_literal(value: str) -> ir.Literal:
    """A literal written without a type annotation."""
    return ir.Literal(value=value, type=pretty_printer.implied_literal_type(value))


@lark_visitors.v_args(inline=True)
class ToIrTransformer(lark_visitors.Transformer):
    def __init__(self, dimensions: Mapping[str, common.DimensionRef]) -> None:
        """
        Args:
            dimensions: The short dimension names the header declares.
        """
        super().__init__(visit_tokens=True)
        self.dimensions = dimensions

    def SYM(self, value: lark_lexer.Token) -> ir.Sym:
        return ir.Sym(id=value.value)

    def SYM_REF(self, value: lark_lexer.Token) -> Union[ir.SymRef, ir.Literal]:
        if value.value in ("True", "False"):
            return _bare_literal(value.value)
        return ir.SymRef(id=value.value)

    def INT_LITERAL(self, value: lark_lexer.Token) -> ir.Literal:
        return _bare_literal(str(int(value.value)))

    def FLOAT_LITERAL(self, value: lark_lexer.Token) -> ir.Literal:
        return _bare_literal(value.value)

    def TYPE_LITERAL(self, value: lark_lexer.Token) -> ts.ScalarType:
        if (kind := pretty_printer.SCALAR_TYPE_KINDS.get(value.value)) is None:
            raise ValueError(
                f"Invalid type '{value}'; expected one of "
                f"{', '.join(sorted(pretty_printer.SCALAR_TYPE_KINDS))}."
            )
        return ts.ScalarType(kind=kind)

    def shaped_scalar_type(self, type_: ts.ScalarType, *shape: ir.Literal) -> ts.ScalarType:
        return ts.ScalarType(kind=type_.kind, shape=[int(s.value) for s in shape])

    def tuple_type(self, *types: ts.DataType) -> ts.TupleType:
        return ts.TupleType(types=list(types))

    def typed_literal(
        self, value: Union[ir.Literal, ir.SymRef], type_: ts.ScalarType
    ) -> ir.Literal:
        if not isinstance(value, ir.Literal):
            raise ValueError(f"Only a literal can carry a type annotation, got '{value.id}'.")
        return ir.Literal(value=value.value, type=type_)

    def OFFSET_LITERAL(self, value: lark_lexer.Token) -> ir.OffsetLiteral:
        v: Union[int, str] = value.value[:-1]
        try:
            v = int(v)
        except ValueError:
            pass
        return ir.OffsetLiteral(value=v)

    def ID_NAME(self, value: lark_lexer.Token) -> str:
        return value.value

    def INFINITY_LITERAL(self, value: lark_lexer.Token) -> ir.InfinityLiteral:
        if value.value == "-∞":
            return ir.InfinityLiteral.NEGATIVE
        assert value.value == "∞"
        return ir.InfinityLiteral.POSITIVE

    def AXIS_LITERAL(self, value: lark_lexer.Token) -> ir.AxisLiteral:
        name, kind, category = _split_axis_literal(value.value)
        if _is_full_tag(name):
            return ir.AxisLiteral(dim=_full_dimension_ref(value.value))
        if (dim := self.dimensions.get(name)) is None:
            raise ValueError(
                f"Dimension '{name}' is not declared. A short dimension name needs a header entry"
                f" '{pretty_printer.DIMENSION_HEADER_KEYWORD} {name} = <module>:<qualname>"
                "<kind><category>;'; otherwise spell the full tag 'module:qualname' in place."
            )
        if dim.kind != kind or (category is not None and dim.category != category):
            raise ValueError(
                f"'{value.value}' does not match the declaration of '{name}':"
                f" '{pretty_printer.format_full_axis_literal(dim)}'."
            )
        return ir.AxisLiteral(dim=dim)

    def dim_decl(self, *args: object) -> None:
        return None  # read by `pparse` before the transformation

    def lam(self, *args: ir.Node) -> ir.Lambda:
        *params, expr = args
        return ir.Lambda(params=params, expr=expr)

    def bool_and(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="and_"), args=[lhs, rhs])

    def bool_or(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="or_"), args=[lhs, rhs])

    def bool_not(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="not_"), args=[arg])

    def bool_xor(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="xor_"), args=[arg])

    def plus(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="plus"), args=[lhs, rhs])

    def minus(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="minus"), args=[lhs, rhs])

    def multiplies(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="multiplies"), args=[lhs, rhs])

    def divides(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="divides"), args=[lhs, rhs])

    def floordiv(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="floordiv"), args=[lhs, rhs])

    def mod(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="mod"), args=[lhs, rhs])

    def eq(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="eq"), args=[lhs, rhs])

    def greater(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="greater"), args=[lhs, rhs])

    def less(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="less"), args=[lhs, rhs])

    def less_equal(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="less_equal"), args=[lhs, rhs])

    def greater_equal(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="greater_equal"), args=[lhs, rhs])

    def deref(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="deref"), args=[arg])

    def lift(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="lift"), args=[arg])

    def as_fieldop(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="as_fieldop"), args=[arg])

    def astype(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="cast_"), args=[arg])

    def shift(self, *offsets: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="shift"), args=list(offsets))

    def tuple_get(self, tup: ir.Expr, idx: ir.Literal) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="tuple_get"), args=[idx, tup])

    def make_tuple(self, *args: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=list(args))

    def named_range(self, name: ir.AxisLiteral, start: ir.Expr, end: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="named_range"), args=[name, start, end])

    def cartesian_offset(
        self, domain: ir.AxisLiteral, codomain: ir.AxisLiteral
    ) -> ir.CartesianOffset:
        return ir.CartesianOffset(domain=domain, codomain=codomain)

    def cartesian_domain(self, *ranges: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=list(ranges))

    def unstructured_domain(self, *ranges: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="unstructured_domain"), args=list(ranges))

    def ifthenelse(self, condition: ir.Expr, then: ir.Expr, otherwise: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="if_"), args=[condition, then, otherwise])

    def call(self, fun: ir.Expr, *args: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=fun, args=list(args))

    def function_definition(self, *args: ir.Node) -> ir.FunctionDefinition:
        fid, *params, expr = args
        return ir.FunctionDefinition(id=fid, params=params, expr=expr)

    def if_stmt(self, cond: ir.Expr, *args):
        found_else_seperator = False
        true_branch = []
        false_branch = []
        for arg in args:
            if isinstance(arg, lark_tree.Tree):
                assert arg.data == "else_branch_seperator"
                found_else_seperator = True
                continue

            if not found_else_seperator:
                true_branch.append(arg)
            else:
                false_branch.append(arg)

        return ir.IfStmt(
            cond=cond,
            true_branch=true_branch,
            false_branch=false_branch,
        )

    def declaration(self, *args: ir.Expr) -> ir.Temporary:
        tid, domain, dtype = args
        return ir.Temporary(id=tid, domain=domain, dtype=dtype)

    def set_at(self, *args: ir.Expr) -> ir.SetAt:
        target, domain, expr = args
        return ir.SetAt(expr=expr, domain=domain, target=target)

    def program(self, fid: str, *args: ir.Node) -> ir.Program:
        params = []
        function_definitions = []
        body = []
        declarations = []
        for arg in args:
            if isinstance(arg, ir.Sym):
                params.append(arg)
            elif isinstance(arg, ir.FunctionDefinition):
                function_definitions.append(arg)
            elif isinstance(arg, ir.Temporary):
                declarations.append(arg)
            else:
                assert isinstance(arg, ir.Stmt)
                body.append(arg)
        return ir.Program(
            id=fid,
            function_definitions=function_definitions,
            params=params,
            body=body,
            declarations=declarations,
        )

    def start(self, *args: ir.Node | None) -> ir.Node:
        *_header, root = args
        assert root is not None
        return root


def _dimension_header(tree: lark_tree.Tree) -> dict[str, common.DimensionRef]:
    dimensions: dict[str, common.DimensionRef] = {}
    for decl in tree.children:
        if not (isinstance(decl, lark_tree.Tree) and decl.data == "dim_decl"):
            continue
        name, literal = (str(token) for token in decl.children)
        if _is_full_tag(name):
            raise ValueError(f"Invalid short dimension name '{name}': it has a module part.")
        if not _is_full_tag(_split_axis_literal(literal)[0]):
            raise ValueError(
                f"The declaration of '{name}' must spell the full tag, got '{literal}'."
            )
        if name in dimensions:
            raise ValueError(f"Dimension '{name}' is declared twice.")
        dimensions[name] = _full_dimension_ref(literal)
    return dimensions


def pparse(pretty_str: str) -> ir.Node:
    """
    Parse the text `pretty_printer.pformat` prints back to the IR node.

    A dimension is either spelled in full in place (`module:qualname` with its kind and
    category suffixes, e.g. `my.dims:IDimₕᵃ`), which keeps hand-written IR easy, or by a short
    name with its kind suffix (`IDimₕ`) declared in the header (`dim IDim = my.dims:IDimₕᵃ;`).
    A short name that the header does not declare is an error: nothing is imported to guess it.

    Raises:
        ValueError: If a dimension is malformed, undeclared or does not match its declaration.
    """
    parser = lark.Lark(GRAMMAR, parser="earley")
    tree = parser.parse(pretty_str)
    try:
        return ToIrTransformer(_dimension_header(tree)).transform(tree)
    except lark_exceptions.VisitError as e:
        raise e.orig_exc from None
