from typing import Set
from lark import Lark, Transformer, v_args
from .dsl import Type, UOp, Ops, TVar, T, Scheme, reduce_redundant

lambda_grammar = r"""
?term          : abstraction
               | application

abstraction    : LAMBDA param+ "." term             -> abstr
application    : atom+                              -> appl

?atom          : literal
               | IDENT                              -> var
               | "(" term ")"
               | list_lit

list_lit       : "[" [literal ("," literal)*] "]"   -> list_lit

?literal       : SIGNED_FLOAT                       -> float_lit
               | SIGNED_INT                         -> int_lit
               | BOOL                               -> bool_lit
               | CHAR                               -> char_lit
               | STRING                             -> string_lit

?param         : IDENT

LAMBDA.2       : "L" | "λ" | "lambda"
BOOL           : "True" | "False"
CHAR           : "'" ( /\\./ | /[^\\']/ ) "'"       // escaped or raw char
STRING         : ESCAPED_STRING

IDENT          : /[a-zA-Z_+\-*\/=%><!&|^~]+/

%import common.SIGNED_INT
%import common.SIGNED_FLOAT
%import common.ESCAPED_STRING
%import common.WS_INLINE
%ignore WS_INLINE
"""

term_parser = Lark(lambda_grammar, parser="lalr", start="term")

@v_args(inline=True)
class TermBuilder(Transformer):
    def int_lit(self, t):
        return UOp(Ops.Val, [int(t)])
    def float_lit(self, t):
        return UOp(Ops.Val, [float(t)])
    def bool_lit(self, t):
        return UOp(Ops.Val, [t == "True"])
    def char_lit(self, t):
        return UOp(Ops.Val, [int(t)])
    def string_lit(self, t):
        return UOp(Ops.Val, [list(eval(t))])
    def var(self, name):
        return UOp(Ops.Var, [name])
    def list_lit(self, *elems):
        elems = [] if elems == (None,) else list(elems)
        return UOp(Ops.Val, [elems])
    def appl(self, fn, *args):
        return UOp(Ops.Appl, [fn, *args])
    def abstr(self, *items):
        _, *params, body = items
        return UOp(Ops.Abstr, [*map(str, params), body])

def term_check(expr, env: Set[str]):
    if not isinstance(expr, UOp):
        return
    if expr.op == Ops.Var:
        var = expr.args[0]
        if var not in env:
            raise SyntaxError(f"Unbound variable: {var}")
    elif expr.op == Ops.Abstr:
        *params, _ = expr.args
        env |= set(params)
    for a in expr.args:
        term_check(a, env)

def parse(src: str, **kwargs) -> UOp:
    env = kwargs.get("known", set()).copy()
    assert isinstance(env, set), "parse(..) parameter `known` is not a set"
    expr = TermBuilder().transform(term_parser.parse(src))
    term_check(expr, env)
    expr = reduce_redundant(expr)
    return expr

type_grammar = r"""
?type          : arrow_type
               | atom_type+

arrow_type     : atom_type+ ("->" type)+     -> arrow

?atom_type     : "Int"                             -> int_t
               | "Float"                           -> float_t
               | "Bool"                            -> bool_t
               | "Char"                            -> char_t
               | "List" atom_type                  -> list_type
               | "(" type ")"
               | IDENT                             -> tvar

IDENT          : /[A-Za-z][A-Za-z0-9_]*/

%import common.WS_INLINE
%ignore WS_INLINE
"""

type_parser = Lark(type_grammar, parser="lalr", start="type")

@v_args(inline=True)
class TypeBuilder(Transformer):
    def int_t(self):
        return Type(T.Int)
    def float_t(self):
        return Type(T.Float)
    def bool_t(self):
        return Type(T.Bool)
    def char_t(self):
        return Type(T.Char)
    def string_t(self):
        return Type(T.List, [T.Char])
    def tvar(self, name):
        return TVar(name)
    def list_type(self, elem):
        return Type(T.List, [elem])
    def arrow(self, *elems):
        *params_ty, body_ty = elems
        return Type.arrow(params_ty, body_ty)

def parse_poly_type(src: str) -> Scheme:
    ty = TypeBuilder().transform(type_parser.parse(src))
    vars = [TVar(tv.name, generic=True) for tv in ty.free_vars()]
    return Scheme(vars, ty)
