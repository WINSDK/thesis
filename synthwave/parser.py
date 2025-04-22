from typing import Set
from lark import Lark, Transformer, v_args, Token
from lark.lexer import TerminalDef, PatternStr
from .dsl import Type, UOp, Ops, TVar, T, Scheme, reduce_redundant

lambda_grammar = r"""
?start       : term

?term        : abstraction
             | application

abstraction  : LAMBDA (IDENT2 (WS IDENT2)*) "." term  -> abstr
application  : atom (WS atom)*                        -> appl

?atom        : literal
             | IDENT1                                 -> var
             | "(" term ")"

list_lit     : "[" "]"
             | "[" literal ("," literal)* "]"

?literal     : FLOAT                                -> float_lit
             | INTEGER                              -> int_lit
             | BOOL                                 -> bool_lit
             | CHAR                                 -> char_lit
             | STRING                               -> string_lit
             | list_lit

LAMBDA.2     : ("lambda" WS) | "L" | "λ"
BOOL         : "True" | "False"
CHAR         : "'" ( /\\./ | /[^\\']/ ) "'"       // escaped or raw char
STRING       : "\"" /[a-zA-Z0-9 ]*/ "\""
FLOAT.3      : ["+"|"-"] DECIMAL
INTEGER.2    : SIGNED_INT

IDENT1       : /[a-zA-Z_+\-*\/=%><!&|^~]+/
IDENT2       : /[a-zA-Z_+\-*\/=%><!&|^~]+/

%import common.SIGNED_INT
%import common.DECIMAL
WS           : " "
"""

term_parser = Lark(lambda_grammar, parser="lalr", start="term")

def skip_ws(items):
    return [i for i in items if not (isinstance(i, Token) and i.type == "WS")]

@v_args(inline=True)
class TermBuilder(Transformer):
    def int_lit(self, t):
        return UOp(Ops.Val, [int(t)])
    def float_lit(self, t):
        return UOp(Ops.Val, [float(t)])
    def bool_lit(self, t):
        return UOp(Ops.Val, [t == "True"])
    def char_lit(self, t):
        t = t[1:-1]
        return UOp(Ops.Val, [t])
    def string_lit(self, t):
        return UOp(Ops.Val, [list(eval(t))])
    def var(self, name):
        return UOp(Ops.Var, [name])
    def list_lit(self, *elems):
        elems = skip_ws(elems)
        return UOp(Ops.Val, [elems])
    def appl(self, *items):
        fn, *args = skip_ws(items)
        return UOp(Ops.Appl, [fn, *args])
    def abstr(self, *items):
        _, *params, body = skip_ws(items)
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

def parse_inc(src: str, **kwargs):
    env = kwargs.get("known", set()).copy()
    assert isinstance(env, set), "parse(..) parameter `known` is not a set"
    ip = term_parser.parse_interactive(src)
    ip.exhaust_lexer()
    allowed = [] 
    for term_name in ip.accepts():
        if term_name == "$END":
            term_def = TerminalDef("$END", PatternStr(""))
            allowed.append(term_def)
            continue
        term_def = next(t for t in term_parser.terminals if t.name == term_name)
        allowed.append(term_def)
    # Couldn't find a way to restrict the grammar to NOT allowing LAMBDA
    # as a valid identifier
    if src in ["L", "λ", "lambda"]:
        allowed = [t for t in allowed if t.name != "$END"]
    return allowed

type_grammar = r"""
?type          : arrow_type
               | atom_type+

arrow_type     : atom_type+ ("->" type)+

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
    def arrow_type(self, *elems):
        *params_ty, body_ty = elems
        return Type.arrow(params_ty, body_ty)

def parse_poly_type(src: str) -> Scheme:
    ty = TypeBuilder().transform(type_parser.parse(src))
    vars = [TVar(tv.name, generic=True) for tv in ty.free_vars()]
    return Scheme(vars, ty)
