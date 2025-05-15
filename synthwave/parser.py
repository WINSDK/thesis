from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Set, Any, Optional
from lark import Lark, Transformer, v_args, Token
from lark.lexer import TerminalDef, PatternStr
from .dsl import Type, UOp, Ops, TVar, T, Scheme, reduce_redundant, fresh_type_var
import string

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
    vars = {}

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
        if name in self.vars:
            return self.vars[name]
        var = fresh_type_var(prefix=name, generic=True)
        self.vars[name] = var
        return var
    def list_type(self, elem):
        return Type(T.List, [elem])
    def arrow_type(self, *elems):
        *params_ty, body_ty = elems
        return Type.arrow(params_ty, body_ty)

def parse_poly_type(src: str) -> Scheme:
    ty = TypeBuilder().transform(type_parser.parse(src))
    return Scheme(ty.free_vars(), ty)

class Modi(IntEnum):
    Single = auto()
    Many = auto()

@dataclass(frozen=True)
class NonTerminal:
    name: str
    chars: frozenset[str]
    modi: Modi = Modi.Single

IDENT = NonTerminal("IDENT", frozenset(string.ascii_letters + "_+-*/=%><!&|^~"), Modi.Many)
LAMBDA_ARG_IDENT = NonTerminal("LAMBDA_ARG_IDENT", frozenset(string.ascii_letters + "_+-*/=%><!&|^~"), Modi.Many)
DIGIT = NonTerminal("DIGIT", frozenset("0123456789"), Modi.Many)
BOOLEAN = NonTerminal("BOOLEAN", frozenset({"True", "False"}))
LAMBDA = NonTerminal("LAMBDA", frozenset({"lambda"}))
QUOTE = NonTerminal("QUOTE", frozenset({"\""}))
COMMA = NonTerminal("COMMA", frozenset({","}))
DOT = NonTerminal("DOT", frozenset({"."}))
WS = NonTerminal("WS", frozenset({" "}))
LBRACKET = NonTerminal("LBRACKET", frozenset({"["}))
RBRACKET = NonTerminal("RBRACKET", frozenset({"]"}))
LPAREN = NonTerminal("LPARAN", frozenset({"("}))
RPAREN = NonTerminal("RPARAN", frozenset({")"}))

class Outcome(IntEnum):
    OK     = auto()
    ERR    = auto()

@dataclass(eq=True)
class Result:
    outcome : Outcome
    value   : Any = None                              # the AST node on success
    expect  : set[NonTerminal] = field(default_factory=set) # what tokens _could_ have matched here
    pos     : int  = -1                               # where the failure occurred
    depth   : int  = 0

    def __bool__(self):
        return self.outcome is Outcome.OK

    @staticmethod
    def ok(value, **kwargs) -> "Result":
        expect = kwargs.get("expect", set)
        return Result(Outcome.OK, value=value, expect=expect)

class Backtrack:
    def __init__(self, p):
        self.p = p
        self.start = p.off
        self.active = True

    def commit(self):
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.active:
            self.p.off = self.start     # rollback

@dataclass
class Parser:
    src: str
    off: int
    env: set[str]
    strict: bool # Require all var's to be known

    def err(self, expect=None) -> Result:
        if expect is None:
            expect = set()
        return Result(Outcome.ERR, expect=expect, pos=self.off)

    def choice(self, *parsers) -> Result:
        best_fail_pos = -1
        best_expects = set()
        for parse in parsers:
            with Backtrack(self) as bt:
                if r := parse():
                    bt.commit()
                    return r
                if r.pos > best_fail_pos:
                    best_fail_pos = r.pos
                    best_expects = r.expect.copy()
                elif r.pos == best_fail_pos:
                    best_expects |= r.expect
        # No branch succeeded.
        return Result(Outcome.ERR, expect=best_expects, pos=best_fail_pos)

    def peek(self, expect):
        if self.off < len(self.src) and self.src[self.off] in expect.chars:
            return Result.ok(self.src[self.off])
        return self.err({expect})

    def at(self, s):
        return self.src[self.off:].startswith(s)

    def consume(self, s, expect: set[NonTerminal]):
        if self.at(s):
            self.off += len(s)
            return Result.ok(s)
        return self.err(expect)

    def consume_while(self, expect):
        start = self.off
        while self.peek(expect):
            self.off += 1
        if self.off == start:
            return self.err({expect})
        return Result.ok(self.src[start:self.off])

    def parse_ident(self, ident=None):
        if not ident:
            ident = IDENT
        with Backtrack(self) as bt:
            if not (p := self.consume_while(ident)):
                return p
            # 'lambda' is not a valid identifier.
            if p.value == "lambda":
                return self.err()
            bt.commit()
            return Result.ok(p.value, expect={ident})

    def parse_abstr(self):
        with Backtrack(self) as bt:
            if not (c := self.consume("lambda", {LAMBDA})):
                return c
            if not (c := self.consume(" ", {WS})):
                return c
            if not (p1 := self.parse_ident(LAMBDA_ARG_IDENT)):
                return p1
            params = [p1.value]
            while True:
                if not self.consume(" ", {WS}):
                    break
                if not (p := self.parse_ident(LAMBDA_ARG_IDENT)):
                    return p
                params.append(p.value)
            # Don't allow parameters with the same name.
            # expects = [e for e in [" ", ".", *IDENT] if e not in params]
            if not (c := self.consume(".", {WS, DOT, IDENT})):
                return c
            # Save the current env
            old_env = old_env = self.env.copy()
            self.env.update(params)
            if not (body := self.parse()):
                return body
            # Restore the previous env
            self.env = old_env
            bt.commit()
            v = UOp(Ops.Abstr, [*params, body.value])
            return Result.ok(v, expect=body.expect)

    def parse_signed_int(self):
        with Backtrack(self) as bt:
            p1 = self.consume("+", set()) or\
                 self.consume("-", set()) or\
                 Result.ok("")
            if not (p2 := self.consume_while(DIGIT)):
                return p2
            bt.commit()
            v = UOp(Ops.Val, [int(f"{p1.value}{p2.value}")])
            return Result.ok(v, expect=DIGIT)

    def parse_float(self):
        with Backtrack(self) as bt:
            if not (p1 := self.parse_signed_int()):
                return p1
            if not (c := self.consume('.', {DOT})):
                return c
            p2 = self.consume_while(DIGIT) or Result.ok(0)
            bt.commit()
            v = UOp(Ops.Val, [float(f"{p1.value}.{p2.value}")])
            return Result.ok(v, expect={DIGIT})

    def parse_bool(self):
        if self.consume("True", {BOOLEAN}):
            v = True
        elif self.consume("False", {BOOLEAN}):
            v = False
        else:
            return self.err({BOOLEAN})
        return Result.ok(UOp(Ops.Val, [v])) 

    def parse_string(self):
        with Backtrack(self) as bt:
            if not (c := self.consume("\"", {QUOTE})):
                return c
            if not (p := self.parse_ident()):
                return p
            if not (c := self.consume("\"", {QUOTE})):
                return c
            bt.commit()
            v = p.value
            return Result.ok(UOp(Ops.Val, [v])) 

    def parse_lit(self):
        with Backtrack(self) as bt:
            if not (c := self.consume("[", {LBRACKET})):
                return c
            p1 = self.parse_literal()
            vals = [p1.value]
            while True:
                if not self.consume(",", {COMMA}):
                    break
                if not (p := self.parse_literal()):
                    return p
                vals.append(p.value)
            if not (c := self.consume("]", {RBRACKET})):
                return c
            bt.commit()
            return Result.ok(UOp(Ops.Val, [vals])) 

    def parse_literal(self):
        return self.choice(
            self.parse_float,
            self.parse_signed_int,
            self.parse_bool,
            self.parse_string,
            self.parse_lit,
        )

    def parse_var(self):
        if (v := self.parse_ident()):
            if self.strict and v.value not in self.env:
                return self.err()
            v = UOp(Ops.Var, [v.value])
            return Result.ok(v)
        else:
            non_terminals = {NonTerminal("VAR", frozenset(self.env))}
            return self.err(non_terminals)

    def parse_braces(self):
        with Backtrack(self) as bt:
            if not (c := self.consume('(', {LPAREN})):
                return c
            if not (p := self.parse()):
                return p
            if not (c := self.consume(')', {RPAREN})):
                return c
            bt.commit()
            return p

    def parse_atom(self):
        return self.choice(
            self.parse_literal,
            self.parse_braces,
            self.parse_var,
        )

    def parse_appl(self):
        from .typing import infer, Type, T
        # A -> B -> C == A -> (B -> C) => [A, B, C]
        def decompose(ty: Type):
            if isinstance(ty, TVar) or ty.t != T.Arrow:
                return [ty]
            head, tail = ty.params
            return [head] + decompose(tail)
        def filter_expects_by_ty(expect, given: Type):
            g_args = decompose(given) # [:-1]
            def filter(expected):
                p = Parser(expected, 0, self.env, self.strict)
                expected_ty = infer(p.parse().value)
                e_args = decompose(expected_ty)
                if len(e_args) > len(g_args):
                    return False
                for g, e in zip(g_args, e_args):
                    # If g or e are generic, we don't check equality.
                    if any(isinstance(x, TVar) and x.generic for x in (g, e)):
                        continue
                    if g != e:
                        return False
                return True
            results = set()
            for e in expect:
                if e == LBRACKET:
                    if isinstance(g_args[0], TVar):
                        continue
                    if g_args[0] == Type(T.List):
                        results.add(e)
                elif e == QUOTE and g_args[0] == Type(T.Int):
                    results.add(e)
                elif e == DIGIT and g_args[0] == Type(T.Int):
                    results.add(e)
                elif e == QUOTE and g_args[0] == Type(T.List, [T.Char]):
                    results.add(e)
                elif e == LPAREN:
                    results.add(e)
                elif e.name == "VAR":
                    for token in e.chars:
                        if filter(token):
                            v = NonTerminal("VAR", frozenset({token}))
                            results.add(v)
            return results
        def flatten(args):
            return args[0] if len(args) == 1 else UOp(Ops.Appl, args)
        with Backtrack(self) as bt:
            if not (p1 := self.parse_atom()):
                return p1
            body = p1.value
            args = [body]
            while True:
                if not self.consume(" ", {WS}):
                    break
                if not (p := self.parse_atom()):
                    ty = infer(flatten(args))
                    p.expect = filter_expects_by_ty(p.expect, ty)
                    return p
                args.append(p.value)
            bt.commit()
            v = flatten(args)
            return Result.ok(v, expect={WS})

    def parse(self):
        return self.choice(self.parse_abstr, self.parse_appl)


def parse2(src: str, **kwargs):
    strict = kwargs.get("strict", False)
    assert isinstance(strict, bool), "parse(..) parameter `strict` is not a bool"
    env = kwargs.get("known", set()).copy()
    assert isinstance(env, set), "parse(..) parameter `known` is not a set"
    p = Parser(src, 0, env, strict)

    r = p.parse()
    if r:
        if p.off != len(p.src):
            return p.err()
        return r.value
    # total failure: r has pos and expect filled
    expected = ""
    if len(r.expect) > 0:
        expect = sorted(list(r.expect))
        expected = ": expected" + ", ".join(expect)
    raise SyntaxError(f"Parse error at column {r.pos}{expected}")

def parse_inc2(src: str, **kwargs) -> tuple[Optional[UOp], set[str]]:
    strict = kwargs.get("strict", False)
    assert isinstance(strict, bool), "parse(..) parameter `strict` is not a bool"
    env = kwargs.get("known", set()).copy()
    assert isinstance(env, set), "parse(..) parameter `known` is not a set"
    p = Parser(src, 0, env, strict)
    r = p.parse()
    return r.value if r else None, r.expect


# -- Numbers
# Y -> (+ Y Y) | (* Y Y) | (- Y Y) | (/ Y Y) | (% Y Y) | (** Y Y) | Float
# X -> (+ X X) | (* X X) | (- X X) | (/ X X) | (% X X) | (** X X) | Int
# Int -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | Int Int
# Float -> Int "." Int | Int "."
# 
# Ex. 1
# {Abstr | Var}
# lambda {ident}
# lambda x{'.'}
# lambda x.{Var :: * -> *}
# lambda x.map {Var | Abstr :: T List}
#
# {Abstr | Var}
# add {Literal | Var | Term :: Int}
# add 100 {Literal | Var | Term :: Int}
# add 100 ({Literal | Var | Term :: Int}
