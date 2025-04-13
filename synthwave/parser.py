from dataclasses import dataclass
import re
from .dsl import UOp, Ops

LITERALS = ("FLOAT", "INT", "BOOL", "CHAR", "STRING")
ATOMS = (*LITERALS, "IDENT", "LPAREN", "LBRACKET", "LAMBDA")
TOKEN_REGEX = r"""
(?P<LAMBDA>lambda|L|λ)              # 'lambda' or 'L' or 'λ'
|(?P<FLOAT>-?(?:\d+\.\d*|\d*\.\d+)) # float literal (e.g. 123.456, -0.5)
|(?P<INT>-?\d+)                     # integer literal
|(?P<BOOL>True|False)               # bool literal
|(?P<CHAR>'(?:\\.|[^\\'])')         # char literal (e.g. 'a', '\n')
|(?P<STRING>"(?:\\.|[^\\"])*")      # string literal (e.g. "hello", "line\nbreak")
|(?P<IDENT>[a-zA-Z_]\w*)            # identifier (variable names, etc.)
|(?P<LPAREN>\()                     # (
|(?P<RPAREN>\))                     # )
|(?P<LBRACKET>\[)                   # [
|(?P<RBRACKET>\])                   # ]
|(?P<COMMA>,)                       # ,
|(?P<DOT>\.)                        # .
|(?P<WHITESPACE>\s+)                # whitespace (ignored)
"""

@dataclass
class Token():
    kind: str
    val: str

def tokenize(src: str) -> list[Token]:
    tokens = []
    for match in re.finditer(TOKEN_REGEX, src, re.VERBOSE):
        kind = match.lastgroup
        text = match.group()
        assert isinstance(kind, str), "Impossible"
        if kind == "WHITESPACE":
            continue
        tokens.append(Token(kind, text))
    return tokens

@dataclass
class Parser():
    tokens: list[Token]
    off: int = 0

    def peek(self):
        if self.off < len(self.tokens):
            return self.tokens[self.off]
        return None

    def peek_no_eof(self):
        if not (t := self.peek()):
            raise SyntaxError("Incomplete statement")
        return t

    def next(self):
        if self.off < len(self.tokens):
            tok = self.tokens[self.off]
            self.off += 1
            return tok
        return None

    def expect(self, kind: str, err: str):
        if not (t := self.next()) or t.kind != kind:
            raise SyntaxError(err)
        return t

    def peek_kind(self, kind: str):
        t = self.peek()
        if not t or t.kind != kind:
            return None
        return t

    def parse_primitive(self, t):
        match t.kind:
            case "INT" | "FLOAT" | "BOOL":
                self.next()
                return eval(t.val)
            case "CHAR" | "STRING":
                self.next()
                return list(eval(t.val))
            case _:
                raise SyntaxError(f"Unknown primitive: {t}")

    def parse_list(self) -> UOp:
        elems = []
        while True:
            if self.peek_kind("RBRACKET"):
                break
            # We currently only support list's of literals
            t = self.peek_no_eof()
            if t.kind in LITERALS:
                prim = self.parse_primitive(t)
                elems.append(prim)
            else:
                raise SyntaxError("List must be made up of literals")
            if self.peek_kind("COMMA"):
                self.next()
        val = [v for v in elems]
        return UOp(Ops.Val, [val])

    def parse_atom(self) -> UOp:
        t = self.peek_no_eof()
        match t.kind:
            case "INT" | "FLOAT" | "BOOL" | "CHAR" | "STRING":
                prim = self.parse_primitive(t)
                return UOp(Ops.Val, [prim])
            case "IDENT":
                self.next()
                return UOp(Ops.Var, [t.val])
            case "LPAREN":
                self.next()
                expr = self.parse_expr()
                self.expect("RPAREN", "Missing closing parenthesis")
                return expr
            case "LBRACKET":
                self.next()
                expr = self.parse_list()
                self.expect("RBRACKET", "Missing closing bracket")
                return expr
            case "LAMBDA":
                return self.parse_abstr()
            case _:
                raise SyntaxError(f"Unexpected token: {t}")

    def parse_abstr(self) -> UOp:
        self.next() # Lambda
        params = []
        while True:
            if t := self.peek_kind("IDENT"):
                self.next()
                params.append(t.val)
            else:
                break
        self.expect("DOT", "Expected '.' after lambda parameter")
        if len(params) == 0:
            raise SyntaxError("Abstractions must have at least one argument")
        body = self.parse_expr()
        return UOp(Ops.Abstr, [*params, body])

    def parse_appl(self) -> UOp:
        body = self.parse_atom()
        args = []
        while True:
            if not (t := self.peek()):
                break
            if t.kind in ATOMS:
                args.append(self.parse_atom())
            else:
                break
        return UOp(Ops.Appl, [body, *args])

    def parse_expr(self) -> UOp:
        if self.peek_kind("LAMBDA"):
            return self.parse_abstr()
        else:
            return self.parse_appl()

def reduce_redundant(expr):
    """Removes redundant bracket's in nested applications"""
    if not isinstance(expr, UOp):
        return expr
    if expr.op == Ops.Appl and len(expr.args) == 1:
        return reduce_redundant(expr.args[0])
    return UOp(expr.op, [reduce_redundant(a) for a in expr.args])

def parse(src: str) -> UOp:
    p = Parser(tokens=tokenize(src))
    expr = p.parse_expr()
    if p.next() is not None:
        raise SyntaxError("Generic syntax error")
    return reduce_redundant(expr)
