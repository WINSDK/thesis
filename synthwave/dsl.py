from typing import Any, Optional
from enum import auto, IntEnum
from dataclasses import dataclass

class ExternalError(Exception):
    pass

class Ops(IntEnum):
    # Wrapped python value
    Val = auto()
    # Anything that's `named`: functions, parameters, etc
    Var = auto()  
    # Application (M N): [body, param1, param2, ..]
    Appl = auto()
    # Abstraction (λx.M): [arg1, arg2, .., body]
    Abstr = auto() 
    # Regular python functions: [arg1, arg2, .., python_fn]
    External = auto() 
    # Context-specific functions: [arg1, arg2, .., body, env]
    Closure = auto()

@dataclass(eq=False, slots=True)
class UOp:
    op: Ops
    args: list[Any]

    def __repr__(self):
        if self.op == Ops.Var or self.op == Ops.Val:
            return str(self.args[0])
        def fmt_arg(arg):
            if not isinstance(arg, UOp) and callable(arg):
                return arg.__name__
            else:
                return str(arg)
        args = ", ".join(map(fmt_arg, self.args))
        return f"{self.op.name}({args})"

    def __call__(self, *args):
        from .eval import evaluate
        match self.op:
            case Ops.Closure:
                cargs = []
                for a in args:
                    # We need to wrap literals in a Ops.Val but
                    # still ignore uop's
                    cargs.append(a if isinstance(a, UOp) else UOp(Ops.Val, [a]))
                return evaluate(UOp(Ops.Appl, [self, *cargs]))
            case Ops.Var:
                return evaluate(self)(*args)
            case _:
                raise ExternalError(f"Can't call uop: {self}")

class T(IntEnum):
    Int = auto()
    Float = auto()
    Char = auto()
    Bool = auto()
    List = auto()
    Arrow = auto()

# TODO: This should be a dataclass
class Type:
    t: T
    params: list["Type"]

    def __init__(self, t: T, params: Optional[list[Any]] = None):
        self.t = t
        if params is not None:
            self.params = params
        else:
            self.params = []

    def __repr__(self):
        match self.t:
            case T.Int:
                return "Int"
            case T.Float:
                return "Float"
            case T.Char:
                return "Char"
            case T.Bool:
                return "Bool"
            case T.List:
                return f"{str(self.params[0])} List"
            case T.Arrow:
                return " -> ".join(map(str, self.params))

    def free_vars(self) -> set:
        if isinstance(self, TVar):
            return {self}
        ft = set()
        for t in self.params:
            ft |= t.free_vars()
        return ft

    @staticmethod
    def arrow(params_ty: list["Type"], body_ty: "Type") -> "Type":
        """Right-fold for function types: param1 -> param2 -> ... -> body"""
        func_ty = body_ty
        for pt in reversed(params_ty):
            func_ty = Type(T.Arrow, [pt, func_ty])
        return func_ty

@dataclass(eq=True, frozen=True)
class TVar(Type):
    name: str
    generic: bool = False

    def __repr__(self):
        s = self.name
        if self.generic:
            # Strip numeric suffix
            while s[-1].isnumeric():
                s = s[:-1]
        return s

@dataclass(eq=True, frozen=True)
class Scheme:
    vars: list[TVar]
    ty: Type

    def free_vars(self) -> set:
        # Free type variables in a scheme are those free in its type minus the quantified ones.
        return self.ty.free_vars() - set(self.vars)

def pretty_print(expr) -> str:
    if isinstance(expr, str):
        return expr
    if isinstance(expr, list) and len(expr) > 0:
        p = (pretty_print(e) for e in expr)
        if isinstance(expr[0], str):
            return '"' + "".join(p) + '"'
        else:
            return "[" + ", ".join(p) + "]"
    return str(expr)
