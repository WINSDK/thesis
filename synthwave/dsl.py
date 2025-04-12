from typing import Any
from enum import auto, IntEnum
from dataclasses import dataclass

class ExternalError(Exception):
    pass

class Ops(IntEnum):
    # Wrapped python value
    Val   = auto()
    # Anything that's `named`: functions, parameters, etc
    Var   = auto()  
    # Application (M N): [body, param1, param2, ..]
    Appl = auto()
    # Abstraction (λx.M): [arg1, arg2, .., body]
    Abstr = auto() 
    # Regular python functions: [arg1, arg2, .., python_fn]
    External = auto() 
    # Context-specific functions: [arg1, arg2, .., body, env]
    Closure = auto()

@dataclass(eq=False, slots=True)
class UOp():
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
