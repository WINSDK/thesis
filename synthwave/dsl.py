from collections.abc import Callable
from typing import Union, Any, Dict, Optional, Tuple
from enum import auto, IntEnum
from dataclasses import dataclass
import typing
import inspect
import re

class ExternalCall(Exception):
    pass

class Ops(IntEnum):
    Val   = auto()
    Var   = auto()  
    # Application (M N): [body, param1, param2, ..]
    Appl = auto()
    # Abstraction (λx.M): [arg1, arg2, .., body]
    Abstr = auto() 
    # Context-specific functions: [args1, args2, .., body, env]
    Closure = auto()
    # Regular old python functions: [arg1, arg2, .., python_fn]
    External = auto() 

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

    # This is technically not always correct, example:
    # => map 1 (map add [1,2 ,3])
    #   => builtin_map(1, [Closure(y, External(x, y, builtin_add), {'x': 1}) ..
    # Expected => [2, 3, 4]
    def __call__(self, *args):
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
                raise ExternalCall(f"Can't call uop: {self}")

def external_fn(f):
    assert callable(f), "Not a valid python function"
    params = inspect.signature(f).parameters.keys()
    return UOp(
        Ops.Closure,
        [*params, UOp(Ops.External, [*params, f]), {}]
    )

def builtin_add(x: int, y: int) -> int:
    return x + y

def builtin_mul(x: int, y: int) -> int:
    return x * y

def builtin_map(f: Callable[[int], int], lst: list[int]) -> list[int]:
    return [f(x) for x in lst]

def builtin_cons(x: int, xs: list[int]) -> list[int]:
    return [x] + xs

def builtin_head(xs: list[int]) -> int:
    return xs[0]

def builtin_tail(xs: list[int]) -> list[int]:
    return xs[1:]

def builtin_append(xs: list[int], x: int) -> list[int]:
    return xs + [x]

def builtin_leq(a: int, b: int) -> bool:
    return a <= b

BUILTINS = {
    "add": external_fn(builtin_add),
    "mul": external_fn(builtin_mul),
    "map": external_fn(builtin_map),
    "cons": external_fn(builtin_cons),
    "head": external_fn(builtin_head),
    "tail": external_fn(builtin_tail),
    "append": external_fn(builtin_append),
    "leq?": external_fn(builtin_leq),
}

def evaluate(expr: UOp, env: Optional[Dict[str, UOp]]=None):
    assert isinstance(expr, UOp), f"Not an expr: {expr}"
    if env is None:
        env = {}
    def lookup(varname):
        if varname in env:
            var = env[varname]
        elif varname in BUILTINS:
            var = BUILTINS[varname]
        else:
            raise NameError(f"Unbound variable: {varname}")
        return evaluate(var, env) if isinstance(var, UOp) else var
    match expr.op:
        case Ops.Val:
            return expr.args[0]
        case Ops.Closure:
            return expr
        case Ops.Var:
            return lookup(expr.args[0])
        case Ops.Abstr:
            return UOp(Ops.Closure, [*expr.args, env])
        case Ops.Appl:
            func, *args = expr.args
            func = evaluate(func, env)
            args = [evaluate(a, env) for a in args]
            while args:
                if not (isinstance(func, UOp) and func.op == Ops.Closure):
                    raise TypeError(f"Cannot call a non-closure: {func}")
                *params, body, closure_env = func.args
                def apply(env, params, args):
                    env = env.copy()
                    for p, v in zip(params, args):
                        env[p] = v
                    return env
                if len(args) < len(params):
                    # Not enough arguments, so we do partial application
                    # which will require creating a new closure
                    not_applied = params[len(args):]
                    closure_env = apply(closure_env, params, args)
                    return UOp(Ops.Closure, [*not_applied, body, closure_env])
                else:
                    # Full application or over-applied 
                    closure_env = apply(closure_env, params, args[:len(params)])
                    func = evaluate(body, closure_env)
                    args = args[len(params):]
            return func
        case Ops.External:
            *params, body = expr.args
            pv = {p: lookup(p) for p in params}
            try:
                return body(**pv)
            except Exception as e:
                params = ", ".join(map(str, pv.values()))
                raise ExternalCall(f"{body.__name__}({params})") from e

@dataclass
class TInt:
    def __repr__(self):
        return "Int"

@dataclass
class TBool:
    def __repr__(self):
        return "Bool"

@dataclass(frozen=True)
class TList:
    elems: "Type"

    def __repr__(self):
        return f"{str(self.elems)} List"

@dataclass(frozen=True)
class TArrow:
    param: "Type"
    result: "Type"

    def __repr__(self):
        return str(self.param) + " -> " + str(self.result)

    @staticmethod
    def of_list(params_ty: list["Type"], body_ty: "Type"):
        """Rfold for function types param1 -> param2 -> ... -> body"""
        func_ty = body_ty
        for pt in reversed(params_ty):
            func_ty = TArrow(pt, func_ty)
        return func_ty

@dataclass(frozen=True)
class TVar:
    name: str

    def __repr__(self):
        return self.name

Type = Union[TInt, TBool, TList, TVar, TArrow]
PRIMITIVES = (TInt, TBool, TList)

def apply_subst(ty: Type, subst: Dict[TVar, Type]) -> Type:
    if isinstance(ty, TVar) and ty in subst:
        return apply_subst(subst[ty], subst)
    elif isinstance(ty, TArrow):
        return TArrow(
            apply_subst(ty.param, subst),
            apply_subst(ty.result, subst)
        )
    else:
        return ty

def occurs_in_type(tv: TVar, ty: Type, subst: Dict[TVar, Type]) -> bool:
    ty = apply_subst(ty, subst)
    if ty == tv:
        return True
    elif isinstance(ty, TArrow):
        return occurs_in_type(tv, ty.param, subst) or \
               occurs_in_type(tv, ty.result, subst)
    else:
        return False

def var_bind(tv: TVar, ty: Type, subst: Dict[TVar, Type]) -> Dict[TVar, Type]:
    """Bind the type var 'tv' to type 'ty' if allowed,"""
    if tv == ty:
        return subst
    # Cannot bind a type variable to a type that contains it
    if occurs_in_type(tv, ty, subst):
        raise TypeError(f"Occurs check fails: {tv} in {ty}")
    # Override the subst
    new_subst = subst.copy()
    new_subst[tv] = ty
    return new_subst

def unify(t1: Type, t2: Type, subst: Dict[TVar, Type]) -> Dict[TVar, Type]:
    """Unify t1 and t2 under the existing substitution 'subst',"""
    # 1) Apply current subst so we see the "true" forms
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)
    # 2) Pattern-match on t1, t2
    if isinstance(t1, TVar):
        return var_bind(t1, t2, subst)
    elif isinstance(t2, TVar):
        return var_bind(t2, t1, subst)
    # This must be manually modified for each new base type
    elif any(isinstance(t1, t) and isinstance(t2, t) for t in PRIMITIVES):
        return subst
    elif isinstance(t1, TArrow) and isinstance(t2, TArrow):
        # unify param, then unify result
        subst2 = unify(t1.param, t2.param, subst)
        return unify(t1.result, t2.result, subst2)
    else:
        raise TypeError(f"Cannot unify {t1} with {t2}")

def fresh_type_var(prefix="a", counter=[0]) -> TVar:
    name = f"{prefix}{counter[0]}"
    counter[0] += 1
    return TVar(name)

def _infer(expr: UOp, env: Dict[str, Type], subst: Dict[TVar, Type]) -> Tuple[Type, Dict[TVar, Type]]:
    assert isinstance(expr, UOp), f"Not an expr: {expr}"
    def var_ty(var):
        varname = var.__name__
        if var is int:
            return TInt()
        if var is bool:
            return TBool()
        elif var is list or var is Callable:
            raise TypeError(f"Type {varname} is missing type spec")
        elif typing.get_origin(var) is list:
            elems = typing.get_args(var)[0]
            return TList(var_ty(elems))
        elif typing.get_origin(var) is Callable:
            params_ty, body_ty = typing.get_args(var)
            params_ty = [var_ty(p) for p in params_ty]
            return TArrow.of_list(params_ty, var_ty(body_ty))
        else:
            raise TypeError(f"Type {varname} isn't supported (yet)")
    def lookup(varname):
        if varname in env:
            # The type in env might be partially substituted
            return (apply_subst(env[varname], subst), subst)
        elif varname in BUILTINS:
            # Built-in functions are always external and therefore
            # don't require substitution
            return _infer(BUILTINS[varname], env, subst)
        else:
            raise NameError(f"Unbound variable: {varname}")
    match expr.op:
        case Ops.Val:
            py_ty = type(expr.args[0])
            # Special case for list's, so that we can infer the type of
            # A value such as: Val([1, 2, 3]) => Int List
            if py_ty is list and len(expr.args[0]) > 0:
                elem = expr.args[0][0]
                return (var_ty(list[type(elem)]), subst)
            return (var_ty(py_ty), subst)
        case Ops.Var:
            return lookup(expr.args[0])
        case Ops.Closure:
            *_, body, _ = expr.args
            return _infer(body, env, subst)
        case Ops.Abstr:
            *params, body = expr.args
            # We create a fresh type variable for each param
            # or you might do something fancy if param type is annotated
            env = env.copy()
            params_ty = []
            for p in params:
                tv = fresh_type_var()
                env[p] = tv
                params_ty.append(tv)
            # Infer body with those param types in env
            (body_ty, s1) = _infer(body, env, subst)
            ty = TArrow.of_list(params_ty, body_ty)
            return (ty, s1)
        case Ops.Appl:
            func, *args = expr.args
            # 1. Infer the type of func
            (func_ty, s1) = _infer(func, env, subst)
            # 2. For each argument, unify the function type with TArrow(...)
            #    and produce the resulting function type for next arg
            cur_subst = s1
            cur_func_ty = func_ty
            for arg in args:
                # The function must be an arrow: param_type -> result_type
                param_ty = fresh_type_var()
                result_ty = fresh_type_var()
                cur_subst = unify(cur_func_ty, TArrow(param_ty, result_ty), cur_subst)
                # Infer arg type
                (arg_ty, cur_subst) = _infer(arg, env, cur_subst)
                # Unify arg type with param
                cur_subst = unify(arg_ty, param_ty, cur_subst)
                cur_func_ty = result_ty
            # After applying all arguments, cur_func_ty is the final type of the application
            return (apply_subst(cur_func_ty, cur_subst), cur_subst)
        case Ops.External:
            *params, body = expr.args
            params_ty = body.__annotations__
            assert len(params_ty) == len(params) + 1, \
                   "All external functions must fully typed, including their return type"
            body_ty = var_ty(params_ty["return"])
            params_ty = [var_ty(params_ty[p]) for p in params]
            ty = TArrow.of_list(params_ty, body_ty)
            return (ty, subst)

def infer(expr: UOp) -> Type:
    ty, s = _infer(expr, {}, {})
    return apply_subst(ty, s)

ATOMS = ("INT", "BOOL", "IDENT", "LPAREN", "LBRACKET", "LAMBDA")
TOKEN_REGEX = r"""
(?P<LAMBDA>lambda|L)            # 'lambda' or 'L'
|(?P<INT>\d+)                   # integer literal
|(?P<BOOL>true|false)           # bool literal
|(?P<IDENT>[a-zA-Z_]\w*)        # identifier (variable names, etc.)
|(?P<LPAREN>\()                 # (
|(?P<RPAREN>\))                 # )
|(?P<LBRACKET>\[)               # [
|(?P<RBRACKET>\])               # ]
|(?P<COMMA>,)                   # ,
|(?P<DOT>\.)                    # .
|(?P<WHITESPACE>\s+)            # whitespace (ignored)
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

    def parse_list(self) -> UOp:
        elems = []
        while True:
            if self.peek_kind("RBRACKET"):
                break
            elems.append(self.parse_atom())
            if self.peek_kind("COMMA"):
                self.next()
        val = [v for v in elems]
        return UOp(Ops.Val, [val])

    def parse_atom(self) -> UOp:
        if not (t := self.peek()):
            raise SyntaxError("Unexpected EOF") 
        match t.kind:
            case "INT" | "BOOL":
                self.next()
                return UOp(Ops.Val, [eval(t.val)])
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
