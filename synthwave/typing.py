from collections.abc import Callable
from typing import Any, Dict, Tuple, Optional
from enum import auto, IntEnum
import typing
from .helpers import fn_parameters
from .eval import BUILTINS
from .dsl import UOp, Ops


class T(IntEnum):
    Var = auto()
    Int = auto()
    Float = auto()
    Char = auto()
    Bool = auto()
    List = auto()
    Arrow = auto()

PRIMITIVES = (T.Int, T.Bool, T.List)

class Type():
    t: T
    params: list[Any] = [] # str or type

    def __init__(self, t: T, params: Optional[list[Any]] = None):
        self.t = t
        if params is not None:
            self.params = params

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
            case T.Var:
                return str(self.params[0])

def apply_subst(ty: Type, subst: Dict[Type, Type]) -> Type:
    if ty.t == T.Var and ty in subst:
        return apply_subst(subst[ty], subst)
    elif ty.t == T.Arrow:
        return Type(T.Arrow, [apply_subst(t, subst) for t in ty.params])
    else:
        return ty

def occurs_in_type(tv: Type, ty: Type, subst: Dict[Type, Type]) -> bool:
    ty = apply_subst(ty, subst)
    if ty == tv:
        return True
    elif ty.t == T.Arrow:
        return any(occurs_in_type(tv, t, subst) for t in ty.params)
    else:
        return False

def var_bind(tv: Type, ty: Type, subst: Dict[Type, Type]) -> Dict[Type, Type]:
    """Bind the type var 'tv' to type 'ty' if allowed,"""
    if tv == ty:
        return subst
    # Cannot bind a type variable to a type that contains it
    if occurs_in_type(tv, ty, subst):
        raise TypeError(f"Occurs check fails: {tv} in {ty}")
    # Override the subst
    subst = subst.copy()
    subst[tv] = ty
    return subst

def unify(t1: Type, t2: Type, subst: Dict[Type, Type]) -> Dict[Type, Type]:
    """Unify t1 and t2 under the existing substitution 'subst',"""
    # 1) Apply current subst so we see the "true" forms
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)
    # 2) Pattern-match on t1, t2
    if t1.t == T.Var:
        return var_bind(t1, t2, subst)
    elif t2.t == T.Var:
        return var_bind(t2, t1, subst)
    # This must be manually modified for each new base type
    elif any(t1.t == t and t2.t == t for t in PRIMITIVES):
        return subst
    elif t1.t == T.Arrow and t2.t == T.Arrow:
        # assert len(t1.params) == len(t2.params)
        for t1, t2 in zip(t1.params, t2.params):
            subst = unify(t1, t2, subst)
        return subst
    else:
        raise TypeError(f"Cannot unify {t1} with {t2}")

def fresh_type_var(prefix="a", counter=[0]) -> Type:
    name = f"{prefix}{counter[0]}"
    counter[0] += 1
    return Type(T.Var, [name])

def infer_py_ty(ty, expr=None) -> Type:
    varname = ty.__name__
    if ty is int:
        return Type(T.Int)
    if ty is bool:
        return Type(T.Bool)
    if ty is float:
        return Type(T.Float)
    if ty is str:
        return Type(T.Char)
    elif ty is list and expr is not None and len(expr) > 0:
        # Special case for list's, so that we can infer the type of
        # A value such as: Val([1, 2, 3]) => Int List
        return infer_py_ty(list[type(expr[0])])
    elif ty is list or ty is Callable:
        return Type(T.List, [fresh_type_var()])
    elif typing.get_origin(ty) is list:
        elems = typing.get_args(ty)[0]
        return Type(T.List, [infer_py_ty(elems)])
    elif typing.get_origin(ty) is Callable:
        params_ty, body_ty = typing.get_args(ty)
        params_ty = [infer_py_ty(p) for p in params_ty]
        return Type(T.Arrow, [*params_ty, infer_py_ty(body_ty)])
    else:
        raise TypeError(f"Type '{varname}' isn't supported (yet)")

def _infer(expr: UOp, env: Dict[str, Type], subst: Dict[Type, Type]) -> Tuple[Type, Dict[Type, Type]]:
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
            ty = type(expr.args[0])
            return (infer_py_ty(ty, expr.args[0]), subst)
        case Ops.Var:
            return lookup(expr.args[0])
        case Ops.Closure:
            *args, body, _ = expr.args
            if isinstance(body, UOp):
                body = UOp(Ops.Abstr, [*args, body])
                return _infer(body, env, subst)
            else:
                body = UOp(Ops.External, [*args, body])
                return _infer(body, env, subst)
        case Ops.Abstr:
            *params, body = expr.args
            # We create a fresh type variable for each param
            # or you might do something fancy if param type is annotated
            env = env.copy()
            param_tys = []
            for p in params:
                tv = fresh_type_var()
                env[p] = tv
                param_tys.append(tv)
            # Infer body with those param types in env
            (body_ty, subst) = _infer(body, env, subst)
            ty = Type(T.Arrow, [*param_tys, body_ty])
            return (ty, subst)
        case Ops.Appl:
            func, *args = expr.args
            # 1. Infer the type of func
            (func_ty, subst) = _infer(func, env, subst)
            for idx, arg in enumerate(args):
                # Infer its type and unify it with the corresponding parameter type.
                (arg_ty, subst) = _infer(arg, env, subst)
                param_ty = apply_subst(func_ty.params[idx], subst)
                subst = unify(arg_ty, param_ty, subst)
            # Determine the type of the application.
            if len(args) == len(func_ty.params) - 1:
                # Fully applied, so last type is the return type
                result_ty = apply_subst(func_ty.params[-1], subst)
            else:
                # Partial application: function is missing for the remaining args.
                remaining_tys = func_ty.params[len(args):]
                result_ty = Type(T.Arrow, remaining_tys)
            return (result_ty, subst)
        case Ops.External:
            *params, body = expr.args
            param_tys = body.__annotations__
            assert "return" in param_tys, \
                  f"Function '{body.__name__}' missing return type annotation"
            assert len(fn_parameters(body)) + 1 == len(param_tys), \
                  f"Function '{body.__name__}' missing parameter annotations"
            body_ty = infer_py_ty(param_tys["return"])
            param_tys = [infer_py_ty(param_tys[p]) for p in params]
            ty = Type(T.Arrow, [*param_tys, body_ty])
            return (ty, subst)

def infer(expr) -> Type:
    if isinstance(expr, UOp):
        ty, s = _infer(expr, {}, {})
        return apply_subst(ty, s)
    else:
        return infer_py_ty(type(expr), expr)
