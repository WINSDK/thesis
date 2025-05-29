from typing import Dict, Tuple
from .eval import BUILTIN_SCHEMES
from .dsl import UOp, Ops, T, Type, Scheme, TVar, fresh_type_var

PRIMITIVES = [t.value for t in T if t.value not in [T.List, T.Arrow]]

Subst = Dict[TVar, Type]
Env = Dict[str, Scheme]

def apply_subst(ty: Type, subst: Subst) -> Type:
    if isinstance(ty, TVar):
        if ty in subst:
            return apply_subst(subst[ty], subst)
        return ty
    return Type(ty.t, [apply_subst(t, subst) for t in ty.params])

def occurs_in_type(tv: TVar, ty: Type, subst: Subst) -> bool:
    ty = apply_subst(ty, subst)
    if ty == tv:
        return True
    if isinstance(ty, TVar):
        return False
    elif len(ty.params) > 0:
        return any(occurs_in_type(tv, t, subst) for t in ty.params)
    else:
        return False

def var_bind(tv: TVar, ty: Type, subst: Subst) -> Subst:
    """Bind the type variable 'tv' to type 'ty' if allowed."""
    if tv == ty:
        return subst
    # Cannot bind a type variable to a type that contains it.
    if occurs_in_type(tv, ty, subst):
        raise TypeError(f"Occurs check fails: {tv} in {ty}")
    new_subst = subst.copy()
    new_subst[tv] = ty
    return new_subst

def unify(t1: Type, t2: Type, subst: Subst) -> Subst:
    """Unify t1 and t2 under the existing substitution."""
    # 1) Apply current substitution so we see the "true" forms.
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)
    # 2) Pattern-match on t1 and t2.
    if isinstance(t1, TVar):
        return var_bind(t1, t2, subst)
    elif isinstance(t2, TVar):
        return var_bind(t2, t1, subst)
    elif any(t1.t == t and t2.t == t for t in PRIMITIVES):
        return subst
    elif t1.t == t2.t:
        for t1_arg, t2_arg in zip(t1.params, t2.params):
            subst = unify(t1_arg, t2_arg, subst)
        return subst
    else:
        raise TypeError(f"Cannot unify {t1} with {t2}")

def generalize(env: Dict[str, Scheme], ty: Type) -> Scheme:
    env_ft = set()
    for scheme in env.values():
        env_ft |= scheme.free_vars()
    ty_ft = ty.free_vars()
    quantified = list(ty_ft - env_ft)
    return Scheme(quantified, ty)

def instantiate(scheme: Scheme) -> Type:
    mapping = {}
    for var in scheme.vars:
        mapping[var] = fresh_type_var(var.name, generic=var.generic)
    return apply_subst(scheme.ty, mapping)

def infer_py_ty(expr, env: Env, subst: Subst) -> Tuple[Type, Subst]:
    if isinstance(expr, int):
        py_ty = Type(T.Int)
    elif isinstance(expr, bool):
        py_ty = Type(T.Bool)
    elif isinstance(expr, float):
        py_ty = Type(T.Float)
    elif isinstance(expr, str):
        py_ty = Type(T.Char)
    elif isinstance(expr, list) and len(expr) > 0:
        # Special case for list of elements.
        # We infer the type of the list through the first item.
        if isinstance(expr[0], UOp):
            elem_ty, subst = _infer(expr[0], env, subst)
        else:
            elem_ty, subst = infer_py_ty(expr[0], env, subst)
        py_ty = Type(T.List, [elem_ty])
    elif isinstance(expr, list):
        py_ty = Type(T.List, [fresh_type_var()])
    else:
        raise TypeError(f"Type {type(expr).__name__} isn't supported (yet)")
    return py_ty, subst

def lookup(varname: str, env: Env, subst: Subst) -> Tuple[Type, Subst]:
    if varname in env:
        # Instantiate the scheme to get a fresh copy of the type.
        ty = instantiate(env[varname])
        return (ty, subst)
    else:
        raise NameError(f"Unbound variable {varname}")

def _infer(expr, env: Env, subst: Subst) -> Tuple[Type, Subst]:
    if not isinstance(expr, UOp):
        return infer_py_ty(expr, env, subst)
    match expr.op:
        case Ops.Val:
            return _infer(expr.args[0], env, subst)
        case Ops.Var:
            return lookup(expr.args[0], env, subst)
        case Ops.Closure:
            *args, body, closure = expr.args
            # Merge captured environment into the type-inference environment.
            env = env.copy()
            for k, v in closure.items():
                # Infer the type of the captured value, then generalize it.
                v_ty, subst = _infer(UOp(Ops.Val, [v]), env, subst)
                env[k] = generalize(env, v_ty)
            if isinstance(body, UOp):
                body = UOp(Ops.Abstr, [*args, body])
                return _infer(body, env, subst)
            else:
                body = UOp(Ops.External, [*args, body])
                return _infer(body, env, subst)
        case Ops.Abstr:
            *args, body = expr.args
            new_env = env.copy()
            args_ty = []
            for a in args:
                tv = fresh_type_var()
                # When a arg is introduced, it is monomorphic until it is generalized.
                new_env[a] = Scheme([], tv)
                args_ty.append(tv)
            body_ty, subst = _infer(body, new_env, subst)
            ty = Type.arrow(args_ty, body_ty)
            return (ty, subst)
        case Ops.External:
            body = expr.args[-1]
            # I don't like requiring builtin's to be defined python fn's
            fn_name = body.__name__.removeprefix("builtin_")
            return lookup(fn_name, env, subst)
        case Ops.Appl:
            func, *args = expr.args
            func_ty, subst = _infer(func, env, subst)
            # Special case for Church boolean application
            func_ty = apply_subst(func_ty, subst)
            if not isinstance(func_ty, TVar) and func_ty.t == T.Bool:
                v = fresh_type_var()
                func_ty = Type.arrow([v, v], v)
            for a in args:
                # For each argument, the function type must be an arrow.
                param_ty = fresh_type_var()
                result_ty = fresh_type_var()
                subst = unify(func_ty, Type(T.Arrow, [param_ty, result_ty]), subst)
                arg_ty, subst = _infer(a, env, subst)
                subst = unify(arg_ty, param_ty, subst)
                func_ty = result_ty
            return (func_ty, subst)

def infer(expr) -> Type:
    ty, subst = _infer(expr, BUILTIN_SCHEMES, {})
    return apply_subst(ty, subst)
