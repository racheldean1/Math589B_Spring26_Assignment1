from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

ValueGrad = Callable[[np.ndarray], Tuple[float, np.ndarray]]

@dataclass
class BFGSResult:
    x: np.ndarray
    f: float
    g: np.ndarray
    n_iter: int
    n_feval: int
    converged: bool
    history: Dict[str, Any]

def backtracking_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_steps: int = 30,
) -> Tuple[float, float, np.ndarray, int]:
        """
    Simple Armijo backtracking line search.
    Returns (alpha, f_new, g_new, n_feval_increment).
    """
    # TODO (students): implement Armijo condition and backtracking.
    # Armijo: f(x + a p) <= f(x) + c1 a g^T p
    
    alpha = float(alpha0)
    gTp = float(g @ p)

    n_feval_inc = 0

    # fallback: if p is not descent, use steepest descent
    if gTp >= 0.0:
        p = -g
        gTp = float(g @ p)

    # Keep the last evaluated (f_new, g_new) in case we never satisfy Armijo
    f_new, g_new = f, g

    for _ in range(max_steps):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        n_feval_inc += 1

        if f_new <= f + c1 * alpha * gTp:
            return alpha, float(f_new), g_new, n_feval_inc

        alpha *= tau

    # returning last try if loop is exited
    return alpha, float(f_new), g_new, n_feval_inc


def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Minimize f(x) with BFGS.

    You should:
    - maintain an approximation H_k to the inverse Hessian
    - compute p_k = -H_k g_k
    - perform a line search to get step alpha_k
    - update x, f, g
    - update H via the BFGS formula (with curvature checks)

    Return BFGSResult with a small iteration history useful for plotting.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    n_feval = 1

    n = x.size
    H = np.eye(n)  # inverse Hessian approximation

    hist = {"f": [f], "gnorm": [np.linalg.norm(g)], "alpha": []}

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                              converged=True, history=hist)

        # Search direction
        p = -H @ g

        # Line search
        # TODO (students): call your line search to get alpha, f_new, g_new.
        # alpha, f_new, g_new, inc = backtracking_line_search(...)
        
        gnorm = float(np.linalg.norm(g))
        alpha_start = float(min(alpha0, 1.0 / max(1.0, gnorm)))

        alpha, f_new, g_new, inc = backtracking_line_search(
            f_and_g=f_and_g,
            x=x,
            f=float(f),
            g=g,
            p=p,
            alpha0=alpha_start,
            c1=1e-4,
            tau=0.5,
            max_steps=12,
        )

        n_feval += int(inc)
        hist["alpha"].append(float(alpha))

        # Update step
        x_new = x + alpha * p
        s = x_new - x
        y = g_new - g

        # TODO (students): BFGS update for H with curvature check y^T s > 0.
        # Hint: Use the standard BFGS inverse-Hessian update.

        ys = float(np.dot(y, s))

        # Curvature / finite checks
        if (np.isfinite(ys) and ys > 1e-12
            and np.all(np.isfinite(s)) and np.all(np.isfinite(y))):

            rho = 1.0 / ys

            # Additional stability guard: avoid extremely large rho updates
            if rho > 1e12:
                H = np.eye(n)
            else:
                I = np.eye(n)
                V = I - rho * np.outer(s, y)
                H = V @ H @ V.T + rho * np.outer(s, s)

                # numerical symmetry cleanup
                H = 0.5 * (H + H.T)

                # optional: if H becomes non-finite, reset
                if not np.all(np.isfinite(H)):
                    H = np.eye(n)
        else:
            H = np.eye(n)


        # Accept iterate
        x = x_new
        f = float(f_new)
        g = g_new

        hist["f"].append(float(f))
        hist["gnorm"].append(float(np.linalg.norm(g)))

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval,
                      converged=False, history=hist)
