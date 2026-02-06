"""
optimiser
=========

Defines the Optimiser class.
"""

import logging
from time import perf_counter as timer
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from aso.logging import format_array_for_logging

from aso.optimisation_problem import OptimisationProblem
from aso.optimisation_result import OptimisationResult

logger = logging.getLogger(__name__)


class Optimiser:
    """
    Contains various optimisation algorithms to solve an `OptimisationProblem`.

    Attributes
    ----------
    problem : OptimisationProblem
        The optimisation problem to be solved.
    x : numpy.ndarray
        Current design variable values.
    n : int
        Number of design variables.
    lm : numpy.ndarray
        Current Lagrange multipliers.
    """

    def __init__(
        self,
        problem: OptimisationProblem,
        x: NDArray,
        lm: NDArray | None = None,
    ) -> None:
        """Initialize an `Optimiser` instance.

        Parameters
        ----------
        problem : OptimisationProblem
            Optimisation problem to solve.
        x : numpy.ndarray
            Initial design variables.
        lm : numpy.ndarray, optional
            Initial Lagrange multipliers.

        Notes
        -----
        The given array of design variables will be modified in place.
        Hence, the optimiser does currently not reuturn the optimised
        design variables but only the number of outer-loop iterations.
        This behavior may change in future versions.
        """
        self.problem = problem
        self.x = x
        self.n = x.size

        # Check and, if necessary, initialise the Lagrange multipliers:
        if lm is None:
            self.lm = np.zeros(problem.m + problem.me)
        elif lm.size != problem.m + problem.me:
            raise ValueError(
                "The number of Lagrange multipliers must match the number of constraints."
            )
        else:
            self.lm = lm

    def optimise(
        self,
        algorithm: Literal[
            "SQP",
            "MMA",
            "STEEPEST_DESCENT",
            "CONJUGATE_GRADIENTS",
        ] = "SQP",
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        Distinguish constrained and unconstrained optimization problems
        and call an appropriate optimisation function.

        Parameters
        ----------
        algorithm : str, default: "SQP"
            Algorithm to use.
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect (intermediate) optimization results.

        Returns
        -------
        iteration : int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If `algorithm` is unknown or not suitable for constrained
            optimisation.
        """

        start = timer()

        if self.problem.constrained:
            match algorithm:
                case "SQP":
                    iteration = self.sqp_constrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case "MMA":
                    iteration = self.mma()
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for constrained optimisation."
                    )
        else:
            match algorithm:
                case "STEEPEST_DESCENT":
                    iteration = self.steepest_descent(
                        iteration_limit=iteration_limit,
                    )
                case "CONJUGATE_GRADIENTS":
                    iteration = self.conjugate_gradients(
                        iteration_limit=iteration_limit,
                    )
                case "SQP":
                    iteration = self.sqp_unconstrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for unconstrained optimisation."
                    )

        end = timer()
        elapsed_ms = round((end - start) * 1000, 3)

        if iteration == -1:
            logger.info(
                f"Algorithm {algorithm} failed to converge in {elapsed_ms} ms after {iteration} "
                f"iterations. Consider using another algorithm or increasing the iteration limit.",
            )
        else:
            logger.info(
                f"Algorithm {algorithm} converged in {elapsed_ms} ms after {iteration} "
                f"iterations. Optimised design variables: {format_array_for_logging(self.x)}",
            )

        return iteration

    def steepest_descent(
        self,
        iteration_limit: int = 1000,
    ) -> int:
        """Steepest-descent algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer loop iterations.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def conjugate_gradients(
        self,
        iteration_limit: int = 1000,
        beta_formula: Literal[
            "FLETCHER-REEVES",
            "POLAK-RIBIERE",
            "HESTENES-STIEFEL",
            "DAI-YUAN",
        ] = "FLETCHER-REEVES",
    ) -> int:
        """Conjugate-gradient algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        beta_formula : str, : optional
            Heuristic formula for computing the conjugation factor beta.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_unconstrained(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """SQP algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : str, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_constrained(
        self,
        iteration_limit: int = 1000,
        working_set: list[int] | None = None,
        working_set_size: int | None = None,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        SQP algorithm with an active-set strategy for constrained
        optimisation.

        Parameters `m_w` and `working_set` are currently ignored.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        working_set : list of int, optional
            Initial working set.
        working_set_size : int, optional
            Size of the working set (ignored if `working_set` is provided).
        callback : callable, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If the size of the working set is too large or too small.

        References
        ----------
        .. [1] K. Schittkowski, "An Active Set Strategy for Solving Optimization Problems with up to 200,000,000 Nonlinear Constraints." Accessed: May 25, 2025. [Online]. Available: https://klaus-schittkowski.de/SC_NLPQLB.pdf
        """
        tolerance = 0.01

        n = self.n
        m = self.problem.m
        me = self.problem.me

        i = 0
        f = self.problem.compute_objective(self.x)
        df = self.problem.compute_grad_objective(self.x)

        g = self.problem.compute_constraints(self.x)
        active = np.nonzero(g[:m] >= 0)[0].tolist() + list(range(m, m + me))
        print('active of inequality', np.nonzero(g[:m] >= 0), 'equality constraints', list(range(m, m + me)))
        dg = self.problem.compute_grad_constraints(self.x)
        # print("d constraints", dg.shape)
        # print("constraints", g, g.shape, "obj", f.shape, "g", df.shape)

        # print("initial lambda", self.lm)
        print("n", n, "m", m)

        # self.lm = np.random.normal(size=m)
        
        V = np.identity(n)

        # print('INITIAL X', self.x, self.problem.lb, self.problem.ub)

        i = 0
        while True:
            i += 1
            print('active', active, 'dg', dg, 'masked', dg[active])
            A = dg[active]

            a_dim = A.shape[0]

            print("SHAPES", V.shape, "A", A.shape, "T", A.T.shape)
            print("Actual", V)
            print('A', A)
            lhs = np.block([[V, A.T], [A, np.zeros((a_dim, a_dim))]])
            print("Final block", lhs)
            print("DFG", df, g)
            rhs = np.concatenate((-df, -g[active]))
            # print("LHS", lhs)
            # print("RHS", rhs)

            pl = np.linalg.solve(lhs, rhs)
            # print("PL", pl)

            p_dir = pl[:n]
            self.lm[active] = pl[n:]

            for li, lm in enumerate(self.lm):
                print('L', li, lm)
                if lm < 0:
                    self.lm[li] = 0

            # print('after', self.lm)
            p_dir = self.mask_gradients(p_dir)

            line_alpha = self.line_search(p_dir)
            print("line result", line_alpha)

            x_old = np.copy(self.x)
            self.x += line_alpha * p_dir

            p = self.x - x_old

            f = self.problem.compute_objective(self.x)
            df = self.problem.compute_grad_objective(self.x)

            g = self.problem.compute_constraints(self.x)
            active = np.nonzero(g[:m] >= 0)[0].tolist() + list(range(m, m + me))
            dg = self.problem.compute_grad_constraints(self.x)

            dl = self.problem.compute_grad_lagrange_function(self.x, self.lm)

            print('tKKT', dl, g)

            #if (np.all(np.abs(dl) < tolerance) and np.all(g <= 0)):
            if self.converged(dl, g):
                print('converged', self.x)
                # print('ub', self.problem.ub, 'lb', self.problem.lb)
                return i

            y = self.problem.compute_grad_lagrange_function(self.x, self.lm) - self.problem.compute_grad_lagrange_function(x_old, self.lm)

            I = np.identity(n)
            # print("shapes: p", p.shape, "y", y.shape, "I", I)
            # print("Y", y, "P", p)
            # print("DOT", np.dot(y, p))
            # V = (I - np.outer(p, y) / np.dot(y, p)) @ V @ (I - np.outer(y, p) / np.dot(y, p)) + np.outer(p, p) / np.dot(y, p)
            denom = y @ p
            if denom <= 1e-8:
                # skip update or reset V
                V = np.identity(n) # V = np.eye(n)
                print("resetting V", i)
            else:
                rho = 1.0 / denom
                V = (I - rho * np.outer(y, p)) @ V @ (I - rho * np.outer(p, y)) + rho * np.outer(y, y)
            # print("NEW V", V)
            # break

            print("i", i, self.x)

            if i > 1000:
                # return 1000
                # raise Exception('Did not converge')
                return -1
            
    def mask_gradients(self, direction):
        if self.problem.lb is None and self.problem.ub is None:
            return direction
        else:
            if np.any(self.x <= self.problem.lb):
                direction[(self.x <= self.problem.lb) & (direction < 0)] = 0
            if np.any(self.x >= self.problem.ub):
                direction[(self.x >= self.problem.ub) & (direction > 0)] = 0

            self.x[:] = np.minimum(np.maximum(self.x, self.problem.lb), self.problem.ub)

            return direction

    def mma(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        MMA algorithm for constrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect intermediate results.
        """
        
        tol = 0.001
        SF = 10

        m, me, n = self.problem.m, self.problem.me, self.n
        print("initials params", m, me, n, "x", self.x)

        pz =  np.zeros((1, n))
        qz =  np.zeros((1, n))

        pm =  np.zeros((m, n))
        qm =  np.zeros((m, n))
        pme = np.zeros((me, n))

        # dL__dx = self.problem.compute_grad_lagrange_function(self.x - 1, self.lm)
        # print("starting dL__dx", dL__dx)

        for outer in range(iteration_limit):
            print("\n<<<OUTER>>>", outer)
            # old_dL__dx = dL__dx
            dL__dx = np.array(self.problem.compute_grad_lagrange_function(self.x, self.lm))
            # uses finite difference, not derivative
            # ddL__ddx = np.array(dL__dx - old_dL__dx)
            # print("old ddL_ddx", np.array(dL__dx - old_dL__dx))
            denom = (self.x - (old_x if outer != 0 else 0))
            denom = np.where(denom == 0, 1, denom)

            ddL__ddx = np.array(dL__dx - self.problem.compute_grad_lagrange_function(old_x, self.lm)) / denom if outer != 0 else np.ones(n)
            
            # enforce convexity
            # ddL__ddx = np.maximum(ddL__ddx, 1e-2)

            old_x = np.copy(self.x)

            print("dL__dx", dL__dx.shape, dL__dx)
            print("ddL__ddx", ddL__ddx.shape, ddL__ddx)

            lm, mu = self.lm[:m], self.lm[m:]

            splm = np.dot(lm, pm)
            spmu = np.dot(mu, pme)
            sqlm = np.dot(lm, qm)
            print("splm", splm.shape, splm)
            print("spmu", spmu.shape, spmu)
            print("sqlm", sqlm.shape, sqlm)

            lm1 = [np.roots([1 / 4 * ddL__ddx[i], + 1 / 2 * dL__dx[i], 0, - splm[i] - spmu[i]]) for i in range(n)]
            lm2 = [np.roots([1 / 4 * ddL__ddx[i], - 1 / 2 * dL__dx[i], 0, - sqlm[i] + spmu[i]]) for i in range(n)]
            print("lm1", lm1)
            print("lm2", lm2)

            # ignore imaginary roots and set minimum value to 1e-2
            lm1_max = np.array([max((x.real for x in a if np.isclose(x.imag, 0) and x.real > 1e-5), default=0) for a in lm1])
            lm2_max = np.array([max((x.real for x in a if np.isclose(x.imag, 0) and x.real > 1e-5), default=0) for a in lm2])
            print("lm1_max", lm1_max, "lm2_max", lm2_max)

            print("true delta", SF * np.max((lm1_max, lm2_max), axis=0))
            delta = np.clip(SF * np.max((lm1_max, lm2_max), axis=0), 1e-2, 1)
            print("delta", delta.shape, delta)

            LB = self.x - delta
            UB = self.x + delta

            # objective function
            pz = 1 / 4 * delta ** 3 * ddL__ddx + 1 / 2 * delta ** 2 * dL__dx - splm - spmu
            qz = 1 / 4 * delta ** 3 * ddL__ddx - 1 / 2 * delta ** 2 * dL__dx - sqlm + spmu
            print("pz", pz.shape, pz)
            print("qz", qz.shape, qz)

            pz = np.maximum(1e-5, pz)
            qz = np.maximum(1e-5, qz)

            # inequality constraints g
            c = self.problem.compute_constraints(self.x)
            print("c", c.shape, c)
            g, h = c[:m], c[m:]

            dc = self.problem.compute_grad_constraints(self.x)
            print("dc", dc.shape, dc)
            dg, dh = dc[:m], dc[m:]

            pm = np.multiply(np.where(dg >  0, dg, 0), + (UB - self.x) ** 2)
            qm = np.multiply(np.where(dg <= 0, dg, 0), - (self.x - LB) ** 2)
            print("pm", pm.shape, pm)
            print("qm", qm.shape, qm)

            rm = g - np.sum(pm / (UB - self.x) + qm / (self.x - LB), axis=-1)
            print("rm", rm.shape, rm)

            # equality constraints h
            pme = np.multiply(dh, 1 / 2 * (UB - self.x) ** 2)
            print("pme", pme.shape, pme)

            rme = h
            print("rme", h.shape, h)

            # check KKT of original function
            # primal feasibility
            if np.all(g < tol) and np.all(np.abs(h) < tol):
                print("original constraints satisfied", g, h)
                # complementary slackness
                cs = self.lm[:m] * g
                if np.all(np.abs(cs) < tol):
                    print("ORIGINAL COMPLEMENTARY SATISFIED", cs, dL__dx)

                    grad_f = self.problem.compute_grad_objective(self.x)
                    J = self.problem.compute_grad_constraints(self.x)
                    active = np.concatenate([g >= -tol, np.ones(me, dtype=bool)])
                    A = J[active]
                    v = np.linalg.lstsq(A.T, -grad_f, rcond=None)[0]
                    res = grad_f + A.T @ v

                    if np.linalg.norm(res, np.inf) < tol:
                        print("ORIGINAL GRADIENT ZERO", dL__dx)

                        print("FOUND", self.x, "LM", self.lm, "AT ITERATION", outer)

                        return outer
            else:
                print("NOT SATISFIED", g, h, dL)

            # minimise in x
            S = np.concatenate((g, h))
            print("S", S.shape, S)

            for inner in range(1000):
                print("\n---OUTER---", outer, "-INNER-", inner)

                lm, mu = self.lm[:m], self.lm[m:]

                splm = np.dot(lm, pm)
                spmu = np.dot(mu, pme)
                sqlm = np.dot(lm, qm)
                print("splm revised", splm.shape, splm)
                print("spmu revised", spmu.shape, spmu)
                print("sqlm revised", sqlm.shape, sqlm)

                P = pz + splm + spmu
                Q = qz + sqlm - spmu

                print("P", P.shape, P)
                print("Q", Q.shape, Q)

                if np.any(P < 0) or np.any(Q < 0):
                    P = np.maximum(P, 1e-2)
                    Q = np.maximum(Q, 1e-2)
                    # raise Exception(f"Zero sqrt P:{P}, Q:{Q}")

                # keep shared reference with test
                self.x[:] = (UB * np.sqrt(Q) + LB * np.sqrt(P)) / (np.sqrt(Q) + np.sqrt(P))
                print("new x", self.x)

                # maximise in lm

                # compute using approximation
                g = rm  + np.sum(pm  / (UB - self.x) + qm  / (self.x - LB), axis=-1)
                h = rme + np.sum(pme / (UB - self.x) - pme / (self.x - LB), axis=-1)

                dg = pm  / (UB - self.x) ** 2 - qm  / (self.x - LB) ** 2
                dh = pme / (UB - self.x) ** 2 + pme / (self.x - LB) ** 2

                dL_dlm = np.array(g)
                dL_dmu = np.array(h)

                # dim n
                dx__dP = - 1 / 2 * (np.sqrt(Q) * (UB - LB)) / (np.sqrt(P) * (np.sqrt(Q) + np.sqrt(P)) ** 2)
                # sign error in slides
                dx__dQ = + 1 / 2 * (np.sqrt(P) * (UB - LB)) / (np.sqrt(Q) * (np.sqrt(Q) + np.sqrt(P)) ** 2)
                print("dx__dP", dx__dP.shape, dx__dP)

                # p dim k, n
                # dim n, k
                dx__dlmk = np.multiply(qm,  dx__dQ).T + np.multiply(pm,  dx__dP).T
                dx__dmuk = np.multiply(pme, dx__dP- dx__dQ).T
                print("dx__dlmk", dx__dlmk.shape, dx__dlmk)
                print("dx__dmuk", dx__dmuk.shape, dx__dmuk)

                # only g first
                # dg -> dim j, n
                # dim j, k
                ddL__dlmj_dlmk = np.matmul(dg, dx__dlmk)
                ddL__dmuj_dmuk = np.matmul(dh, dx__dmuk)
                print("ddL__dlmj_dlmk", ddL__dlmj_dlmk.shape, ddL__dlmj_dlmk)
                print("ddL__dmuj_dmuk", ddL__dmuj_dmuk.shape, ddL__dmuj_dmuk)

                # cross terms
                ddL__dlmj_dmuk = np.matmul(dg, dx__dmuk)
                ddL__dmuj_dlmk = np.matmul(dh, dx__dlmk)
                print("ddL__dlmj_dmuk", ddL__dlmj_dmuk.shape, ddL__dlmj_dmuk)
                print("ddL__dmuj_dlmk", ddL__dmuj_dlmk.shape, ddL__dmuj_dlmk)

                dL = np.concatenate((dL_dlm, dL_dmu))
                print("dL", dL.shape, dL)
                # ddL = np.block([[ddL__dlmj_dlmk, np.zeros((m, me))], [np.zeros((me, m)), ddL__dmuj_dmuk]])
                ddL = np.block([[ddL__dlmj_dlmk, ddL__dlmj_dmuk], [ddL__dmuj_dlmk, ddL__dmuj_dmuk]])
                print("ddL", ddL.shape, ddL)

                
                # check KKT
                if np.all(g < tol) and np.all(np.abs(h) < tol):
                    print("CONSTRAINTS SATISFIED", g, h)
                    cs = self.lm[:m] * g
                    if np.all(np.abs(cs) < tol):
                        print("COMPLEMENTARY SATISFIED", cs)
                        if np.all(np.abs(dL) < tol):
                            print("GRADIENT ZERO", dL)

                            print("FOUND", self.x, "LM", self.lm)

                            break
                else:
                    print("NOT SATISFIED", g, h, dL)


                # perform step
                alpha_opt = - np.dot(dL, S) / (S @ ddL @ S)
                print("alpha_opt", alpha_opt.shape, alpha_opt)

                self.lm = self.lm + alpha_opt * S
                print("new lm", self.lm)
                
                # mask inequality constraints, could make S unstable, might want to return to grad descrent if reset lm
                self.lm[:m] = np.maximum(self.lm[:m], 0)

                # reset search direction if mask lm
                if np.any(self.lm[:m] == 0):
                    S = dL.copy()
                
                print("masked lm", self.lm)

                print("components", (S @ ddL @ dL), "SEP", (S @ ddL @ S))

                # might want to check or clamp beta values
                beta = - (S @ ddL @ dL) / (S @ ddL @ S)
                print("beta", beta.shape, beta)
                S = dL + beta * S
                print("S", S.shape, S)

        print("last x", self.x)
        return -1



    def converged(
        self,
        gradient: NDArray,
        constraints: NDArray | None = None,
        gradient_tol: float = 1e-5,
        constraint_tol: float = 1e-5,
        complementarity_tol: float = 1e-5,
    ) -> bool:
        """
        Check convergence according to the first-order necessary (KKT)
        conditions assuming LICQ.

        See, for example, Theorem 12.1 in [1]_.

        Parameters
        ----------
        gradient : numpy.ndarray
            Current gradient of the Lagrange function with respect to
            the design variables.
        constraints : numpy.ndarray, optional
            Current constraint values.
        gradient_tol : float, default: 1e-5
            Tolerance applied to each component of the gradient.
        constraint_tol : float, default: 1e-5
            Tolerance applied to each constraint.
        complementarity_tol : float, default: 1e-5
            Tolerance applied to each complementarity condition.

        References
        ----------
        .. [1] J. Nocedal and S. J. Wright, Numerical Optimization. Springer New York, 2006. doi: https://doi.org/10.1007/978-0-387-40065-5.
        """
        ...
        """
        Check a simplified version of the first-order KKT conditions:
        - stationarity of the Lagrangian
        - primal feasibility of constraints

        Inequalities are assumed as g(x) <= 0, equalities as h(x) = 0.
        """

        # --- Stationarity: ||∇_x L||_inf <= gradient_tol ---
        if np.linalg.norm(gradient, np.inf) > gradient_tol:
            return False

        # --- Primal feasibility ---
        if constraints is not None:
            m = self.problem.m   # inequalities
            me = self.problem.me # equalities

            if m > 0:
                g = constraints[:m]
                # allow small positive violation up to constraint_tol
                if np.any(g > constraint_tol):
                    return False

            if me > 0:
                h = constraints[m:m + me]
                if np.any(np.abs(h) > constraint_tol):
                    return False

        # We ignore complementarity and λ >= 0 in this project.
        return True

    def line_search(
            self,
            direction: NDArray,
            alpha_ini: float = 1,
            alpha_min: float = 1e-5,
            alpha_max: float = 1,
            algorithm: Literal[
                "WOLFE",
                "STRONG_WOLFE",
                "GOLDSTEIN-PRICE",
            ] = "STRONG_WOLFE",
            m1: float = 0.01,
            m2: float = 0.90,
            callback: Callable[[OptimisationResult], Any] | None = None,
            callback_iteration: int | None = None,
        ) -> float:
            """
            Perform a line search and returns an approximately optimal step size.

            Parameters
            ----------
            direction : numpy.ndarray
                Search direction.
            alpha_ini : float
                Initial step size.
            alpha_min : float, optional
                Minimum step size.
            alpha_max : float
                Maximum step size.
            algorithm : str, optional
                Line search algorithm to use.
            m1 : float, optional
                Parameter for the sufficient decrease condition.
            m2 : float, optional
                Parameter for the curvature condition.
            callback : callable, optional
                Callback function for collecting intermediate results.
            callback_iteration : int, optional
                Iteration number for the callback function.

            Returns
            -------
            float
                Approximately optimal step size.
            """

            # print("DIR", direction)
            alpha_l = 0
            alpha_u = float('inf')

            alpha = alpha_ini #  * 0.1

            # print('TEST D', self.problem.compute_lagrange_function(self.x, self.lm), self.problem.compute_grad_lagrange_function(self.x, self.lm), np.dot(self.problem.compute_grad_lagrange_function(self.x, self.lm), direction))

            phi_zero = self.problem.compute_lagrange_function(self.x, self.lm)
            phi_zero_p = np.dot(self.problem.compute_grad_lagrange_function(self.x, self.lm), direction)

            # if phi_zero_p >= 0:
                # print("ALARM", phi_zero_p)
                # direction = - self.problem.compute_grad_lagrange_function(self.x, self.lm)
                # phi_zero_p = np.dot(self.problem.compute_grad_lagrange_function(self.x, self.lm), direction)
                # print('new direction', direction, 'NPZP', phi_zero_p)
                # raise Exception('was not descent direction')
                # return -1
            for i in range(1000):
                new_x = self.x + alpha * direction

                phi = self.problem.compute_lagrange_function(new_x, self.lm)
                phi_p = np.dot(self.problem.compute_grad_lagrange_function(new_x, self.lm), direction)

                expected_descent = phi_zero + m1 * alpha * phi_zero_p

                if phi <= expected_descent and abs(phi_p) <= m2 * abs(phi_zero_p):
                    print('EXIT', alpha_min, alpha, alpha_max)
                    return min(alpha, alpha_max)
                elif phi > expected_descent:
                    # print('did not descent enough', alpha, phi_zero, phi, expected_descent, 'p', phi_p, 'z', phi_zero_p)
                    alpha_u = alpha
                    alpha = (alpha_l + alpha_u) / 2
                else:
                    if phi_p < 0:
                        # print('left of minimum', alpha, phi, expected_descent, 'p', phi_p)
                        if alpha_u == float('inf'):
                            alpha += alpha_ini
                        else:
                            alpha_l = alpha
                            alpha = (alpha_l + alpha_u) / 2
                    else:
                        # print('right of minimum', alpha, phi, expected_descent, 'p', phi_p, 'b', alpha_l, alpha_u)
                        alpha_u = alpha
                        alpha = (alpha_l + alpha_u) / 2
                        
                if alpha < alpha_min:
                    print('gave up after alpha too small')
                    return alpha_min

            print('failed', alpha_l, alpha, alpha_u)
            return alpha
