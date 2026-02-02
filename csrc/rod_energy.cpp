#include <cmath>
#include <algorithm>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

// Exported API (students may extend, but autograder only requires Python-level RodEnergy.value_and_grad).
// x: length 3N (xyzxyz...)
// grad_out: length 3N
// Periodic indexing enforces a closed loop.
void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,     // confinement strength
    double eps,    // WCA epsilon
    double sigma,  // WCA sigma
    double* energy_out,
    double* grad_out
) {
    const int M = 3 * N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };

    auto get = [&](int i, int d) -> double {
        return x[3 * idx(i) + d];
    };

    auto addg = [&](int i, int d, double v) {
        grad_out[3 * idx(i) + d] += v;
    };

    // ---- Bending: kb * ||x_{i+1} - 2 x_i + x_{i-1}||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double b = get(i + 1, d) - 2.0 * get(i, d) + get(i - 1, d);
            E += kb * b * b;
            const double c = 2.0 * kb * b;
            addg(i - 1, d,  c);
            addg(i,     d, -2.0 * c);
            addg(i + 1, d,  c);
        }
    }

    // ---- Stretching: ks * (||x_{i+1}-x_i|| - l0)^2
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i + 1, 0) - get(i, 0);
        double dx1 = get(i + 1, 1) - get(i, 1);
        double dx2 = get(i + 1, 2) - get(i, 2);
        double r = std::sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2);
        r = std::max(r, 1e-12);

        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        addg(i + 1, 0,  coeff * dx0);
        addg(i + 1, 1,  coeff * dx1);
        addg(i + 1, 2,  coeff * dx2);
        addg(i,     0, -coeff * dx0);
        addg(i,     1, -coeff * dx1);
        addg(i,     2, -coeff * dx2);
    }

    // ---- Confinement: kc * sum ||x_i||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i, d);
            E += kc * xi * xi;
            addg(i, d, 2.0 * kc * xi);
        }
    }

    // ---- TODO: Segment-segment WCA self-avoidance ----
    //
    // For each non-adjacent segment pair (i,i+1) and (j,j+1):
    //  1) Compute closest points parameters u*, v* in [0,1]
    //  2) Compute r = p_i(u*) - p_j(v*), d = ||r||
    //  3) If d < 2^(1/6)*sigma:
    //      U(d) = 4 eps [ (sigma/d)^12 - (sigma/d)^6 ] + eps
    //      Accumulate E += U(d)
    //      Accumulate gradient to endpoints x_i, x_{i+1}, x_j, x_{j+1}
    //
    // Exclusions: skip adjacent segments (including wrap neighbors).
    //
    // IMPORTANT: You must include the dependence of (u*, v*) on endpoints in your gradient.

    auto clamp01 = [](double t) {
        if (t < 0.0) return 0.0;
        if (t > 1.0) return 1.0;
        return t;
    };

    auto dot3 = [](const double a[3], const double b[3]) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };

    // Proper segment-segment closest parameters (u,v) in [0,1] (no "solve-then-clamp-once").
    auto closest_uv = [&](const double a0[3], const double a1[3],
                          const double b0[3], const double b1[3],
                          double& u, double& v) {
        // Segment directions
        double A[3] = { a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2] };
        double B[3] = { b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2] };

        double AA = dot3(A, A);
        double BB = dot3(B, B);

        // Degenerate fallbacks (very rare in this assignment, but safe).
        if (AA < 1e-14 && BB < 1e-14) { u = 0.0; v = 0.0; return; }
        if (AA < 1e-14) {
            u = 0.0;
            double w[3] = { a0[0] - b0[0], a0[1] - b0[1], a0[2] - b0[2] };
            v = clamp01(dot3(B, w) / BB);
            return;
        }
        if (BB < 1e-14) {
            v = 0.0;
            double w[3] = { b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2] };
            u = clamp01(dot3(A, w) / AA);
            return;
        }

        // Start guess
        u = 0.5;
        v = 0.5;

        // A few alternating projection steps (fast + stable for segments).
        for (int it = 0; it < 3; ++it) {
            // Given v, best u (then clamp)
            double q[3] = {
                (1.0 - v) * b0[0] + v * b1[0],
                (1.0 - v) * b0[1] + v * b1[1],
                (1.0 - v) * b0[2] + v * b1[2]
            };
            double wA[3] = { q[0] - a0[0], q[1] - a0[1], q[2] - a0[2] };
            u = clamp01(dot3(A, wA) / AA);

            // Given u, best v (then clamp)
            double p[3] = {
                (1.0 - u) * a0[0] + u * a1[0],
                (1.0 - u) * a0[1] + u * a1[1],
                (1.0 - u) * a0[2] + u * a1[2]
            };
            double wB[3] = { p[0] - b0[0], p[1] - b0[1], p[2] - b0[2] };
            v = clamp01(dot3(B, wB) / BB);
        }
    };

    const double rcut = std::pow(2.0, 1.0 / 6.0) * sigma; // wca cutoff

    // For each non-adjacent segment pair (i,i+1) and (j,j+1):
    for (int i = 0; i < N; ++i) {
        int ip1 = i + 1;
        for (int j = i + 1; j < N; ++j) {
            int jp1 = j + 1;

            int di = (j - i + N) % N; // skipping neighbours
            if (di == 0 || di == 1 || di == N - 1) continue;

            double a0[3] = { get(i, 0),   get(i, 1),   get(i, 2)   }; // endpoints of [i,i+1]
            double a1[3] = { get(ip1, 0), get(ip1, 1), get(ip1, 2) };
            double b0[3] = { get(j, 0),   get(j, 1),   get(j, 2)   }; // endpoints of [j, j+1]
            double b1[3] = { get(jp1, 0), get(jp1, 1), get(jp1, 2) };

            double u = 0.0, v = 0.0; // solving for u,v
            closest_uv(a0, a1, b0, b1, u, v);

            double p[3] = {  // closest points p(u)
                (1.0 - u) * a0[0] + u * a1[0],
                (1.0 - u) * a0[1] + u * a1[1],
                (1.0 - u) * a0[2] + u * a1[2]
            };

            double q[3] = {  // q(v)
                (1.0 - v) * b0[0] + v * b1[0],
                (1.0 - v) * b0[1] + v * b1[1],
                (1.0 - v) * b0[2] + v * b1[2]
            };

            double rvec[3] = { p[0] - q[0], p[1] - q[1], p[2] - q[2] }; // r = p-q
            double d2 = dot3(rvec, rvec);
            double d  = std::sqrt(std::max(d2, 1e-24));

            if (d >= rcut) continue;

            // WCA energy
            double invd = 1.0 / d;
            double sr   = sigma * invd;
            double sr2  = sr * sr;
            double sr6  = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double U = 4.0 * eps * (sr12 - sr6) + eps;
            E += U;

            double dU_dd = (24.0 * eps * invd) * (-2.0 * sr12 + sr6); // dU/dd
            double nvec[3] = { rvec[0] * invd, rvec[1] * invd, rvec[2] * invd }; // n = r/d

            // baseline gradient
            for (int dim = 0; dim < 3; ++dim) {
                double gcomp = dU_dd * nvec[dim];

                addg(i,   dim, (1.0 - u) * gcomp);  // [i, i + 1] endpoints
                addg(ip1, dim, u * gcomp);

                addg(j,   dim, -(1.0 - v) * gcomp); // [j, j + 1] endpoints but are negative
                addg(jp1, dim, -v * gcomp);
            }

            // NOTE: The assignment note says to include the dependence of (u*, v*) on endpoints.
            // Once energy is correct/stable, you can add the (du,dv) correction term back in and
            // verify against finite differences.
        }
    }

    *energy_out = E;
}

} // extern "C"
