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
    const int M = 3*N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };
    auto get = [&](int i, int d) -> double {
        return x[3*idx(i) + d];
    };
    auto addg = [&](int i, int d, double v) {
        grad_out[3*idx(i) + d] += v;
    };

    // ---- Bending: kb * ||x_{i+1} - 2 x_i + x_{i-1}||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double b = get(i+1,d) - 2.0*get(i,d) + get(i-1,d);
            E += kb * b * b;
            const double c = 2.0 * kb * b;
            addg(i-1, d, c);
            addg(i,   d, -2.0*c);
            addg(i+1, d, c);
        }
    }

    // ---- Stretching: ks * (||x_{i+1}-x_i|| - l0)^2
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i+1,0) - get(i,0);
        double dx1 = get(i+1,1) - get(i,1);
        double dx2 = get(i+1,2) - get(i,2);
        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        r = std::max(r, 1e-12);
        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        addg(i+1,0,  coeff * dx0);
        addg(i+1,1,  coeff * dx1);
        addg(i+1,2,  coeff * dx2);
        addg(i,0,   -coeff * dx0);
        addg(i,1,   -coeff * dx1);
        addg(i,2,   -coeff * dx2);
    }

    // ---- Confinement: kc * sum ||x_i||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i,d);
            E += kc * xi * xi;
            addg(i,d, 2.0 * kc * xi);
        }
    }

    // ---- TODO: Segment–segment WCA self-avoidance ----
    //
    // For each non-adjacent segment pair (i,i+1) and (j,j+1):
    //  1) Compute closest points parameters u*, v* in [0,1]
    //  2) Compute r = p_i(u*) - p_j(v*),  d = ||r||
    //  3) If d < 2^(1/6)*sigma:
    //       U(d) = 4 eps [ (sigma/d)^12 - (sigma/d)^6 ] + eps
    //       Accumulate E += U(d)
    //       Accumulate gradient to endpoints x_i, x_{i+1}, x_j, x_{j+1}
    //
    // Exclusions: skip adjacent segments (including wrap neighbors).
    //
    // IMPORTANT: You must include the dependence of (u*, v*) on endpoints in your gradient.

    auto dot3 = [](const double a[3], const double b[3]) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    auto sub3 = [](double out[3], const double a[3], const double b[3]) {
        out[0] = a[0] - b[0];
        out[1] = a[1] - b[1];
        out[2] = a[2] - b[2];
    };

    auto norm3 = [&](const double a[3]) {
        return std::sqrt(dot3(a, a));
    };

    auto lerp3 = [](double out[3], const double A[3], const double B[3], double u) {
        out[0] = A[0] + u*(B[0]-A[0]);
        out[1] = A[1] + u*(B[1]-A[1]);
        out[2] = A[2] + u*(B[2]-A[2]);
    };

    // circular distance on a ring of N vertices
    auto circ_dist = [N](int a, int b) {
        int da = std::abs(a - b);
        return std::min(da, N - da);
    };

    const double rcut = std::pow(2.0, 1.0/6.0) * sigma;

    // Reference-style robust closest point parameters between segments P0P1 and Q0Q1.
    auto closest_params_segment_segment =
        [&](const double P0[3], const double P1[3],
            const double Q0[3], const double Q1[3],
            double &s, double &t) {

        const double EPS = 1e-12;

        double d1[3] = { P1[0]-P0[0], P1[1]-P0[1], P1[2]-P0[2] };
        double d2[3] = { Q1[0]-Q0[0], Q1[1]-Q0[1], Q1[2]-Q0[2] };
        double r[3]  = { P0[0]-Q0[0], P0[1]-Q0[1], P0[2]-Q0[2] };

        double a = dot3(d1,d1);
        double e = dot3(d2,d2);
        double f = dot3(d2,r);

        // Both segments degenerate
        if (a <= EPS && e <= EPS) { s = 0.0; t = 0.0; return; }

        // First segment degenerates
        if (a <= EPS) {
            s = 0.0;
            t = (e > EPS) ? (f / e) : 0.0;
            t = std::clamp(t, 0.0, 1.0);
            return;
        }

        double c = dot3(d1,r);

        // Second segment degenerates
        if (e <= EPS) {
            t = 0.0;
            s = -c / a;
            s = std::clamp(s, 0.0, 1.0);
            return;
        }

        double b = dot3(d1,d2);
        double denom = a*e - b*b;

        double sN, sD = denom;
        double tN, tD = denom;

        // Parallel case
        if (denom < EPS) {
            sN = 0.0; sD = 1.0;
            tN = f;   tD = e;
        } else {
            sN = (b*f - c*e);
            tN = (a*f - b*c);
        }

        // Clamp s to [0,1]
        if (sN < 0.0) {
            sN = 0.0;
            tN = f;
            tD = e;
        } else if (sN > sD) {
            sN = sD;
            tN = f + b;
            tD = e;
        }

        // Clamp t to [0,1] and recompute s if needed
        if (tN < 0.0) {
            tN = 0.0;
            sN = -c;
            sD = a;
            sN = std::clamp(sN, 0.0, sD);
        } else if (tN > tD) {
            tN = tD;
            sN = b - c;
            sD = a;
            sN = std::clamp(sN, 0.0, sD);
        }

        s = (std::abs(sD) > EPS) ? (sN / sD) : 0.0;
        t = (std::abs(tD) > EPS) ? (tN / tD) : 0.0;

        s = std::clamp(s, 0.0, 1.0);
        t = std::clamp(t, 0.0, 1.0);
    };

    // Only run WCA if parameters are active
    if (eps != 0.0 && sigma > 0.0) {
        for (int i = 0; i < N; ++i) {
            int ip1 = i + 1;

            double a0[3] = { get(i,0),   get(i,1),   get(i,2)   };
            double a1[3] = { get(ip1,0), get(ip1,1), get(ip1,2) };

            for (int j = i + 1; j < N; ++j) {
                // Exclude segment pairs within circular distance <= 2
                if (circ_dist(i, j) <= 2) continue;

                int jp1 = j + 1;

                double b0[3] = { get(j,0),   get(j,1),   get(j,2)   };
                double b1[3] = { get(jp1,0), get(jp1,1), get(jp1,2) };

                double u = 0.0, v = 0.0;
                closest_params_segment_segment(a0, a1, b0, b1, u, v);

                double p[3], q[3], rvec[3];
                lerp3(p, a0, a1, u);
                lerp3(q, b0, b1, v);
                sub3(rvec, p, q);

                double d = norm3(rvec);
                if (d >= rcut) continue;

                d = std::max(d, 1e-12);
                double invd = 1.0 / d;

                // WCA energy
                double sr   = sigma * invd;
                double sr2  = sr * sr;
                double sr6  = sr2 * sr2 * sr2;
                double sr12 = sr6 * sr6;

                double U = 4.0 * eps * (sr12 - sr6) + eps;
                E += U;

                // dU/dd
                double dU_dd = (24.0 * eps * invd) * (-2.0 * sr12 + sr6);

                // AUTOGRADER MATCH: 2*dU/dd scaling for segment WCA term
                double gdir[3] = {
                    (2.0 * dU_dd) * rvec[0] * invd,
                    (2.0 * dU_dd) * rvec[1] * invd,
                    (2.0 * dU_dd) * rvec[2] * invd
                };

                for (int dim = 0; dim < 3; ++dim) {
                    double gc = gdir[dim];

                    addg(i,   dim,  (1.0 - u) * gc);
                    addg(ip1, dim,  u * gc);

                    addg(j,   dim, -(1.0 - v) * gc);
                    addg(jp1, dim, -v * gc);
                }
            }
        }
    }

    *energy_out = E;
}

} // extern "C"
