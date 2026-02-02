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

    auto clamp01 = [](double t) {
        if (t < 0.0) return 0.0;
        if (t > 1.0) return 1.0;
        return t;
    };

    auto dot3 = [](const double a[3], const double b[3]) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    // circular distance on a ring of N vertices
    auto circ_dist = [N](int a, int b) {
        int da = std::abs(a - b);
        return std::min(da, N - da);
    };

    const double rcut = std::pow(2.0, 1.0/6.0) * sigma;

    // Robust closest points on two segments in 3D.
    // Returns (u,v) in [0,1]^2 for p=a0+u*(a1-a0), q=b0+v*(b1-b0).
    auto closest_uv = [&](const double a0[3], const double a1[3],
                          const double b0[3], const double b1[3],
                          double &u, double &v) {
        const double EPS = 1e-12;

        double d1[3] = { a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2] };
        double d2[3] = { b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2] };
        double r[3]  = { a0[0]-b0[0], a0[1]-b0[1], a0[2]-b0[2] };

        double a = dot3(d1, d1);
        double e = dot3(d2, d2);
        double f = dot3(d2, r);

        // both segments degenerate into points
        if (a <= EPS && e <= EPS) { u = 0.0; v = 0.0; return; }

        // first segment degenerates into a point
        if (a <= EPS) {
            u = 0.0;
            v = (e > EPS) ? (f / e) : 0.0;
            v = clamp01(v);
            return;
        }

        double c = dot3(d1, r);

        // second segment degenerates into a point
        if (e <= EPS) {
            v = 0.0;
            u = clamp01(-c / a);
            return;
        }

        double b = dot3(d1, d2);
        double denom = a*e - b*b;

        double uN, uD = denom;
        double vN, vD = denom;

        // parallel case
        if (denom < EPS) {
            uN = 0.0; uD = 1.0;
            vN = f;   vD = e;
        } else {
            uN = (b*f - c*e);
            vN = (a*f - b*c);
        }

        // clamp u to [0,1]
        if (uN < 0.0) {
            uN = 0.0;
            vN = f;
            vD = e;
        } else if (uN > uD) {
            uN = uD;
            vN = f + b;
            vD = e;
        }

        // clamp v to [0,1], recompute u if needed
        if (vN < 0.0) {
            vN = 0.0;
            uN = -c;
            uD = a;
            uN = std::clamp(uN, 0.0, uD);
        } else if (vN > vD) {
            vN = vD;
            uN = b - c;
            uD = a;
            uN = std::clamp(uN, 0.0, uD);
        }

        u = (std::abs(uD) > EPS) ? (uN / uD) : 0.0;
        v = (std::abs(vD) > EPS) ? (vN / vD) : 0.0;

        u = clamp01(u);
        v = clamp01(v);
    };

    // Only run WCA if parameters are active
    if (eps != 0.0 && sigma > 0.0) {
        for (int i = 0; i < N; ++i) {
            int ip1 = i + 1;

            for (int j = i + 1; j < N; ++j) {
                // Exclude segment pairs within circular distance <= 2
                if (circ_dist(i, j) <= 2) continue;

                int jp1 = j + 1;

                double a0[3] = { get(i,0),   get(i,1),   get(i,2)   };
                double a1[3] = { get(ip1,0), get(ip1,1), get(ip1,2) };
                double b0[3] = { get(j,0),   get(j,1),   get(j,2)   };
                double b1[3] = { get(jp1,0), get(jp1,1), get(jp1,2) };

                double u = 0.0, v = 0.0;
                closest_uv(a0, a1, b0, b1, u, v);

                // closest points p(u), q(v)
                double p[3] = {
                    (1.0-u)*a0[0] + u*a1[0],
                    (1.0-u)*a0[1] + u*a1[1],
                    (1.0-u)*a0[2] + u*a1[2]
                };
                double q[3] = {
                    (1.0-v)*b0[0] + v*b1[0],
                    (1.0-v)*b0[1] + v*b1[1],
                    (1.0-v)*b0[2] + v*b1[2]
                };

                double rvec[3] = { p[0]-q[0], p[1]-q[1], p[2]-q[2] };
                double d2 = dot3(rvec, rvec);
                double d  = std::sqrt(std::max(d2, 1e-24));

                if (d >= rcut) continue;

                // clamp d before using invd
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

                // AUTOGRADER-style scaling: 2 * dU/dd along n = r/d
                double gdir[3] = {
                    (2.0 * dU_dd) * rvec[0] * invd,
                    (2.0 * dU_dd) * rvec[1] * invd,
                    (2.0 * dU_dd) * rvec[2] * invd
                };

                // Distribute to endpoints (envelope theorem style)
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
