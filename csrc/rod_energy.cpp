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
            addg(i - 1, d, c);
            addg(i,     d, -2.0 * c);
            addg(i + 1, d, c);
        }
    }

    // ---- Stretching: ks * (||x_{i+1}-x_i|| - l0)^2
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i + 1, 0) - get(i, 0);
        double dx1 = get(i + 1, 1) - get(i, 1);
        double dx2 = get(i + 1, 2) - get(i, 2);

        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
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

    // ---- Segment-segment WCA self-avoidance (closest points; NO implicit du/dv differentiation)

    auto clamp01 = [](double t) {
        if (t < 0.0) return 0.0;
        if (t > 1.0) return 1.0;
        return t;
    };

    auto dot3 = [](const double a[3], const double b[3]) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    const double rcut = std::pow(2.0, 1.0/6.0) * sigma; // WCA cutoff

    // Robust closest points on two segments in 3D:
    // Returns (u,v) in [0,1]^2 for p=a0+u(A), q=b0+v(B).
    auto closest_uv = [&](const double a0[3], const double a1[3],
                          const double b0[3], const double b1[3],
                          double &u, double &v) {
        const double EPS = 1e-14;

        double A[3] = { a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2] };
        double B[3] = { b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2] };
        double r[3] = { a0[0]-b0[0], a0[1]-b0[1], a0[2]-b0[2] };

        double a = dot3(A, A); // |A|^2
        double e = dot3(B, B); // |B|^2
        double b = dot3(A, B);
        double c = dot3(A, r);
        double f = dot3(B, r);

        // Degenerate segments
        if (a <= EPS && e <= EPS) { u = 0.0; v = 0.0; return; }
        if (a <= EPS) { u = 0.0; v = clamp01(f / e); return; }
        if (e <= EPS) { v = 0.0; u = clamp01(-c / a); return; }

        double denom = a*e - b*b;

        if (std::abs(denom) > EPS) {
            u = clamp01((b*f - c*e) / denom);
        } else {
            // Parallel / near-parallel
            u = 0.0;
        }

        v = (b*u + f) / e;

        if (v < 0.0) {
            v = 0.0;
            u = clamp01(-c / a);
        } else if (v > 1.0) {
            v = 1.0;
            u = clamp01((b - c) / a);
        }

        // Optional boundary refinement
        if (u <= 0.0) {
            u = 0.0;
            v = clamp01(f / e);
        } else if (u >= 1.0) {
            u = 1.0;
            double r1[3] = { a1[0]-b0[0], a1[1]-b0[1], a1[2]-b0[2] };
            v = clamp01(dot3(B, r1) / e);
        }
    };

    for (int i = 0; i < N; ++i) {
        int ip1 = i + 1;

        for (int j = i + 1; j < N; ++j) {
            int jp1 = j + 1;

            // Exclusions: skip adjacent segments (including wrap neighbors)
            int di = (j - i + N) % N;
            if (di <= 2 || di >= N - 2) continue;

            double a0[3] = { get(i,0),   get(i,1),   get(i,2)   };
            double a1[3] = { get(ip1,0), get(ip1,1), get(ip1,2) };
            double b0[3] = { get(j,0),   get(j,1),   get(j,2)   };
            double b1[3] = { get(jp1,0), get(jp1,1), get(jp1,2) };

            double u = 0.0, v = 0.0;
            closest_uv(a0, a1, b0, b1, u, v);

            // Closest points
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

            // WCA energy
            double invd = 1.0 / d;
            double sr   = sigma * invd;
            double sr2  = sr * sr;
            double sr6  = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double U = 4.0 * eps * (sr12 - sr6) + eps;
            E += U;

            // dU/dd
            double dU_dd = (24.0 * eps * invd) * (-2.0 * sr12 + sr6);

            // unit direction n = r / d
            double nvec[3] = { rvec[0]*invd, rvec[1]*invd, rvec[2]*invd };

            // Gradient contribution to endpoints (treat u,v as the closest-point weights)
            for (int dim = 0; dim < 3; ++dim) {
                double gcomp = dU_dd * nvec[dim];

                // p = (1-u)a0 + u a1
                addg(i,   dim, (1.0 - u) * gcomp);
                addg(i+1, dim, u * gcomp);

                // q = (1-v)b0 + v b1, and r = p - q => negative contribution for b endpoints
                addg(j,   dim, -(1.0 - v) * gcomp);
                addg(j+1, dim, -v * gcomp);
            }
        }
    }

    *energy_out = E;
}

} // extern "C"
