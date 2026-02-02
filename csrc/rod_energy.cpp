#include <cmath>
#include <algorithm>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

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

    auto dot3 = [](const double a[3], const double b[3]) -> double {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    auto clamp01 = [](double t) -> double {
        if (t < 0.0) return 0.0;
        if (t > 1.0) return 1.0;
        return t;
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
        double dx[3] = {
            get(i + 1, 0) - get(i, 0),
            get(i + 1, 1) - get(i, 1),
            get(i + 1, 2) - get(i, 2)
        };

        double r = std::sqrt(dot3(dx, dx));
        r = std::max(r, 1e-12);

        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        for (int d = 0; d < 3; ++d) {
            addg(i + 1, d,  coeff * dx[d]);
            addg(i,     d, -coeff * dx[d]);
        }
    }

    // ---- Confinement: kc * sum ||x_i||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i, d);
            E += kc * xi * xi;
            addg(i, d, 2.0 * kc * xi);
        }
    }

    // ---- Segment-segment WCA self-avoidance (closest features) ----
    //
    // We compute closest points p(u) on segment [a0,a1] and q(v) on [b0,b1],
    // with u,v clamped to [0,1]. Then:
    //   r = p - q, d = ||r||
    // If d < rcut = 2^(1/6) sigma:
    //   U(d) = 4 eps [ (sigma/d)^12 - (sigma/d)^6 ] + eps
    // Gradient uses envelope-theorem style (treat u,v fixed after closest-feature selection):
    //   dU/dr = (dU/dd) * r/d
    // distribute to endpoints with barycentric weights.
    //
    const double rcut = std::pow(2.0, 1.0/6.0) * sigma;

    // Robust closest points for two segments in 3D (Ericson-style).
    // Returns u,v in [0,1] for p=a0 + u*(a1-a0), q=b0 + v*(b1-b0).
    auto closest_uv = [&](const double a0[3], const double a1[3],
                          const double b0[3], const double b1[3],
                          double &u, double &v) {
        const double EPS = 1e-14;

        double d1[3] = { a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2] };
        double d2[3] = { b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2] };
        double r[3]  = { a0[0]-b0[0], a0[1]-b0[1], a0[2]-b0[2] };

        double a = dot3(d1, d1);
        double e = dot3(d2, d2);
        double f = dot3(d2, r);

        // If both segments degenerate into points
        if (a <= EPS && e <= EPS) { u = 0.0; v = 0.0; return; }

        // If first segment degenerates into a point
        if (a <= EPS) {
            u = 0.0;
            v = clamp01(f / e);
            return;
        }

        double c = dot3(d1, r);

        // If second segment degenerates into a point
        if (e <= EPS) {
            v = 0.0;
            u = clamp01(-c / a);
            return;
        }

        double b = dot3(d1, d2);
        double denom = a*e - b*b;

        // Compute u on infinite lines, then clamp to segment
        if (std::abs(denom) > EPS) {
            u = clamp01((b*f - c*e) / denom);
        } else {
            // Nearly parallel
            u = 0.0;
        }

        // Compute v from u, then clamp; recompute u if clamped v hits boundary
        v = (b*u + f) / e;

        if (v < 0.0) {
            v = 0.0;
            u = clamp01(-c / a);
        } else if (v > 1.0) {
            v = 1.0;
            u = clamp01((b - c) / a);
        } else {
            // v already interior
            v = clamp01(v);
        }
    };

    // Iterate segment pairs (i,i+1) and (j,j+1).
    // Exclude adjacent segments including wrap neighbors.
    for (int i = 0; i < N; ++i) {
        int ip1 = i + 1;
        for (int j = i + 1; j < N; ++j) {
            int jp1 = j + 1;

            int di = (j - i + N) % N;
            if (di <= 2 || di >= N-2) continue;


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

            // dU/dr = dU/dd * r/d
            double gdir[3] = { dU_dd * rvec[0] * invd,
                               dU_dd * rvec[1] * invd,
                               dU_dd * rvec[2] * invd };

            // Distribute to endpoints with barycentric weights
            for (int dim = 0; dim < 3; ++dim) {
                double gc = gdir[dim];

                addg(i,   dim,  (1.0 - u) * gc);
                addg(ip1, dim,  u * gc);

                addg(j,   dim, -(1.0 - v) * gc);
                addg(jp1, dim, -v * gc);
            }
        }
    }

    *energy_out = E;
}

} // extern "C"
