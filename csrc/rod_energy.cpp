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

    // ---- TODO: Segment-segment WCA self-avoidance ----
    //
    // For each non-adjacent segment pair (i,i+1) and (j,j+1):
    // 1) Compute closest points parameters u*, v* in [0,1]
    // 2) Compute r = p_i(u*) - p_j(v*), d = ||r||
    // 3) If d < 2^(1/6)*sigma:
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
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    const double rcut = std::pow(2.0, 1.0/6.0) * sigma; // wca cutoff

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
        if (a <= EPS) { // first is a point
            u = 0.0;
            v = clamp01(f / e);
            return;
        }
        if (e <= EPS) { // second is a point
            v = 0.0;
            u = clamp01(-c / a);
            return;
        }

        double denom = a*e - b*b;

        // If not parallel, solve; otherwise fall back robustly
        if (std::abs(denom) > EPS) {
            u = clamp01((b*f - c*e) / denom);
        } else {
            // Parallel: pick u=0 initially (will be clamped), then compute v from that
            u = 0.0;
        }

        // Given u, solve best v and clamp
        v = (b*u + f) / e;
        if (v < 0.0) {
            v = 0.0;
            u = clamp01(-c / a);
        } else if (v > 1.0) {
            v = 1.0;
            u = clamp01((b - c) / a);
        }

        // One more pass if u ended up on boundary to refine v
        // (helps with parallel / near-parallel cases)
        if (u <= 0.0) {
            u = 0.0;
            v = clamp01(f / e);
        } else if (u >= 1.0) {
            u = 1.0;
            double r1[3] = { a1[0]-b0[0], a1[1]-b0[1], a1[2]-b0[2] };
            v = clamp01(dot3(B, r1) / e);
        }
    };

    // For each non-adjacent segment pair (i,i+1) and (j,j+1):
    for (int i = 0; i < N; ++i) {
        int ip1 = i + 1;
        for (int j = i + 1; j < N; ++j) {
            int jp1 = j + 1;

            // Exclusions: skip adjacent segments (including wrap neighbors).
            int di = (j - i + N) % N; // skipping neighbours
            if (di == 0 || di == 1 || di == N-1) continue;

            double a0[3] = { get(i,0),   get(i,1),   get(i,2)   }; // endpoints of [i, i+1]
            double a1[3] = { get(ip1,0), get(ip1,1), get(ip1,2) };
            double b0[3] = { get(j,0),   get(j,1),   get(j,2)   }; // endpoints of [j, j+1]
            double b1[3] = { get(jp1,0), get(jp1,1), get(jp1,2) };

            double A[3]  = { a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2] }; // direction vectors
            double B[3]  = { b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2] };
            double r0[3] = { a0[0]-b0[0], a0[1]-b0[1], a0[2]-b0[2] };

            // ---- Robust closest points parameters u*, v* in [0,1]
            double u = 0.0, v = 0.0;
            closest_uv(a0, a1, b0, b1, u, v);

            // closest points
            double p[3] = { // p(u)
                (1.0-u)*a0[0] + u*a1[0],
                (1.0-u)*a0[1] + u*a1[1],
                (1.0-u)*a0[2] + u*a1[2]
            };

            double q[3] = { // q(v)
                (1.0-v)*b0[0] + v*b1[0],
                (1.0-v)*b0[1] + v*b1[1],
                (1.0-v)*b0[2] + v*b1[2]
            };

            double rvec[3] = { p[0]-q[0], p[1]-q[1], p[2]-q[2] }; // r = p - q
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
            double nvec[3] = { rvec[0]*invd, rvec[1]*invd, rvec[2]*invd }; // n = r/d

            // baseline gradient (treating u,v as constants)
            for (int dim = 0; dim < 3; ++dim) {
                double gcomp = dU_dd * nvec[dim];

                addg(i,   dim, (1.0 - u) * gcomp);   // [i, i+1] endpoints
                addg(i+1, dim, u * gcomp);

                addg(j,   dim, -(1.0 - v) * gcomp);  // [j, j+1] endpoints but are negative
                addg(j+1, dim, -v * gcomp);
            }

            // Dependence of (u*,v*) on endpoints:
            // Only add implicit correction if interior (not clamped) and not near-parallel.
            // (Parallel / boundary cases are non-smooth; skipping is standard for this assignment.)
            double BB = dot3(A, A);
            double DD = dot3(B, B);
            double EE = dot3(A, B);
            double det = BB*DD - EE*EE;

            bool u_free = (u > 1e-12 && u < 1.0 - 1e-12);
            bool v_free = (v > 1e-12 && v < 1.0 - 1e-12);
            bool can_diff = u_free && v_free && (std::abs(det) > 1e-14);

            double inv00=0.0, inv01=0.0, inv10=0.0, inv11=0.0; // inverse of [BB -EE; -EE DD] is (1/det)[DD EE; EE BB]
            if (can_diff) {
                inv00 = DD / det;
                inv01 = EE / det;
                inv10 = EE / det;
                inv11 = BB / det;

                // For each endpoint coordinate perturbation, compute induced du,dv, then add energy derivative to that coord.
                for (int endpoint = 0; endpoint < 4; ++endpoint) {  // 0=a0, 1=a1, 2=b0, 3=b1
                    for (int dim = 0; dim < 3; ++dim) {

                        double da0[3] = {0,0,0}, da1[3] = {0,0,0}, db0[3] = {0,0,0}, db1[3] = {0,0,0}; // unit perturbation applied to one endpoint
                        if (endpoint == 0) da0[dim] = 1.0;
                        if (endpoint == 1) da1[dim] = 1.0;
                        if (endpoint == 2) db0[dim] = 1.0;
                        if (endpoint == 3) db1[dim] = 1.0;

                        double dA[3]  = { da1[0]-da0[0], da1[1]-da0[1], da1[2]-da0[2] }; // variations of A,B,r0
                        double dB[3]  = { db1[0]-db0[0], db1[1]-db0[1], db1[2]-db0[2] };
                        double dr0v[3]= { da0[0]-db0[0], da0[1]-db0[1], da0[2]-db0[2] };

                        double rhs0 = -dot3(A, r0);
                        double rhs1 =  dot3(B, r0);

                        double drhs0 = -(dot3(dA, r0) + dot3(A, dr0v)); // d(rhs)
                        double drhs1 =  (dot3(dB, r0) + dot3(B, dr0v));

                        double dBB = 2.0 * dot3(A, dA);      // d(matrix entries)
                        double dDD = 2.0 * dot3(B, dB);
                        double dEE = dot3(dA, B) + dot3(A, dB);

                        // M = [BB -EE; -EE DD], rhs = [rhs0; rhs1]
                        // dM*[u;v] = [dBB*u + (-dEE)*v; (-dEE)*u + dDD*v]
                        double dMt0 = dBB*u + (-dEE)*v;
                        double dMt1 = (-dEE)*u + dDD*v;

                        double b0s = drhs0 - dMt0; // solving: [du;dv] = invM * (drhs - dM*[u;v])
                        double b1s = drhs1 - dMt1;

                        double du = inv00*b0s + inv01*b1s;
                        double dv = inv10*b0s + inv11*b1s;

                        // r = r0 + uA - vB, so dr from du,dv is du*A - dv*B
                        double dr_uv[3] = {
                            du*A[0] - dv*B[0],
                            du*A[1] - dv*B[1],
                            du*A[2] - dv*B[2]
                        };

                        double dd_uv = nvec[0]*dr_uv[0] + nvec[1]*dr_uv[1] + nvec[2]*dr_uv[2];
                        double dU_uv = dU_dd * dd_uv;

                        if (endpoint == 0) addg(i,   dim, dU_uv); // so it adds this correction into the right grad entry
                        if (endpoint == 1) addg(i+1, dim, dU_uv);
                        if (endpoint == 2) addg(j,   dim, dU_uv);
                        if (endpoint == 3) addg(j+1, dim, dU_uv);
                    }
                }
            }
        }
    }

    *energy_out = E;
}

} // extern "C"
