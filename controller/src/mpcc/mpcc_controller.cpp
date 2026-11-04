#include "quadrotor_mpc/mpcc/mpcc_controller.hpp"

#include "acados_solver_quadrotor_mpcc.h"
#include "acados_c/ocp_nlp_interface.h"

#include <cstring>
#include <cstdio>

namespace quadrotor_mpc {

const int MpccController::N = QUADROTOR_MPCC_N;

MpccController::MpccController() = default;

MpccController::~MpccController() {
    free();
}

bool MpccController::init() {
    capsule_ = quadrotor_mpcc_acados_create_capsule();
    int status = quadrotor_mpcc_acados_create_with_discretization(capsule_, N, nullptr);
    if (status != 0) {
        std::fprintf(stderr, "[MpccController] acados_create failed with status %d\n", status);
        return false;
    }
    return true;
}

void MpccController::set_x0(const State16& x0) {
    ocp_nlp_config* config = quadrotor_mpcc_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims   = quadrotor_mpcc_acados_get_nlp_dims(capsule_);
    ocp_nlp_in*     in     = quadrotor_mpcc_acados_get_nlp_in(capsule_);
    ocp_nlp_out*    out    = quadrotor_mpcc_acados_get_nlp_out(capsule_);

    double lbx0[NX], ubx0[NX];
    for (int i = 0; i < NX; ++i) {
        lbx0[i] = x0(i);
        ubx0[i] = x0(i);
    }
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "ubx", ubx0);
}

void MpccController::weights_to_param(double* p, const MpccWeights& w) {
    p[0]  = w.Q_ec(0);   p[1]  = w.Q_ec(1);   p[2]  = w.Q_ec(2);
    p[3]  = w.Q_el(0);   p[4]  = w.Q_el(1);   p[5]  = w.Q_el(2);
    p[6]  = w.Q_q(0);    p[7]  = w.Q_q(1);    p[8]  = w.Q_q(2);
    p[9]  = w.U_mat(0);  p[10] = w.U_mat(1);  p[11] = w.U_mat(2);  p[12] = w.U_mat(3);
    p[13] = w.Q_s;
    p[14] = w.vtheta_max;
    p[15] = w.W_df;
}

void MpccController::set_params(int stage, const MpccWeights& w) {
    double p[NP];
    weights_to_param(p, w);
    quadrotor_mpcc_acados_update_params(capsule_, stage, p, NP);
}

void MpccController::set_params_all(const MpccWeights& w) {
    double p[NP];
    weights_to_param(p, w);
    for (int k = 0; k <= N; ++k) {
        quadrotor_mpcc_acados_update_params(capsule_, k, p, NP);
    }
}

int MpccController::solve() {
    return quadrotor_mpcc_acados_solve(capsule_);
}

Control5 MpccController::get_u0() const {
    ocp_nlp_config* config = quadrotor_mpcc_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims   = quadrotor_mpcc_acados_get_nlp_dims(capsule_);
    ocp_nlp_out*    out    = quadrotor_mpcc_acados_get_nlp_out(capsule_);

    double u[NU];
    ocp_nlp_out_get(config, dims, out, 0, "u", u);
    return Control5(u[0], u[1], u[2], u[3], u[4]);
}

State16 MpccController::get_x(int k) const {
    ocp_nlp_config* config = quadrotor_mpcc_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims   = quadrotor_mpcc_acados_get_nlp_dims(capsule_);
    ocp_nlp_out*    out    = quadrotor_mpcc_acados_get_nlp_out(capsule_);

    double x[NX];
    ocp_nlp_out_get(config, dims, out, k, "x", x);
    return Eigen::Map<State16>(x);
}

void MpccController::set_x_init(int k, const State16& x) {
    ocp_nlp_config* config = quadrotor_mpcc_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims   = quadrotor_mpcc_acados_get_nlp_dims(capsule_);
    ocp_nlp_out*    out    = quadrotor_mpcc_acados_get_nlp_out(capsule_);
    ocp_nlp_in*     in     = quadrotor_mpcc_acados_get_nlp_in(capsule_);

    double xd[NX];
    for (int i = 0; i < NX; ++i) xd[i] = x(i);
    ocp_nlp_out_set(config, dims, out, in, k, "x", xd);
}

void MpccController::set_u_init(int k, const Control5& u) {
    ocp_nlp_config* config = quadrotor_mpcc_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims   = quadrotor_mpcc_acados_get_nlp_dims(capsule_);
    ocp_nlp_out*    out    = quadrotor_mpcc_acados_get_nlp_out(capsule_);
    ocp_nlp_in*     in     = quadrotor_mpcc_acados_get_nlp_in(capsule_);

    double ud[NU];
    for (int i = 0; i < NU; ++i) ud[i] = u(i);
    ocp_nlp_out_set(config, dims, out, in, k, "u", ud);
}

double MpccController::get_solve_time() const {
    ocp_nlp_solver* solver = quadrotor_mpcc_acados_get_nlp_solver(capsule_);
    double t;
    ocp_nlp_get(solver, "time_tot", &t);
    return t;
}

void MpccController::free() {
    if (capsule_) {
        quadrotor_mpcc_acados_free(capsule_);
        quadrotor_mpcc_acados_free_capsule(capsule_);
        capsule_ = nullptr;
    }
}

}  // namespace quadrotor_mpc
