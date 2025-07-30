/* mpc_set_x_x -- Set complex number real and imaginary parts from parameters
   whose type is known by mpfr.

Copyright (C) 2009, 2011 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include "config.h"

#ifdef HAVE_INTTYPES_H
# include <inttypes.h> /* for intmax_t */
#else
# ifdef HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#include "mpc-impl.h"

#define MPC_SET_X_X(type, z, real_value, imag_value, rnd)       \
  MPC_SET_X_Y (type, type, z, real_value, imag_value, rnd)

int
mpc_set_d_d (mpc_ptr z, double a, double b, mpc_rnd_t rnd)
  MPC_SET_X_X (d, z, a, b, rnd)

int
mpc_set_f_f (mpc_ptr z, mpf_srcptr a, mpf_srcptr b, mpc_rnd_t rnd)
  MPC_SET_X_X (f, z, a, b, rnd)

int
mpc_set_fr_fr (mpc_ptr z, mpfr_srcptr a, mpfr_srcptr b, mpc_rnd_t rnd)
  MPC_SET_X_X (fr, z, a, b, rnd)

int
mpc_set_ld_ld (mpc_ptr z, long double a, long double b, mpc_rnd_t rnd)
  MPC_SET_X_X (ld, z, a, b, rnd)

int
mpc_set_q_q (mpc_ptr z, mpq_srcptr a, mpq_srcptr b, mpc_rnd_t rnd)
  MPC_SET_X_X (q, z, a, b, rnd)

int
mpc_set_si_si (mpc_ptr z, long int a, long int b, mpc_rnd_t rnd)
  MPC_SET_X_X (si, z, a, b, rnd)

int
mpc_set_ui_ui (mpc_ptr z, unsigned long int a, unsigned long int b,
               mpc_rnd_t rnd)
  MPC_SET_X_X (ui, z, a, b, rnd)

int
mpc_set_z_z (mpc_ptr z, mpz_srcptr a, mpz_srcptr b, mpc_rnd_t rnd)
  MPC_SET_X_X (z, z, a, b, rnd)

#ifdef _MPC_H_HAVE_INTMAX_T
int
mpc_set_uj_uj (mpc_ptr z, uintmax_t a, uintmax_t b, mpc_rnd_t rnd)
  MPC_SET_X_X (uj, z, a, b, rnd)

int
mpc_set_sj_sj (mpc_ptr z, intmax_t a, intmax_t b, mpc_rnd_t rnd)
  MPC_SET_X_X (sj, z, a, b, rnd)
#endif
