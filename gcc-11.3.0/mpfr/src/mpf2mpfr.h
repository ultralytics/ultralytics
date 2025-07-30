/* mpf2mpfr.h -- Compatibility include file with mpf.

Copyright 1999-2002, 2004-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#ifndef __MPFR_FROM_MPF__
#define __MPFR_FROM_MPF__

/* types */
#define mpf_t mpfr_t
#define mpf_srcptr mpfr_srcptr
#define mpf_ptr mpfr_ptr

/* Get current Rounding Mode */
#ifndef MPFR_DEFAULT_RND
# define MPFR_DEFAULT_RND mpfr_get_default_rounding_mode ()
#endif

/* mpf_init initalizes at 0 */
#undef mpf_init
#define mpf_init(x) mpfr_init_set_ui ((x), 0, MPFR_DEFAULT_RND)
#undef mpf_init2
#define mpf_init2(x,p) (mpfr_init2((x),(p)), mpfr_set_ui ((x), 0, MPFR_DEFAULT_RND))

/* functions which don't take as argument the rounding mode */
#undef mpf_ceil
#define mpf_ceil mpfr_ceil
#undef mpf_clear
#define mpf_clear mpfr_clear
#undef mpf_cmp
#define mpf_cmp mpfr_cmp
#undef mpf_cmp_si
#define mpf_cmp_si mpfr_cmp_si
#undef mpf_cmp_ui
#define mpf_cmp_ui mpfr_cmp_ui
#undef mpf_cmp_d
#define mpf_cmp_d mpfr_cmp_d
#undef mpf_eq
#define mpf_eq mpfr_eq
#undef mpf_floor
#define mpf_floor mpfr_floor
#undef mpf_get_prec
#define mpf_get_prec mpfr_get_prec
#undef mpf_integer_p
#define mpf_integer_p mpfr_integer_p
#undef mpf_random2
#define mpf_random2 mpfr_random2
#undef mpf_set_default_prec
#define mpf_set_default_prec mpfr_set_default_prec
#undef mpf_get_default_prec
#define mpf_get_default_prec mpfr_get_default_prec
#undef mpf_set_prec
#define mpf_set_prec mpfr_set_prec
#undef mpf_set_prec_raw
#define mpf_set_prec_raw(x,p) mpfr_prec_round(x,p,MPFR_DEFAULT_RND)
#undef mpf_trunc
#define mpf_trunc mpfr_trunc
#undef mpf_sgn
#define mpf_sgn mpfr_sgn
#undef mpf_swap
#define mpf_swap mpfr_swap
#undef mpf_dump
#define mpf_dump mpfr_dump

/* functions which take as argument the rounding mode */
#undef mpf_abs
#define mpf_abs(x,y) mpfr_abs(x,y,MPFR_DEFAULT_RND)
#undef mpf_add
#define mpf_add(x,y,z) mpfr_add(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_add_ui
#define mpf_add_ui(x,y,z) mpfr_add_ui(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_div
#define mpf_div(x,y,z) mpfr_div(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_div_ui
#define mpf_div_ui(x,y,z) mpfr_div_ui(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_div_2exp
#define mpf_div_2exp(x,y,z) mpfr_div_2exp(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_fits_slong_p
#define mpf_fits_slong_p(x) mpfr_fits_slong_p(x,MPFR_DEFAULT_RND)
#undef mpf_fits_ulong_p
#define mpf_fits_ulong_p(x) mpfr_fits_ulong_p(x,MPFR_DEFAULT_RND)
#undef mpf_fits_sint_p
#define mpf_fits_sint_p(x) mpfr_fits_sint_p(x,MPFR_DEFAULT_RND)
#undef mpf_fits_uint_p
#define mpf_fits_uint_p(x) mpfr_fits_uint_p(x,MPFR_DEFAULT_RND)
#undef mpf_fits_sshort_p
#define mpf_fits_sshort_p(x) mpfr_fits_sshort_p(x,MPFR_DEFAULT_RND)
#undef mpf_fits_ushort_p
#define mpf_fits_ushort_p(x) mpfr_fits_ushort_p(x,MPFR_DEFAULT_RND)
#undef mpf_get_str
#define mpf_get_str(x,y,z,t,u) mpfr_get_str(x,y,z,t,u,MPFR_DEFAULT_RND)
#undef mpf_get_d
#define mpf_get_d(x) mpfr_get_d(x,MPFR_DEFAULT_RND)
#undef mpf_get_d_2exp
#define mpf_get_d_2exp(e,x) mpfr_get_d_2exp(e,x,MPFR_DEFAULT_RND)
#undef mpf_get_ui
#define mpf_get_ui(x) mpfr_get_ui(x,MPFR_DEFAULT_RND)
#undef mpf_get_si
#define mpf_get_si(x) mpfr_get_si(x,MPFR_DEFAULT_RND)
#undef mpf_inp_str
#define mpf_inp_str(x,y,z) mpfr_inp_str(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_set_str
#define mpf_set_str(x,y,z) mpfr_set_str(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_init_set
#define mpf_init_set(x,y) mpfr_init_set(x,y,MPFR_DEFAULT_RND)
#undef mpf_init_set_d
#define mpf_init_set_d(x,y) mpfr_init_set_d(x,y,MPFR_DEFAULT_RND)
#undef mpf_init_set_si
#define mpf_init_set_si(x,y) mpfr_init_set_si(x,y,MPFR_DEFAULT_RND)
#undef mpf_init_set_str
#define mpf_init_set_str(x,y,z) mpfr_init_set_str(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_init_set_ui
#define mpf_init_set_ui(x,y) mpfr_init_set_ui(x,y,MPFR_DEFAULT_RND)
#undef mpf_mul
#define mpf_mul(x,y,z) mpfr_mul(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_mul_2exp
#define mpf_mul_2exp(x,y,z) mpfr_mul_2exp(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_mul_ui
#define mpf_mul_ui(x,y,z) mpfr_mul_ui(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_neg
#define mpf_neg(x,y) mpfr_neg(x,y,MPFR_DEFAULT_RND)
#undef mpf_out_str
#define mpf_out_str(x,y,z,t) mpfr_out_str(x,y,z,t,MPFR_DEFAULT_RND)
#undef mpf_pow_ui
#define mpf_pow_ui(x,y,z) mpfr_pow_ui(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_reldiff
#define mpf_reldiff(x,y,z) mpfr_reldiff(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_set
#define mpf_set(x,y) mpfr_set(x,y,MPFR_DEFAULT_RND)
#undef mpf_set_d
#define mpf_set_d(x,y) mpfr_set_d(x,y,MPFR_DEFAULT_RND)
#undef mpf_set_q
#define mpf_set_q(x,y) mpfr_set_q(x,y,MPFR_DEFAULT_RND)
#undef mpf_set_si
#define mpf_set_si(x,y) mpfr_set_si(x,y,MPFR_DEFAULT_RND)
#undef mpf_set_ui
#define mpf_set_ui(x,y) mpfr_set_ui(x,y,MPFR_DEFAULT_RND)
#undef mpf_set_z
#define mpf_set_z(x,y) mpfr_set_z(x,y,MPFR_DEFAULT_RND)
#undef mpf_sqrt
#define mpf_sqrt(x,y) mpfr_sqrt(x,y,MPFR_DEFAULT_RND)
#undef mpf_sqrt_ui
#define mpf_sqrt_ui(x,y) mpfr_sqrt_ui(x,y,MPFR_DEFAULT_RND)
#undef mpf_sub
#define mpf_sub(x,y,z) mpfr_sub(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_sub_ui
#define mpf_sub_ui(x,y,z) mpfr_sub_ui(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_ui_div
#define mpf_ui_div(x,y,z) mpfr_ui_div(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_ui_sub
#define mpf_ui_sub(x,y,z) mpfr_ui_sub(x,y,z,MPFR_DEFAULT_RND)
#undef mpf_urandomb
#define mpf_urandomb(x,y,n) mpfr_urandomb(x,y)

#undef mpz_set_f
#define mpz_set_f(z,f) mpfr_get_z(z,f,MPFR_DEFAULT_RND)

#endif /* __MPFR_FROM_MPF__ */
