#ifndef ISL_INT_IMATH_H
#define ISL_INT_IMATH_H

#include "isl_hide_deprecated.h"

#include <isl_imath.h>

/* isl_int is the basic integer type, implemented with imath's mp_int. */
typedef mp_int isl_int;

#define isl_int_init(i)		i = mp_int_alloc()
#define isl_int_clear(i)	mp_int_free(i)

#define isl_int_set(r,i)	impz_set(r,i)
#define isl_int_set_si(r,i)	impz_set_si(r,i)
#define isl_int_set_ui(r,i)	impz_set_ui(r,i)
#define isl_int_fits_slong(r)	isl_imath_fits_slong_p(r)
#define isl_int_get_si(r)	impz_get_si(r)
#define isl_int_fits_ulong(r)	isl_imath_fits_ulong_p(r)
#define isl_int_get_ui(r)	impz_get_ui(r)
#define isl_int_get_d(r)	impz_get_si(r)
#define isl_int_get_str(r)	impz_get_str(0, 10, r)
#define isl_int_abs(r,i)	impz_abs(r,i)
#define isl_int_neg(r,i)	impz_neg(r,i)
#define isl_int_swap(i,j)	impz_swap(i,j)
#define isl_int_swap_or_set(i,j)	impz_swap(i,j)
#define isl_int_add_ui(r,i,j)	impz_add_ui(r,i,j)
#define isl_int_sub_ui(r,i,j)	impz_sub_ui(r,i,j)

#define isl_int_add(r,i,j)	impz_add(r,i,j)
#define isl_int_sub(r,i,j)	impz_sub(r,i,j)
#define isl_int_mul(r,i,j)	impz_mul(r,i,j)
#define isl_int_mul_2exp(r,i,j)	impz_mul_2exp(r,i,j)
#define isl_int_mul_si(r,i,j)	mp_int_mul_value(i,j,r)
#define isl_int_mul_ui(r,i,j)	impz_mul_ui(r,i,j)
#define isl_int_pow_ui(r,i,j)	impz_pow_ui(r,i,j)
#define isl_int_addmul(r,i,j)	impz_addmul(r,i,j)
#define isl_int_addmul_ui(r,i,j)	isl_imath_addmul_ui(r,i,j)
#define isl_int_submul(r,i,j)	impz_submul(r,i,j)
#define isl_int_submul_ui(r,i,j)	isl_imath_submul_ui(r,i,j)

#define isl_int_gcd(r,i,j)	impz_gcd(r,i,j)
#define isl_int_lcm(r,i,j)	impz_lcm(r,i,j)
#define isl_int_divexact(r,i,j)	impz_divexact(r,i,j)
#define isl_int_divexact_ui(r,i,j)	impz_divexact_ui(r,i,j)
#define isl_int_tdiv_q(r,i,j)	impz_tdiv_q(r,i,j)
#define isl_int_cdiv_q(r,i,j)	impz_cdiv_q(r,i,j)
#define isl_int_fdiv_q(r,i,j)	impz_fdiv_q(r,i,j)
#define isl_int_fdiv_r(r,i,j)	impz_fdiv_r(r,i,j)
#define isl_int_fdiv_q_ui(r,i,j)	impz_fdiv_q_ui(r,i,j)

#define isl_int_read(r,s)	impz_set_str(r,s,10)
#define isl_int_sgn(i)		impz_sgn(i)
#define isl_int_cmp(i,j)	impz_cmp(i,j)
#define isl_int_cmp_si(i,si)	impz_cmp_si(i,si)
#define isl_int_eq(i,j)		(impz_cmp(i,j) == 0)
#define isl_int_ne(i,j)		(impz_cmp(i,j) != 0)
#define isl_int_lt(i,j)		(impz_cmp(i,j) < 0)
#define isl_int_le(i,j)		(impz_cmp(i,j) <= 0)
#define isl_int_gt(i,j)		(impz_cmp(i,j) > 0)
#define isl_int_ge(i,j)		(impz_cmp(i,j) >= 0)
#define isl_int_abs_cmp(i,j)	impz_cmpabs(i,j)
#define isl_int_abs_eq(i,j)	(impz_cmpabs(i,j) == 0)
#define isl_int_abs_ne(i,j)	(impz_cmpabs(i,j) != 0)
#define isl_int_abs_lt(i,j)	(impz_cmpabs(i,j) < 0)
#define isl_int_abs_gt(i,j)	(impz_cmpabs(i,j) > 0)
#define isl_int_abs_ge(i,j)	(impz_cmpabs(i,j) >= 0)
#define isl_int_is_divisible_by(i,j)	impz_divisible_p(i,j)

uint32_t isl_imath_hash(mp_int v, uint32_t hash);
#define isl_int_hash(v,h)	isl_imath_hash(v,h)

typedef void (*isl_int_print_mp_free_t)(void *, size_t);
#define isl_int_free_str(s)	free(s)

#endif /* ISL_INT_IMATH_H */
