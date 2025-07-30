/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_SEQ_H
#define ISL_SEQ_H

#include <sys/types.h>
#include <isl_int.h>
#include <isl/ctx.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Some common operations on sequences of isl_int's */

void isl_seq_clr(isl_int *p, unsigned len);
void isl_seq_set(isl_int *p, isl_int v, unsigned len);
void isl_seq_set_si(isl_int *p, int v, unsigned len);
void isl_seq_neg(isl_int *dst, isl_int *src, unsigned len);
void isl_seq_cpy(isl_int *dst, isl_int *src, unsigned len);
void isl_seq_addmul(isl_int *dst, isl_int f, isl_int *src, unsigned len);
void isl_seq_submul(isl_int *dst, isl_int f, isl_int *src, unsigned len);
void isl_seq_swp_or_cpy(isl_int *dst, isl_int *src, unsigned len);
void isl_seq_scale(isl_int *dst, isl_int *src, isl_int f, unsigned len);
void isl_seq_scale_down(isl_int *dst, isl_int *src, isl_int f, unsigned len);
void isl_seq_cdiv_q(isl_int *dst, isl_int *src, isl_int m, unsigned len);
void isl_seq_fdiv_q(isl_int *dst, isl_int *src, isl_int m, unsigned len);
void isl_seq_fdiv_r(isl_int *dst, isl_int *src, isl_int m, unsigned len);
void isl_seq_combine(isl_int *dst, isl_int m1, isl_int *src1,
			isl_int m2, isl_int *src2, unsigned len);
void isl_seq_elim(isl_int *dst, isl_int *src, unsigned pos, unsigned len,
		  isl_int *m);
void isl_seq_abs_max(isl_int *p, unsigned len, isl_int *max);
void isl_seq_gcd(isl_int *p, unsigned len, isl_int *gcd);
void isl_seq_lcm(isl_int *p, unsigned len, isl_int *lcm);
void isl_seq_normalize(struct isl_ctx *ctx, isl_int *p, unsigned len);
void isl_seq_inner_product(isl_int *p1, isl_int *p2, unsigned len,
			   isl_int *prod);
int isl_seq_first_non_zero(isl_int *p, unsigned len);
int isl_seq_last_non_zero(isl_int *p, unsigned len);
int isl_seq_abs_min_non_zero(isl_int *p, unsigned len);
int isl_seq_eq(isl_int *p1, isl_int *p2, unsigned len);
int isl_seq_cmp(isl_int *p1, isl_int *p2, unsigned len);
int isl_seq_is_neg(isl_int *p1, isl_int *p2, unsigned len);

uint32_t isl_seq_get_hash(isl_int *p, unsigned len);
uint32_t isl_seq_get_hash_bits(isl_int *p, unsigned len, unsigned bits);

#if defined(__cplusplus)
}
#endif

#endif
