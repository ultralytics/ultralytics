/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <isl_ctx_private.h>
#include <isl_seq.h>

void isl_seq_clr(isl_int *p, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_set_si(p[i], 0);
}

void isl_seq_set_si(isl_int *p, int v, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_set_si(p[i], v);
}

void isl_seq_set(isl_int *p, isl_int v, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_set(p[i], v);
}

void isl_seq_neg(isl_int *dst, isl_int *src, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_neg(dst[i], src[i]);
}

void isl_seq_cpy(isl_int *dst, isl_int *src, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_set(dst[i], src[i]);
}

void isl_seq_submul(isl_int *dst, isl_int f, isl_int *src, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_submul(dst[i], f, src[i]);
}

void isl_seq_addmul(isl_int *dst, isl_int f, isl_int *src, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_addmul(dst[i], f, src[i]);
}

void isl_seq_swp_or_cpy(isl_int *dst, isl_int *src, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_swap_or_set(dst[i], src[i]);
}

void isl_seq_scale(isl_int *dst, isl_int *src, isl_int m, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_mul(dst[i], src[i], m);
}

void isl_seq_scale_down(isl_int *dst, isl_int *src, isl_int m, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_divexact(dst[i], src[i], m);
}

void isl_seq_cdiv_q(isl_int *dst, isl_int *src, isl_int m, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_cdiv_q(dst[i], src[i], m);
}

void isl_seq_fdiv_q(isl_int *dst, isl_int *src, isl_int m, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_fdiv_q(dst[i], src[i], m);
}

void isl_seq_fdiv_r(isl_int *dst, isl_int *src, isl_int m, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		isl_int_fdiv_r(dst[i], src[i], m);
}

void isl_seq_combine(isl_int *dst, isl_int m1, isl_int *src1,
			isl_int m2, isl_int *src2, unsigned len)
{
	int i;
	isl_int tmp;

	if (dst == src1 && isl_int_is_one(m1)) {
		if (isl_int_is_zero(m2))
			return;
		for (i = 0; i < len; ++i)
			isl_int_addmul(src1[i], m2, src2[i]);
		return;
	}

	isl_int_init(tmp);
	for (i = 0; i < len; ++i) {
		isl_int_mul(tmp, m1, src1[i]);
		isl_int_addmul(tmp, m2, src2[i]);
		isl_int_set(dst[i], tmp);
	}
	isl_int_clear(tmp);
}

/*
 * Let d = dst[pos] and s = src[pos]
 * dst is replaced by |s| dst - sgn(s)d src
 */
void isl_seq_elim(isl_int *dst, isl_int *src, unsigned pos, unsigned len,
		  isl_int *m)
{
	isl_int a;
	isl_int b;

	if (isl_int_is_zero(dst[pos]))
		return;

	isl_int_init(a);
	isl_int_init(b);

	isl_int_gcd(a, src[pos], dst[pos]);
	isl_int_divexact(b, dst[pos], a);
	if (isl_int_is_pos(src[pos]))
		isl_int_neg(b, b);
	isl_int_divexact(a, src[pos], a);
	isl_int_abs(a, a);
	isl_seq_combine(dst, a, dst, b, src, len);

	if (m)
		isl_int_mul(*m, *m, a);

	isl_int_clear(a);
	isl_int_clear(b);
}

int isl_seq_eq(isl_int *p1, isl_int *p2, unsigned len)
{
	int i;
	for (i = 0; i < len; ++i)
		if (isl_int_ne(p1[i], p2[i]))
			return 0;
	return 1;
}

int isl_seq_cmp(isl_int *p1, isl_int *p2, unsigned len)
{
	int i;
	int cmp;
	for (i = 0; i < len; ++i)
		if ((cmp = isl_int_cmp(p1[i], p2[i])) != 0)
			return cmp;
	return 0;
}

int isl_seq_is_neg(isl_int *p1, isl_int *p2, unsigned len)
{
	int i;

	for (i = 0; i < len; ++i) {
		if (isl_int_abs_ne(p1[i], p2[i]))
			return 0;
		if (isl_int_is_zero(p1[i]))
			continue;
		if (isl_int_eq(p1[i], p2[i]))
			return 0;
	}
	return 1;
}

int isl_seq_first_non_zero(isl_int *p, unsigned len)
{
	int i;

	for (i = 0; i < len; ++i)
		if (!isl_int_is_zero(p[i]))
			return i;
	return -1;
}

int isl_seq_last_non_zero(isl_int *p, unsigned len)
{
	int i;

	for (i = len - 1; i >= 0; --i)
		if (!isl_int_is_zero(p[i]))
			return i;
	return -1;
}

void isl_seq_abs_max(isl_int *p, unsigned len, isl_int *max)
{
	int i;

	isl_int_set_si(*max, 0);

	for (i = 0; i < len; ++i)
		if (isl_int_abs_gt(p[i], *max))
			isl_int_abs(*max, p[i]);
}

int isl_seq_abs_min_non_zero(isl_int *p, unsigned len)
{
	int i, min = isl_seq_first_non_zero(p, len);
	if (min < 0)
		return -1;
	for (i = min + 1; i < len; ++i) {
		if (isl_int_is_zero(p[i]))
			continue;
		if (isl_int_abs_lt(p[i], p[min]))
			min = i;
	}
	return min;
}

void isl_seq_gcd(isl_int *p, unsigned len, isl_int *gcd)
{
	int i, min = isl_seq_abs_min_non_zero(p, len);

	if (min < 0) {
		isl_int_set_si(*gcd, 0);
		return;
	}
	isl_int_abs(*gcd, p[min]);
	for (i = 0; isl_int_cmp_si(*gcd, 1) > 0 && i < len; ++i) {
		if (i == min)
			continue;
		if (isl_int_is_zero(p[i]))
			continue;
		isl_int_gcd(*gcd, *gcd, p[i]);
	}
}

void isl_seq_normalize(struct isl_ctx *ctx, isl_int *p, unsigned len)
{
	if (len == 0)
		return;
	isl_seq_gcd(p, len, &ctx->normalize_gcd);
	if (!isl_int_is_zero(ctx->normalize_gcd) &&
	    !isl_int_is_one(ctx->normalize_gcd))
		isl_seq_scale_down(p, p, ctx->normalize_gcd, len);
}

void isl_seq_lcm(isl_int *p, unsigned len, isl_int *lcm)
{
	int i;

	if (len == 0) {
		isl_int_set_si(*lcm, 1);
		return;
	}
	isl_int_set(*lcm, p[0]);
	for (i = 1; i < len; ++i)
		isl_int_lcm(*lcm, *lcm, p[i]);
}

void isl_seq_inner_product(isl_int *p1, isl_int *p2, unsigned len,
			   isl_int *prod)
{
	int i;
	if (len == 0) {
		isl_int_set_si(*prod, 0);
		return;
	}
	isl_int_mul(*prod, p1[0], p2[0]);
	for (i = 1; i < len; ++i)
		isl_int_addmul(*prod, p1[i], p2[i]);
}

uint32_t isl_seq_hash(isl_int *p, unsigned len, uint32_t hash)
{
	int i;
	for (i = 0; i < len; ++i) {
		if (isl_int_is_zero(p[i]))
			continue;
		hash *= 16777619;
		hash ^= (i & 0xFF);
		hash = isl_int_hash(p[i], hash);
	}
	return hash;
}

uint32_t isl_seq_get_hash(isl_int *p, unsigned len)
{
	uint32_t hash = isl_hash_init();

	return isl_seq_hash(p, len, hash);
}

uint32_t isl_seq_get_hash_bits(isl_int *p, unsigned len, unsigned bits)
{
	uint32_t hash;

	hash = isl_seq_get_hash(p, len);
	return isl_hash_bits(hash, bits);
}

void isl_seq_dump(isl_int *p, unsigned len)
{
	int i;

	for (i = 0; i < len; ++i) {
		if (i)
			fprintf(stderr, " ");
		isl_int_print(stderr, p[i], 0);
	}
	fprintf(stderr, "\n");
}
