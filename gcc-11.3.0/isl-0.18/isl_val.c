/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_int.h>
#include <isl_ctx_private.h>
#include <isl_val_private.h>

#undef BASE
#define BASE val

#include <isl_list_templ.c>

/* Allocate an isl_val object with indeterminate value.
 */
__isl_give isl_val *isl_val_alloc(isl_ctx *ctx)
{
	isl_val *v;

	v = isl_alloc_type(ctx, struct isl_val);
	if (!v)
		return NULL;

	v->ctx = ctx;
	isl_ctx_ref(ctx);
	v->ref = 1;
	isl_int_init(v->n);
	isl_int_init(v->d);

	return v;
}

/* Return a reference to an isl_val representing zero.
 */
__isl_give isl_val *isl_val_zero(isl_ctx *ctx)
{
	return isl_val_int_from_si(ctx, 0);
}

/* Return a reference to an isl_val representing one.
 */
__isl_give isl_val *isl_val_one(isl_ctx *ctx)
{
	return isl_val_int_from_si(ctx, 1);
}

/* Return a reference to an isl_val representing negative one.
 */
__isl_give isl_val *isl_val_negone(isl_ctx *ctx)
{
	return isl_val_int_from_si(ctx, -1);
}

/* Return a reference to an isl_val representing NaN.
 */
__isl_give isl_val *isl_val_nan(isl_ctx *ctx)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set_si(v->n, 0);
	isl_int_set_si(v->d, 0);

	return v;
}

/* Change "v" into a NaN.
 */
__isl_give isl_val *isl_val_set_nan(__isl_take isl_val *v)
{
	if (!v)
		return NULL;
	if (isl_val_is_nan(v))
		return v;
	v = isl_val_cow(v);
	if (!v)
		return NULL;

	isl_int_set_si(v->n, 0);
	isl_int_set_si(v->d, 0);

	return v;
}

/* Return a reference to an isl_val representing +infinity.
 */
__isl_give isl_val *isl_val_infty(isl_ctx *ctx)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set_si(v->n, 1);
	isl_int_set_si(v->d, 0);

	return v;
}

/* Return a reference to an isl_val representing -infinity.
 */
__isl_give isl_val *isl_val_neginfty(isl_ctx *ctx)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set_si(v->n, -1);
	isl_int_set_si(v->d, 0);

	return v;
}

/* Return a reference to an isl_val representing the integer "i".
 */
__isl_give isl_val *isl_val_int_from_si(isl_ctx *ctx, long i)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set_si(v->n, i);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Change the value of "v" to be equal to the integer "i".
 */
__isl_give isl_val *isl_val_set_si(__isl_take isl_val *v, long i)
{
	if (!v)
		return NULL;
	if (isl_val_is_int(v) && isl_int_cmp_si(v->n, i) == 0)
		return v;
	v = isl_val_cow(v);
	if (!v)
		return NULL;

	isl_int_set_si(v->n, i);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Change the value of "v" to be equal to zero.
 */
__isl_give isl_val *isl_val_set_zero(__isl_take isl_val *v)
{
	return isl_val_set_si(v, 0);
}

/* Return a reference to an isl_val representing the unsigned integer "u".
 */
__isl_give isl_val *isl_val_int_from_ui(isl_ctx *ctx, unsigned long u)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set_ui(v->n, u);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Return a reference to an isl_val representing the integer "n".
 */
__isl_give isl_val *isl_val_int_from_isl_int(isl_ctx *ctx, isl_int n)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set(v->n, n);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Return a reference to an isl_val representing the rational value "n"/"d".
 * Normalizing the isl_val (if needed) is left to the caller.
 */
__isl_give isl_val *isl_val_rat_from_isl_int(isl_ctx *ctx,
	isl_int n, isl_int d)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set(v->n, n);
	isl_int_set(v->d, d);

	return v;
}

/* Return a new reference to "v".
 */
__isl_give isl_val *isl_val_copy(__isl_keep isl_val *v)
{
	if (!v)
		return NULL;

	v->ref++;
	return v;
}

/* Return a fresh copy of "val".
 */
__isl_give isl_val *isl_val_dup(__isl_keep isl_val *val)
{
	isl_val *dup;

	if (!val)
		return NULL;

	dup = isl_val_alloc(isl_val_get_ctx(val));
	if (!dup)
		return NULL;

	isl_int_set(dup->n, val->n);
	isl_int_set(dup->d, val->d);

	return dup;
}

/* Return an isl_val that is equal to "val" and that has only
 * a single reference.
 */
__isl_give isl_val *isl_val_cow(__isl_take isl_val *val)
{
	if (!val)
		return NULL;

	if (val->ref == 1)
		return val;
	val->ref--;
	return isl_val_dup(val);
}

/* Free "v" and return NULL.
 */
__isl_null isl_val *isl_val_free(__isl_take isl_val *v)
{
	if (!v)
		return NULL;

	if (--v->ref > 0)
		return NULL;

	isl_ctx_deref(v->ctx);
	isl_int_clear(v->n);
	isl_int_clear(v->d);
	free(v);
	return NULL;
}

/* Extract the numerator of a rational value "v" as an integer.
 *
 * If "v" is not a rational value, then the result is undefined.
 */
long isl_val_get_num_si(__isl_keep isl_val *v)
{
	if (!v)
		return 0;
	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return 0);
	if (!isl_int_fits_slong(v->n))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"numerator too large", return 0);
	return isl_int_get_si(v->n);
}

/* Extract the numerator of a rational value "v" as an isl_int.
 *
 * If "v" is not a rational value, then the result is undefined.
 */
int isl_val_get_num_isl_int(__isl_keep isl_val *v, isl_int *n)
{
	if (!v)
		return -1;
	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return -1);
	isl_int_set(*n, v->n);
	return 0;
}

/* Extract the denominator of a rational value "v" as an integer.
 *
 * If "v" is not a rational value, then the result is undefined.
 */
long isl_val_get_den_si(__isl_keep isl_val *v)
{
	if (!v)
		return 0;
	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return 0);
	if (!isl_int_fits_slong(v->d))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"denominator too large", return 0);
	return isl_int_get_si(v->d);
}

/* Extract the denominator of a rational value "v" as an isl_val.
 *
 * If "v" is not a rational value, then the result is undefined.
 */
__isl_give isl_val *isl_val_get_den_val(__isl_keep isl_val *v)
{
	if (!v)
		return NULL;
	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return NULL);
	return isl_val_int_from_isl_int(isl_val_get_ctx(v), v->d);
}

/* Return an approximation of "v" as a double.
 */
double isl_val_get_d(__isl_keep isl_val *v)
{
	if (!v)
		return 0;
	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return 0);
	return isl_int_get_d(v->n) / isl_int_get_d(v->d);
}

/* Return the isl_ctx to which "val" belongs.
 */
isl_ctx *isl_val_get_ctx(__isl_keep isl_val *val)
{
	return val ? val->ctx : NULL;
}

/* Return a hash value that digests "val".
 */
uint32_t isl_val_get_hash(__isl_keep isl_val *val)
{
	uint32_t hash;

	if (!val)
		return 0;

	hash = isl_hash_init();
	hash = isl_int_hash(val->n, hash);
	hash = isl_int_hash(val->d, hash);

	return hash;
}

/* Normalize "v".
 *
 * In particular, make sure that the denominator of a rational value
 * is positive and the numerator and denominator do not have any
 * common divisors.
 *
 * This function should not be called by an external user
 * since it will only be given normalized values.
 */
__isl_give isl_val *isl_val_normalize(__isl_take isl_val *v)
{
	isl_ctx *ctx;

	if (!v)
		return NULL;
	if (isl_val_is_int(v))
		return v;
	if (!isl_val_is_rat(v))
		return v;
	if (isl_int_is_neg(v->d)) {
		isl_int_neg(v->d, v->d);
		isl_int_neg(v->n, v->n);
	}
	ctx = isl_val_get_ctx(v);
	isl_int_gcd(ctx->normalize_gcd, v->n, v->d);
	if (isl_int_is_one(ctx->normalize_gcd))
		return v;
	isl_int_divexact(v->n, v->n, ctx->normalize_gcd);
	isl_int_divexact(v->d, v->d, ctx->normalize_gcd);
	return v;
}

/* Return the opposite of "v".
 */
__isl_give isl_val *isl_val_neg(__isl_take isl_val *v)
{
	if (!v)
		return NULL;
	if (isl_val_is_nan(v))
		return v;
	if (isl_val_is_zero(v))
		return v;

	v = isl_val_cow(v);
	if (!v)
		return NULL;
	isl_int_neg(v->n, v->n);

	return v;
}

/* Return the inverse of "v".
 */
__isl_give isl_val *isl_val_inv(__isl_take isl_val *v)
{
	if (!v)
		return NULL;
	if (isl_val_is_nan(v))
		return v;
	if (isl_val_is_zero(v)) {
		isl_ctx *ctx = isl_val_get_ctx(v);
		isl_val_free(v);
		return isl_val_nan(ctx);
	}
	if (isl_val_is_infty(v) || isl_val_is_neginfty(v)) {
		isl_ctx *ctx = isl_val_get_ctx(v);
		isl_val_free(v);
		return isl_val_zero(ctx);
	}

	v = isl_val_cow(v);
	if (!v)
		return NULL;
	isl_int_swap(v->n, v->d);

	return isl_val_normalize(v);
}

/* Return the absolute value of "v".
 */
__isl_give isl_val *isl_val_abs(__isl_take isl_val *v)
{
	if (!v)
		return NULL;
	if (isl_val_is_nan(v))
		return v;
	if (isl_val_is_nonneg(v))
		return v;
	return isl_val_neg(v);
}

/* Return the "floor" (greatest integer part) of "v".
 * That is, return the result of rounding towards -infinity.
 */
__isl_give isl_val *isl_val_floor(__isl_take isl_val *v)
{
	if (!v)
		return NULL;
	if (isl_val_is_int(v))
		return v;
	if (!isl_val_is_rat(v))
		return v;

	v = isl_val_cow(v);
	if (!v)
		return NULL;
	isl_int_fdiv_q(v->n, v->n, v->d);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Return the "ceiling" of "v".
 * That is, return the result of rounding towards +infinity.
 */
__isl_give isl_val *isl_val_ceil(__isl_take isl_val *v)
{
	if (!v)
		return NULL;
	if (isl_val_is_int(v))
		return v;
	if (!isl_val_is_rat(v))
		return v;

	v = isl_val_cow(v);
	if (!v)
		return NULL;
	isl_int_cdiv_q(v->n, v->n, v->d);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Truncate "v".
 * That is, return the result of rounding towards zero.
 */
__isl_give isl_val *isl_val_trunc(__isl_take isl_val *v)
{
	if (!v)
		return NULL;
	if (isl_val_is_int(v))
		return v;
	if (!isl_val_is_rat(v))
		return v;

	v = isl_val_cow(v);
	if (!v)
		return NULL;
	isl_int_tdiv_q(v->n, v->n, v->d);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Return 2^v, where v is an integer (that is not too large).
 */
__isl_give isl_val *isl_val_2exp(__isl_take isl_val *v)
{
	unsigned long exp;
	int neg;

	v = isl_val_cow(v);
	if (!v)
		return NULL;
	if (!isl_val_is_int(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"can only compute integer powers",
			return isl_val_free(v));
	neg = isl_val_is_neg(v);
	if (neg)
		isl_int_neg(v->n, v->n);
	if (!isl_int_fits_ulong(v->n))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"exponent too large", return isl_val_free(v));
	exp = isl_int_get_ui(v->n);
	if (neg) {
		isl_int_mul_2exp(v->d, v->d, exp);
		isl_int_set_si(v->n, 1);
	} else {
		isl_int_mul_2exp(v->n, v->d, exp);
	}

	return v;
}

/* Return the minimum of "v1" and "v2".
 */
__isl_give isl_val *isl_val_min(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;

	if (isl_val_is_nan(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_nan(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if (isl_val_le(v1, v2)) {
		isl_val_free(v2);
		return v1;
	} else {
		isl_val_free(v1);
		return v2;
	}
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Return the maximum of "v1" and "v2".
 */
__isl_give isl_val *isl_val_max(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;

	if (isl_val_is_nan(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_nan(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if (isl_val_ge(v1, v2)) {
		isl_val_free(v2);
		return v1;
	} else {
		isl_val_free(v1);
		return v2;
	}
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Return the sum of "v1" and "v2".
 */
__isl_give isl_val *isl_val_add(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;
	if (isl_val_is_nan(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_nan(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if ((isl_val_is_infty(v1) && isl_val_is_neginfty(v2)) ||
	    (isl_val_is_neginfty(v1) && isl_val_is_infty(v2))) {
		isl_val_free(v2);
		return isl_val_set_nan(v1);
	}
	if (isl_val_is_infty(v1) || isl_val_is_neginfty(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_infty(v2) || isl_val_is_neginfty(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if (isl_val_is_zero(v1)) {
		isl_val_free(v1);
		return v2;
	}
	if (isl_val_is_zero(v2)) {
		isl_val_free(v2);
		return v1;
	}

	v1 = isl_val_cow(v1);
	if (!v1)
		goto error;
	if (isl_val_is_int(v1) && isl_val_is_int(v2))
		isl_int_add(v1->n, v1->n, v2->n);
	else {
		if (isl_int_eq(v1->d, v2->d))
			isl_int_add(v1->n, v1->n, v2->n);
		else {
			isl_int_mul(v1->n, v1->n, v2->d);
			isl_int_addmul(v1->n, v2->n, v1->d);
			isl_int_mul(v1->d, v1->d, v2->d);
		}
		v1 = isl_val_normalize(v1);
	}
	isl_val_free(v2);
	return v1;
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Return the sum of "v1" and "v2".
 */
__isl_give isl_val *isl_val_add_ui(__isl_take isl_val *v1, unsigned long v2)
{
	if (!v1)
		return NULL;
	if (!isl_val_is_rat(v1))
		return v1;
	if (v2 == 0)
		return v1;
	v1 = isl_val_cow(v1);
	if (!v1)
		return NULL;

	isl_int_addmul_ui(v1->n, v1->d, v2);

	return v1;
}

/* Subtract "v2" from "v1".
 */
__isl_give isl_val *isl_val_sub(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;
	if (isl_val_is_nan(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_nan(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if ((isl_val_is_infty(v1) && isl_val_is_infty(v2)) ||
	    (isl_val_is_neginfty(v1) && isl_val_is_neginfty(v2))) {
		isl_val_free(v2);
		return isl_val_set_nan(v1);
	}
	if (isl_val_is_infty(v1) || isl_val_is_neginfty(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_infty(v2) || isl_val_is_neginfty(v2)) {
		isl_val_free(v1);
		return isl_val_neg(v2);
	}
	if (isl_val_is_zero(v2)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_zero(v1)) {
		isl_val_free(v1);
		return isl_val_neg(v2);
	}

	v1 = isl_val_cow(v1);
	if (!v1)
		goto error;
	if (isl_val_is_int(v1) && isl_val_is_int(v2))
		isl_int_sub(v1->n, v1->n, v2->n);
	else {
		if (isl_int_eq(v1->d, v2->d))
			isl_int_sub(v1->n, v1->n, v2->n);
		else {
			isl_int_mul(v1->n, v1->n, v2->d);
			isl_int_submul(v1->n, v2->n, v1->d);
			isl_int_mul(v1->d, v1->d, v2->d);
		}
		v1 = isl_val_normalize(v1);
	}
	isl_val_free(v2);
	return v1;
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Subtract "v2" from "v1".
 */
__isl_give isl_val *isl_val_sub_ui(__isl_take isl_val *v1, unsigned long v2)
{
	if (!v1)
		return NULL;
	if (!isl_val_is_rat(v1))
		return v1;
	if (v2 == 0)
		return v1;
	v1 = isl_val_cow(v1);
	if (!v1)
		return NULL;

	isl_int_submul_ui(v1->n, v1->d, v2);

	return v1;
}

/* Return the product of "v1" and "v2".
 */
__isl_give isl_val *isl_val_mul(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;
	if (isl_val_is_nan(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_nan(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if ((!isl_val_is_rat(v1) && isl_val_is_zero(v2)) ||
	    (isl_val_is_zero(v1) && !isl_val_is_rat(v2))) {
		isl_val_free(v2);
		return isl_val_set_nan(v1);
	}
	if (isl_val_is_zero(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_zero(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if (isl_val_is_infty(v1) || isl_val_is_neginfty(v1)) {
		if (isl_val_is_neg(v2))
			v1 = isl_val_neg(v1);
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_infty(v2) || isl_val_is_neginfty(v2)) {
		if (isl_val_is_neg(v1))
			v2 = isl_val_neg(v2);
		isl_val_free(v1);
		return v2;
	}

	v1 = isl_val_cow(v1);
	if (!v1)
		goto error;
	if (isl_val_is_int(v1) && isl_val_is_int(v2))
		isl_int_mul(v1->n, v1->n, v2->n);
	else {
		isl_int_mul(v1->n, v1->n, v2->n);
		isl_int_mul(v1->d, v1->d, v2->d);
		v1 = isl_val_normalize(v1);
	}
	isl_val_free(v2);
	return v1;
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Return the product of "v1" and "v2".
 *
 * This is a private copy of isl_val_mul for use in the generic
 * isl_multi_*_scale_val instantiated for isl_val.
 */
__isl_give isl_val *isl_val_scale_val(__isl_take isl_val *v1,
	__isl_take isl_val *v2)
{
	return isl_val_mul(v1, v2);
}

/* Return the product of "v1" and "v2".
 */
__isl_give isl_val *isl_val_mul_ui(__isl_take isl_val *v1, unsigned long v2)
{
	if (!v1)
		return NULL;
	if (isl_val_is_nan(v1))
		return v1;
	if (!isl_val_is_rat(v1)) {
		if (v2 == 0)
			v1 = isl_val_set_nan(v1);
		return v1;
	}
	if (v2 == 1)
		return v1;
	v1 = isl_val_cow(v1);
	if (!v1)
		return NULL;

	isl_int_mul_ui(v1->n, v1->n, v2);

	return isl_val_normalize(v1);
}

/* Divide "v1" by "v2".
 */
__isl_give isl_val *isl_val_div(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;
	if (isl_val_is_nan(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_nan(v2)) {
		isl_val_free(v1);
		return v2;
	}
	if (isl_val_is_zero(v2) ||
	    (!isl_val_is_rat(v1) && !isl_val_is_rat(v2))) {
		isl_val_free(v2);
		return isl_val_set_nan(v1);
	}
	if (isl_val_is_zero(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_infty(v1) || isl_val_is_neginfty(v1)) {
		if (isl_val_is_neg(v2))
			v1 = isl_val_neg(v1);
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_infty(v2) || isl_val_is_neginfty(v2)) {
		isl_val_free(v2);
		return isl_val_set_zero(v1);
	}

	v1 = isl_val_cow(v1);
	if (!v1)
		goto error;
	if (isl_val_is_int(v2)) {
		isl_int_mul(v1->d, v1->d, v2->n);
		v1 = isl_val_normalize(v1);
	} else {
		isl_int_mul(v1->d, v1->d, v2->n);
		isl_int_mul(v1->n, v1->n, v2->d);
		v1 = isl_val_normalize(v1);
	}
	isl_val_free(v2);
	return v1;
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Divide "v1" by "v2".
 *
 * This is a private copy of isl_val_div for use in the generic
 * isl_multi_*_scale_down_val instantiated for isl_val.
 */
__isl_give isl_val *isl_val_scale_down_val(__isl_take isl_val *v1,
	__isl_take isl_val *v2)
{
	return isl_val_div(v1, v2);
}

/* Given two integer values "v1" and "v2", check if "v1" is divisible by "v2".
 */
isl_bool isl_val_is_divisible_by(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	if (!v1 || !v2)
		return isl_bool_error;

	if (!isl_val_is_int(v1) || !isl_val_is_int(v2))
		isl_die(isl_val_get_ctx(v1), isl_error_invalid,
			"expecting two integers", return isl_bool_error);

	return isl_int_is_divisible_by(v1->n, v2->n);
}

/* Given two integer values "v1" and "v2", return the residue of "v1"
 * modulo "v2".
 */
__isl_give isl_val *isl_val_mod(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;
	if (!isl_val_is_int(v1) || !isl_val_is_int(v2))
		isl_die(isl_val_get_ctx(v1), isl_error_invalid,
			"expecting two integers", goto error);
	if (isl_val_is_nonneg(v1) && isl_val_lt(v1, v2)) {
		isl_val_free(v2);
		return v1;
	}
	v1 = isl_val_cow(v1);
	if (!v1)
		goto error;
	isl_int_fdiv_r(v1->n, v1->n, v2->n);
	isl_val_free(v2);
	return v1;
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Given two integer values "v1" and "v2", return the residue of "v1"
 * modulo "v2".
 *
 * This is a private copy of isl_val_mod for use in the generic
 * isl_multi_*_mod_multi_val instantiated for isl_val.
 */
__isl_give isl_val *isl_val_mod_val(__isl_take isl_val *v1,
	__isl_take isl_val *v2)
{
	return isl_val_mod(v1, v2);
}

/* Given two integer values, return their greatest common divisor.
 */
__isl_give isl_val *isl_val_gcd(__isl_take isl_val *v1, __isl_take isl_val *v2)
{
	if (!v1 || !v2)
		goto error;
	if (!isl_val_is_int(v1) || !isl_val_is_int(v2))
		isl_die(isl_val_get_ctx(v1), isl_error_invalid,
			"expecting two integers", goto error);
	if (isl_val_eq(v1, v2)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_one(v1)) {
		isl_val_free(v2);
		return v1;
	}
	if (isl_val_is_one(v2)) {
		isl_val_free(v1);
		return v2;
	}
	v1 = isl_val_cow(v1);
	if (!v1)
		goto error;
	isl_int_gcd(v1->n, v1->n, v2->n);
	isl_val_free(v2);
	return v1;
error:
	isl_val_free(v1);
	isl_val_free(v2);
	return NULL;
}

/* Compute x, y and g such that g = gcd(a,b) and a*x+b*y = g.
 */
static void isl_int_gcdext(isl_int *g, isl_int *x, isl_int *y,
	isl_int a, isl_int b)
{
	isl_int d, tmp;
	isl_int a_copy, b_copy;

	isl_int_init(a_copy);
	isl_int_init(b_copy);
	isl_int_init(d);
	isl_int_init(tmp);
	isl_int_set(a_copy, a);
	isl_int_set(b_copy, b);
	isl_int_abs(*g, a_copy);
	isl_int_abs(d, b_copy);
	isl_int_set_si(*x, 1);
	isl_int_set_si(*y, 0);
	while (isl_int_is_pos(d)) {
		isl_int_fdiv_q(tmp, *g, d);
		isl_int_submul(*x, tmp, *y);
		isl_int_submul(*g, tmp, d);
		isl_int_swap(*g, d);
		isl_int_swap(*x, *y);
	}
	if (isl_int_is_zero(a_copy))
		isl_int_set_si(*x, 0);
	else if (isl_int_is_neg(a_copy))
		isl_int_neg(*x, *x);
	if (isl_int_is_zero(b_copy))
		isl_int_set_si(*y, 0);
	else {
		isl_int_mul(tmp, a_copy, *x);
		isl_int_sub(tmp, *g, tmp);
		isl_int_divexact(*y, tmp, b_copy);
	}
	isl_int_clear(d);
	isl_int_clear(tmp);
	isl_int_clear(a_copy);
	isl_int_clear(b_copy);
}

/* Given two integer values v1 and v2, return their greatest common divisor g,
 * as well as two integers x and y such that x * v1 + y * v2 = g.
 */
__isl_give isl_val *isl_val_gcdext(__isl_take isl_val *v1,
	__isl_take isl_val *v2, __isl_give isl_val **x, __isl_give isl_val **y)
{
	isl_ctx *ctx;
	isl_val *a = NULL, *b = NULL;

	if (!x && !y)
		return isl_val_gcd(v1, v2);

	if (!v1 || !v2)
		goto error;

	ctx = isl_val_get_ctx(v1);
	if (!isl_val_is_int(v1) || !isl_val_is_int(v2))
		isl_die(ctx, isl_error_invalid,
			"expecting two integers", goto error);

	v1 = isl_val_cow(v1);
	a = isl_val_alloc(ctx);
	b = isl_val_alloc(ctx);
	if (!v1 || !a || !b)
		goto error;
	isl_int_gcdext(&v1->n, &a->n, &b->n, v1->n, v2->n);
	if (x) {
		isl_int_set_si(a->d, 1);
		*x = a;
	} else
		isl_val_free(a);
	if (y) {
		isl_int_set_si(b->d, 1);
		*y = b;
	} else
		isl_val_free(b);
	isl_val_free(v2);
	return v1;
error:
	isl_val_free(v1);
	isl_val_free(v2);
	isl_val_free(a);
	isl_val_free(b);
	if (x)
		*x = NULL;
	if (y)
		*y = NULL;
	return NULL;
}

/* Does "v" represent an integer value?
 */
isl_bool isl_val_is_int(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_one(v->d);
}

/* Does "v" represent a rational value?
 */
isl_bool isl_val_is_rat(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return !isl_int_is_zero(v->d);
}

/* Does "v" represent NaN?
 */
isl_bool isl_val_is_nan(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_zero(v->n) && isl_int_is_zero(v->d);
}

/* Does "v" represent +infinity?
 */
isl_bool isl_val_is_infty(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_pos(v->n) && isl_int_is_zero(v->d);
}

/* Does "v" represent -infinity?
 */
isl_bool isl_val_is_neginfty(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_neg(v->n) && isl_int_is_zero(v->d);
}

/* Does "v" represent the integer zero?
 */
isl_bool isl_val_is_zero(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_zero(v->n) && !isl_int_is_zero(v->d);
}

/* Does "v" represent the integer one?
 */
isl_bool isl_val_is_one(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_eq(v->n, v->d);
}

/* Does "v" represent the integer negative one?
 */
isl_bool isl_val_is_negone(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_neg(v->n) && isl_int_abs_eq(v->n, v->d);
}

/* Is "v" (strictly) positive?
 */
isl_bool isl_val_is_pos(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_pos(v->n);
}

/* Is "v" (strictly) negative?
 */
isl_bool isl_val_is_neg(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	return isl_int_is_neg(v->n);
}

/* Is "v" non-negative?
 */
isl_bool isl_val_is_nonneg(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	if (isl_val_is_nan(v))
		return isl_bool_false;

	return isl_int_is_nonneg(v->n);
}

/* Is "v" non-positive?
 */
isl_bool isl_val_is_nonpos(__isl_keep isl_val *v)
{
	if (!v)
		return isl_bool_error;

	if (isl_val_is_nan(v))
		return isl_bool_false;

	return isl_int_is_nonpos(v->n);
}

/* Return the sign of "v".
 *
 * The sign of NaN is undefined.
 */
int isl_val_sgn(__isl_keep isl_val *v)
{
	if (!v)
		return 0;
	if (isl_val_is_zero(v))
		return 0;
	if (isl_val_is_pos(v))
		return 1;
	return -1;
}

/* Is "v1" (strictly) less than "v2"?
 */
isl_bool isl_val_lt(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	isl_int t;
	isl_bool lt;

	if (!v1 || !v2)
		return isl_bool_error;
	if (isl_val_is_int(v1) && isl_val_is_int(v2))
		return isl_int_lt(v1->n, v2->n);
	if (isl_val_is_nan(v1) || isl_val_is_nan(v2))
		return isl_bool_false;
	if (isl_val_eq(v1, v2))
		return isl_bool_false;
	if (isl_val_is_infty(v2))
		return isl_bool_true;
	if (isl_val_is_infty(v1))
		return isl_bool_false;
	if (isl_val_is_neginfty(v1))
		return isl_bool_true;
	if (isl_val_is_neginfty(v2))
		return isl_bool_false;

	isl_int_init(t);
	isl_int_mul(t, v1->n, v2->d);
	isl_int_submul(t, v2->n, v1->d);
	lt = isl_int_is_neg(t);
	isl_int_clear(t);

	return lt;
}

/* Is "v1" (strictly) greater than "v2"?
 */
isl_bool isl_val_gt(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	return isl_val_lt(v2, v1);
}

/* Is "v1" less than or equal to "v2"?
 */
isl_bool isl_val_le(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	isl_int t;
	isl_bool le;

	if (!v1 || !v2)
		return isl_bool_error;
	if (isl_val_is_int(v1) && isl_val_is_int(v2))
		return isl_int_le(v1->n, v2->n);
	if (isl_val_is_nan(v1) || isl_val_is_nan(v2))
		return isl_bool_false;
	if (isl_val_eq(v1, v2))
		return isl_bool_true;
	if (isl_val_is_infty(v2))
		return isl_bool_true;
	if (isl_val_is_infty(v1))
		return isl_bool_false;
	if (isl_val_is_neginfty(v1))
		return isl_bool_true;
	if (isl_val_is_neginfty(v2))
		return isl_bool_false;

	isl_int_init(t);
	isl_int_mul(t, v1->n, v2->d);
	isl_int_submul(t, v2->n, v1->d);
	le = isl_int_is_nonpos(t);
	isl_int_clear(t);

	return le;
}

/* Is "v1" greater than or equal to "v2"?
 */
isl_bool isl_val_ge(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	return isl_val_le(v2, v1);
}

/* How does "v" compare to "i"?
 *
 * Return 1 if v is greater, -1 if v is smaller and 0 if v is equal to i.
 *
 * If v is NaN (or NULL), then the result is undefined.
 */
int isl_val_cmp_si(__isl_keep isl_val *v, long i)
{
	isl_int t;
	int cmp;

	if (!v)
		return 0;
	if (isl_val_is_int(v))
		return isl_int_cmp_si(v->n, i);
	if (isl_val_is_nan(v))
		return 0;
	if (isl_val_is_infty(v))
		return 1;
	if (isl_val_is_neginfty(v))
		return -1;

	isl_int_init(t);
	isl_int_mul_si(t, v->d, i);
	isl_int_sub(t, v->n, t);
	cmp = isl_int_sgn(t);
	isl_int_clear(t);

	return cmp;
}

/* Is "v1" equal to "v2"?
 */
isl_bool isl_val_eq(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	if (!v1 || !v2)
		return isl_bool_error;
	if (isl_val_is_nan(v1) || isl_val_is_nan(v2))
		return isl_bool_false;

	return isl_int_eq(v1->n, v2->n) && isl_int_eq(v1->d, v2->d);
}

/* Is "v1" equal to "v2" in absolute value?
 */
isl_bool isl_val_abs_eq(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	if (!v1 || !v2)
		return isl_bool_error;
	if (isl_val_is_nan(v1) || isl_val_is_nan(v2))
		return isl_bool_false;

	return isl_int_abs_eq(v1->n, v2->n) && isl_int_eq(v1->d, v2->d);
}

/* Is "v1" different from "v2"?
 */
isl_bool isl_val_ne(__isl_keep isl_val *v1, __isl_keep isl_val *v2)
{
	if (!v1 || !v2)
		return isl_bool_error;
	if (isl_val_is_nan(v1) || isl_val_is_nan(v2))
		return isl_bool_false;

	return isl_int_ne(v1->n, v2->n) || isl_int_ne(v1->d, v2->d);
}

/* Print a textual representation of "v" onto "p".
 */
__isl_give isl_printer *isl_printer_print_val(__isl_take isl_printer *p,
	__isl_keep isl_val *v)
{
	int neg;

	if (!p || !v)
		return isl_printer_free(p);

	neg = isl_int_is_neg(v->n);
	if (neg) {
		p = isl_printer_print_str(p, "-");
		isl_int_neg(v->n, v->n);
	}
	if (isl_int_is_zero(v->d)) {
		int sgn = isl_int_sgn(v->n);
		p = isl_printer_print_str(p, sgn < 0 ? "-infty" :
					    sgn == 0 ? "NaN" : "infty");
	} else
		p = isl_printer_print_isl_int(p, v->n);
	if (neg)
		isl_int_neg(v->n, v->n);
	if (!isl_int_is_zero(v->d) && !isl_int_is_one(v->d)) {
		p = isl_printer_print_str(p, "/");
		p = isl_printer_print_isl_int(p, v->d);
	}

	return p;
}

/* Is "val1" (obviously) equal to "val2"?
 *
 * This is a private copy of isl_val_eq for use in the generic
 * isl_multi_*_plain_is_equal instantiated for isl_val.
 */
int isl_val_plain_is_equal(__isl_keep isl_val *val1, __isl_keep isl_val *val2)
{
	return isl_val_eq(val1, val2);
}

/* Does "v" have any non-zero coefficients
 * for any dimension in the given range?
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have any coefficients, this function
 * always return 0.
 */
int isl_val_involves_dims(__isl_keep isl_val *v, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	if (!v)
		return -1;

	return 0;
}

/* Insert "n" dimensions of type "type" at position "first".
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * does not do anything.
 */
__isl_give isl_val *isl_val_insert_dims(__isl_take isl_val *v,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return v;
}

/* Drop the the "n" first dimensions of type "type" at position "first".
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * does not do anything.
 */
__isl_give isl_val *isl_val_drop_dims(__isl_take isl_val *v,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return v;
}

/* Change the name of the dimension of type "type" at position "pos" to "s".
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * does not do anything.
 */
__isl_give isl_val *isl_val_set_dim_name(__isl_take isl_val *v,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	return v;
}

/* Return the space of "v".
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  The conditions surrounding the call to this function make sure
 * that this function will never actually get called.  We return a valid
 * space anyway, just in case.
 */
__isl_give isl_space *isl_val_get_space(__isl_keep isl_val *v)
{
	if (!v)
		return NULL;

	return isl_space_params_alloc(isl_val_get_ctx(v), 0);
}

/* Reset the domain space of "v" to "space".
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * does not do anything, apart from error handling and cleaning up memory.
 */
__isl_give isl_val *isl_val_reset_domain_space(__isl_take isl_val *v,
	__isl_take isl_space *space)
{
	if (!space)
		return isl_val_free(v);
	isl_space_free(space);
	return v;
}

/* Align the parameters of "v" to those of "space".
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * does not do anything, apart from error handling and cleaning up memory.
 * Note that the conditions surrounding the call to this function make sure
 * that this function will never actually get called.
 */
__isl_give isl_val *isl_val_align_params(__isl_take isl_val *v,
	__isl_take isl_space *space)
{
	if (!space)
		return isl_val_free(v);
	isl_space_free(space);
	return v;
}

/* Reorder the dimensions of the domain of "v" according
 * to the given reordering.
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * does not do anything, apart from error handling and cleaning up memory.
 */
__isl_give isl_val *isl_val_realign_domain(__isl_take isl_val *v,
	__isl_take isl_reordering *r)
{
	if (!r)
		return isl_val_free(v);
	isl_reordering_free(r);
	return v;
}

/* Return an isl_val that is zero on "ls".
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * simply returns a zero isl_val in the same context as "ls".
 */
__isl_give isl_val *isl_val_zero_on_domain(__isl_take isl_local_space *ls)
{
	isl_ctx *ctx;

	if (!ls)
		return NULL;
	ctx = isl_local_space_get_ctx(ls);
	isl_local_space_free(ls);
	return isl_val_zero(ctx);
}

/* Do the parameters of "v" match those of "space"?
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * simply returns 1, except if "v" or "space" are NULL.
 */
int isl_val_matching_params(__isl_keep isl_val *v, __isl_keep isl_space *space)
{
	if (!v || !space)
		return -1;
	return 1;
}

/* Check that the domain space of "v" matches "space".
 *
 * Return 0 on success and -1 on error.
 *
 * This function is only meant to be used in the generic isl_multi_*
 * functions which have to deal with base objects that have an associated
 * space.  Since an isl_val does not have an associated space, this function
 * simply returns 0, except if "v" or "space" are NULL.
 */
int isl_val_check_match_domain_space(__isl_keep isl_val *v,
	__isl_keep isl_space *space)
{
	if (!v || !space)
		return -1;
	return 0;
}

#undef BASE
#define BASE val

#define NO_DOMAIN
#define NO_IDENTITY
#define NO_FROM_BASE
#define NO_MOVE_DIMS
#include <isl_multi_templ.c>

/* Apply "fn" to each of the elements of "mv" with as second argument "v".
 */
static __isl_give isl_multi_val *isl_multi_val_fn_val(
	__isl_take isl_multi_val *mv,
	__isl_give isl_val *(*fn)(__isl_take isl_val *v1,
					__isl_take isl_val *v2),
	__isl_take isl_val *v)
{
	int i;

	mv = isl_multi_val_cow(mv);
	if (!mv || !v)
		goto error;

	for (i = 0; i < mv->n; ++i) {
		mv->p[i] = fn(mv->p[i], isl_val_copy(v));
		if (!mv->p[i])
			goto error;
	}

	isl_val_free(v);
	return mv;
error:
	isl_val_free(v);
	isl_multi_val_free(mv);
	return NULL;
}

/* Add "v" to each of the elements of "mv".
 */
__isl_give isl_multi_val *isl_multi_val_add_val(__isl_take isl_multi_val *mv,
	__isl_take isl_val *v)
{
	if (!v)
		return isl_multi_val_free(mv);
	if (isl_val_is_zero(v)) {
		isl_val_free(v);
		return mv;
	}
	return isl_multi_val_fn_val(mv, &isl_val_add, v);
}

/* Reduce the elements of "mv" modulo "v".
 */
__isl_give isl_multi_val *isl_multi_val_mod_val(__isl_take isl_multi_val *mv,
	__isl_take isl_val *v)
{
	return isl_multi_val_fn_val(mv, &isl_val_mod, v);
}
