#include <isl_val_private.h>

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Compute the given non-zero power of "map" and return the result.
 * If the exponent "exp" is negative, then the -exp th power of the inverse
 * relation is computed.
 */
__isl_give TYPE *FN(TYPE,fixed_power)(__isl_take TYPE *map, isl_int exp)
{
	isl_ctx *ctx;
	TYPE *res = NULL;
	isl_int r;

	if (!map)
		return NULL;

	ctx = FN(TYPE,get_ctx)(map);
	if (isl_int_is_zero(exp))
		isl_die(ctx, isl_error_invalid,
			"expecting non-zero exponent", goto error);

	if (isl_int_is_neg(exp)) {
		isl_int_neg(exp, exp);
		map = FN(TYPE,reverse)(map);
		return FN(TYPE,fixed_power)(map, exp);
	}

	isl_int_init(r);
	for (;;) {
		isl_int_fdiv_r(r, exp, ctx->two);

		if (!isl_int_is_zero(r)) {
			if (!res)
				res = FN(TYPE,copy)(map);
			else {
				res = FN(TYPE,apply_range)(res,
							  FN(TYPE,copy)(map));
				res = FN(TYPE,coalesce)(res);
			}
			if (!res)
				break;
		}

		isl_int_fdiv_q(exp, exp, ctx->two);
		if (isl_int_is_zero(exp))
			break;

		map = FN(TYPE,apply_range)(map, FN(TYPE,copy)(map));
		map = FN(TYPE,coalesce)(map);
	}
	isl_int_clear(r);

	FN(TYPE,free)(map);
	return res;
error:
	FN(TYPE,free)(map);
	return NULL;
}

/* Compute the given non-zero power of "map" and return the result.
 * If the exponent "exp" is negative, then the -exp th power of the inverse
 * relation is computed.
 */
__isl_give TYPE *FN(TYPE,fixed_power_val)(__isl_take TYPE *map,
	__isl_take isl_val *exp)
{
	if (!map || !exp)
		goto error;
	if (!isl_val_is_int(exp))
		isl_die(FN(TYPE,get_ctx)(map), isl_error_invalid,
			"expecting integer exponent", goto error);
	map = FN(TYPE,fixed_power)(map, exp->n);
	isl_val_free(exp);
	return map;
error:
	FN(TYPE,free)(map);
	isl_val_free(exp);
	return NULL;
}
