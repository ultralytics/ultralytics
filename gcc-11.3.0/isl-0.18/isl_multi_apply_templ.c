/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl_multi_macro.h>

/* Transform the elements of "multi" by applying "fn" to them
 * with extra argument "set".
 *
 * The parameters of "multi" and "set" are assumed to have been aligned.
 */
__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),apply_aligned),APPLY_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take APPLY_DOM *set,
	__isl_give EL *(*fn)(EL *el, __isl_take APPLY_DOM *set))
{
	int i;

	if (!multi || !set)
		goto error;

	if (multi->n == 0) {
		FN(APPLY_DOM,free)(set);
		return multi;
	}

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = fn(multi->p[i], FN(APPLY_DOM,copy)(set));
		if (!multi->p[i])
			goto error;
	}

	FN(APPLY_DOM,free)(set);
	return multi;
error:
	FN(APPLY_DOM,free)(set);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

/* Transform the elements of "multi" by applying "fn" to them
 * with extra argument "set".
 *
 * Align the parameters if needed and call apply_set_aligned.
 */
static __isl_give MULTI(BASE) *FN(FN(MULTI(BASE),apply),APPLY_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take APPLY_DOM *set,
	__isl_give EL *(*fn)(EL *el, __isl_take APPLY_DOM *set))
{
	isl_ctx *ctx;

	if (!multi || !set)
		goto error;

	if (isl_space_match(multi->space, isl_dim_param,
			    set->dim, isl_dim_param))
		return FN(FN(MULTI(BASE),apply_aligned),APPLY_DOMBASE)(multi,
								    set, fn);
	ctx = FN(MULTI(BASE),get_ctx)(multi);
	if (!isl_space_has_named_params(multi->space) ||
	    !isl_space_has_named_params(set->dim))
		isl_die(ctx, isl_error_invalid,
			"unaligned unnamed parameters", goto error);
	multi = FN(MULTI(BASE),align_params)(multi,
						FN(APPLY_DOM,get_space)(set));
	set = FN(APPLY_DOM,align_params)(set, FN(MULTI(BASE),get_space)(multi));
	return FN(FN(MULTI(BASE),apply_aligned),APPLY_DOMBASE)(multi, set, fn);
error:
	FN(MULTI(BASE),free)(multi);
	FN(APPLY_DOM,free)(set);
	return NULL;
}
