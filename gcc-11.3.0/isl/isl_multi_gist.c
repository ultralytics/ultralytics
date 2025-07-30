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

/* Compute the gist of "multi" with respect to the domain constraints
 * of "context".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),gist)(__isl_take MULTI(BASE) *multi,
	__isl_take DOM *context)
{
	return FN(FN(MULTI(BASE),apply),DOMBASE)(multi, context, &FN(EL,gist));
}

/* Compute the gist of "multi" with respect to the parameter constraints
 * of "context".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),gist_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *context)
{
	return FN(MULTI(BASE),apply_set)(multi, context, &FN(EL,gist_params));
}
