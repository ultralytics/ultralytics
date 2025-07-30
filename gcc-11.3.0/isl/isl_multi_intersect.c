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

/* Intersect the domain of "multi" with "domain".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take DOM *domain)
{
	return FN(FN(MULTI(BASE),apply),DOMBASE)(multi, domain,
					&FN(EL,intersect_domain));
}

/* Intersect the parameter domain of "multi" with "domain".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *domain)
{
	return FN(MULTI(BASE),apply_set)(multi, domain,
					&FN(EL,intersect_params));
}
