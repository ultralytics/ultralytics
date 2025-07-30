/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_multi_macro.h>

/* Coalesce the elements of "multi".
 *
 * Note that such coalescing does not change the meaning of "multi"
 * so there is no need to cow.  We do need to be careful not to
 * destroy any other copies of "multi" in case of failure.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),coalesce)(__isl_take MULTI(BASE) *multi)
{
	int i;

	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		EL *el = FN(EL,copy)(multi->p[i]);
		el = FN(EL,coalesce)(el);
		if (!el)
			return FN(MULTI(BASE),free)(multi);
		FN(EL,free)(multi->p[i]);
		multi->p[i] = el;
	}

	return multi;
}
