/*
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#include <isl_multi_macro.h>

/* Compare two multi expressions.
 *
 * Return -1 if "multi1" is "smaller" than "multi2", 1 if "multi1" is "greater"
 * than "multi2" and 0 if they are equal.
 */
int FN(MULTI(BASE),plain_cmp)(__isl_keep MULTI(BASE) *multi1,
	__isl_keep MULTI(BASE) *multi2)
{
	int i;
	int cmp;

	if (multi1 == multi2)
		return 0;
	if (!multi1)
		return -1;
	if (!multi2)
		return 1;

	cmp = isl_space_cmp(multi1->space, multi2->space);
	if (cmp != 0)
		return cmp;

	for (i = 0; i < multi1->n; ++i) {
		cmp = FN(EL,plain_cmp)(multi1->p[i], multi2->p[i]);
		if (cmp != 0)
			return cmp;
	}

	return 0;
}
