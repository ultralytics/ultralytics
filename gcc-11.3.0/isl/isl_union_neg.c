/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <isl_union_macro.h>

/* Return the opposite of "part".
 */
static __isl_give PART *FN(UNION,neg_entry)(__isl_take PART *part, void *user)
{
	return FN(PART,neg)(part);
}

/* Return the opposite of "u".
 */
__isl_give UNION *FN(UNION,neg)(__isl_take UNION *u)
{
	return FN(UNION,transform_inplace)(u, &FN(UNION,neg_entry), NULL);
}
