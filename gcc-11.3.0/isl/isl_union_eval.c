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

/* Is the domain space of "entry" equal to "space"?
 */
static int FN(UNION,has_domain_space)(const void *entry, const void *val)
{
	PART *part = (PART *)entry;
	isl_space *space = (isl_space *) val;

	if (isl_space_is_params(space))
		return isl_space_is_set(part->dim);

	return isl_space_tuple_is_equal(part->dim, isl_dim_in,
					space, isl_dim_set);
}

__isl_give isl_val *FN(UNION,eval)(__isl_take UNION *u,
	__isl_take isl_point *pnt)
{
	uint32_t hash;
	struct isl_hash_table_entry *entry;
	isl_space *space;
	isl_val *v;

	if (!u || !pnt)
		goto error;

	space = isl_space_copy(pnt->dim);
	if (!space)
		goto error;
	hash = isl_space_get_hash(space);
	entry = isl_hash_table_find(u->space->ctx, &u->table,
				    hash, &FN(UNION,has_domain_space),
				    space, 0);
	isl_space_free(space);
	if (!entry) {
		v = isl_val_zero(isl_point_get_ctx(pnt));
		isl_point_free(pnt);
	} else {
		v = FN(PART,eval)(FN(PART,copy)(entry->data), pnt);
	}
	FN(UNION,free)(u);
	return v;
error:
	FN(UNION,free)(u);
	isl_point_free(pnt);
	return NULL;
}
