/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2013-2014 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <stdlib.h>
#include <string.h>
#include <isl_space_private.h>
#include <isl_id_private.h>
#include <isl_reordering.h>

isl_ctx *isl_space_get_ctx(__isl_keep isl_space *dim)
{
	return dim ? dim->ctx : NULL;
}

__isl_give isl_space *isl_space_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned n_in, unsigned n_out)
{
	isl_space *dim;

	dim = isl_alloc_type(ctx, struct isl_space);
	if (!dim)
		return NULL;

	dim->ctx = ctx;
	isl_ctx_ref(ctx);
	dim->ref = 1;
	dim->nparam = nparam;
	dim->n_in = n_in;
	dim->n_out = n_out;

	dim->tuple_id[0] = NULL;
	dim->tuple_id[1] = NULL;

	dim->nested[0] = NULL;
	dim->nested[1] = NULL;

	dim->n_id = 0;
	dim->ids = NULL;

	return dim;
}

/* Mark the space as being that of a set, by setting the domain tuple
 * to isl_id_none.
 */
static __isl_give isl_space *mark_as_set(__isl_take isl_space *space)
{
	space = isl_space_cow(space);
	if (!space)
		return NULL;
	space = isl_space_set_tuple_id(space, isl_dim_in, &isl_id_none);
	return space;
}

/* Is the space that of a set?
 */
isl_bool isl_space_is_set(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;
	if (space->n_in != 0 || space->nested[0])
		return isl_bool_false;
	if (space->tuple_id[0] != &isl_id_none)
		return isl_bool_false;
	return isl_bool_true;
}

/* Is the given space that of a map?
 */
isl_bool isl_space_is_map(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;
	return space->tuple_id[0] != &isl_id_none &&
		space->tuple_id[1] != &isl_id_none;
}

__isl_give isl_space *isl_space_set_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned dim)
{
	isl_space *space;
	space = isl_space_alloc(ctx, nparam, 0, dim);
	space = mark_as_set(space);
	return space;
}

/* Mark the space as being that of a parameter domain, by setting
 * both tuples to isl_id_none.
 */
static __isl_give isl_space *mark_as_params(isl_space *space)
{
	if (!space)
		return NULL;
	space = isl_space_set_tuple_id(space, isl_dim_in, &isl_id_none);
	space = isl_space_set_tuple_id(space, isl_dim_out, &isl_id_none);
	return space;
}

/* Is the space that of a parameter domain?
 */
isl_bool isl_space_is_params(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;
	if (space->n_in != 0 || space->nested[0] ||
	    space->n_out != 0 || space->nested[1])
		return isl_bool_false;
	if (space->tuple_id[0] != &isl_id_none)
		return isl_bool_false;
	if (space->tuple_id[1] != &isl_id_none)
		return isl_bool_false;
	return isl_bool_true;
}

/* Create a space for a parameter domain.
 */
__isl_give isl_space *isl_space_params_alloc(isl_ctx *ctx, unsigned nparam)
{
	isl_space *space;
	space = isl_space_alloc(ctx, nparam, 0, 0);
	space = mark_as_params(space);
	return space;
}

static unsigned global_pos(__isl_keep isl_space *dim,
				 enum isl_dim_type type, unsigned pos)
{
	struct isl_ctx *ctx = dim->ctx;

	switch (type) {
	case isl_dim_param:
		isl_assert(ctx, pos < dim->nparam,
			    return isl_space_dim(dim, isl_dim_all));
		return pos;
	case isl_dim_in:
		isl_assert(ctx, pos < dim->n_in,
			    return isl_space_dim(dim, isl_dim_all));
		return pos + dim->nparam;
	case isl_dim_out:
		isl_assert(ctx, pos < dim->n_out,
			    return isl_space_dim(dim, isl_dim_all));
		return pos + dim->nparam + dim->n_in;
	default:
		isl_assert(ctx, 0, return isl_space_dim(dim, isl_dim_all));
	}
	return isl_space_dim(dim, isl_dim_all);
}

/* Extend length of ids array to the total number of dimensions.
 */
static __isl_give isl_space *extend_ids(__isl_take isl_space *dim)
{
	isl_id **ids;
	int i;

	if (isl_space_dim(dim, isl_dim_all) <= dim->n_id)
		return dim;

	if (!dim->ids) {
		dim->ids = isl_calloc_array(dim->ctx,
				isl_id *, isl_space_dim(dim, isl_dim_all));
		if (!dim->ids)
			goto error;
	} else {
		ids = isl_realloc_array(dim->ctx, dim->ids,
				isl_id *, isl_space_dim(dim, isl_dim_all));
		if (!ids)
			goto error;
		dim->ids = ids;
		for (i = dim->n_id; i < isl_space_dim(dim, isl_dim_all); ++i)
			dim->ids[i] = NULL;
	}

	dim->n_id = isl_space_dim(dim, isl_dim_all);

	return dim;
error:
	isl_space_free(dim);
	return NULL;
}

static __isl_give isl_space *set_id(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	dim = isl_space_cow(dim);

	if (!dim)
		goto error;

	pos = global_pos(dim, type, pos);
	if (pos == isl_space_dim(dim, isl_dim_all))
		goto error;

	if (pos >= dim->n_id) {
		if (!id)
			return dim;
		dim = extend_ids(dim);
		if (!dim)
			goto error;
	}

	dim->ids[pos] = id;

	return dim;
error:
	isl_id_free(id);
	isl_space_free(dim);
	return NULL;
}

static __isl_keep isl_id *get_id(__isl_keep isl_space *dim,
				 enum isl_dim_type type, unsigned pos)
{
	if (!dim)
		return NULL;

	pos = global_pos(dim, type, pos);
	if (pos == isl_space_dim(dim, isl_dim_all))
		return NULL;
	if (pos >= dim->n_id)
		return NULL;
	return dim->ids[pos];
}

static unsigned offset(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:	return 0;
	case isl_dim_in:	return dim->nparam;
	case isl_dim_out:	return dim->nparam + dim->n_in;
	default:		return 0;
	}
}

static unsigned n(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:	return dim->nparam;
	case isl_dim_in:	return dim->n_in;
	case isl_dim_out:	return dim->n_out;
	case isl_dim_all:	return dim->nparam + dim->n_in + dim->n_out;
	default:		return 0;
	}
}

unsigned isl_space_dim(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	if (!dim)
		return 0;
	return n(dim, type);
}

unsigned isl_space_offset(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	if (!dim)
		return 0;
	return offset(dim, type);
}

static __isl_give isl_space *copy_ids(__isl_take isl_space *dst,
	enum isl_dim_type dst_type, unsigned offset, __isl_keep isl_space *src,
	enum isl_dim_type src_type)
{
	int i;
	isl_id *id;

	if (!dst)
		return NULL;

	for (i = 0; i < n(src, src_type); ++i) {
		id = get_id(src, src_type, i);
		if (!id)
			continue;
		dst = set_id(dst, dst_type, offset + i, isl_id_copy(id));
		if (!dst)
			return NULL;
	}
	return dst;
}

__isl_take isl_space *isl_space_dup(__isl_keep isl_space *dim)
{
	isl_space *dup;
	if (!dim)
		return NULL;
	dup = isl_space_alloc(dim->ctx, dim->nparam, dim->n_in, dim->n_out);
	if (!dup)
		return NULL;
	if (dim->tuple_id[0] &&
	    !(dup->tuple_id[0] = isl_id_copy(dim->tuple_id[0])))
		goto error;
	if (dim->tuple_id[1] &&
	    !(dup->tuple_id[1] = isl_id_copy(dim->tuple_id[1])))
		goto error;
	if (dim->nested[0] && !(dup->nested[0] = isl_space_copy(dim->nested[0])))
		goto error;
	if (dim->nested[1] && !(dup->nested[1] = isl_space_copy(dim->nested[1])))
		goto error;
	if (!dim->ids)
		return dup;
	dup = copy_ids(dup, isl_dim_param, 0, dim, isl_dim_param);
	dup = copy_ids(dup, isl_dim_in, 0, dim, isl_dim_in);
	dup = copy_ids(dup, isl_dim_out, 0, dim, isl_dim_out);
	return dup;
error:
	isl_space_free(dup);
	return NULL;
}

__isl_give isl_space *isl_space_cow(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;

	if (dim->ref == 1)
		return dim;
	dim->ref--;
	return isl_space_dup(dim);
}

__isl_give isl_space *isl_space_copy(__isl_keep isl_space *dim)
{
	if (!dim)
		return NULL;

	dim->ref++;
	return dim;
}

__isl_null isl_space *isl_space_free(__isl_take isl_space *space)
{
	int i;

	if (!space)
		return NULL;

	if (--space->ref > 0)
		return NULL;

	isl_id_free(space->tuple_id[0]);
	isl_id_free(space->tuple_id[1]);

	isl_space_free(space->nested[0]);
	isl_space_free(space->nested[1]);

	for (i = 0; i < space->n_id; ++i)
		isl_id_free(space->ids[i]);
	free(space->ids);
	isl_ctx_deref(space->ctx);
	
	free(space);

	return NULL;
}

/* Check if "s" is a valid dimension or tuple name.
 * We currently only forbid names that look like a number.
 *
 * s is assumed to be non-NULL.
 */
static int name_ok(isl_ctx *ctx, const char *s)
{
	char *p;
	long dummy;

	dummy = strtol(s, &p, 0);
	if (p != s)
		isl_die(ctx, isl_error_invalid, "name looks like a number",
			return 0);

	return 1;
}

/* Is it possible for the given dimension type to have a tuple id?
 */
static int space_can_have_id(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	if (!space)
		return 0;
	if (isl_space_is_params(space))
		isl_die(space->ctx, isl_error_invalid,
			"parameter spaces don't have tuple ids", return 0);
	if (isl_space_is_set(space) && type != isl_dim_set)
		isl_die(space->ctx, isl_error_invalid,
			"set spaces can only have a set id", return 0);
	if (type != isl_dim_in && type != isl_dim_out)
		isl_die(space->ctx, isl_error_invalid,
			"only input, output and set tuples can have ids",
			return 0);

	return 1;
}

/* Does the tuple have an id?
 */
isl_bool isl_space_has_tuple_id(__isl_keep isl_space *dim,
	enum isl_dim_type type)
{
	if (!space_can_have_id(dim, type))
		return isl_bool_error;
	return dim->tuple_id[type - isl_dim_in] != NULL;
}

__isl_give isl_id *isl_space_get_tuple_id(__isl_keep isl_space *dim,
	enum isl_dim_type type)
{
	int has_id;

	if (!dim)
		return NULL;
	has_id = isl_space_has_tuple_id(dim, type);
	if (has_id < 0)
		return NULL;
	if (!has_id)
		isl_die(dim->ctx, isl_error_invalid,
			"tuple has no id", return NULL);
	return isl_id_copy(dim->tuple_id[type - isl_dim_in]);
}

__isl_give isl_space *isl_space_set_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type, __isl_take isl_id *id)
{
	dim = isl_space_cow(dim);
	if (!dim || !id)
		goto error;
	if (type != isl_dim_in && type != isl_dim_out)
		isl_die(dim->ctx, isl_error_invalid,
			"only input, output and set tuples can have names",
			goto error);

	isl_id_free(dim->tuple_id[type - isl_dim_in]);
	dim->tuple_id[type - isl_dim_in] = id;

	return dim;
error:
	isl_id_free(id);
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_reset_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type)
{
	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;
	if (type != isl_dim_in && type != isl_dim_out)
		isl_die(dim->ctx, isl_error_invalid,
			"only input, output and set tuples can have names",
			goto error);

	isl_id_free(dim->tuple_id[type - isl_dim_in]);
	dim->tuple_id[type - isl_dim_in] = NULL;

	return dim;
error:
	isl_space_free(dim);
	return NULL;
}

/* Set the id of the given dimension of "space" to "id".
 * If the dimension already has an id, then it is replaced.
 * If the dimension is a parameter, then we need to change it
 * in the nested spaces (if any) as well.
 */
__isl_give isl_space *isl_space_set_dim_id(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	space = isl_space_cow(space);
	if (!space || !id)
		goto error;

	if (type == isl_dim_param) {
		int i;

		for (i = 0; i < 2; ++i) {
			if (!space->nested[i])
				continue;
			space->nested[i] =
				isl_space_set_dim_id(space->nested[i],
						type, pos, isl_id_copy(id));
			if (!space->nested[i])
				goto error;
		}
	}

	isl_id_free(get_id(space, type, pos));
	return set_id(space, type, pos, id);
error:
	isl_id_free(id);
	isl_space_free(space);
	return NULL;
}

/* Reset the id of the given dimension of "space".
 * If the dimension already has an id, then it is removed.
 * If the dimension is a parameter, then we need to reset it
 * in the nested spaces (if any) as well.
 */
__isl_give isl_space *isl_space_reset_dim_id(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned pos)
{
	space = isl_space_cow(space);
	if (!space)
		goto error;

	if (type == isl_dim_param) {
		int i;

		for (i = 0; i < 2; ++i) {
			if (!space->nested[i])
				continue;
			space->nested[i] =
				isl_space_reset_dim_id(space->nested[i],
							type, pos);
			if (!space->nested[i])
				goto error;
		}
	}

	isl_id_free(get_id(space, type, pos));
	return set_id(space, type, pos, NULL);
error:
	isl_space_free(space);
	return NULL;
}

isl_bool isl_space_has_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos)
{
	if (!dim)
		return isl_bool_error;
	return get_id(dim, type, pos) != NULL;
}

__isl_give isl_id *isl_space_get_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos)
{
	if (!dim)
		return NULL;
	if (!get_id(dim, type, pos))
		isl_die(dim->ctx, isl_error_invalid,
			"dim has no id", return NULL);
	return isl_id_copy(get_id(dim, type, pos));
}

__isl_give isl_space *isl_space_set_tuple_name(__isl_take isl_space *dim,
	enum isl_dim_type type, const char *s)
{
	isl_id *id;

	if (!dim)
		return NULL;

	if (!s)
		return isl_space_reset_tuple_id(dim, type);

	if (!name_ok(dim->ctx, s))
		goto error;

	id = isl_id_alloc(dim->ctx, s, NULL);
	return isl_space_set_tuple_id(dim, type, id);
error:
	isl_space_free(dim);
	return NULL;
}

/* Does the tuple have a name?
 */
isl_bool isl_space_has_tuple_name(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	isl_id *id;

	if (!space_can_have_id(space, type))
		return isl_bool_error;
	id = space->tuple_id[type - isl_dim_in];
	return id && id->name;
}

const char *isl_space_get_tuple_name(__isl_keep isl_space *dim,
	 enum isl_dim_type type)
{
	isl_id *id;
	if (!dim)
		return NULL;
	if (type != isl_dim_in && type != isl_dim_out)
		return NULL;
	id = dim->tuple_id[type - isl_dim_in];
	return id ? id->name : NULL;
}

__isl_give isl_space *isl_space_set_dim_name(__isl_take isl_space *dim,
				 enum isl_dim_type type, unsigned pos,
				 const char *s)
{
	isl_id *id;

	if (!dim)
		return NULL;
	if (!s)
		return isl_space_reset_dim_id(dim, type, pos);
	if (!name_ok(dim->ctx, s))
		goto error;
	id = isl_id_alloc(dim->ctx, s, NULL);
	return isl_space_set_dim_id(dim, type, pos, id);
error:
	isl_space_free(dim);
	return NULL;
}

/* Does the given dimension have a name?
 */
isl_bool isl_space_has_dim_name(__isl_keep isl_space *space,
	enum isl_dim_type type, unsigned pos)
{
	isl_id *id;

	if (!space)
		return isl_bool_error;
	id = get_id(space, type, pos);
	return id && id->name;
}

__isl_keep const char *isl_space_get_dim_name(__isl_keep isl_space *dim,
				 enum isl_dim_type type, unsigned pos)
{
	isl_id *id = get_id(dim, type, pos);
	return id ? id->name : NULL;
}

int isl_space_find_dim_by_id(__isl_keep isl_space *dim, enum isl_dim_type type,
	__isl_keep isl_id *id)
{
	int i;
	int offset;
	int n;

	if (!dim || !id)
		return -1;

	offset = isl_space_offset(dim, type);
	n = isl_space_dim(dim, type);
	for (i = 0; i < n && offset + i < dim->n_id; ++i)
		if (dim->ids[offset + i] == id)
			return i;

	return -1;
}

int isl_space_find_dim_by_name(__isl_keep isl_space *space,
	enum isl_dim_type type, const char *name)
{
	int i;
	int offset;
	int n;

	if (!space || !name)
		return -1;

	offset = isl_space_offset(space, type);
	n = isl_space_dim(space, type);
	for (i = 0; i < n && offset + i < space->n_id; ++i) {
		isl_id *id = get_id(space, type, i);
		if (id && id->name && !strcmp(id->name, name))
			return i;
	}

	return -1;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of "space".
 */
__isl_give isl_space *isl_space_reset_user(__isl_take isl_space *space)
{
	int i;
	isl_ctx *ctx;
	isl_id *id;
	const char *name;

	if (!space)
		return NULL;

	ctx = isl_space_get_ctx(space);

	for (i = 0; i < space->nparam && i < space->n_id; ++i) {
		if (!isl_id_get_user(space->ids[i]))
			continue;
		space = isl_space_cow(space);
		if (!space)
			return NULL;
		name = isl_id_get_name(space->ids[i]);
		id = isl_id_alloc(ctx, name, NULL);
		isl_id_free(space->ids[i]);
		space->ids[i] = id;
		if (!id)
			return isl_space_free(space);
	}

	for (i = 0; i < 2; ++i) {
		if (!space->tuple_id[i])
			continue;
		if (!isl_id_get_user(space->tuple_id[i]))
			continue;
		space = isl_space_cow(space);
		if (!space)
			return NULL;
		name = isl_id_get_name(space->tuple_id[i]);
		id = isl_id_alloc(ctx, name, NULL);
		isl_id_free(space->tuple_id[i]);
		space->tuple_id[i] = id;
		if (!id)
			return isl_space_free(space);
	}

	for (i = 0; i < 2; ++i) {
		if (!space->nested[i])
			continue;
		space = isl_space_cow(space);
		if (!space)
			return NULL;
		space->nested[i] = isl_space_reset_user(space->nested[i]);
		if (!space->nested[i])
			return isl_space_free(space);
	}

	return space;
}

static __isl_keep isl_id *tuple_id(__isl_keep isl_space *dim,
	enum isl_dim_type type)
{
	if (!dim)
		return NULL;
	if (type == isl_dim_in)
		return dim->tuple_id[0];
	if (type == isl_dim_out)
		return dim->tuple_id[1];
	return NULL;
}

static __isl_keep isl_space *nested(__isl_keep isl_space *dim,
	enum isl_dim_type type)
{
	if (!dim)
		return NULL;
	if (type == isl_dim_in)
		return dim->nested[0];
	if (type == isl_dim_out)
		return dim->nested[1];
	return NULL;
}

/* Are the two spaces the same, apart from positions and names of parameters?
 */
isl_bool isl_space_has_equal_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space1 || !space2)
		return isl_bool_error;
	if (space1 == space2)
		return isl_bool_true;
	return isl_space_tuple_is_equal(space1, isl_dim_in,
					space2, isl_dim_in) &&
	       isl_space_tuple_is_equal(space1, isl_dim_out,
					space2, isl_dim_out);
}

/* Check if the tuple of type "type1" of "space1" is the same as
 * the tuple of type "type2" of "space2".
 *
 * That is, check if the tuples have the same identifier, the same dimension
 * and the same internal structure.
 * The identifiers of the dimensions inside the tuples do not affect the result.
 *
 * Note that this function only checks the tuples themselves.
 * If nested tuples are involved, then we need to be careful not
 * to have result affected by possibly differing parameters
 * in those nested tuples.
 */
isl_bool isl_space_tuple_is_equal(__isl_keep isl_space *space1,
	enum isl_dim_type type1, __isl_keep isl_space *space2,
	enum isl_dim_type type2)
{
	isl_id *id1, *id2;
	isl_space *nested1, *nested2;

	if (!space1 || !space2)
		return isl_bool_error;

	if (space1 == space2 && type1 == type2)
		return isl_bool_true;

	if (n(space1, type1) != n(space2, type2))
		return isl_bool_false;
	id1 = tuple_id(space1, type1);
	id2 = tuple_id(space2, type2);
	if (!id1 ^ !id2)
		return isl_bool_false;
	if (id1 && id1 != id2)
		return isl_bool_false;
	nested1 = nested(space1, type1);
	nested2 = nested(space2, type2);
	if (!nested1 ^ !nested2)
		return isl_bool_false;
	if (nested1 && !isl_space_has_equal_tuples(nested1, nested2))
		return isl_bool_false;
	return isl_bool_true;
}

/* This is the old, undocumented, name for isl_space_tuple_is_equal.
 * It will be removed at some point.
 */
int isl_space_tuple_match(__isl_keep isl_space *space1, enum isl_dim_type type1,
	__isl_keep isl_space *space2, enum isl_dim_type type2)
{
	return isl_space_tuple_is_equal(space1, type1, space2, type2);
}

static int match(__isl_keep isl_space *space1, enum isl_dim_type type1,
	__isl_keep isl_space *space2, enum isl_dim_type type2)
{
	int i;

	if (space1 == space2 && type1 == type2)
		return 1;

	if (!isl_space_tuple_is_equal(space1, type1, space2, type2))
		return 0;

	if (!space1->ids && !space2->ids)
		return 1;

	for (i = 0; i < n(space1, type1); ++i) {
		if (get_id(space1, type1, i) != get_id(space2, type2, i))
			return 0;
	}
	return 1;
}

isl_bool isl_space_match(__isl_keep isl_space *space1, enum isl_dim_type type1,
	__isl_keep isl_space *space2, enum isl_dim_type type2)
{
	if (!space1 || !space2)
		return isl_bool_error;

	return match(space1, type1, space2, type2);
}

static void get_ids(__isl_keep isl_space *dim, enum isl_dim_type type,
	unsigned first, unsigned n, __isl_keep isl_id **ids)
{
	int i;

	for (i = 0; i < n ; ++i)
		ids[i] = get_id(dim, type, first + i);
}

__isl_give isl_space *isl_space_extend(__isl_take isl_space *space,
			unsigned nparam, unsigned n_in, unsigned n_out)
{
	isl_id **ids = NULL;

	if (!space)
		return NULL;
	if (space->nparam == nparam &&
	    space->n_in == n_in && space->n_out == n_out)
		return space;

	isl_assert(space->ctx, space->nparam <= nparam, goto error);
	isl_assert(space->ctx, space->n_in <= n_in, goto error);
	isl_assert(space->ctx, space->n_out <= n_out, goto error);

	space = isl_space_cow(space);
	if (!space)
		goto error;

	if (space->ids) {
		unsigned n;
		n = nparam + n_in + n_out;
		if (n < nparam || n < n_in || n < n_out)
			isl_die(isl_space_get_ctx(space), isl_error_invalid,
				"overflow in total number of dimensions",
				goto error);
		ids = isl_calloc_array(space->ctx, isl_id *, n);
		if (!ids)
			goto error;
		get_ids(space, isl_dim_param, 0, space->nparam, ids);
		get_ids(space, isl_dim_in, 0, space->n_in, ids + nparam);
		get_ids(space, isl_dim_out, 0, space->n_out,
			ids + nparam + n_in);
		free(space->ids);
		space->ids = ids;
		space->n_id = nparam + n_in + n_out;
	}
	space->nparam = nparam;
	space->n_in = n_in;
	space->n_out = n_out;

	return space;
error:
	free(ids);
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_add_dims(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned n)
{
	dim = isl_space_reset(dim, type);
	if (!dim)
		return NULL;
	switch (type) {
	case isl_dim_param:
		dim = isl_space_extend(dim,
					dim->nparam + n, dim->n_in, dim->n_out);
		if (dim && dim->nested[0] &&
		    !(dim->nested[0] = isl_space_add_dims(dim->nested[0],
						    isl_dim_param, n)))
			goto error;
		if (dim && dim->nested[1] &&
		    !(dim->nested[1] = isl_space_add_dims(dim->nested[1],
						    isl_dim_param, n)))
			goto error;
		return dim;
	case isl_dim_in:
		return isl_space_extend(dim,
					dim->nparam, dim->n_in + n, dim->n_out);
	case isl_dim_out:
		return isl_space_extend(dim,
					dim->nparam, dim->n_in, dim->n_out + n);
	default:
		isl_die(dim->ctx, isl_error_invalid,
			"cannot add dimensions of specified type", goto error);
	}
error:
	isl_space_free(dim);
	return NULL;
}

static int valid_dim_type(enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:
	case isl_dim_in:
	case isl_dim_out:
		return 1;
	default:
		return 0;
	}
}

/* Insert "n" dimensions of type "type" at position "pos".
 * If we are inserting parameters, then they are also inserted in
 * any nested spaces.
 */
__isl_give isl_space *isl_space_insert_dims(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	isl_id **ids = NULL;

	if (!dim)
		return NULL;
	if (n == 0)
		return isl_space_reset(dim, type);

	if (!valid_dim_type(type))
		isl_die(dim->ctx, isl_error_invalid,
			"cannot insert dimensions of specified type",
			goto error);

	isl_assert(dim->ctx, pos <= isl_space_dim(dim, type), goto error);

	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;

	if (dim->ids) {
		enum isl_dim_type t, o = isl_dim_param;
		int off;
		int s[3];
		ids = isl_calloc_array(dim->ctx, isl_id *,
				     dim->nparam + dim->n_in + dim->n_out + n);
		if (!ids)
			goto error;
		off = 0;
		s[isl_dim_param - o] = dim->nparam;
		s[isl_dim_in - o] = dim->n_in;
		s[isl_dim_out - o] = dim->n_out;
		for (t = isl_dim_param; t <= isl_dim_out; ++t) {
			if (t != type) {
				get_ids(dim, t, 0, s[t - o], ids + off);
				off += s[t - o];
			} else {
				get_ids(dim, t, 0, pos, ids + off);
				off += pos + n;
				get_ids(dim, t, pos, s[t - o] - pos, ids + off);
				off += s[t - o] - pos;
			}
		}
		free(dim->ids);
		dim->ids = ids;
		dim->n_id = dim->nparam + dim->n_in + dim->n_out + n;
	}
	switch (type) {
	case isl_dim_param:	dim->nparam += n; break;
	case isl_dim_in:	dim->n_in += n; break;
	case isl_dim_out:	dim->n_out += n; break;
	default:		;
	}
	dim = isl_space_reset(dim, type);

	if (type == isl_dim_param) {
		if (dim && dim->nested[0] &&
		    !(dim->nested[0] = isl_space_insert_dims(dim->nested[0],
						    isl_dim_param, pos, n)))
			goto error;
		if (dim && dim->nested[1] &&
		    !(dim->nested[1] = isl_space_insert_dims(dim->nested[1],
						    isl_dim_param, pos, n)))
			goto error;
	}

	return dim;
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_move_dims(__isl_take isl_space *dim,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	int i;

	if (!dim)
		return NULL;
	if (n == 0) {
		dim = isl_space_reset(dim, src_type);
		return isl_space_reset(dim, dst_type);
	}

	isl_assert(dim->ctx, src_pos + n <= isl_space_dim(dim, src_type),
		goto error);

	if (dst_type == src_type && dst_pos == src_pos)
		return dim;

	isl_assert(dim->ctx, dst_type != src_type, goto error);

	dim = isl_space_reset(dim, src_type);
	dim = isl_space_reset(dim, dst_type);

	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;

	if (dim->ids) {
		isl_id **ids;
		enum isl_dim_type t, o = isl_dim_param;
		int off;
		int s[3];
		ids = isl_calloc_array(dim->ctx, isl_id *,
					 dim->nparam + dim->n_in + dim->n_out);
		if (!ids)
			goto error;
		off = 0;
		s[isl_dim_param - o] = dim->nparam;
		s[isl_dim_in - o] = dim->n_in;
		s[isl_dim_out - o] = dim->n_out;
		for (t = isl_dim_param; t <= isl_dim_out; ++t) {
			if (t == dst_type) {
				get_ids(dim, t, 0, dst_pos, ids + off);
				off += dst_pos;
				get_ids(dim, src_type, src_pos, n, ids + off);
				off += n;
				get_ids(dim, t, dst_pos, s[t - o] - dst_pos,
						ids + off);
				off += s[t - o] - dst_pos;
			} else if (t == src_type) {
				get_ids(dim, t, 0, src_pos, ids + off);
				off += src_pos;
				get_ids(dim, t, src_pos + n,
					    s[t - o] - src_pos - n, ids + off);
				off += s[t - o] - src_pos - n;
			} else {
				get_ids(dim, t, 0, s[t - o], ids + off);
				off += s[t - o];
			}
		}
		free(dim->ids);
		dim->ids = ids;
		dim->n_id = dim->nparam + dim->n_in + dim->n_out;
	}

	switch (dst_type) {
	case isl_dim_param:	dim->nparam += n; break;
	case isl_dim_in:	dim->n_in += n; break;
	case isl_dim_out:	dim->n_out += n; break;
	default:		;
	}

	switch (src_type) {
	case isl_dim_param:	dim->nparam -= n; break;
	case isl_dim_in:	dim->n_in -= n; break;
	case isl_dim_out:	dim->n_out -= n; break;
	default:		;
	}

	if (dst_type != isl_dim_param && src_type != isl_dim_param)
		return dim;

	for (i = 0; i < 2; ++i) {
		if (!dim->nested[i])
			continue;
		dim->nested[i] = isl_space_replace(dim->nested[i],
						 isl_dim_param, dim);
		if (!dim->nested[i])
			goto error;
	}

	return dim;
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_join(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *dim;

	if (!left || !right)
		goto error;

	isl_assert(left->ctx, match(left, isl_dim_param, right, isl_dim_param),
			goto error);
	isl_assert(left->ctx,
		isl_space_tuple_is_equal(left, isl_dim_out, right, isl_dim_in),
		goto error);

	dim = isl_space_alloc(left->ctx, left->nparam, left->n_in, right->n_out);
	if (!dim)
		goto error;

	dim = copy_ids(dim, isl_dim_param, 0, left, isl_dim_param);
	dim = copy_ids(dim, isl_dim_in, 0, left, isl_dim_in);
	dim = copy_ids(dim, isl_dim_out, 0, right, isl_dim_out);

	if (dim && left->tuple_id[0] &&
	    !(dim->tuple_id[0] = isl_id_copy(left->tuple_id[0])))
		goto error;
	if (dim && right->tuple_id[1] &&
	    !(dim->tuple_id[1] = isl_id_copy(right->tuple_id[1])))
		goto error;
	if (dim && left->nested[0] &&
	    !(dim->nested[0] = isl_space_copy(left->nested[0])))
		goto error;
	if (dim && right->nested[1] &&
	    !(dim->nested[1] = isl_space_copy(right->nested[1])))
		goto error;

	isl_space_free(left);
	isl_space_free(right);

	return dim;
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

/* Given two map spaces { A -> C } and { B -> D }, construct the space
 * { [A -> B] -> [C -> D] }.
 * Given two set spaces { A } and { B }, construct the space { [A -> B] }.
 */
__isl_give isl_space *isl_space_product(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *dom1, *dom2, *nest1, *nest2;
	int is_set;

	if (!left || !right)
		goto error;

	is_set = isl_space_is_set(left);
	if (is_set != isl_space_is_set(right))
		isl_die(isl_space_get_ctx(left), isl_error_invalid,
			"expecting either two set spaces or two map spaces",
			goto error);
	if (is_set)
		return isl_space_range_product(left, right);

	isl_assert(left->ctx, match(left, isl_dim_param, right, isl_dim_param),
			goto error);

	dom1 = isl_space_domain(isl_space_copy(left));
	dom2 = isl_space_domain(isl_space_copy(right));
	nest1 = isl_space_wrap(isl_space_join(isl_space_reverse(dom1), dom2));

	dom1 = isl_space_range(left);
	dom2 = isl_space_range(right);
	nest2 = isl_space_wrap(isl_space_join(isl_space_reverse(dom1), dom2));

	return isl_space_join(isl_space_reverse(nest1), nest2);
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

/* Given two spaces { A -> C } and { B -> C }, construct the space
 * { [A -> B] -> C }
 */
__isl_give isl_space *isl_space_domain_product(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *ran, *dom1, *dom2, *nest;

	if (!left || !right)
		goto error;

	if (!match(left, isl_dim_param, right, isl_dim_param))
		isl_die(left->ctx, isl_error_invalid,
			"parameters need to match", goto error);
	if (!isl_space_tuple_is_equal(left, isl_dim_out, right, isl_dim_out))
		isl_die(left->ctx, isl_error_invalid,
			"ranges need to match", goto error);

	ran = isl_space_range(isl_space_copy(left));

	dom1 = isl_space_domain(left);
	dom2 = isl_space_domain(right);
	nest = isl_space_wrap(isl_space_join(isl_space_reverse(dom1), dom2));

	return isl_space_join(isl_space_reverse(nest), ran);
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

__isl_give isl_space *isl_space_range_product(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *dom, *ran1, *ran2, *nest;

	if (!left || !right)
		goto error;

	isl_assert(left->ctx, match(left, isl_dim_param, right, isl_dim_param),
			goto error);
	if (!isl_space_tuple_is_equal(left, isl_dim_in, right, isl_dim_in))
		isl_die(left->ctx, isl_error_invalid,
			"domains need to match", goto error);

	dom = isl_space_domain(isl_space_copy(left));

	ran1 = isl_space_range(left);
	ran2 = isl_space_range(right);
	nest = isl_space_wrap(isl_space_join(isl_space_reverse(ran1), ran2));

	return isl_space_join(isl_space_reverse(dom), nest);
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

/* Given a space of the form [A -> B] -> [C -> D], return the space A -> C.
 */
__isl_give isl_space *isl_space_factor_domain(__isl_take isl_space *space)
{
	space = isl_space_domain_factor_domain(space);
	space = isl_space_range_factor_domain(space);
	return space;
}

/* Given a space of the form [A -> B] -> C, return the space A -> C.
 */
__isl_give isl_space *isl_space_domain_factor_domain(
	__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *domain;

	if (!space)
		return NULL;
	if (!isl_space_domain_is_wrapping(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"domain not a product", return isl_space_free(space));

	nested = space->nested[0];
	domain = isl_space_copy(space);
	domain = isl_space_drop_dims(domain, isl_dim_in,
					nested->n_in, nested->n_out);
	if (!domain)
		return isl_space_free(space);
	if (nested->tuple_id[0]) {
		domain->tuple_id[0] = isl_id_copy(nested->tuple_id[0]);
		if (!domain->tuple_id[0])
			goto error;
	}
	if (nested->nested[0]) {
		domain->nested[0] = isl_space_copy(nested->nested[0]);
		if (!domain->nested[0])
			goto error;
	}

	isl_space_free(space);
	return domain;
error:
	isl_space_free(space);
	isl_space_free(domain);
	return NULL;
}

/* Given a space of the form [A -> B] -> C, return the space B -> C.
 */
__isl_give isl_space *isl_space_domain_factor_range(
	__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *range;

	if (!space)
		return NULL;
	if (!isl_space_domain_is_wrapping(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"domain not a product", return isl_space_free(space));

	nested = space->nested[0];
	range = isl_space_copy(space);
	range = isl_space_drop_dims(range, isl_dim_in, 0, nested->n_in);
	if (!range)
		return isl_space_free(space);
	if (nested->tuple_id[1]) {
		range->tuple_id[0] = isl_id_copy(nested->tuple_id[1]);
		if (!range->tuple_id[0])
			goto error;
	}
	if (nested->nested[1]) {
		range->nested[0] = isl_space_copy(nested->nested[1]);
		if (!range->nested[0])
			goto error;
	}

	isl_space_free(space);
	return range;
error:
	isl_space_free(space);
	isl_space_free(range);
	return NULL;
}

/* Given a space of the form A -> [B -> C], return the space A -> B.
 */
__isl_give isl_space *isl_space_range_factor_domain(
	__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *domain;

	if (!space)
		return NULL;
	if (!isl_space_range_is_wrapping(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"range not a product", return isl_space_free(space));

	nested = space->nested[1];
	domain = isl_space_copy(space);
	domain = isl_space_drop_dims(domain, isl_dim_out,
					nested->n_in, nested->n_out);
	if (!domain)
		return isl_space_free(space);
	if (nested->tuple_id[0]) {
		domain->tuple_id[1] = isl_id_copy(nested->tuple_id[0]);
		if (!domain->tuple_id[1])
			goto error;
	}
	if (nested->nested[0]) {
		domain->nested[1] = isl_space_copy(nested->nested[0]);
		if (!domain->nested[1])
			goto error;
	}

	isl_space_free(space);
	return domain;
error:
	isl_space_free(space);
	isl_space_free(domain);
	return NULL;
}

/* Internal function that selects the range of the map that is
 * embedded in either a set space or the range of a map space.
 * In particular, given a space of the form [A -> B], return the space B.
 * Given a space of the form A -> [B -> C], return the space A -> C.
 */
static __isl_give isl_space *range_factor_range(__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *range;

	if (!space)
		return NULL;

	nested = space->nested[1];
	range = isl_space_copy(space);
	range = isl_space_drop_dims(range, isl_dim_out, 0, nested->n_in);
	if (!range)
		return isl_space_free(space);
	if (nested->tuple_id[1]) {
		range->tuple_id[1] = isl_id_copy(nested->tuple_id[1]);
		if (!range->tuple_id[1])
			goto error;
	}
	if (nested->nested[1]) {
		range->nested[1] = isl_space_copy(nested->nested[1]);
		if (!range->nested[1])
			goto error;
	}

	isl_space_free(space);
	return range;
error:
	isl_space_free(space);
	isl_space_free(range);
	return NULL;
}

/* Given a space of the form A -> [B -> C], return the space A -> C.
 */
__isl_give isl_space *isl_space_range_factor_range(
	__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_range_is_wrapping(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"range not a product", return isl_space_free(space));

	return range_factor_range(space);
}

/* Given a space of the form [A -> B], return the space B.
 */
static __isl_give isl_space *set_factor_range(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_is_wrapping(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a product", return isl_space_free(space));

	return range_factor_range(space);
}

/* Given a space of the form [A -> B] -> [C -> D], return the space B -> D.
 * Given a space of the form [A -> B], return the space B.
 */
__isl_give isl_space *isl_space_factor_range(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (isl_space_is_set(space))
		return set_factor_range(space);
	space = isl_space_domain_factor_range(space);
	space = isl_space_range_factor_range(space);
	return space;
}

__isl_give isl_space *isl_space_map_from_set(__isl_take isl_space *dim)
{
	isl_ctx *ctx;
	isl_id **ids = NULL;

	if (!dim)
		return NULL;
	ctx = isl_space_get_ctx(dim);
	if (!isl_space_is_set(dim))
		isl_die(ctx, isl_error_invalid, "not a set space", goto error);
	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;
	if (dim->ids) {
		ids = isl_calloc_array(dim->ctx, isl_id *,
					dim->nparam + dim->n_out + dim->n_out);
		if (!ids)
			goto error;
		get_ids(dim, isl_dim_param, 0, dim->nparam, ids);
		get_ids(dim, isl_dim_out, 0, dim->n_out, ids + dim->nparam);
	}
	dim->n_in = dim->n_out;
	if (ids) {
		free(dim->ids);
		dim->ids = ids;
		dim->n_id = dim->nparam + dim->n_out + dim->n_out;
		dim = copy_ids(dim, isl_dim_out, 0, dim, isl_dim_in);
	}
	isl_id_free(dim->tuple_id[0]);
	dim->tuple_id[0] = isl_id_copy(dim->tuple_id[1]);
	isl_space_free(dim->nested[0]);
	dim->nested[0] = isl_space_copy(dim->nested[1]);
	return dim;
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_map_from_domain_and_range(
	__isl_take isl_space *domain, __isl_take isl_space *range)
{
	if (!domain || !range)
		goto error;
	if (!isl_space_is_set(domain))
		isl_die(isl_space_get_ctx(domain), isl_error_invalid,
			"domain is not a set space", goto error);
	if (!isl_space_is_set(range))
		isl_die(isl_space_get_ctx(range), isl_error_invalid,
			"range is not a set space", goto error);
	return isl_space_join(isl_space_reverse(domain), range);
error:
	isl_space_free(domain);
	isl_space_free(range);
	return NULL;
}

static __isl_give isl_space *set_ids(__isl_take isl_space *dim,
	enum isl_dim_type type,
	unsigned first, unsigned n, __isl_take isl_id **ids)
{
	int i;

	for (i = 0; i < n ; ++i)
		dim = set_id(dim, type, first + i, ids[i]);

	return dim;
}

__isl_give isl_space *isl_space_reverse(__isl_take isl_space *dim)
{
	unsigned t;
	isl_space *nested;
	isl_id **ids = NULL;
	isl_id *id;

	if (!dim)
		return NULL;
	if (match(dim, isl_dim_in, dim, isl_dim_out))
		return dim;

	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;

	id = dim->tuple_id[0];
	dim->tuple_id[0] = dim->tuple_id[1];
	dim->tuple_id[1] = id;

	nested = dim->nested[0];
	dim->nested[0] = dim->nested[1];
	dim->nested[1] = nested;

	if (dim->ids) {
		int n_id = dim->n_in + dim->n_out;
		ids = isl_alloc_array(dim->ctx, isl_id *, n_id);
		if (n_id && !ids)
			goto error;
		get_ids(dim, isl_dim_in, 0, dim->n_in, ids);
		get_ids(dim, isl_dim_out, 0, dim->n_out, ids + dim->n_in);
	}

	t = dim->n_in;
	dim->n_in = dim->n_out;
	dim->n_out = t;

	if (dim->ids) {
		dim = set_ids(dim, isl_dim_out, 0, dim->n_out, ids);
		dim = set_ids(dim, isl_dim_in, 0, dim->n_in, ids + dim->n_out);
		free(ids);
	}

	return dim;
error:
	free(ids);
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_drop_dims(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned first, unsigned num)
{
	int i;

	if (!dim)
		return NULL;

	if (num == 0)
		return isl_space_reset(dim, type);

	if (!valid_dim_type(type))
		isl_die(dim->ctx, isl_error_invalid,
			"cannot drop dimensions of specified type", goto error);

	if (first + num > n(dim, type) || first + num < first)
		isl_die(isl_space_get_ctx(dim), isl_error_invalid,
			"index out of bounds", return isl_space_free(dim));
	dim = isl_space_cow(dim);
	if (!dim)
		goto error;
	if (dim->ids) {
		dim = extend_ids(dim);
		if (!dim)
			goto error;
		for (i = 0; i < num; ++i)
			isl_id_free(get_id(dim, type, first + i));
		for (i = first+num; i < n(dim, type); ++i)
			set_id(dim, type, i - num, get_id(dim, type, i));
		switch (type) {
		case isl_dim_param:
			get_ids(dim, isl_dim_in, 0, dim->n_in,
				dim->ids + offset(dim, isl_dim_in) - num);
		case isl_dim_in:
			get_ids(dim, isl_dim_out, 0, dim->n_out,
				dim->ids + offset(dim, isl_dim_out) - num);
		default:
			;
		}
		dim->n_id -= num;
	}
	switch (type) {
	case isl_dim_param:	dim->nparam -= num; break;
	case isl_dim_in:	dim->n_in -= num; break;
	case isl_dim_out:	dim->n_out -= num; break;
	default:		;
	}
	dim = isl_space_reset(dim, type);
	if (type == isl_dim_param) {
		if (dim && dim->nested[0] &&
		    !(dim->nested[0] = isl_space_drop_dims(dim->nested[0],
						    isl_dim_param, first, num)))
			goto error;
		if (dim && dim->nested[1] &&
		    !(dim->nested[1] = isl_space_drop_dims(dim->nested[1],
						    isl_dim_param, first, num)))
			goto error;
	}
	return dim;
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_drop_inputs(__isl_take isl_space *dim,
		unsigned first, unsigned n)
{
	if (!dim)
		return NULL;
	return isl_space_drop_dims(dim, isl_dim_in, first, n);
}

__isl_give isl_space *isl_space_drop_outputs(__isl_take isl_space *dim,
		unsigned first, unsigned n)
{
	if (!dim)
		return NULL;
	return isl_space_drop_dims(dim, isl_dim_out, first, n);
}

__isl_give isl_space *isl_space_domain(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	dim = isl_space_drop_outputs(dim, 0, dim->n_out);
	dim = isl_space_reverse(dim);
	dim = mark_as_set(dim);
	return dim;
}

__isl_give isl_space *isl_space_from_domain(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	if (!isl_space_is_set(dim))
		isl_die(isl_space_get_ctx(dim), isl_error_invalid,
			"not a set space", goto error);
	dim = isl_space_reverse(dim);
	dim = isl_space_reset(dim, isl_dim_out);
	return dim;
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_range(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	dim = isl_space_drop_inputs(dim, 0, dim->n_in);
	dim = mark_as_set(dim);
	return dim;
}

__isl_give isl_space *isl_space_from_range(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	if (!isl_space_is_set(dim))
		isl_die(isl_space_get_ctx(dim), isl_error_invalid,
			"not a set space", goto error);
	return isl_space_reset(dim, isl_dim_in);
error:
	isl_space_free(dim);
	return NULL;
}

/* Given a map space A -> B, return the map space [A -> B] -> A.
 */
__isl_give isl_space *isl_space_domain_map(__isl_take isl_space *space)
{
	isl_space *domain;

	domain = isl_space_from_range(isl_space_domain(isl_space_copy(space)));
	space = isl_space_from_domain(isl_space_wrap(space));
	space = isl_space_join(space, domain);

	return space;
}

/* Given a map space A -> B, return the map space [A -> B] -> B.
 */
__isl_give isl_space *isl_space_range_map(__isl_take isl_space *space)
{
	isl_space *range;

	range = isl_space_from_range(isl_space_range(isl_space_copy(space)));
	space = isl_space_from_domain(isl_space_wrap(space));
	space = isl_space_join(space, range);

	return space;
}

__isl_give isl_space *isl_space_params(__isl_take isl_space *space)
{
	if (isl_space_is_params(space))
		return space;
	space = isl_space_drop_dims(space,
			    isl_dim_in, 0, isl_space_dim(space, isl_dim_in));
	space = isl_space_drop_dims(space,
			    isl_dim_out, 0, isl_space_dim(space, isl_dim_out));
	space = mark_as_params(space);
	return space;
}

__isl_give isl_space *isl_space_set_from_params(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_is_params(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a parameter space", goto error);
	return isl_space_reset(space, isl_dim_set);
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_as_set_space(__isl_take isl_space *dim)
{
	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;

	dim->n_out += dim->n_in;
	dim->n_in = 0;
	dim = isl_space_reset(dim, isl_dim_in);
	dim = isl_space_reset(dim, isl_dim_out);

	return dim;
}

__isl_give isl_space *isl_space_underlying(__isl_take isl_space *dim,
	unsigned n_div)
{
	int i;

	if (!dim)
		return NULL;
	if (n_div == 0 &&
	    dim->nparam == 0 && dim->n_in == 0 && dim->n_id == 0)
		return isl_space_reset(isl_space_reset(dim, isl_dim_in), isl_dim_out);
	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;
	dim->n_out += dim->nparam + dim->n_in + n_div;
	dim->nparam = 0;
	dim->n_in = 0;

	for (i = 0; i < dim->n_id; ++i)
		isl_id_free(get_id(dim, isl_dim_out, i));
	dim->n_id = 0;
	dim = isl_space_reset(dim, isl_dim_in);
	dim = isl_space_reset(dim, isl_dim_out);

	return dim;
}

/* Are the two spaces the same, including positions and names of parameters?
 */
isl_bool isl_space_is_equal(__isl_keep isl_space *dim1,
	__isl_keep isl_space *dim2)
{
	if (!dim1 || !dim2)
		return isl_bool_error;
	if (dim1 == dim2)
		return isl_bool_true;
	return match(dim1, isl_dim_param, dim2, isl_dim_param) &&
	       isl_space_tuple_is_equal(dim1, isl_dim_in, dim2, isl_dim_in) &&
	       isl_space_tuple_is_equal(dim1, isl_dim_out, dim2, isl_dim_out);
}

/* Is space1 equal to the domain of space2?
 *
 * In the internal version we also allow space2 to be the space of a set,
 * provided space1 is a parameter space.
 */
isl_bool isl_space_is_domain_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space1 || !space2)
		return isl_bool_error;
	if (!isl_space_is_set(space1))
		return isl_bool_false;
	return match(space1, isl_dim_param, space2, isl_dim_param) &&
	       isl_space_tuple_is_equal(space1, isl_dim_set,
					space2, isl_dim_in);
}

/* Is space1 equal to the domain of space2?
 */
isl_bool isl_space_is_domain(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space2)
		return isl_bool_error;
	if (!isl_space_is_map(space2))
		return isl_bool_false;
	return isl_space_is_domain_internal(space1, space2);
}

/* Is space1 equal to the range of space2?
 *
 * In the internal version, space2 is allowed to be the space of a set,
 * in which case it should be equal to space1.
 */
isl_bool isl_space_is_range_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space1 || !space2)
		return isl_bool_error;
	if (!isl_space_is_set(space1))
		return isl_bool_false;
	return match(space1, isl_dim_param, space2, isl_dim_param) &&
	       isl_space_tuple_is_equal(space1, isl_dim_set,
					space2, isl_dim_out);
}

/* Is space1 equal to the range of space2?
 */
isl_bool isl_space_is_range(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space2)
		return isl_bool_error;
	if (!isl_space_is_map(space2))
		return isl_bool_false;
	return isl_space_is_range_internal(space1, space2);
}

int isl_space_compatible_internal(__isl_keep isl_space *dim1,
	__isl_keep isl_space *dim2)
{
	return dim1->nparam == dim2->nparam &&
	       dim1->n_in + dim1->n_out == dim2->n_in + dim2->n_out;
}

int isl_space_compatible(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	return isl_space_compatible_internal(space1, space2);
}

/* Update "hash" by hashing in "space".
 * Changes in this function should be reflected in isl_hash_space_domain.
 */
static uint32_t isl_hash_space(uint32_t hash, __isl_keep isl_space *space)
{
	int i;
	isl_id *id;

	if (!space)
		return hash;

	isl_hash_byte(hash, space->nparam % 256);
	isl_hash_byte(hash, space->n_in % 256);
	isl_hash_byte(hash, space->n_out % 256);

	for (i = 0; i < space->nparam; ++i) {
		id = get_id(space, isl_dim_param, i);
		hash = isl_hash_id(hash, id);
	}

	id = tuple_id(space, isl_dim_in);
	hash = isl_hash_id(hash, id);
	id = tuple_id(space, isl_dim_out);
	hash = isl_hash_id(hash, id);

	hash = isl_hash_space(hash, space->nested[0]);
	hash = isl_hash_space(hash, space->nested[1]);

	return hash;
}

/* Update "hash" by hashing in the domain of "space".
 * The result of this function is equal to the result of applying
 * isl_hash_space to the domain of "space".
 */
static uint32_t isl_hash_space_domain(uint32_t hash,
	__isl_keep isl_space *space)
{
	int i;
	isl_id *id;

	if (!space)
		return hash;

	isl_hash_byte(hash, space->nparam % 256);
	isl_hash_byte(hash, 0);
	isl_hash_byte(hash, space->n_in % 256);

	for (i = 0; i < space->nparam; ++i) {
		id = get_id(space, isl_dim_param, i);
		hash = isl_hash_id(hash, id);
	}

	hash = isl_hash_id(hash, &isl_id_none);
	id = tuple_id(space, isl_dim_in);
	hash = isl_hash_id(hash, id);

	hash = isl_hash_space(hash, space->nested[0]);

	return hash;
}

uint32_t isl_space_get_hash(__isl_keep isl_space *dim)
{
	uint32_t hash;

	if (!dim)
		return 0;

	hash = isl_hash_init();
	hash = isl_hash_space(hash, dim);

	return hash;
}

/* Return the hash value of the domain of "space".
 * That is, isl_space_get_domain_hash(space) is equal to
 * isl_space_get_hash(isl_space_domain(space)).
 */
uint32_t isl_space_get_domain_hash(__isl_keep isl_space *space)
{
	uint32_t hash;

	if (!space)
		return 0;

	hash = isl_hash_init();
	hash = isl_hash_space_domain(hash, space);

	return hash;
}

isl_bool isl_space_is_wrapping(__isl_keep isl_space *dim)
{
	if (!dim)
		return isl_bool_error;

	if (!isl_space_is_set(dim))
		return isl_bool_false;

	return dim->nested[1] != NULL;
}

/* Is "space" the space of a map where the domain is a wrapped map space?
 */
isl_bool isl_space_domain_is_wrapping(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;

	if (isl_space_is_set(space))
		return isl_bool_false;

	return space->nested[0] != NULL;
}

/* Is "space" the space of a map where the range is a wrapped map space?
 */
isl_bool isl_space_range_is_wrapping(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;

	if (isl_space_is_set(space))
		return isl_bool_false;

	return space->nested[1] != NULL;
}

__isl_give isl_space *isl_space_wrap(__isl_take isl_space *dim)
{
	isl_space *wrap;

	if (!dim)
		return NULL;

	wrap = isl_space_set_alloc(dim->ctx,
				    dim->nparam, dim->n_in + dim->n_out);

	wrap = copy_ids(wrap, isl_dim_param, 0, dim, isl_dim_param);
	wrap = copy_ids(wrap, isl_dim_set, 0, dim, isl_dim_in);
	wrap = copy_ids(wrap, isl_dim_set, dim->n_in, dim, isl_dim_out);

	if (!wrap)
		goto error;

	wrap->nested[1] = dim;

	return wrap;
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_space *isl_space_unwrap(__isl_take isl_space *dim)
{
	isl_space *unwrap;

	if (!dim)
		return NULL;

	if (!isl_space_is_wrapping(dim))
		isl_die(dim->ctx, isl_error_invalid, "not a wrapping space",
			goto error);

	unwrap = isl_space_copy(dim->nested[1]);
	isl_space_free(dim);

	return unwrap;
error:
	isl_space_free(dim);
	return NULL;
}

int isl_space_is_named_or_nested(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	if (type != isl_dim_in && type != isl_dim_out)
		return 0;
	if (!dim)
		return -1;
	if (dim->tuple_id[type - isl_dim_in])
		return 1;
	if (dim->nested[type - isl_dim_in])
		return 1;
	return 0;
}

int isl_space_may_be_set(__isl_keep isl_space *dim)
{
	if (!dim)
		return -1;
	if (isl_space_is_set(dim))
		return 1;
	if (isl_space_dim(dim, isl_dim_in) != 0)
		return 0;
	if (isl_space_is_named_or_nested(dim, isl_dim_in))
		return 0;
	return 1;
}

__isl_give isl_space *isl_space_reset(__isl_take isl_space *dim,
	enum isl_dim_type type)
{
	if (!isl_space_is_named_or_nested(dim, type))
		return dim;

	dim = isl_space_cow(dim);
	if (!dim)
		return NULL;

	isl_id_free(dim->tuple_id[type - isl_dim_in]);
	dim->tuple_id[type - isl_dim_in] = NULL;
	isl_space_free(dim->nested[type - isl_dim_in]);
	dim->nested[type - isl_dim_in] = NULL;

	return dim;
}

__isl_give isl_space *isl_space_flatten(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	if (!dim->nested[0] && !dim->nested[1])
		return dim;

	if (dim->nested[0])
		dim = isl_space_reset(dim, isl_dim_in);
	if (dim && dim->nested[1])
		dim = isl_space_reset(dim, isl_dim_out);

	return dim;
}

__isl_give isl_space *isl_space_flatten_domain(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	if (!dim->nested[0])
		return dim;

	return isl_space_reset(dim, isl_dim_in);
}

__isl_give isl_space *isl_space_flatten_range(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	if (!dim->nested[1])
		return dim;

	return isl_space_reset(dim, isl_dim_out);
}

/* Replace the dimensions of the given type of dst by those of src.
 */
__isl_give isl_space *isl_space_replace(__isl_take isl_space *dst,
	enum isl_dim_type type, __isl_keep isl_space *src)
{
	dst = isl_space_cow(dst);

	if (!dst || !src)
		goto error;

	dst = isl_space_drop_dims(dst, type, 0, isl_space_dim(dst, type));
	dst = isl_space_add_dims(dst, type, isl_space_dim(src, type));
	dst = copy_ids(dst, type, 0, src, type);

	if (dst && type == isl_dim_param) {
		int i;
		for (i = 0; i <= 1; ++i) {
			if (!dst->nested[i])
				continue;
			dst->nested[i] = isl_space_replace(dst->nested[i],
							 type, src);
			if (!dst->nested[i])
				goto error;
		}
	}

	return dst;
error:
	isl_space_free(dst);
	return NULL;
}

/* Given a dimension specification "dim" of a set, create a dimension
 * specification for the lift of the set.  In particular, the result
 * is of the form [dim -> local[..]], with n_local variables in the
 * range of the wrapped map.
 */
__isl_give isl_space *isl_space_lift(__isl_take isl_space *dim, unsigned n_local)
{
	isl_space *local_dim;

	if (!dim)
		return NULL;

	local_dim = isl_space_dup(dim);
	local_dim = isl_space_drop_dims(local_dim, isl_dim_set, 0, dim->n_out);
	local_dim = isl_space_add_dims(local_dim, isl_dim_set, n_local);
	local_dim = isl_space_set_tuple_name(local_dim, isl_dim_set, "local");
	dim = isl_space_join(isl_space_from_domain(dim),
			    isl_space_from_range(local_dim));
	dim = isl_space_wrap(dim);
	dim = isl_space_set_tuple_name(dim, isl_dim_set, "lifted");

	return dim;
}

isl_bool isl_space_can_zip(__isl_keep isl_space *dim)
{
	if (!dim)
		return isl_bool_error;

	return dim->nested[0] && dim->nested[1];
}

__isl_give isl_space *isl_space_zip(__isl_take isl_space *dim)
{
	isl_space *dom, *ran;
	isl_space *dom_dom, *dom_ran, *ran_dom, *ran_ran;

	if (!isl_space_can_zip(dim))
		isl_die(dim->ctx, isl_error_invalid, "dim cannot be zipped",
			goto error);

	if (!dim)
		return NULL;
	dom = isl_space_unwrap(isl_space_domain(isl_space_copy(dim)));
	ran = isl_space_unwrap(isl_space_range(dim));
	dom_dom = isl_space_domain(isl_space_copy(dom));
	dom_ran = isl_space_range(dom);
	ran_dom = isl_space_domain(isl_space_copy(ran));
	ran_ran = isl_space_range(ran);
	dom = isl_space_join(isl_space_from_domain(dom_dom),
			   isl_space_from_range(ran_dom));
	ran = isl_space_join(isl_space_from_domain(dom_ran),
			   isl_space_from_range(ran_ran));
	return isl_space_join(isl_space_from_domain(isl_space_wrap(dom)),
			    isl_space_from_range(isl_space_wrap(ran)));
error:
	isl_space_free(dim);
	return NULL;
}

/* Can we apply isl_space_curry to "space"?
 * That is, does it have a nested relation in its domain?
 */
isl_bool isl_space_can_curry(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;

	return !!space->nested[0];
}

/* Given a space (A -> B) -> C, return the corresponding space
 * A -> (B -> C).
 */
__isl_give isl_space *isl_space_curry(__isl_take isl_space *space)
{
	isl_space *dom, *ran;
	isl_space *dom_dom, *dom_ran;

	if (!space)
		return NULL;

	if (!isl_space_can_curry(space))
		isl_die(space->ctx, isl_error_invalid,
			"space cannot be curried", goto error);

	dom = isl_space_unwrap(isl_space_domain(isl_space_copy(space)));
	ran = isl_space_range(space);
	dom_dom = isl_space_domain(isl_space_copy(dom));
	dom_ran = isl_space_range(dom);
	ran = isl_space_join(isl_space_from_domain(dom_ran),
			   isl_space_from_range(ran));
	return isl_space_join(isl_space_from_domain(dom_dom),
			    isl_space_from_range(isl_space_wrap(ran)));
error:
	isl_space_free(space);
	return NULL;
}

/* Can isl_space_range_curry be applied to "space"?
 * That is, does it have a nested relation in its range,
 * the domain of which is itself a nested relation?
 */
isl_bool isl_space_can_range_curry(__isl_keep isl_space *space)
{
	isl_bool can;

	if (!space)
		return isl_bool_error;
	can = isl_space_range_is_wrapping(space);
	if (can < 0 || !can)
		return can;
	return isl_space_can_curry(space->nested[1]);
}

/* Given a space A -> ((B -> C) -> D), return the corresponding space
 * A -> (B -> (C -> D)).
 */
__isl_give isl_space *isl_space_range_curry(__isl_take isl_space *space)
{
	if (!space)
		return NULL;

	if (!isl_space_can_range_curry(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"space range cannot be curried",
			return isl_space_free(space));

	space = isl_space_cow(space);
	if (!space)
		return NULL;
	space->nested[1] = isl_space_curry(space->nested[1]);
	if (!space->nested[1])
		return isl_space_free(space);

	return space;
}

/* Can we apply isl_space_uncurry to "space"?
 * That is, does it have a nested relation in its range?
 */
isl_bool isl_space_can_uncurry(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;

	return !!space->nested[1];
}

/* Given a space A -> (B -> C), return the corresponding space
 * (A -> B) -> C.
 */
__isl_give isl_space *isl_space_uncurry(__isl_take isl_space *space)
{
	isl_space *dom, *ran;
	isl_space *ran_dom, *ran_ran;

	if (!space)
		return NULL;

	if (!isl_space_can_uncurry(space))
		isl_die(space->ctx, isl_error_invalid,
			"space cannot be uncurried",
			return isl_space_free(space));

	dom = isl_space_domain(isl_space_copy(space));
	ran = isl_space_unwrap(isl_space_range(space));
	ran_dom = isl_space_domain(isl_space_copy(ran));
	ran_ran = isl_space_range(ran);
	dom = isl_space_join(isl_space_from_domain(dom),
			   isl_space_from_range(ran_dom));
	return isl_space_join(isl_space_from_domain(isl_space_wrap(dom)),
			    isl_space_from_range(ran_ran));
}

int isl_space_has_named_params(__isl_keep isl_space *dim)
{
	int i;
	unsigned off;

	if (!dim)
		return -1;
	if (dim->nparam == 0)
		return 1;
	off = isl_space_offset(dim, isl_dim_param);
	if (off + dim->nparam > dim->n_id)
		return 0;
	for (i = 0; i < dim->nparam; ++i)
		if (!dim->ids[off + i])
			return 0;
	return 1;
}

/* Align the initial parameters of dim1 to match the order in dim2.
 */
__isl_give isl_space *isl_space_align_params(__isl_take isl_space *dim1,
	__isl_take isl_space *dim2)
{
	isl_reordering *exp;

	if (!isl_space_has_named_params(dim1) || !isl_space_has_named_params(dim2))
		isl_die(isl_space_get_ctx(dim1), isl_error_invalid,
			"parameter alignment requires named parameters",
			goto error);

	dim2 = isl_space_params(dim2);
	exp = isl_parameter_alignment_reordering(dim1, dim2);
	exp = isl_reordering_extend_space(exp, dim1);
	isl_space_free(dim2);
	if (!exp)
		return NULL;
	dim1 = isl_space_copy(exp->dim);
	isl_reordering_free(exp);
	return dim1;
error:
	isl_space_free(dim1);
	isl_space_free(dim2);
	return NULL;
}

/* Given the space of set (domain), construct a space for a map
 * with as domain the given space and as range the range of "model".
 */
__isl_give isl_space *isl_space_extend_domain_with_range(
	__isl_take isl_space *space, __isl_take isl_space *model)
{
	if (!model)
		goto error;

	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out,
				    isl_space_dim(model, isl_dim_out));
	if (isl_space_has_tuple_id(model, isl_dim_out))
		space = isl_space_set_tuple_id(space, isl_dim_out,
				isl_space_get_tuple_id(model, isl_dim_out));
	if (!space)
		goto error;
	if (model->nested[1]) {
		isl_space *nested = isl_space_copy(model->nested[1]);
		int n_nested, n_space;
		nested = isl_space_align_params(nested, isl_space_copy(space));
		n_nested = isl_space_dim(nested, isl_dim_param);
		n_space = isl_space_dim(space, isl_dim_param);
		if (n_nested > n_space)
			nested = isl_space_drop_dims(nested, isl_dim_param,
						n_space, n_nested - n_space);
		if (!nested)
			goto error;
		space->nested[1] = nested;
	}
	isl_space_free(model);
	return space;
error:
	isl_space_free(model);
	isl_space_free(space);
	return NULL;
}

/* Compare the "type" dimensions of two isl_spaces.
 *
 * The order is fairly arbitrary.
 */
static int isl_space_cmp_type(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2, enum isl_dim_type type)
{
	int cmp;
	isl_space *nested1, *nested2;

	if (isl_space_dim(space1, type) != isl_space_dim(space2, type))
		return isl_space_dim(space1, type) -
			isl_space_dim(space2, type);

	cmp = isl_id_cmp(tuple_id(space1, type), tuple_id(space2, type));
	if (cmp != 0)
		return cmp;

	nested1 = nested(space1, type);
	nested2 = nested(space2, type);
	if (!nested1 != !nested2)
		return !nested1 - !nested2;

	if (nested1)
		return isl_space_cmp(nested1, nested2);

	return 0;
}

/* Compare two isl_spaces.
 *
 * The order is fairly arbitrary.
 */
int isl_space_cmp(__isl_keep isl_space *space1, __isl_keep isl_space *space2)
{
	int i;
	int cmp;

	if (space1 == space2)
		return 0;
	if (!space1)
		return -1;
	if (!space2)
		return 1;

	cmp = isl_space_cmp_type(space1, space2, isl_dim_param);
	if (cmp != 0)
		return cmp;
	cmp = isl_space_cmp_type(space1, space2, isl_dim_in);
	if (cmp != 0)
		return cmp;
	cmp = isl_space_cmp_type(space1, space2, isl_dim_out);
	if (cmp != 0)
		return cmp;

	if (!space1->ids && !space2->ids)
		return 0;

	for (i = 0; i < n(space1, isl_dim_param); ++i) {
		cmp = isl_id_cmp(get_id(space1, isl_dim_param, i),
				 get_id(space2, isl_dim_param, i));
		if (cmp != 0)
			return cmp;
	}

	return 0;
}
