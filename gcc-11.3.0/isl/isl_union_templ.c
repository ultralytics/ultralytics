/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

__isl_give UNION *FN(UNION,cow)(__isl_take UNION *u);

isl_ctx *FN(UNION,get_ctx)(__isl_keep UNION *u)
{
	return u ? u->space->ctx : NULL;
}

__isl_give isl_space *FN(UNION,get_space)(__isl_keep UNION *u)
{
	if (!u)
		return NULL;
	return isl_space_copy(u->space);
}

/* Return the number of parameters of "u", where "type"
 * is required to be set to isl_dim_param.
 */
unsigned FN(UNION,dim)(__isl_keep UNION *u, enum isl_dim_type type)
{
	if (!u)
		return 0;

	if (type != isl_dim_param)
		isl_die(FN(UNION,get_ctx)(u), isl_error_invalid,
			"can only reference parameters", return 0);

	return isl_space_dim(u->space, type);
}

/* Return the position of the parameter with the given name
 * in "u".
 * Return -1 if no such dimension can be found.
 */
int FN(UNION,find_dim_by_name)(__isl_keep UNION *u, enum isl_dim_type type,
	const char *name)
{
	if (!u)
		return -1;
	return isl_space_find_dim_by_name(u->space, type, name);
}

#ifdef HAS_TYPE
static __isl_give UNION *FN(UNION,alloc)(__isl_take isl_space *dim,
	enum isl_fold type, int size)
#else
static __isl_give UNION *FN(UNION,alloc)(__isl_take isl_space *dim, int size)
#endif
{
	UNION *u;

	dim = isl_space_params(dim);
	if (!dim)
		return NULL;

	u = isl_calloc_type(dim->ctx, UNION);
	if (!u)
		goto error;

	u->ref = 1;
#ifdef HAS_TYPE
	u->type = type;
#endif
	u->space = dim;
	if (isl_hash_table_init(dim->ctx, &u->table, size) < 0)
		return FN(UNION,free)(u);

	return u;
error:
	isl_space_free(dim);
	return NULL;
}

#ifdef HAS_TYPE
__isl_give UNION *FN(UNION,ZERO)(__isl_take isl_space *dim, enum isl_fold type)
{
	return FN(UNION,alloc)(dim, type, 16);
}
#else
__isl_give UNION *FN(UNION,ZERO)(__isl_take isl_space *dim)
{
	return FN(UNION,alloc)(dim, 16);
}
#endif

__isl_give UNION *FN(UNION,copy)(__isl_keep UNION *u)
{
	if (!u)
		return NULL;

	u->ref++;
	return u;
}

/* Extract the element of "u" living in "space" (ignoring parameters).
 *
 * Return the ZERO element if "u" does not contain any element
 * living in "space".
 */
__isl_give PART *FN(FN(UNION,extract),PARTS)(__isl_keep UNION *u,
	__isl_take isl_space *space)
{
	struct isl_hash_table_entry *entry;

	if (!u || !space)
		goto error;
	if (!isl_space_match(u->space, isl_dim_param, space, isl_dim_param)) {
		space = isl_space_drop_dims(space, isl_dim_param,
					0, isl_space_dim(space, isl_dim_param));
		space = isl_space_align_params(space,
					FN(UNION,get_space)(u));
		if (!space)
			goto error;
	}

	entry = FN(UNION,find_part_entry)(u, space, 0);
	if (!entry)
		goto error;
	if (entry == isl_hash_table_entry_none)
#ifdef HAS_TYPE
		return FN(PART,ZERO)(space, u->type);
#else
		return FN(PART,ZERO)(space);
#endif
	isl_space_free(space);
	return FN(PART,copy)(entry->data);
error:
	isl_space_free(space);
	return NULL;
}

/* Add "part" to "u".
 * If "disjoint" is set, then "u" is not allowed to already have
 * a part that is defined over a domain that overlaps with the domain
 * of "part".
 * Otherwise, compute the union sum of "part" and the part in "u"
 * defined on the same space.
 */
static __isl_give UNION *FN(UNION,add_part_generic)(__isl_take UNION *u,
	__isl_take PART *part, int disjoint)
{
	int empty;
	struct isl_hash_table_entry *entry;

	if (!part)
		goto error;

	empty = FN(PART,IS_ZERO)(part);
	if (empty < 0)
		goto error;
	if (empty) {
		FN(PART,free)(part);
		return u;
	}

	u = FN(UNION,align_params)(u, FN(PART,get_space)(part));
	part = FN(PART,align_params)(part, FN(UNION,get_space)(u));

	u = FN(UNION,cow)(u);

	if (!u)
		goto error;

	if (FN(UNION,check_disjoint_domain_other)(u, part) < 0)
		goto error;
	entry = FN(UNION,find_part_entry)(u, part->dim, 1);
	if (!entry)
		goto error;

	if (!entry->data)
		entry->data = part;
	else {
		if (disjoint &&
		    FN(UNION,check_disjoint_domain)(entry->data, part) < 0)
			goto error;
		entry->data = FN(PART,union_add_)(entry->data,
						FN(PART,copy)(part));
		if (!entry->data)
			goto error;
		empty = FN(PART,IS_ZERO)(part);
		if (empty < 0)
			goto error;
		if (empty)
			u = FN(UNION,remove_part_entry)(u, entry);
		FN(PART,free)(part);
	}

	return u;
error:
	FN(PART,free)(part);
	FN(UNION,free)(u);
	return NULL;
}

/* Add "part" to "u", where "u" is assumed not to already have
 * a part that is defined on the same space as "part".
 */
__isl_give UNION *FN(FN(UNION,add),PARTS)(__isl_take UNION *u,
	__isl_take PART *part)
{
	return FN(UNION,add_part_generic)(u, part, 1);
}

#ifdef HAS_TYPE
/* Allocate a UNION with the same type and the same size as "u" and
 * with space "space".
 */
static __isl_give UNION *FN(UNION,alloc_same_size_on_space)(__isl_keep UNION *u,
	__isl_take isl_space *space)
{
	if (!u)
		goto error;
	return FN(UNION,alloc)(space, u->type, u->table.n);
error:
	isl_space_free(space);
	return NULL;
}
#else
/* Allocate a UNION with the same size as "u" and with space "space".
 */
static __isl_give UNION *FN(UNION,alloc_same_size_on_space)(__isl_keep UNION *u,
	__isl_take isl_space *space)
{
	if (!u)
		goto error;
	return FN(UNION,alloc)(space, u->table.n);
error:
	isl_space_free(space);
	return NULL;
}
#endif

/* Allocate a UNION with the same space, the same type (if any) and
 * the same size as "u".
 */
static __isl_give UNION *FN(UNION,alloc_same_size)(__isl_keep UNION *u)
{
	return FN(UNION,alloc_same_size_on_space)(u, FN(UNION,get_space)(u));
}

/* Internal data structure for isl_union_*_transform_space.
 * "fn' is applied to each entry in the input.
 * "res" collects the results.
 */
S(UNION,transform_data)
{
	__isl_give PART *(*fn)(__isl_take PART *part, void *user);
	void *user;

	UNION *res;
};

/* Apply data->fn to "part" and add the result to data->res.
 */
static isl_stat FN(UNION,transform_entry)(__isl_take PART *part, void *user)
{
	S(UNION,transform_data) *data = (S(UNION,transform_data) *)user;

	part = data->fn(part, data->user);
	data->res = FN(FN(UNION,add),PARTS)(data->res, part);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Return a UNION living in "space" that is obtained by applying "fn"
 * to each of the entries in "u".
 */
static __isl_give UNION *FN(UNION,transform_space)(__isl_take UNION *u,
	isl_space *space,
	__isl_give PART *(*fn)(__isl_take PART *part, void *user), void *user)
{
	S(UNION,transform_data) data = { fn, user };

	data.res = FN(UNION,alloc_same_size_on_space)(u, space);
	if (FN(FN(UNION,foreach),PARTS)(u,
					&FN(UNION,transform_entry), &data) < 0)
		data.res = FN(UNION,free)(data.res);
	FN(UNION,free)(u);
	return data.res;
}

/* Return a UNION that lives in the same space as "u" and that is obtained
 * by applying "fn" to each of the entries in "u".
 */
static __isl_give UNION *FN(UNION,transform)(__isl_take UNION *u,
	__isl_give PART *(*fn)(__isl_take PART *part, void *user), void *user)
{
	return FN(UNION,transform_space)(u, FN(UNION,get_space)(u), fn, user);
}

/* Apply data->fn to *part and store the result back into *part.
 */
static isl_stat FN(UNION,transform_inplace_entry)(void **part, void *user)
{
	S(UNION,transform_data) *data = (S(UNION,transform_data) *) user;

	*part = data->fn(*part, data->user);
	if (!*part)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Update "u" by applying "fn" to each entry.
 * This operation is assumed not to change the number of entries nor
 * the spaces of the entries.
 *
 * If there is only one reference to "u", then change "u" inplace.
 * Otherwise, create a new UNION from "u" and discard the original.
 */
static __isl_give UNION *FN(UNION,transform_inplace)(__isl_take UNION *u,
	__isl_give PART *(*fn)(__isl_take PART *part, void *user), void *user)
{
	isl_bool single_ref;

	single_ref = FN(UNION,has_single_reference)(u);
	if (single_ref < 0)
		return FN(UNION,free)(u);
	if (single_ref) {
		S(UNION,transform_data) data = { fn, user };
		if (FN(UNION,foreach_inplace)(u,
				&FN(UNION,transform_inplace_entry), &data) < 0)
			return FN(UNION,free)(u);
		return u;
	}
	return FN(UNION,transform)(u, fn, user);
}

/* An isl_union_*_transform callback for use in isl_union_*_dup
 * that simply returns "part".
 */
static __isl_give PART *FN(UNION,copy_part)(__isl_take PART *part, void *user)
{
	return part;
}

__isl_give UNION *FN(UNION,dup)(__isl_keep UNION *u)
{
	u = FN(UNION,copy)(u);
	return FN(UNION,transform)(u, &FN(UNION,copy_part), NULL);
}

__isl_give UNION *FN(UNION,cow)(__isl_take UNION *u)
{
	if (!u)
		return NULL;

	if (u->ref == 1)
		return u;
	u->ref--;
	return FN(UNION,dup)(u);
}

__isl_null UNION *FN(UNION,free)(__isl_take UNION *u)
{
	if (!u)
		return NULL;

	if (--u->ref > 0)
		return NULL;

	isl_hash_table_foreach(u->space->ctx, &u->table,
				&FN(UNION,free_u_entry), NULL);
	isl_hash_table_clear(&u->table);
	isl_space_free(u->space);
	free(u);
	return NULL;
}

static __isl_give PART *FN(UNION,align_entry)(__isl_take PART *part, void *user)
{
	isl_reordering *exp = user;

	exp = isl_reordering_extend_space(isl_reordering_copy(exp),
				    FN(PART,get_domain_space)(part));
	return FN(PART,realign_domain)(part, exp);
}

/* Reorder the parameters of "u" according to the given reordering.
 */
static __isl_give UNION *FN(UNION,realign_domain)(__isl_take UNION *u,
	__isl_take isl_reordering *r)
{
	isl_space *space;

	if (!u || !r)
		goto error;

	space = isl_space_copy(r->dim);
	u = FN(UNION,transform_space)(u, space, &FN(UNION,align_entry), r);
	isl_reordering_free(r);
	return u;
error:
	FN(UNION,free)(u);
	isl_reordering_free(r);
	return NULL;
}

/* Align the parameters of "u" to those of "model".
 */
__isl_give UNION *FN(UNION,align_params)(__isl_take UNION *u,
	__isl_take isl_space *model)
{
	isl_reordering *r;

	if (!u || !model)
		goto error;

	if (isl_space_match(u->space, isl_dim_param, model, isl_dim_param)) {
		isl_space_free(model);
		return u;
	}

	model = isl_space_params(model);
	r = isl_parameter_alignment_reordering(u->space, model);
	isl_space_free(model);

	return FN(UNION,realign_domain)(u, r);
error:
	isl_space_free(model);
	FN(UNION,free)(u);
	return NULL;
}

/* Add "part" to *u, taking the union sum if "u" already has
 * a part defined on the same space as "part".
 */
static isl_stat FN(UNION,union_add_part)(__isl_take PART *part, void *user)
{
	UNION **u = (UNION **)user;

	*u = FN(UNION,add_part_generic)(*u, part, 0);

	return isl_stat_ok;
}

/* Compute the sum of "u1" and "u2" on the union of their domains,
 * with the actual sum on the shared domain and
 * the defined expression on the symmetric difference of the domains.
 *
 * This is an internal function that is exposed under different
 * names depending on whether the base expressions have a zero default
 * value.
 * If they do, then this function is called "add".
 * Otherwise, it is called "union_add".
 */
static __isl_give UNION *FN(UNION,union_add_)(__isl_take UNION *u1,
	__isl_take UNION *u2)
{
	u1 = FN(UNION,align_params)(u1, FN(UNION,get_space)(u2));
	u2 = FN(UNION,align_params)(u2, FN(UNION,get_space)(u1));

	u1 = FN(UNION,cow)(u1);

	if (!u1 || !u2)
		goto error;

	if (FN(FN(UNION,foreach),PARTS)(u2, &FN(UNION,union_add_part), &u1) < 0)
		goto error;

	FN(UNION,free)(u2);

	return u1;
error:
	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	return NULL;
}

__isl_give UNION *FN(FN(UNION,from),PARTS)(__isl_take PART *part)
{
	isl_space *dim;
	UNION *u;

	if (!part)
		return NULL;

	dim = FN(PART,get_space)(part);
	dim = isl_space_drop_dims(dim, isl_dim_in, 0, isl_space_dim(dim, isl_dim_in));
	dim = isl_space_drop_dims(dim, isl_dim_out, 0, isl_space_dim(dim, isl_dim_out));
#ifdef HAS_TYPE
	u = FN(UNION,ZERO)(dim, part->type);
#else
	u = FN(UNION,ZERO)(dim);
#endif
	u = FN(FN(UNION,add),PARTS)(u, part);

	return u;
}

S(UNION,match_bin_data) {
	UNION *u2;
	UNION *res;
	__isl_give PART *(*fn)(__isl_take PART *, __isl_take PART *);
};

/* Check if data->u2 has an element living in the same space as "part".
 * If so, call data->fn on the two elements and add the result to
 * data->res.
 */
static isl_stat FN(UNION,match_bin_entry)(__isl_take PART *part, void *user)
{
	S(UNION,match_bin_data) *data = user;
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	PART *part2;

	space = FN(PART,get_space)(part);
	entry2 = FN(UNION,find_part_entry)(data->u2, space, 0);
	isl_space_free(space);
	if (!entry2)
		goto error;
	if (entry2 == isl_hash_table_entry_none) {
		FN(PART,free)(part);
		return isl_stat_ok;
	}

	part2 = entry2->data;
	if (!isl_space_tuple_is_equal(part->dim, isl_dim_out,
					part2->dim, isl_dim_out))
		isl_die(FN(UNION,get_ctx)(data->u2), isl_error_invalid,
			"entries should have the same range space",
			goto error);

	part = data->fn(part, FN(PART, copy)(entry2->data));

	data->res = FN(FN(UNION,add),PARTS)(data->res, part);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
error:
	FN(PART,free)(part);
	return isl_stat_error;
}

/* This function is currently only used from isl_polynomial.c
 * and not from isl_fold.c.
 */
static __isl_give UNION *FN(UNION,match_bin_op)(__isl_take UNION *u1,
	__isl_take UNION *u2,
	__isl_give PART *(*fn)(__isl_take PART *, __isl_take PART *))
	__attribute__ ((unused));
/* For each pair of elements in "u1" and "u2" living in the same space,
 * call "fn" and collect the results.
 */
static __isl_give UNION *FN(UNION,match_bin_op)(__isl_take UNION *u1,
	__isl_take UNION *u2,
	__isl_give PART *(*fn)(__isl_take PART *, __isl_take PART *))
{
	S(UNION,match_bin_data) data = { NULL, NULL, fn };

	u1 = FN(UNION,align_params)(u1, FN(UNION,get_space)(u2));
	u2 = FN(UNION,align_params)(u2, FN(UNION,get_space)(u1));

	if (!u1 || !u2)
		goto error;

	data.u2 = u2;
	data.res = FN(UNION,alloc_same_size)(u1);
	if (FN(FN(UNION,foreach),PARTS)(u1,
				    &FN(UNION,match_bin_entry), &data) < 0)
		goto error;

	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	return data.res;
error:
	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	FN(UNION,free)(data.res);
	return NULL;
}

/* Compute the sum of "u1" and "u2".
 *
 * If the base expressions have a default zero value, then the sum
 * is computed on the union of the domains of "u1" and "u2".
 * Otherwise, it is computed on their shared domains.
 */
__isl_give UNION *FN(UNION,add)(__isl_take UNION *u1, __isl_take UNION *u2)
{
#if DEFAULT_IS_ZERO
	return FN(UNION,union_add_)(u1, u2);
#else
	return FN(UNION,match_bin_op)(u1, u2, &FN(PART,add));
#endif
}

#ifndef NO_SUB
/* Subtract "u2" from "u1" and return the result.
 */
__isl_give UNION *FN(UNION,sub)(__isl_take UNION *u1, __isl_take UNION *u2)
{
	return FN(UNION,match_bin_op)(u1, u2, &FN(PART,sub));
}
#endif

S(UNION,any_set_data) {
	isl_set *set;
	__isl_give PW *(*fn)(__isl_take PW*, __isl_take isl_set*);
};

static __isl_give PART *FN(UNION,any_set_entry)(__isl_take PART *part,
	void *user)
{
	S(UNION,any_set_data) *data = user;

	return data->fn(part, isl_set_copy(data->set));
}

/* Update each element of "u" by calling "fn" on the element and "set".
 */
static __isl_give UNION *FN(UNION,any_set_op)(__isl_take UNION *u,
	__isl_take isl_set *set,
	__isl_give PW *(*fn)(__isl_take PW*, __isl_take isl_set*))
{
	S(UNION,any_set_data) data = { NULL, fn };

	u = FN(UNION,align_params)(u, isl_set_get_space(set));
	set = isl_set_align_params(set, FN(UNION,get_space)(u));

	if (!u || !set)
		goto error;

	data.set = set;
	u = FN(UNION,transform)(u, &FN(UNION,any_set_entry), &data);
	isl_set_free(set);
	return u;
error:
	FN(UNION,free)(u);
	isl_set_free(set);
	return NULL;
}

/* Intersect the domain of "u" with the parameter domain "context".
 */
__isl_give UNION *FN(UNION,intersect_params)(__isl_take UNION *u,
	__isl_take isl_set *set)
{
	return FN(UNION,any_set_op)(u, set, &FN(PW,intersect_params));
}

/* Compute the gist of the domain of "u" with respect to
 * the parameter domain "context".
 */
__isl_give UNION *FN(UNION,gist_params)(__isl_take UNION *u,
	__isl_take isl_set *set)
{
	return FN(UNION,any_set_op)(u, set, &FN(PW,gist_params));
}

S(UNION,match_domain_data) {
	isl_union_set *uset;
	UNION *res;
	__isl_give PW *(*fn)(__isl_take PW*, __isl_take isl_set*);
};

static int FN(UNION,set_has_dim)(const void *entry, const void *val)
{
	isl_set *set = (isl_set *)entry;
	isl_space *dim = (isl_space *)val;

	return isl_space_is_equal(set->dim, dim);
}

/* Find the set in data->uset that lives in the same space as the domain
 * of "part", apply data->fn to *entry and this set (if any), and add
 * the result to data->res.
 */
static isl_stat FN(UNION,match_domain_entry)(__isl_take PART *part, void *user)
{
	S(UNION,match_domain_data) *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_space *space;

	space = FN(PART,get_domain_space)(part);
	hash = isl_space_get_hash(space);
	entry2 = isl_hash_table_find(data->uset->dim->ctx, &data->uset->table,
				     hash, &FN(UNION,set_has_dim), space, 0);
	isl_space_free(space);
	if (!entry2) {
		FN(PART,free)(part);
		return isl_stat_ok;
	}

	part = data->fn(part, isl_set_copy(entry2->data));

	data->res = FN(FN(UNION,add),PARTS)(data->res, part);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Apply fn to each pair of PW in u and set in uset such that
 * the set lives in the same space as the domain of PW
 * and collect the results.
 */
static __isl_give UNION *FN(UNION,match_domain_op)(__isl_take UNION *u,
	__isl_take isl_union_set *uset,
	__isl_give PW *(*fn)(__isl_take PW*, __isl_take isl_set*))
{
	S(UNION,match_domain_data) data = { NULL, NULL, fn };

	u = FN(UNION,align_params)(u, isl_union_set_get_space(uset));
	uset = isl_union_set_align_params(uset, FN(UNION,get_space)(u));

	if (!u || !uset)
		goto error;

	data.uset = uset;
	data.res = FN(UNION,alloc_same_size)(u);
	if (FN(FN(UNION,foreach),PARTS)(u,
				   &FN(UNION,match_domain_entry), &data) < 0)
		goto error;

	FN(UNION,free)(u);
	isl_union_set_free(uset);
	return data.res;
error:
	FN(UNION,free)(u);
	isl_union_set_free(uset);
	FN(UNION,free)(data.res);
	return NULL;
}

/* Intersect the domain of "u" with "uset".
 * If "uset" is a parameters domain, then intersect the parameter
 * domain of "u" with this set.
 */
__isl_give UNION *FN(UNION,intersect_domain)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	if (isl_union_set_is_params(uset))
		return FN(UNION,intersect_params)(u,
						isl_set_from_union_set(uset));
	return FN(UNION,match_domain_op)(u, uset, &FN(PW,intersect_domain));
}

/* Take the set (which may be empty) in data->uset that lives
 * in the same space as the domain of "pw", subtract it from the domain
 * of "part" and return the result.
 */
static __isl_give PART *FN(UNION,subtract_domain_entry)(__isl_take PART *part,
	void *user)
{
	isl_union_set *uset = user;
	isl_space *space;
	isl_set *set;

	space = FN(PART,get_domain_space)(part);
	set = isl_union_set_extract_set(uset, space);
	return FN(PART,subtract_domain)(part, set);
}

/* Subtract "uset' from the domain of "u".
 */
__isl_give UNION *FN(UNION,subtract_domain)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	u = FN(UNION,transform)(u, &FN(UNION,subtract_domain_entry), uset);
	isl_union_set_free(uset);
	return u;
}

__isl_give UNION *FN(UNION,gist)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	if (isl_union_set_is_params(uset))
		return FN(UNION,gist_params)(u, isl_set_from_union_set(uset));
	return FN(UNION,match_domain_op)(u, uset, &FN(PW,gist));
}

/* Coalesce an entry in a UNION.  Coalescing is performed in-place.
 * Since the UNION may have several references, the entry is only
 * replaced if the coalescing is successful.
 */
static isl_stat FN(UNION,coalesce_entry)(void **entry, void *user)
{
	PART **part_p = (PART **) entry;
	PART *part;

	part = FN(PART,copy)(*part_p);
	part = FN(PW,coalesce)(part);
	if (!part)
		return isl_stat_error;
	FN(PART,free)(*part_p);
	*part_p = part;

	return isl_stat_ok;
}

__isl_give UNION *FN(UNION,coalesce)(__isl_take UNION *u)
{
	if (FN(UNION,foreach_inplace)(u, &FN(UNION,coalesce_entry), NULL) < 0)
		goto error;

	return u;
error:
	FN(UNION,free)(u);
	return NULL;
}

static isl_stat FN(UNION,domain_entry)(__isl_take PART *part, void *user)
{
	isl_union_set **uset = (isl_union_set **)user;

	*uset = isl_union_set_add_set(*uset, FN(PART,domain)(part));

	return isl_stat_ok;
}

__isl_give isl_union_set *FN(UNION,domain)(__isl_take UNION *u)
{
	isl_union_set *uset;

	uset = isl_union_set_empty(FN(UNION,get_space)(u));
	if (FN(FN(UNION,foreach),PARTS)(u, &FN(UNION,domain_entry), &uset) < 0)
		goto error;

	FN(UNION,free)(u);
	
	return uset;
error:
	isl_union_set_free(uset);
	FN(UNION,free)(u);
	return NULL;
}

#ifdef HAS_TYPE
/* Negate the type of "u".
 */
static __isl_give UNION *FN(UNION,negate_type)(__isl_take UNION *u)
{
	u = FN(UNION,cow)(u);
	if (!u)
		return NULL;
	u->type = isl_fold_type_negate(u->type);
	return u;
}
#else
/* Negate the type of "u".
 * Since "u" does not have a type, do nothing.
 */
static __isl_give UNION *FN(UNION,negate_type)(__isl_take UNION *u)
{
	return u;
}
#endif

static __isl_give PART *FN(UNION,mul_isl_int_entry)(__isl_take PART *part,
	void *user)
{
	isl_int *v = user;

	return FN(PW,mul_isl_int)(part, *v);
}

__isl_give UNION *FN(UNION,mul_isl_int)(__isl_take UNION *u, isl_int v)
{
	if (isl_int_is_one(v))
		return u;

	if (DEFAULT_IS_ZERO && u && isl_int_is_zero(v)) {
		UNION *zero;
		isl_space *dim = FN(UNION,get_space)(u);
#ifdef HAS_TYPE
		zero = FN(UNION,ZERO)(dim, u->type);
#else
		zero = FN(UNION,ZERO)(dim);
#endif
		FN(UNION,free)(u);
		return zero;
	}

	u = FN(UNION,transform_inplace)(u, &FN(UNION,mul_isl_int_entry), &v);
	if (isl_int_is_neg(v))
		u = FN(UNION,negate_type)(u);

	return u;
}

/* Multiply "part" by the isl_val "user" and return the result.
 */
static __isl_give PART *FN(UNION,scale_val_entry)(__isl_take PART *part,
	void *user)
{
	isl_val *v = user;

	return FN(PART,scale_val)(part, isl_val_copy(v));
}

/* Multiply "u" by "v" and return the result.
 */
__isl_give UNION *FN(UNION,scale_val)(__isl_take UNION *u,
	__isl_take isl_val *v)
{
	if (!u || !v)
		goto error;
	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return u;
	}

	if (DEFAULT_IS_ZERO && u && isl_val_is_zero(v)) {
		UNION *zero;
		isl_space *space = FN(UNION,get_space)(u);
#ifdef HAS_TYPE
		zero = FN(UNION,ZERO)(space, u->type);
#else
		zero = FN(UNION,ZERO)(space);
#endif
		FN(UNION,free)(u);
		isl_val_free(v);
		return zero;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational factor", goto error);

	u = FN(UNION,transform_inplace)(u, &FN(UNION,scale_val_entry), v);
	if (isl_val_is_neg(v))
		u = FN(UNION,negate_type)(u);

	isl_val_free(v);
	return u;
error:
	isl_val_free(v);
	FN(UNION,free)(u);
	return NULL;
}

/* Divide "part" by the isl_val "user" and return the result.
 */
static __isl_give PART *FN(UNION,scale_down_val_entry)(__isl_take PART *part,
	void *user)
{
	isl_val *v = user;

	return FN(PART,scale_down_val)(part, isl_val_copy(v));
}

/* Divide "u" by "v" and return the result.
 */
__isl_give UNION *FN(UNION,scale_down_val)(__isl_take UNION *u,
	__isl_take isl_val *v)
{
	if (!u || !v)
		goto error;
	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return u;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational factor", goto error);
	if (isl_val_is_zero(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"cannot scale down by zero", goto error);

	u = FN(UNION,transform_inplace)(u, &FN(UNION,scale_down_val_entry), v);
	if (isl_val_is_neg(v))
		u = FN(UNION,negate_type)(u);

	isl_val_free(v);
	return u;
error:
	isl_val_free(v);
	FN(UNION,free)(u);
	return NULL;
}

S(UNION,plain_is_equal_data)
{
	UNION *u2;
	isl_bool is_equal;
};

static isl_stat FN(UNION,plain_is_equal_entry)(void **entry, void *user)
{
	S(UNION,plain_is_equal_data) *data = user;
	struct isl_hash_table_entry *entry2;
	PW *pw = *entry;

	entry2 = FN(UNION,find_part_entry)(data->u2, pw->dim, 0);
	if (!entry2 || entry2 == isl_hash_table_entry_none) {
		if (!entry2)
			data->is_equal = isl_bool_error;
		else
			data->is_equal = isl_bool_false;
		return isl_stat_error;
	}

	data->is_equal = FN(PW,plain_is_equal)(pw, entry2->data);
	if (data->is_equal < 0 || !data->is_equal)
		return isl_stat_error;

	return isl_stat_ok;
}

isl_bool FN(UNION,plain_is_equal)(__isl_keep UNION *u1, __isl_keep UNION *u2)
{
	S(UNION,plain_is_equal_data) data = { NULL, isl_bool_true };
	int n1, n2;

	if (!u1 || !u2)
		return isl_bool_error;
	if (u1 == u2)
		return isl_bool_true;
	if (u1->table.n != u2->table.n)
		return isl_bool_false;
	n1 = FN(FN(UNION,n),PARTS)(u1);
	n2 = FN(FN(UNION,n),PARTS)(u2);
	if (n1 < 0 || n2 < 0)
		return isl_bool_error;
	if (n1 != n2)
		return isl_bool_false;

	u1 = FN(UNION,copy)(u1);
	u2 = FN(UNION,copy)(u2);
	u1 = FN(UNION,align_params)(u1, FN(UNION,get_space)(u2));
	u2 = FN(UNION,align_params)(u2, FN(UNION,get_space)(u1));
	if (!u1 || !u2)
		goto error;

	data.u2 = u2;
	if (FN(UNION,foreach_inplace)(u1,
			       &FN(UNION,plain_is_equal_entry), &data) < 0 &&
	    data.is_equal)
		goto error;

	FN(UNION,free)(u1);
	FN(UNION,free)(u2);

	return data.is_equal;
error:
	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	return isl_bool_error;
}

/* Internal data structure for isl_union_*_drop_dims.
 * type, first and n are passed to isl_*_drop_dims.
 */
S(UNION,drop_dims_data) {
	enum isl_dim_type type;
	unsigned first;
	unsigned n;
};

/* Drop the parameters specified by "data" from "part" and return the result.
 */
static __isl_give PART *FN(UNION,drop_dims_entry)(__isl_take PART *part,
	void *user)
{
	S(UNION,drop_dims_data) *data = user;

	return FN(PART,drop_dims)(part, data->type, data->first, data->n);
}

/* Drop the specified parameters from "u".
 * That is, type is required to be isl_dim_param.
 */
__isl_give UNION *FN(UNION,drop_dims)( __isl_take UNION *u,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_space *space;
	S(UNION,drop_dims_data) data = { type, first, n };

	if (!u)
		return NULL;

	if (type != isl_dim_param)
		isl_die(FN(UNION,get_ctx)(u), isl_error_invalid,
			"can only project out parameters",
			return FN(UNION,free)(u));

	space = FN(UNION,get_space)(u);
	space = isl_space_drop_dims(space, type, first, n);
	return FN(UNION,transform_space)(u, space, &FN(UNION,drop_dims_entry),
					&data);
}

/* Internal data structure for isl_union_*_set_dim_name.
 * pos is the position of the parameter that needs to be renamed.
 * s is the new name.
 */
S(UNION,set_dim_name_data) {
	unsigned pos;
	const char *s;
};

/* Change the name of the parameter at position data->pos of "part" to data->s
 * and return the result.
 */
static __isl_give PART *FN(UNION,set_dim_name_entry)(__isl_take PART *part,
	void *user)
{
	S(UNION,set_dim_name_data) *data = user;

	return FN(PART,set_dim_name)(part, isl_dim_param, data->pos, data->s);
}

/* Change the name of the parameter at position "pos" to "s".
 * That is, type is required to be isl_dim_param.
 */
__isl_give UNION *FN(UNION,set_dim_name)(__isl_take UNION *u,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	S(UNION,set_dim_name_data) data = { pos, s };
	isl_space *space;

	if (!u)
		return NULL;

	if (type != isl_dim_param)
		isl_die(FN(UNION,get_ctx)(u), isl_error_invalid,
			"can only set parameter names",
			return FN(UNION,free)(u));

	space = FN(UNION,get_space)(u);
	space = isl_space_set_dim_name(space, type, pos, s);
	return FN(UNION,transform_space)(u, space,
					&FN(UNION,set_dim_name_entry), &data);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of "part" and return the result.
 */
static __isl_give PART *FN(UNION,reset_user_entry)(__isl_take PART *part,
	void *user)
{
	return FN(PART,reset_user)(part);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the spaces of "u".
 */
__isl_give UNION *FN(UNION,reset_user)(__isl_take UNION *u)
{
	isl_space *space;

	space = FN(UNION,get_space)(u);
	space = isl_space_reset_user(space);
	return FN(UNION,transform_space)(u, space, &FN(UNION,reset_user_entry),
					NULL);
}
