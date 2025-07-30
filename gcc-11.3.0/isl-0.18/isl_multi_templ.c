/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl_space_private.h>
#include <isl/set.h>
#include <isl_reordering.h>

#include <isl_multi_macro.h>

#define MULTI_NAME(BASE) "isl_multi_" #BASE
#define xLIST(EL) EL ## _list
#define LIST(EL) xLIST(EL)

isl_ctx *FN(MULTI(BASE),get_ctx)(__isl_keep MULTI(BASE) *multi)
{
	return multi ? isl_space_get_ctx(multi->space) : NULL;
}

__isl_give isl_space *FN(MULTI(BASE),get_space)(__isl_keep MULTI(BASE) *multi)
{
	return multi ? isl_space_copy(multi->space) : NULL;
}

/* Return the position of the dimension of the given type and name
 * in "multi".
 * Return -1 if no such dimension can be found.
 */
int FN(MULTI(BASE),find_dim_by_name)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, const char *name)
{
	if (!multi)
		return -1;
	return isl_space_find_dim_by_name(multi->space, type, name);
}

__isl_give isl_space *FN(MULTI(BASE),get_domain_space)(
	__isl_keep MULTI(BASE) *multi)
{
	return multi ? isl_space_domain(isl_space_copy(multi->space)) : NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),alloc)(__isl_take isl_space *space)
{
	isl_ctx *ctx;
	int n;
	MULTI(BASE) *multi;

	if (!space)
		return NULL;

	ctx = isl_space_get_ctx(space);
	n = isl_space_dim(space, isl_dim_out);
	multi = isl_calloc(ctx, MULTI(BASE),
			 sizeof(MULTI(BASE)) + (n - 1) * sizeof(struct EL *));
	if (!multi)
		goto error;

	multi->space = space;
	multi->n = n;
	multi->ref = 1;
	return multi;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),dup)(__isl_keep MULTI(BASE) *multi)
{
	int i;
	MULTI(BASE) *dup;

	if (!multi)
		return NULL;

	dup = FN(MULTI(BASE),alloc)(isl_space_copy(multi->space));
	if (!dup)
		return NULL;

	for (i = 0; i < multi->n; ++i)
		dup = FN(FN(MULTI(BASE),set),BASE)(dup, i,
						    FN(EL,copy)(multi->p[i]));

	return dup;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),cow)(__isl_take MULTI(BASE) *multi)
{
	if (!multi)
		return NULL;

	if (multi->ref == 1)
		return multi;

	multi->ref--;
	return FN(MULTI(BASE),dup)(multi);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),copy)(__isl_keep MULTI(BASE) *multi)
{
	if (!multi)
		return NULL;

	multi->ref++;
	return multi;
}

__isl_null MULTI(BASE) *FN(MULTI(BASE),free)(__isl_take MULTI(BASE) *multi)
{
	int i;

	if (!multi)
		return NULL;

	if (--multi->ref > 0)
		return NULL;

	isl_space_free(multi->space);
	for (i = 0; i < multi->n; ++i)
		FN(EL,free)(multi->p[i]);
	free(multi);

	return NULL;
}

#ifndef NO_DIMS
/* Check whether "multi" has non-zero coefficients for any dimension
 * in the given range or if any of these dimensions appear
 * with non-zero coefficients in any of the integer divisions involved.
 */
isl_bool FN(MULTI(BASE),involves_dims)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!multi)
		return isl_bool_error;
	if (multi->n == 0 || n == 0)
		return isl_bool_false;

	for (i = 0; i < multi->n; ++i) {
		isl_bool involves;

		involves = FN(EL,involves_dims)(multi->p[i], type, first, n);
		if (involves < 0 || involves)
			return involves;
	}

	return isl_bool_false;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),insert_dims)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!multi)
		return NULL;
	if (type == isl_dim_out)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"cannot insert output/set dimensions",
			return FN(MULTI(BASE),free)(multi));
	if (n == 0 && !isl_space_is_named_or_nested(multi->space, type))
		return multi;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	multi->space = isl_space_insert_dims(multi->space, type, first, n);
	if (!multi->space)
		return FN(MULTI(BASE),free)(multi);

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,insert_dims)(multi->p[i], type, first, n);
		if (!multi->p[i])
			return FN(MULTI(BASE),free)(multi);
	}

	return multi;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),add_dims)(__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned n)
{
	unsigned pos;

	pos = FN(MULTI(BASE),dim)(multi, type);

	return FN(MULTI(BASE),insert_dims)(multi, type, pos, n);
}
#endif

unsigned FN(MULTI(BASE),dim)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	return multi ? isl_space_dim(multi->space, type) : 0;
}

/* Return the position of the first dimension of "type" with id "id".
 * Return -1 if there is no such dimension.
 */
int FN(MULTI(BASE),find_dim_by_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, __isl_keep isl_id *id)
{
	if (!multi)
		return -1;
	return isl_space_find_dim_by_id(multi->space, type, id);
}

/* Return the id of the given dimension.
 */
__isl_give isl_id *FN(MULTI(BASE),get_dim_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos)
{
	return multi ? isl_space_get_dim_id(multi->space, type, pos) : NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),set_dim_name)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	int i;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	multi->space = isl_space_set_dim_name(multi->space, type, pos, s);
	if (!multi->space)
		return FN(MULTI(BASE),free)(multi);

	if (type == isl_dim_out)
		return multi;
	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,set_dim_name)(multi->p[i], type, pos, s);
		if (!multi->p[i])
			return FN(MULTI(BASE),free)(multi);
	}

	return multi;
}

const char *FN(MULTI(BASE),get_tuple_name)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	return multi ? isl_space_get_tuple_name(multi->space, type) : NULL;
}

/* Does the specified tuple have an id?
 */
isl_bool FN(MULTI(BASE),has_tuple_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	if (!multi)
		return isl_bool_error;
	return isl_space_has_tuple_id(multi->space, type);
}

/* Return the id of the specified tuple.
 */
__isl_give isl_id *FN(MULTI(BASE),get_tuple_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	return multi ? isl_space_get_tuple_id(multi->space, type) : NULL;
}

__isl_give EL *FN(FN(MULTI(BASE),get),BASE)(__isl_keep MULTI(BASE) *multi,
	int pos)
{
	isl_ctx *ctx;

	if (!multi)
		return NULL;
	ctx = FN(MULTI(BASE),get_ctx)(multi);
	if (pos < 0 || pos >= multi->n)
		isl_die(ctx, isl_error_invalid,
			"index out of bounds", return NULL);
	return FN(EL,copy)(multi->p[pos]);
}

__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),set),BASE)(
	__isl_take MULTI(BASE) *multi, int pos, __isl_take EL *el)
{
	isl_space *multi_space = NULL;
	isl_space *el_space = NULL;
	int match;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi || !el)
		goto error;

	multi_space = FN(MULTI(BASE),get_space)(multi);
	match = FN(EL,matching_params)(el, multi_space);
	if (match < 0)
		goto error;
	if (!match) {
		multi = FN(MULTI(BASE),align_params)(multi,
						    FN(EL,get_space)(el));
		isl_space_free(multi_space);
		multi_space = FN(MULTI(BASE),get_space)(multi);
		el = FN(EL,align_params)(el, isl_space_copy(multi_space));
	}
	if (FN(EL,check_match_domain_space)(el, multi_space) < 0)
		goto error;

	if (pos < 0 || pos >= multi->n)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"index out of bounds", goto error);

	FN(EL,free)(multi->p[pos]);
	multi->p[pos] = el;

	isl_space_free(multi_space);
	isl_space_free(el_space);

	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	FN(EL,free)(el);
	isl_space_free(multi_space);
	isl_space_free(el_space);
	return NULL;
}

/* Reset the space of "multi".  This function is called from isl_pw_templ.c
 * and doesn't know if the space of an element object is represented
 * directly or through its domain.  It therefore passes along both,
 * which we pass along to the element function since we don't how
 * that is represented either.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_space_and_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space,
	__isl_take isl_space *domain)
{
	int i;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi || !space || !domain)
		goto error;

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,reset_domain_space)(multi->p[i],
				 isl_space_copy(domain));
		if (!multi->p[i])
			goto error;
	}
	isl_space_free(domain);
	isl_space_free(multi->space);
	multi->space = space;

	return multi;
error:
	isl_space_free(domain);
	isl_space_free(space);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_domain_space)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *domain)
{
	isl_space *space;

	space = isl_space_extend_domain_with_range(isl_space_copy(domain),
						isl_space_copy(multi->space));
	return FN(MULTI(BASE),reset_space_and_domain)(multi, space, domain);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_space)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space)
{
	isl_space *domain;

	domain = isl_space_domain(isl_space_copy(space));
	return FN(MULTI(BASE),reset_space_and_domain)(multi, space, domain);
}

/* Set the id of the given dimension of "multi" to "id".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),set_dim_id)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	isl_space *space;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi || !id)
		goto error;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_set_dim_id(space, type, pos, id);

	return FN(MULTI(BASE),reset_space)(multi, space);
error:
	isl_id_free(id);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),set_tuple_name)(
	__isl_keep MULTI(BASE) *multi, enum isl_dim_type type,
	const char *s)
{
	isl_space *space;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_set_tuple_name(space, type, s);

	return FN(MULTI(BASE),reset_space)(multi, space);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),set_tuple_id)(
	__isl_take MULTI(BASE) *multi, enum isl_dim_type type,
	__isl_take isl_id *id)
{
	isl_space *space;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_set_tuple_id(space, type, id);

	return FN(MULTI(BASE),reset_space)(multi, space);
error:
	isl_id_free(id);
	return NULL;
}

/* Drop the id on the specified tuple.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_tuple_id)(
	__isl_take MULTI(BASE) *multi, enum isl_dim_type type)
{
	isl_space *space;

	if (!multi)
		return NULL;
	if (!FN(MULTI(BASE),has_tuple_id)(multi, type))
		return multi;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_reset_tuple_id(space, type);

	return FN(MULTI(BASE),reset_space)(multi, space);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of "multi".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_user)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_reset_user(space);

	return FN(MULTI(BASE),reset_space)(multi, space);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),realign_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_reordering *exp)
{
	int i;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi || !exp)
		goto error;

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,realign_domain)(multi->p[i],
						isl_reordering_copy(exp));
		if (!multi->p[i])
			goto error;
	}

	multi = FN(MULTI(BASE),reset_domain_space)(multi,
						    isl_space_copy(exp->dim));

	isl_reordering_free(exp);
	return multi;
error:
	isl_reordering_free(exp);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

/* Align the parameters of "multi" to those of "model".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),align_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *model)
{
	isl_ctx *ctx;
	isl_reordering *exp;

	if (!multi || !model)
		goto error;

	if (isl_space_match(multi->space, isl_dim_param,
			     model, isl_dim_param)) {
		isl_space_free(model);
		return multi;
	}

	ctx = isl_space_get_ctx(model);
	if (!isl_space_has_named_params(model))
		isl_die(ctx, isl_error_invalid,
			"model has unnamed parameters", goto error);
	if (!isl_space_has_named_params(multi->space))
		isl_die(ctx, isl_error_invalid,
			"input has unnamed parameters", goto error);

	model = isl_space_params(model);
	exp = isl_parameter_alignment_reordering(multi->space, model);
	exp = isl_reordering_extend_space(exp,
				    FN(MULTI(BASE),get_domain_space)(multi));
	multi = FN(MULTI(BASE),realign_domain)(multi, exp);

	isl_space_free(model);
	return multi;
error:
	isl_space_free(model);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),from),LIST(BASE))(
	__isl_take isl_space *space, __isl_take LIST(EL) *list)
{
	int i;
	int n;
	isl_ctx *ctx;
	MULTI(BASE) *multi;

	if (!space || !list)
		goto error;

	ctx = isl_space_get_ctx(space);
	n = FN(FN(LIST(EL),n),BASE)(list);
	if (n != isl_space_dim(space, isl_dim_out))
		isl_die(ctx, isl_error_invalid,
			"invalid number of elements in list", goto error);

	multi = FN(MULTI(BASE),alloc)(isl_space_copy(space));
	for (i = 0; i < n; ++i) {
		multi = FN(FN(MULTI(BASE),set),BASE)(multi, i,
					FN(FN(LIST(EL),get),BASE)(list, i));
	}

	isl_space_free(space);
	FN(LIST(EL),free)(list);
	return multi;
error:
	isl_space_free(space);
	FN(LIST(EL),free)(list);
	return NULL;
}

#ifndef NO_IDENTITY
/* Create a multi expression in the given space that maps each
 * input dimension to the corresponding output dimension.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),identity)(__isl_take isl_space *space)
{
	int i, n;
	isl_local_space *ls;
	MULTI(BASE) *multi;

	if (!space)
		return NULL;

	if (isl_space_is_set(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting map space", goto error);

	n = isl_space_dim(space, isl_dim_out);
	if (n != isl_space_dim(space, isl_dim_in))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"number of input and output dimensions needs to be "
			"the same", goto error);

	multi = FN(MULTI(BASE),alloc)(isl_space_copy(space));

	if (!n) {
		isl_space_free(space);
		return multi;
	}

	space = isl_space_domain(space);
	ls = isl_local_space_from_space(space);

	for (i = 0; i < n; ++i) {
		EL *el;
		el = FN(EL,var_on_domain)(isl_local_space_copy(ls),
						isl_dim_set, i);
		multi = FN(FN(MULTI(BASE),set),BASE)(multi, i, el);
	}

	isl_local_space_free(ls);

	return multi;
error:
	isl_space_free(space);
	return NULL;
}
#endif

#ifndef NO_ZERO
/* Construct a multi expression in the given space with value zero in
 * each of the output dimensions.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),zero)(__isl_take isl_space *space)
{
	int n;
	MULTI(BASE) *multi;

	if (!space)
		return NULL;

	n = isl_space_dim(space , isl_dim_out);
	multi = FN(MULTI(BASE),alloc)(isl_space_copy(space));

	if (!n)
		isl_space_free(space);
	else {
		int i;
		isl_local_space *ls;
		EL *el;

		space = isl_space_domain(space);
		ls = isl_local_space_from_space(space);
		el = FN(EL,zero_on_domain)(ls);

		for (i = 0; i < n; ++i)
			multi = FN(FN(MULTI(BASE),set),BASE)(multi, i,
							    FN(EL,copy)(el));

		FN(EL,free)(el);
	}

	return multi;
}
#endif

#ifndef NO_FROM_BASE
/* Create a multiple expression with a single output/set dimension
 * equal to "el".
 * For most multiple expression types, the base type has a single
 * output/set dimension and the space of the result is therefore
 * the same as the space of the input.
 * In the case of isl_multi_union_pw_aff, however, the base type
 * lives in a parameter space and we therefore need to add
 * a single set dimension.
 */
__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),from),BASE)(__isl_take EL *el)
{
	isl_space *space;
	MULTI(BASE) *multi;

	space = FN(EL,get_space(el));
	if (isl_space_is_params(space)) {
		space = isl_space_set_from_params(space);
		space = isl_space_add_dims(space, isl_dim_set, 1);
	}
	multi = FN(MULTI(BASE),alloc)(space);
	multi = FN(FN(MULTI(BASE),set),BASE)(multi, 0, el);

	return multi;
}
#endif

__isl_give MULTI(BASE) *FN(MULTI(BASE),drop_dims)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	unsigned dim;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	dim = FN(MULTI(BASE),dim)(multi, type);
	if (first + n > dim || first + n < first)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"index out of bounds",
			return FN(MULTI(BASE),cow)(multi));

	multi->space = isl_space_drop_dims(multi->space, type, first, n);
	if (!multi->space)
		return FN(MULTI(BASE),cow)(multi);

	if (type == isl_dim_out) {
		for (i = 0; i < n; ++i)
			FN(EL,free)(multi->p[first + i]);
		for (i = first; i + n < multi->n; ++i)
			multi->p[i] = multi->p[i + n];
		multi->n -= n;

		return multi;
	}

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,drop_dims)(multi->p[i], type, first, n);
		if (!multi->p[i])
			return FN(MULTI(BASE),cow)(multi);
	}

	return multi;
}

/* Align the parameters of "multi1" and "multi2" (if needed) and call "fn".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),align_params_multi_multi_and)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2,
	__isl_give MULTI(BASE) *(*fn)(__isl_take MULTI(BASE) *multi1,
		__isl_take MULTI(BASE) *multi2))
{
	isl_ctx *ctx;

	if (!multi1 || !multi2)
		goto error;
	if (isl_space_match(multi1->space, isl_dim_param,
			    multi2->space, isl_dim_param))
		return fn(multi1, multi2);
	ctx = FN(MULTI(BASE),get_ctx)(multi1);
	if (!isl_space_has_named_params(multi1->space) ||
	    !isl_space_has_named_params(multi2->space))
		isl_die(ctx, isl_error_invalid,
			"unaligned unnamed parameters", goto error);
	multi1 = FN(MULTI(BASE),align_params)(multi1,
					    FN(MULTI(BASE),get_space)(multi2));
	multi2 = FN(MULTI(BASE),align_params)(multi2,
					    FN(MULTI(BASE),get_space)(multi1));
	return fn(multi1, multi2);
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}

/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) (A * C) -> [B -> D].
 *
 * The parameters are assumed to have been aligned.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),range_product_aligned)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	int i, n1, n2;
	EL *el;
	isl_space *space;
	MULTI(BASE) *res;

	if (!multi1 || !multi2)
		goto error;

	space = isl_space_range_product(FN(MULTI(BASE),get_space)(multi1),
					FN(MULTI(BASE),get_space)(multi2));
	res = FN(MULTI(BASE),alloc)(space);

	n1 = FN(MULTI(BASE),dim)(multi1, isl_dim_out);
	n2 = FN(MULTI(BASE),dim)(multi2, isl_dim_out);

	for (i = 0; i < n1; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi1, i);
		res = FN(FN(MULTI(BASE),set),BASE)(res, i, el);
	}

	for (i = 0; i < n2; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi2, i);
		res = FN(FN(MULTI(BASE),set),BASE)(res, n1 + i, el);
	}

	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return res;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}

/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) (A * C) -> [B -> D].
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_product)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),align_params_multi_multi_and)(multi1, multi2,
					&FN(MULTI(BASE),range_product_aligned));
}

/* Is the range of "multi" a wrapped relation?
 */
isl_bool FN(MULTI(BASE),range_is_wrapping)(__isl_keep MULTI(BASE) *multi)
{
	if (!multi)
		return isl_bool_error;
	return isl_space_range_is_wrapping(multi->space);
}

/* Given a function A -> [B -> C], extract the function A -> B.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_factor_domain)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;
	int total, keep;

	if (!multi)
		return NULL;
	if (!isl_space_range_is_wrapping(multi->space))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"range is not a product",
			return FN(MULTI(BASE),free)(multi));

	space = FN(MULTI(BASE),get_space)(multi);
	total = isl_space_dim(space, isl_dim_out);
	space = isl_space_range_factor_domain(space);
	keep = isl_space_dim(space, isl_dim_out);
	multi = FN(MULTI(BASE),drop_dims)(multi,
					isl_dim_out, keep, total - keep);
	multi = FN(MULTI(BASE),reset_space)(multi, space);

	return multi;
}

/* Given a function A -> [B -> C], extract the function A -> C.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_factor_range)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;
	int total, keep;

	if (!multi)
		return NULL;
	if (!isl_space_range_is_wrapping(multi->space))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"range is not a product",
			return FN(MULTI(BASE),free)(multi));

	space = FN(MULTI(BASE),get_space)(multi);
	total = isl_space_dim(space, isl_dim_out);
	space = isl_space_range_factor_range(space);
	keep = isl_space_dim(space, isl_dim_out);
	multi = FN(MULTI(BASE),drop_dims)(multi, isl_dim_out, 0, total - keep);
	multi = FN(MULTI(BASE),reset_space)(multi, space);

	return multi;
}

/* Given a function [B -> C], extract the function C.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),factor_range)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;
	int total, keep;

	if (!multi)
		return NULL;
	if (!isl_space_is_wrapping(multi->space))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"not a product", return FN(MULTI(BASE),free)(multi));

	space = FN(MULTI(BASE),get_space)(multi);
	total = isl_space_dim(space, isl_dim_out);
	space = isl_space_factor_range(space);
	keep = isl_space_dim(space, isl_dim_out);
	multi = FN(MULTI(BASE),drop_dims)(multi, isl_dim_out, 0, total - keep);
	multi = FN(MULTI(BASE),reset_space)(multi, space);

	return multi;
}

#ifndef NO_PRODUCT
/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) [A -> C] -> [B -> D].
 *
 * The parameters are assumed to have been aligned.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),product_aligned)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	int i;
	EL *el;
	isl_space *space;
	MULTI(BASE) *res;
	int in1, in2, out1, out2;

	in1 = FN(MULTI(BASE),dim)(multi1, isl_dim_in);
	in2 = FN(MULTI(BASE),dim)(multi2, isl_dim_in);
	out1 = FN(MULTI(BASE),dim)(multi1, isl_dim_out);
	out2 = FN(MULTI(BASE),dim)(multi2, isl_dim_out);
	space = isl_space_product(FN(MULTI(BASE),get_space)(multi1),
				  FN(MULTI(BASE),get_space)(multi2));
	res = FN(MULTI(BASE),alloc)(isl_space_copy(space));
	space = isl_space_domain(space);

	for (i = 0; i < out1; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi1, i);
		el = FN(EL,insert_dims)(el, isl_dim_in, in1, in2);
		el = FN(EL,reset_domain_space)(el, isl_space_copy(space));
		res = FN(FN(MULTI(BASE),set),BASE)(res, i, el);
	}

	for (i = 0; i < out2; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi2, i);
		el = FN(EL,insert_dims)(el, isl_dim_in, 0, in1);
		el = FN(EL,reset_domain_space)(el, isl_space_copy(space));
		res = FN(FN(MULTI(BASE),set),BASE)(res, out1 + i, el);
	}

	isl_space_free(space);
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return res;
}

/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) [A -> C] -> [B -> D].
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),product)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),align_params_multi_multi_and)(multi1, multi2,
					&FN(MULTI(BASE),product_aligned));
}
#endif

__isl_give MULTI(BASE) *FN(MULTI(BASE),flatten_range)(
	__isl_take MULTI(BASE) *multi)
{
	if (!multi)
		return NULL;

	if (!multi->space->nested[1])
		return multi;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	multi->space = isl_space_flatten_range(multi->space);
	if (!multi->space)
		return FN(MULTI(BASE),free)(multi);

	return multi;
}

/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) (A * C) -> (B, D).
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),flat_range_product)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	MULTI(BASE) *multi;

	multi = FN(MULTI(BASE),range_product)(multi1, multi2);
	multi = FN(MULTI(BASE),flatten_range)(multi);
	return multi;
}

/* Given two multi expressions, "multi1"
 *
 *	[A] -> [B1 B2]
 *
 * where B2 starts at position "pos", and "multi2"
 *
 *	[A] -> [D]
 *
 * return the multi expression
 *
 *	[A] -> [B1 D B2]
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_splice)(
	__isl_take MULTI(BASE) *multi1, unsigned pos,
	__isl_take MULTI(BASE) *multi2)
{
	MULTI(BASE) *res;
	unsigned dim;

	if (!multi1 || !multi2)
		goto error;

	dim = FN(MULTI(BASE),dim)(multi1, isl_dim_out);
	if (pos > dim)
		isl_die(FN(MULTI(BASE),get_ctx)(multi1), isl_error_invalid,
			"index out of bounds", goto error);

	res = FN(MULTI(BASE),copy)(multi1);
	res = FN(MULTI(BASE),drop_dims)(res, isl_dim_out, pos, dim - pos);
	multi1 = FN(MULTI(BASE),drop_dims)(multi1, isl_dim_out, 0, pos);

	res = FN(MULTI(BASE),flat_range_product)(res, multi2);
	res = FN(MULTI(BASE),flat_range_product)(res, multi1);

	return res;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}

#ifndef NO_SPLICE
/* Given two multi expressions, "multi1"
 *
 *	[A1 A2] -> [B1 B2]
 *
 * where A2 starts at position "in_pos" and B2 starts at position "out_pos",
 * and "multi2"
 *
 *	[C] -> [D]
 *
 * return the multi expression
 *
 *	[A1 C A2] -> [B1 D B2]
 *
 * We first insert input dimensions to obtain
 *
 *	[A1 C A2] -> [B1 B2]
 *
 * and
 *
 *	[A1 C A2] -> [D]
 *
 * and then apply range_splice.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),splice)(
	__isl_take MULTI(BASE) *multi1, unsigned in_pos, unsigned out_pos,
	__isl_take MULTI(BASE) *multi2)
{
	unsigned n_in1;
	unsigned n_in2;

	if (!multi1 || !multi2)
		goto error;

	n_in1 = FN(MULTI(BASE),dim)(multi1, isl_dim_in);
	if (in_pos > n_in1)
		isl_die(FN(MULTI(BASE),get_ctx)(multi1), isl_error_invalid,
			"index out of bounds", goto error);

	n_in2 = FN(MULTI(BASE),dim)(multi2, isl_dim_in);

	multi1 = FN(MULTI(BASE),insert_dims)(multi1, isl_dim_in, in_pos, n_in2);
	multi2 = FN(MULTI(BASE),insert_dims)(multi2, isl_dim_in, n_in2,
						n_in1 - in_pos);
	multi2 = FN(MULTI(BASE),insert_dims)(multi2, isl_dim_in, 0, in_pos);

	return FN(MULTI(BASE),range_splice)(multi1, out_pos, multi2);
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}
#endif

/* This function is currently only used from isl_aff.c
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),bin_op)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2,
	__isl_give EL *(*fn)(__isl_take EL *, __isl_take EL *))
	__attribute__ ((unused));

/* Pairwise perform "fn" to the elements of "multi1" and "multi2" and
 * return the result.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),bin_op)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2,
	__isl_give EL *(*fn)(__isl_take EL *, __isl_take EL *))
{
	int i;
	isl_ctx *ctx;

	multi1 = FN(MULTI(BASE),cow)(multi1);
	if (!multi1 || !multi2)
		goto error;

	ctx = FN(MULTI(BASE),get_ctx)(multi1);
	if (!isl_space_is_equal(multi1->space, multi2->space))
		isl_die(ctx, isl_error_invalid,
			"spaces don't match", goto error);

	for (i = 0; i < multi1->n; ++i) {
		multi1->p[i] = fn(multi1->p[i], FN(EL,copy)(multi2->p[i]));
		if (!multi1->p[i])
			goto error;
	}

	FN(MULTI(BASE),free)(multi2);
	return multi1;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}

/* Add "multi2" from "multi1" and return the result.
 *
 * The parameters of "multi1" and "multi2" are assumed to have been aligned.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),add_aligned)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),bin_op)(multi1, multi2, &FN(EL,add));
}

/* Add "multi2" from "multi1" and return the result.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),add)(__isl_take MULTI(BASE) *multi1,
	__isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),align_params_multi_multi_and)(multi1, multi2,
						&FN(MULTI(BASE),add_aligned));
}

/* Subtract "multi2" from "multi1" and return the result.
 *
 * The parameters of "multi1" and "multi2" are assumed to have been aligned.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),sub_aligned)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),bin_op)(multi1, multi2, &FN(EL,sub));
}

/* Subtract "multi2" from "multi1" and return the result.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),sub)(__isl_take MULTI(BASE) *multi1,
	__isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),align_params_multi_multi_and)(multi1, multi2,
						&FN(MULTI(BASE),sub_aligned));
}

/* Multiply the elements of "multi" by "v" and return the result.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),scale_val)(__isl_take MULTI(BASE) *multi,
	__isl_take isl_val *v)
{
	int i;

	if (!multi || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return multi;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational factor", goto error);

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,scale_val)(multi->p[i], isl_val_copy(v));
		if (!multi->p[i])
			goto error;
	}

	isl_val_free(v);
	return multi;
error:
	isl_val_free(v);
	return FN(MULTI(BASE),free)(multi);
}

/* Divide the elements of "multi" by "v" and return the result.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),scale_down_val)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_val *v)
{
	int i;

	if (!multi || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return multi;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational factor", goto error);
	if (isl_val_is_zero(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"cannot scale down by zero", goto error);

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,scale_down_val)(multi->p[i],
						    isl_val_copy(v));
		if (!multi->p[i])
			goto error;
	}

	isl_val_free(v);
	return multi;
error:
	isl_val_free(v);
	return FN(MULTI(BASE),free)(multi);
}

/* Multiply the elements of "multi" by the corresponding element of "mv"
 * and return the result.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),scale_multi_val)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_multi_val *mv)
{
	int i;

	if (!multi || !mv)
		goto error;

	if (!isl_space_tuple_is_equal(multi->space, isl_dim_out,
					mv->space, isl_dim_set))
		isl_die(isl_multi_val_get_ctx(mv), isl_error_invalid,
			"spaces don't match", goto error);

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;

	for (i = 0; i < multi->n; ++i) {
		isl_val *v;

		v = isl_multi_val_get_val(mv, i);
		multi->p[i] = FN(EL,scale_val)(multi->p[i], v);
		if (!multi->p[i])
			goto error;
	}

	isl_multi_val_free(mv);
	return multi;
error:
	isl_multi_val_free(mv);
	return FN(MULTI(BASE),free)(multi);
}

/* Divide the elements of "multi" by the corresponding element of "mv"
 * and return the result.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),scale_down_multi_val)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_multi_val *mv)
{
	int i;

	if (!multi || !mv)
		goto error;

	if (!isl_space_tuple_is_equal(multi->space, isl_dim_out,
					mv->space, isl_dim_set))
		isl_die(isl_multi_val_get_ctx(mv), isl_error_invalid,
			"spaces don't match", goto error);

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		isl_val *v;

		v = isl_multi_val_get_val(mv, i);
		multi->p[i] = FN(EL,scale_down_val)(multi->p[i], v);
		if (!multi->p[i])
			goto error;
	}

	isl_multi_val_free(mv);
	return multi;
error:
	isl_multi_val_free(mv);
	return FN(MULTI(BASE),free)(multi);
}

/* Compute the residues of the elements of "multi" modulo
 * the corresponding element of "mv" and return the result.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),mod_multi_val)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_multi_val *mv)
{
	int i;

	if (!multi || !mv)
		goto error;

	if (!isl_space_tuple_is_equal(multi->space, isl_dim_out,
					mv->space, isl_dim_set))
		isl_die(isl_multi_val_get_ctx(mv), isl_error_invalid,
			"spaces don't match", goto error);

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		isl_val *v;

		v = isl_multi_val_get_val(mv, i);
		multi->p[i] = FN(EL,mod_val)(multi->p[i], v);
		if (!multi->p[i])
			goto error;
	}

	isl_multi_val_free(mv);
	return multi;
error:
	isl_multi_val_free(mv);
	return FN(MULTI(BASE),free)(multi);
}

#ifndef NO_MOVE_DIMS
/* Move the "n" dimensions of "src_type" starting at "src_pos" of "multi"
 * to dimensions of "dst_type" at "dst_pos".
 *
 * We only support moving input dimensions to parameters and vice versa.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),move_dims)(__isl_take MULTI(BASE) *multi,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	int i;

	if (!multi)
		return NULL;

	if (n == 0 &&
	    !isl_space_is_named_or_nested(multi->space, src_type) &&
	    !isl_space_is_named_or_nested(multi->space, dst_type))
		return multi;

	if (dst_type == isl_dim_out || src_type == isl_dim_out)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"cannot move output/set dimension",
			return FN(MULTI(BASE),free)(multi));
	if (dst_type == isl_dim_div || src_type == isl_dim_div)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"cannot move divs",
			return FN(MULTI(BASE),free)(multi));
	if (src_pos + n > isl_space_dim(multi->space, src_type))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"range out of bounds",
			return FN(MULTI(BASE),free)(multi));
	if (dst_type == src_type)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_unsupported,
			"moving dims within the same type not supported",
			return FN(MULTI(BASE),free)(multi));

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	multi->space = isl_space_move_dims(multi->space, dst_type, dst_pos,
						src_type, src_pos, n);
	if (!multi->space)
		return FN(MULTI(BASE),free)(multi);

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,move_dims)(multi->p[i], dst_type, dst_pos,
						src_type, src_pos, n);
		if (!multi->p[i])
			return FN(MULTI(BASE),free)(multi);
	}

	return multi;
}
#endif

/* Convert a multiple expression defined over a parameter domain
 * into one that is defined over a zero-dimensional set.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),from_range)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;

	if (!multi)
		return NULL;
	if (!isl_space_is_set(multi->space))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"not living in a set space",
			return FN(MULTI(BASE),free)(multi));

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_from_range(space);
	multi = FN(MULTI(BASE),reset_space)(multi, space);

	return multi;
}

/* Are "multi1" and "multi2" obviously equal?
 */
isl_bool FN(MULTI(BASE),plain_is_equal)(__isl_keep MULTI(BASE) *multi1,
	__isl_keep MULTI(BASE) *multi2)
{
	int i;
	isl_bool equal;

	if (!multi1 || !multi2)
		return isl_bool_error;
	if (multi1->n != multi2->n)
		return isl_bool_false;
	equal = isl_space_is_equal(multi1->space, multi2->space);
	if (equal < 0 || !equal)
		return equal;

	for (i = 0; i < multi1->n; ++i) {
		equal = FN(EL,plain_is_equal)(multi1->p[i], multi2->p[i]);
		if (equal < 0 || !equal)
			return equal;
	}

	return isl_bool_true;
}

#ifndef NO_DOMAIN
/* Return the shared domain of the elements of "multi".
 */
__isl_give isl_set *FN(MULTI(BASE),domain)(__isl_take MULTI(BASE) *multi)
{
	int i;
	isl_set *dom;

	if (!multi)
		return NULL;

	dom = isl_set_universe(FN(MULTI(BASE),get_domain_space)(multi));
	for (i = 0; i < multi->n; ++i) {
		isl_set *dom_i;

		dom_i = FN(EL,domain)(FN(FN(MULTI(BASE),get),BASE)(multi, i));
		dom = isl_set_intersect(dom, dom_i);
	}

	FN(MULTI(BASE),free)(multi);
	return dom;
}
#endif

#ifndef NO_NEG
/* Return the opposite of "multi".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),neg)(__isl_take MULTI(BASE) *multi)
{
	int i;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		multi->p[i] = FN(EL,neg)(multi->p[i]);
		if (!multi->p[i])
			return FN(MULTI(BASE),free)(multi);
	}

	return multi;
}
#endif
