/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 */

#include <isl_map_private.h>
#include <isl_constraint_private.h>
#include <isl_space_private.h>
#include <isl_seq.h>
#include <isl_aff_private.h>
#include <isl_local_space_private.h>
#include <isl_val_private.h>
#include <isl_vec_private.h>
#include <isl/deprecated/constraint_int.h>

#include <bset_to_bmap.c>
#include <bset_from_bmap.c>

#undef BASE
#define BASE constraint

#include <isl_list_templ.c>

isl_ctx *isl_constraint_get_ctx(__isl_keep isl_constraint *c)
{
	return c ? isl_local_space_get_ctx(c->ls) : NULL;
}

static unsigned n(struct isl_constraint *c, enum isl_dim_type type)
{
	return isl_local_space_dim(c->ls, type);
}

static unsigned offset(struct isl_constraint *c, enum isl_dim_type type)
{
	return isl_local_space_offset(c->ls, type);
}

static unsigned basic_map_offset(__isl_keep isl_basic_map *bmap,
							enum isl_dim_type type)
{
	return type == isl_dim_div ? 1 + isl_space_dim(bmap->dim, isl_dim_all)
				   : 1 + isl_space_offset(bmap->dim, type);
}

static unsigned basic_set_offset(struct isl_basic_set *bset,
							enum isl_dim_type type)
{
	isl_space *dim = bset->dim;
	switch (type) {
	case isl_dim_param:	return 1;
	case isl_dim_in:	return 1 + dim->nparam;
	case isl_dim_out:	return 1 + dim->nparam + dim->n_in;
	case isl_dim_div:	return 1 + dim->nparam + dim->n_in + dim->n_out;
	default:		return 0;
	}
}

__isl_give isl_constraint *isl_constraint_alloc_vec(int eq,
	__isl_take isl_local_space *ls, __isl_take isl_vec *v)
{
	isl_constraint *constraint;

	if (!ls || !v)
		goto error;

	constraint = isl_alloc_type(isl_vec_get_ctx(v), isl_constraint);
	if (!constraint)
		goto error;

	constraint->ref = 1;
	constraint->eq = eq;
	constraint->ls = ls;
	constraint->v = v;

	return constraint;
error:
	isl_local_space_free(ls);
	isl_vec_free(v);
	return NULL;
}

__isl_give isl_constraint *isl_constraint_alloc(int eq,
	__isl_take isl_local_space *ls)
{
	isl_ctx *ctx;
	isl_vec *v;

	if (!ls)
		return NULL;

	ctx = isl_local_space_get_ctx(ls);
	v = isl_vec_alloc(ctx, 1 + isl_local_space_dim(ls, isl_dim_all));
	v = isl_vec_clr(v);
	return isl_constraint_alloc_vec(eq, ls, v);
}

struct isl_constraint *isl_basic_map_constraint(struct isl_basic_map *bmap,
	isl_int **line)
{
	int eq;
	isl_ctx *ctx;
	isl_vec *v;
	isl_local_space *ls = NULL;
	isl_constraint *constraint;

	if (!bmap || !line)
		goto error;

	eq = line >= bmap->eq;

	ctx = isl_basic_map_get_ctx(bmap);
	ls = isl_basic_map_get_local_space(bmap);
	v = isl_vec_alloc(ctx, 1 + isl_local_space_dim(ls, isl_dim_all));
	if (!v)
		goto error;
	isl_seq_cpy(v->el, line[0], v->size);
	constraint = isl_constraint_alloc_vec(eq, ls, v);

	isl_basic_map_free(bmap);
	return constraint;
error:
	isl_local_space_free(ls);
	isl_basic_map_free(bmap);
	return NULL;
}

struct isl_constraint *isl_basic_set_constraint(struct isl_basic_set *bset,
	isl_int **line)
{
	return isl_basic_map_constraint(bset_to_bmap(bset), line);
}

__isl_give isl_constraint *isl_constraint_alloc_equality(
	__isl_take isl_local_space *ls)
{
	return isl_constraint_alloc(1, ls);
}

__isl_give isl_constraint *isl_constraint_alloc_inequality(
	__isl_take isl_local_space *ls)
{
	return isl_constraint_alloc(0, ls);
}

struct isl_constraint *isl_constraint_dup(struct isl_constraint *c)
{
	if (!c)
		return NULL;

	return isl_constraint_alloc_vec(c->eq, isl_local_space_copy(c->ls),
						isl_vec_copy(c->v));
}

struct isl_constraint *isl_constraint_cow(struct isl_constraint *c)
{
	if (!c)
		return NULL;

	if (c->ref == 1)
		return c;
	c->ref--;
	return isl_constraint_dup(c);
}

struct isl_constraint *isl_constraint_copy(struct isl_constraint *constraint)
{
	if (!constraint)
		return NULL;

	constraint->ref++;
	return constraint;
}

__isl_null isl_constraint *isl_constraint_free(__isl_take isl_constraint *c)
{
	if (!c)
		return NULL;

	if (--c->ref > 0)
		return NULL;

	isl_local_space_free(c->ls);
	isl_vec_free(c->v);
	free(c);

	return NULL;
}

/* Return the number of constraints in "bmap", i.e., the
 * number of times isl_basic_map_foreach_constraint will
 * call the callback.
 */
int isl_basic_map_n_constraint(__isl_keep isl_basic_map *bmap)
{
	if (!bmap)
		return -1;

	return bmap->n_eq + bmap->n_ineq;
}

/* Return the number of constraints in "bset", i.e., the
 * number of times isl_basic_set_foreach_constraint will
 * call the callback.
 */
int isl_basic_set_n_constraint(__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_n_constraint(bset);
}

isl_stat isl_basic_map_foreach_constraint(__isl_keep isl_basic_map *bmap,
	isl_stat (*fn)(__isl_take isl_constraint *c, void *user), void *user)
{
	int i;
	struct isl_constraint *c;

	if (!bmap)
		return isl_stat_error;

	isl_assert(bmap->ctx, ISL_F_ISSET(bmap, ISL_BASIC_MAP_FINAL),
			return isl_stat_error);

	for (i = 0; i < bmap->n_eq; ++i) {
		c = isl_basic_map_constraint(isl_basic_map_copy(bmap),
						&bmap->eq[i]);
		if (!c)
			return isl_stat_error;
		if (fn(c, user) < 0)
			return isl_stat_error;
	}

	for (i = 0; i < bmap->n_ineq; ++i) {
		c = isl_basic_map_constraint(isl_basic_map_copy(bmap),
						&bmap->ineq[i]);
		if (!c)
			return isl_stat_error;
		if (fn(c, user) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

isl_stat isl_basic_set_foreach_constraint(__isl_keep isl_basic_set *bset,
	isl_stat (*fn)(__isl_take isl_constraint *c, void *user), void *user)
{
	return isl_basic_map_foreach_constraint(bset_to_bmap(bset), fn, user);
}

/* Add the constraint to the list that "user" points to, if it is not
 * a div constraint.
 */
static isl_stat collect_constraint(__isl_take isl_constraint *constraint,
	void *user)
{
	isl_constraint_list **list = user;

	if (isl_constraint_is_div_constraint(constraint))
		isl_constraint_free(constraint);
	else
		*list = isl_constraint_list_add(*list, constraint);

	return isl_stat_ok;
}

/* Return a list of constraints that, when combined, are equivalent
 * to "bmap".  The input is required to have only known divs.
 *
 * There is no need to include the div constraints as they are
 * implied by the div expressions.
 */
__isl_give isl_constraint_list *isl_basic_map_get_constraint_list(
	__isl_keep isl_basic_map *bmap)
{
	int n;
	int known;
	isl_ctx *ctx;
	isl_constraint_list *list;

	known = isl_basic_map_divs_known(bmap);
	if (known < 0)
		return NULL;
	ctx = isl_basic_map_get_ctx(bmap);
	if (!known)
		isl_die(ctx, isl_error_invalid,
			"input involves unknown divs", return NULL);

	n = isl_basic_map_n_constraint(bmap);
	list = isl_constraint_list_alloc(ctx, n);
	if (isl_basic_map_foreach_constraint(bmap,
					    &collect_constraint, &list) < 0)
		list = isl_constraint_list_free(list);

	return list;
}

/* Return a list of constraints that, when combined, are equivalent
 * to "bset".  The input is required to have only known divs.
 */
__isl_give isl_constraint_list *isl_basic_set_get_constraint_list(
	__isl_keep isl_basic_set *bset)
{
	return isl_basic_map_get_constraint_list(bset);
}

int isl_constraint_is_equal(struct isl_constraint *constraint1,
	struct isl_constraint *constraint2)
{
	int equal;

	if (!constraint1 || !constraint2)
		return 0;
	if (constraint1->eq != constraint2->eq)
		return 0;
	equal = isl_local_space_is_equal(constraint1->ls, constraint2->ls);
	if (equal < 0 || !equal)
		return equal;
	return isl_vec_is_equal(constraint1->v, constraint2->v);
}

struct isl_basic_map *isl_basic_map_add_constraint(
	struct isl_basic_map *bmap, struct isl_constraint *constraint)
{
	isl_ctx *ctx;
	isl_space *dim;
	int equal_space;

	if (!bmap || !constraint)
		goto error;

	ctx = isl_constraint_get_ctx(constraint);
	dim = isl_constraint_get_space(constraint);
	equal_space = isl_space_is_equal(bmap->dim, dim);
	isl_space_free(dim);
	isl_assert(ctx, equal_space, goto error);

	bmap = isl_basic_map_intersect(bmap,
				isl_basic_map_from_constraint(constraint));
	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_constraint_free(constraint);
	return NULL;
}

struct isl_basic_set *isl_basic_set_add_constraint(
	struct isl_basic_set *bset, struct isl_constraint *constraint)
{
	return bset_from_bmap(isl_basic_map_add_constraint(bset_to_bmap(bset),
							    constraint));
}

__isl_give isl_map *isl_map_add_constraint(__isl_take isl_map *map,
	__isl_take isl_constraint *constraint)
{
	isl_basic_map *bmap;

	bmap = isl_basic_map_from_constraint(constraint);
	map = isl_map_intersect(map, isl_map_from_basic_map(bmap));

	return map;
}

__isl_give isl_set *isl_set_add_constraint(__isl_take isl_set *set,
	__isl_take isl_constraint *constraint)
{
	return isl_map_add_constraint(set, constraint);
}

__isl_give isl_space *isl_constraint_get_space(
	__isl_keep isl_constraint *constraint)
{
	return constraint ? isl_local_space_get_space(constraint->ls) : NULL;
}

__isl_give isl_local_space *isl_constraint_get_local_space(
	__isl_keep isl_constraint *constraint)
{
	return constraint ? isl_local_space_copy(constraint->ls) : NULL;
}

int isl_constraint_dim(struct isl_constraint *constraint,
	enum isl_dim_type type)
{
	if (!constraint)
		return -1;
	return n(constraint, type);
}

isl_bool isl_constraint_involves_dims(__isl_keep isl_constraint *constraint,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	isl_ctx *ctx;
	int *active = NULL;
	isl_bool involves = isl_bool_false;

	if (!constraint)
		return isl_bool_error;
	if (n == 0)
		return isl_bool_false;

	ctx = isl_constraint_get_ctx(constraint);
	if (first + n > isl_constraint_dim(constraint, type))
		isl_die(ctx, isl_error_invalid,
			"range out of bounds", return isl_bool_error);

	active = isl_local_space_get_active(constraint->ls,
					    constraint->v->el + 1);
	if (!active)
		goto error;

	first += isl_local_space_offset(constraint->ls, type) - 1;
	for (i = 0; i < n; ++i)
		if (active[first + i]) {
			involves = isl_bool_true;
			break;
		}

	free(active);

	return involves;
error:
	free(active);
	return isl_bool_error;
}

/* Does the given constraint represent a lower bound on the given
 * dimension?
 */
isl_bool isl_constraint_is_lower_bound(__isl_keep isl_constraint *constraint,
	enum isl_dim_type type, unsigned pos)
{
	if (!constraint)
		return isl_bool_error;

	if (pos >= isl_local_space_dim(constraint->ls, type))
		isl_die(isl_constraint_get_ctx(constraint), isl_error_invalid,
			"position out of bounds", return isl_bool_error);

	pos += isl_local_space_offset(constraint->ls, type);
	return isl_int_is_pos(constraint->v->el[pos]);
}

/* Does the given constraint represent an upper bound on the given
 * dimension?
 */
isl_bool isl_constraint_is_upper_bound(__isl_keep isl_constraint *constraint,
	enum isl_dim_type type, unsigned pos)
{
	if (!constraint)
		return isl_bool_error;

	if (pos >= isl_local_space_dim(constraint->ls, type))
		isl_die(isl_constraint_get_ctx(constraint), isl_error_invalid,
			"position out of bounds", return isl_bool_error);

	pos += isl_local_space_offset(constraint->ls, type);
	return isl_int_is_neg(constraint->v->el[pos]);
}

const char *isl_constraint_get_dim_name(__isl_keep isl_constraint *constraint,
	enum isl_dim_type type, unsigned pos)
{
	return constraint ?
	    isl_local_space_get_dim_name(constraint->ls, type, pos) : NULL;
}

void isl_constraint_get_constant(struct isl_constraint *constraint, isl_int *v)
{
	if (!constraint)
		return;
	isl_int_set(*v, constraint->v->el[0]);
}

/* Return the constant term of "constraint".
 */
__isl_give isl_val *isl_constraint_get_constant_val(
	__isl_keep isl_constraint *constraint)
{
	isl_ctx *ctx;

	if (!constraint)
		return NULL;

	ctx = isl_constraint_get_ctx(constraint);
	return isl_val_int_from_isl_int(ctx, constraint->v->el[0]);
}

void isl_constraint_get_coefficient(struct isl_constraint *constraint,
	enum isl_dim_type type, int pos, isl_int *v)
{
	if (!constraint)
		return;

	if (pos >= isl_local_space_dim(constraint->ls, type))
		isl_die(constraint->v->ctx, isl_error_invalid,
			"position out of bounds", return);

	pos += isl_local_space_offset(constraint->ls, type);
	isl_int_set(*v, constraint->v->el[pos]);
}

/* Return the coefficient of the variable of type "type" at position "pos"
 * of "constraint".
 */
__isl_give isl_val *isl_constraint_get_coefficient_val(
	__isl_keep isl_constraint *constraint, enum isl_dim_type type, int pos)
{
	isl_ctx *ctx;

	if (!constraint)
		return NULL;

	ctx = isl_constraint_get_ctx(constraint);
	if (pos < 0 || pos >= isl_local_space_dim(constraint->ls, type))
		isl_die(ctx, isl_error_invalid,
			"position out of bounds", return NULL);

	pos += isl_local_space_offset(constraint->ls, type);
	return isl_val_int_from_isl_int(ctx, constraint->v->el[pos]);
}

__isl_give isl_aff *isl_constraint_get_div(__isl_keep isl_constraint *constraint,
	int pos)
{
	if (!constraint)
		return NULL;

	return isl_local_space_get_div(constraint->ls, pos);
}

__isl_give isl_constraint *isl_constraint_set_constant(
	__isl_take isl_constraint *constraint, isl_int v)
{
	constraint = isl_constraint_cow(constraint);
	if (!constraint)
		return NULL;

	constraint->v = isl_vec_cow(constraint->v);
	if (!constraint->v)
		return isl_constraint_free(constraint);

	isl_int_set(constraint->v->el[0], v);
	return constraint;
}

/* Replace the constant term of "constraint" by "v".
 */
__isl_give isl_constraint *isl_constraint_set_constant_val(
	__isl_take isl_constraint *constraint, __isl_take isl_val *v)
{
	constraint = isl_constraint_cow(constraint);
	if (!constraint || !v)
		goto error;
	if (!isl_val_is_int(v))
		isl_die(isl_constraint_get_ctx(constraint), isl_error_invalid,
			"expecting integer value", goto error);
	constraint->v = isl_vec_set_element_val(constraint->v, 0, v);
	if (!constraint->v)
		constraint = isl_constraint_free(constraint);
	return constraint;
error:
	isl_val_free(v);
	return isl_constraint_free(constraint);
}

__isl_give isl_constraint *isl_constraint_set_constant_si(
	__isl_take isl_constraint *constraint, int v)
{
	constraint = isl_constraint_cow(constraint);
	if (!constraint)
		return NULL;

	constraint->v = isl_vec_cow(constraint->v);
	if (!constraint->v)
		return isl_constraint_free(constraint);

	isl_int_set_si(constraint->v->el[0], v);
	return constraint;
}

__isl_give isl_constraint *isl_constraint_set_coefficient(
	__isl_take isl_constraint *constraint,
	enum isl_dim_type type, int pos, isl_int v)
{
	constraint = isl_constraint_cow(constraint);
	if (!constraint)
		return NULL;

	if (pos >= isl_local_space_dim(constraint->ls, type))
		isl_die(constraint->v->ctx, isl_error_invalid,
			"position out of bounds",
			return isl_constraint_free(constraint));

	constraint = isl_constraint_cow(constraint);
	if (!constraint)
		return NULL;

	constraint->v = isl_vec_cow(constraint->v);
	if (!constraint->v)
		return isl_constraint_free(constraint);

	pos += isl_local_space_offset(constraint->ls, type);
	isl_int_set(constraint->v->el[pos], v);

	return constraint;
}

/* Replace the coefficient of the variable of type "type" at position "pos"
 * of "constraint" by "v".
 */
__isl_give isl_constraint *isl_constraint_set_coefficient_val(
	__isl_take isl_constraint *constraint,
	enum isl_dim_type type, int pos, __isl_take isl_val *v)
{
	constraint = isl_constraint_cow(constraint);
	if (!constraint || !v)
		goto error;
	if (!isl_val_is_int(v))
		isl_die(isl_constraint_get_ctx(constraint), isl_error_invalid,
			"expecting integer value", goto error);

	if (pos >= isl_local_space_dim(constraint->ls, type))
		isl_die(isl_constraint_get_ctx(constraint), isl_error_invalid,
			"position out of bounds", goto error);

	pos += isl_local_space_offset(constraint->ls, type);
	constraint->v = isl_vec_set_element_val(constraint->v, pos, v);
	if (!constraint->v)
		constraint = isl_constraint_free(constraint);
	return constraint;
error:
	isl_val_free(v);
	return isl_constraint_free(constraint);
}

__isl_give isl_constraint *isl_constraint_set_coefficient_si(
	__isl_take isl_constraint *constraint,
	enum isl_dim_type type, int pos, int v)
{
	constraint = isl_constraint_cow(constraint);
	if (!constraint)
		return NULL;

	if (pos >= isl_local_space_dim(constraint->ls, type))
		isl_die(constraint->v->ctx, isl_error_invalid,
			"position out of bounds",
			return isl_constraint_free(constraint));

	constraint = isl_constraint_cow(constraint);
	if (!constraint)
		return NULL;

	constraint->v = isl_vec_cow(constraint->v);
	if (!constraint->v)
		return isl_constraint_free(constraint);

	pos += isl_local_space_offset(constraint->ls, type);
	isl_int_set_si(constraint->v->el[pos], v);

	return constraint;
}

/* Drop any constraint from "bset" that is identical to "constraint".
 * In particular, this means that the local spaces of "bset" and
 * "constraint" need to be the same.
 *
 * We manually set ISL_BASIC_SET_FINAL instead of calling
 * isl_basic_set_finalize because this function is called by CLooG,
 * which does not expect any variables to disappear.
 */
__isl_give isl_basic_set *isl_basic_set_drop_constraint(
	__isl_take isl_basic_set *bset, __isl_take isl_constraint *constraint)
{
	int i;
	unsigned n;
	isl_int **row;
	unsigned total;
	isl_local_space *ls1;
	int equal;
	int equality;

	if (!bset || !constraint)
		goto error;

	ls1 = isl_basic_set_get_local_space(bset);
	equal = isl_local_space_is_equal(ls1, constraint->ls);
	isl_local_space_free(ls1);
	if (equal < 0)
		goto error;
	if (!equal) {
		isl_constraint_free(constraint);
		return bset;
	}

	bset = isl_basic_set_cow(bset);
	if (!bset)
		goto error;

	equality = isl_constraint_is_equality(constraint);
	if (equality) {
		n = bset->n_eq;
		row = bset->eq;
	} else {
		n = bset->n_ineq;
		row = bset->ineq;
	}

	total = isl_constraint_dim(constraint, isl_dim_all);
	for (i = 0; i < n; ++i) {
		if (!isl_seq_eq(row[i], constraint->v->el, 1 + total))
			continue;
		if (equality && isl_basic_set_drop_equality(bset, i) < 0)
			goto error;
		if (!equality && isl_basic_set_drop_inequality(bset, i) < 0)
			goto error;
		break;
	}
			
	isl_constraint_free(constraint);
	ISL_F_SET(bset, ISL_BASIC_SET_FINAL);
	return bset;
error:
	isl_constraint_free(constraint);
	isl_basic_set_free(bset);
	return NULL;
}

struct isl_constraint *isl_constraint_negate(struct isl_constraint *constraint)
{
	isl_ctx *ctx;

	constraint = isl_constraint_cow(constraint);
	if (!constraint)
		return NULL;

	ctx = isl_constraint_get_ctx(constraint);
	if (isl_constraint_is_equality(constraint))
		isl_die(ctx, isl_error_invalid, "cannot negate equality",
			return isl_constraint_free(constraint));
	constraint->v = isl_vec_neg(constraint->v);
	constraint->v = isl_vec_cow(constraint->v);
	if (!constraint->v)
		return isl_constraint_free(constraint);
	isl_int_sub_ui(constraint->v->el[0], constraint->v->el[0], 1);
	return constraint;
}

isl_bool isl_constraint_is_equality(struct isl_constraint *constraint)
{
	if (!constraint)
		return isl_bool_error;
	return constraint->eq;
}

int isl_constraint_is_div_constraint(__isl_keep isl_constraint *constraint)
{
	int i;
	int n_div;

	if (!constraint)
		return -1;
	if (isl_constraint_is_equality(constraint))
		return 0;
	n_div = isl_constraint_dim(constraint, isl_dim_div);
	for (i = 0; i < n_div; ++i) {
		if (isl_local_space_is_div_constraint(constraint->ls,
							constraint->v->el, i))
			return 1;
	}

	return 0;
}

/* We manually set ISL_BASIC_SET_FINAL instead of calling
 * isl_basic_map_finalize because we want to keep the position
 * of the divs and we therefore do not want to throw away redundant divs.
 * This is arguably a bit fragile.
 */
__isl_give isl_basic_map *isl_basic_map_from_constraint(
	__isl_take isl_constraint *constraint)
{
	int k;
	isl_local_space *ls;
	struct isl_basic_map *bmap;
	isl_int *c;
	unsigned total;

	if (!constraint)
		return NULL;

	ls = isl_local_space_copy(constraint->ls);
	bmap = isl_basic_map_from_local_space(ls);
	bmap = isl_basic_map_extend_constraints(bmap, 1, 1);
	if (isl_constraint_is_equality(constraint)) {
		k = isl_basic_map_alloc_equality(bmap);
		if (k < 0)
			goto error;
		c = bmap->eq[k];
	}
	else {
		k = isl_basic_map_alloc_inequality(bmap);
		if (k < 0)
			goto error;
		c = bmap->ineq[k];
	}
	total = isl_basic_map_total_dim(bmap);
	isl_seq_cpy(c, constraint->v->el, 1 + total);
	isl_constraint_free(constraint);
	if (bmap)
		ISL_F_SET(bmap, ISL_BASIC_SET_FINAL);
	return bmap;
error:
	isl_constraint_free(constraint);
	isl_basic_map_free(bmap);
	return NULL;
}

struct isl_basic_set *isl_basic_set_from_constraint(
	struct isl_constraint *constraint)
{
	if (!constraint)
		return NULL;

	if (isl_constraint_dim(constraint, isl_dim_in) != 0)
		isl_die(isl_constraint_get_ctx(constraint), isl_error_invalid,
			"not a set constraint", goto error);
	return bset_from_bmap(isl_basic_map_from_constraint(constraint));
error:
	isl_constraint_free(constraint);
	return NULL;
}

/* Is the variable of "type" at position "pos" of "bmap" defined
 * in terms of earlier dimensions through an equality?
 *
 * If so, and if c is not NULL, then return a copy of this equality in *c.
 */
int isl_basic_map_has_defining_equality(
	__isl_keep isl_basic_map *bmap, enum isl_dim_type type, int pos,
	__isl_give isl_constraint **c)
{
	int i;
	unsigned offset;
	unsigned total;

	if (!bmap)
		return -1;
	offset = basic_map_offset(bmap, type);
	total = isl_basic_map_total_dim(bmap);
	isl_assert(bmap->ctx, pos < isl_basic_map_dim(bmap, type), return -1);
	for (i = 0; i < bmap->n_eq; ++i) {
		if (isl_int_is_zero(bmap->eq[i][offset + pos]) ||
		    isl_seq_first_non_zero(bmap->eq[i]+offset+pos+1,
					   1+total-offset-pos-1) != -1)
			continue;
		if (c)
			*c = isl_basic_map_constraint(isl_basic_map_copy(bmap),
								&bmap->eq[i]);
		return 1;
	}
	return 0;
}

/* Is the variable of "type" at position "pos" of "bset" defined
 * in terms of earlier dimensions through an equality?
 *
 * If so, and if c is not NULL, then return a copy of this equality in *c.
 */
int isl_basic_set_has_defining_equality(
	__isl_keep isl_basic_set *bset, enum isl_dim_type type, int pos,
	__isl_give isl_constraint **c)
{
	return isl_basic_map_has_defining_equality(bset_to_bmap(bset),
						    type, pos, c);
}

int isl_basic_set_has_defining_inequalities(
	struct isl_basic_set *bset, enum isl_dim_type type, int pos,
	struct isl_constraint **lower,
	struct isl_constraint **upper)
{
	int i, j;
	unsigned offset;
	unsigned total;
	isl_int m;
	isl_int **lower_line, **upper_line;

	if (!bset)
		return -1;
	offset = basic_set_offset(bset, type);
	total = isl_basic_set_total_dim(bset);
	isl_assert(bset->ctx, pos < isl_basic_set_dim(bset, type), return -1);
	isl_int_init(m);
	for (i = 0; i < bset->n_ineq; ++i) {
		if (isl_int_is_zero(bset->ineq[i][offset + pos]))
			continue;
		if (isl_int_is_one(bset->ineq[i][offset + pos]))
			continue;
		if (isl_int_is_negone(bset->ineq[i][offset + pos]))
			continue;
		if (isl_seq_first_non_zero(bset->ineq[i]+offset+pos+1,
						1+total-offset-pos-1) != -1)
			continue;
		for (j = i + 1; j < bset->n_ineq; ++j) {
			if (!isl_seq_is_neg(bset->ineq[i]+1, bset->ineq[j]+1,
					    total))
				continue;
			isl_int_add(m, bset->ineq[i][0], bset->ineq[j][0]);
			if (isl_int_abs_ge(m, bset->ineq[i][offset+pos]))
				continue;

			if (isl_int_is_pos(bset->ineq[i][offset+pos])) {
				lower_line = &bset->ineq[i];
				upper_line = &bset->ineq[j];
			} else {
				lower_line = &bset->ineq[j];
				upper_line = &bset->ineq[i];
			}
			*lower = isl_basic_set_constraint(
					isl_basic_set_copy(bset), lower_line);
			*upper = isl_basic_set_constraint(
					isl_basic_set_copy(bset), upper_line);
			isl_int_clear(m);
			return 1;
		}
	}
	*lower = NULL;
	*upper = NULL;
	isl_int_clear(m);
	return 0;
}

/* Given two constraints "a" and "b" on the variable at position "abs_pos"
 * (in "a" and "b"), add a constraint to "bset" that ensures that the
 * bound implied by "a" is (strictly) larger than the bound implied by "b".
 *
 * If both constraints imply lower bounds, then this means that "a" is
 * active in the result.
 * If both constraints imply upper bounds, then this means that "b" is
 * active in the result.
 */
static __isl_give isl_basic_set *add_larger_bound_constraint(
	__isl_take isl_basic_set *bset, isl_int *a, isl_int *b,
	unsigned abs_pos, int strict)
{
	int k;
	isl_int t;
	unsigned total;

	k = isl_basic_set_alloc_inequality(bset);
	if (k < 0)
		goto error;

	total = isl_basic_set_dim(bset, isl_dim_all);

	isl_int_init(t);
	isl_int_neg(t, b[1 + abs_pos]);

	isl_seq_combine(bset->ineq[k], t, a, a[1 + abs_pos], b, 1 + abs_pos);
	isl_seq_combine(bset->ineq[k] + 1 + abs_pos,
		t, a + 1 + abs_pos + 1, a[1 + abs_pos], b + 1 + abs_pos + 1,
		total - abs_pos);

	if (strict)
		isl_int_sub_ui(bset->ineq[k][0], bset->ineq[k][0], 1);

	isl_int_clear(t);

	return bset;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Add constraints to "context" that ensure that "u" is the smallest
 * (and therefore active) upper bound on "abs_pos" in "bset" and return
 * the resulting basic set.
 */
static __isl_give isl_basic_set *set_smallest_upper_bound(
	__isl_keep isl_basic_set *context,
	__isl_keep isl_basic_set *bset, unsigned abs_pos, int n_upper, int u)
{
	int j;

	context = isl_basic_set_copy(context);
	context = isl_basic_set_cow(context);

	context = isl_basic_set_extend_constraints(context, 0, n_upper - 1);

	for (j = 0; j < bset->n_ineq; ++j) {
		if (j == u)
			continue;
		if (!isl_int_is_neg(bset->ineq[j][1 + abs_pos]))
			continue;
		context = add_larger_bound_constraint(context,
			bset->ineq[j], bset->ineq[u], abs_pos, j > u);
	}

	context = isl_basic_set_simplify(context);
	context = isl_basic_set_finalize(context);

	return context;
}

/* Add constraints to "context" that ensure that "u" is the largest
 * (and therefore active) upper bound on "abs_pos" in "bset" and return
 * the resulting basic set.
 */
static __isl_give isl_basic_set *set_largest_lower_bound(
	__isl_keep isl_basic_set *context,
	__isl_keep isl_basic_set *bset, unsigned abs_pos, int n_lower, int l)
{
	int j;

	context = isl_basic_set_copy(context);
	context = isl_basic_set_cow(context);

	context = isl_basic_set_extend_constraints(context, 0, n_lower - 1);

	for (j = 0; j < bset->n_ineq; ++j) {
		if (j == l)
			continue;
		if (!isl_int_is_pos(bset->ineq[j][1 + abs_pos]))
			continue;
		context = add_larger_bound_constraint(context,
			bset->ineq[l], bset->ineq[j], abs_pos, j > l);
	}

	context = isl_basic_set_simplify(context);
	context = isl_basic_set_finalize(context);

	return context;
}

static isl_stat foreach_upper_bound(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned abs_pos,
	__isl_take isl_basic_set *context, int n_upper,
	isl_stat (*fn)(__isl_take isl_constraint *lower,
		  __isl_take isl_constraint *upper,
		  __isl_take isl_basic_set *bset, void *user), void *user)
{
	isl_basic_set *context_i;
	isl_constraint *upper = NULL;
	int i;

	for (i = 0; i < bset->n_ineq; ++i) {
		if (isl_int_is_zero(bset->ineq[i][1 + abs_pos]))
			continue;

		context_i = set_smallest_upper_bound(context, bset,
							abs_pos, n_upper, i);
		if (isl_basic_set_is_empty(context_i)) {
			isl_basic_set_free(context_i);
			continue;
		}
		upper = isl_basic_set_constraint(isl_basic_set_copy(bset),
						&bset->ineq[i]);
		if (!upper || !context_i)
			goto error;
		if (fn(NULL, upper, context_i, user) < 0)
			break;
	}

	isl_basic_set_free(context);

	if (i < bset->n_ineq)
		return isl_stat_error;

	return isl_stat_ok;
error:
	isl_constraint_free(upper);
	isl_basic_set_free(context_i);
	isl_basic_set_free(context);
	return isl_stat_error;
}

static isl_stat foreach_lower_bound(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned abs_pos,
	__isl_take isl_basic_set *context, int n_lower,
	isl_stat (*fn)(__isl_take isl_constraint *lower,
		  __isl_take isl_constraint *upper,
		  __isl_take isl_basic_set *bset, void *user), void *user)
{
	isl_basic_set *context_i;
	isl_constraint *lower = NULL;
	int i;

	for (i = 0; i < bset->n_ineq; ++i) {
		if (isl_int_is_zero(bset->ineq[i][1 + abs_pos]))
			continue;

		context_i = set_largest_lower_bound(context, bset,
							abs_pos, n_lower, i);
		if (isl_basic_set_is_empty(context_i)) {
			isl_basic_set_free(context_i);
			continue;
		}
		lower = isl_basic_set_constraint(isl_basic_set_copy(bset),
						&bset->ineq[i]);
		if (!lower || !context_i)
			goto error;
		if (fn(lower, NULL, context_i, user) < 0)
			break;
	}

	isl_basic_set_free(context);

	if (i < bset->n_ineq)
		return isl_stat_error;

	return isl_stat_ok;
error:
	isl_constraint_free(lower);
	isl_basic_set_free(context_i);
	isl_basic_set_free(context);
	return isl_stat_error;
}

static isl_stat foreach_bound_pair(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned abs_pos,
	__isl_take isl_basic_set *context, int n_lower, int n_upper,
	isl_stat (*fn)(__isl_take isl_constraint *lower,
		  __isl_take isl_constraint *upper,
		  __isl_take isl_basic_set *bset, void *user), void *user)
{
	isl_basic_set *context_i, *context_j;
	isl_constraint *lower = NULL;
	isl_constraint *upper = NULL;
	int i, j;

	for (i = 0; i < bset->n_ineq; ++i) {
		if (!isl_int_is_pos(bset->ineq[i][1 + abs_pos]))
			continue;

		context_i = set_largest_lower_bound(context, bset,
							abs_pos, n_lower, i);
		if (isl_basic_set_is_empty(context_i)) {
			isl_basic_set_free(context_i);
			continue;
		}

		for (j = 0; j < bset->n_ineq; ++j) {
			if (!isl_int_is_neg(bset->ineq[j][1 + abs_pos]))
				continue;

			context_j = set_smallest_upper_bound(context_i, bset,
							    abs_pos, n_upper, j);
			context_j = isl_basic_set_extend_constraints(context_j,
									0, 1);
			context_j = add_larger_bound_constraint(context_j,
				bset->ineq[i], bset->ineq[j], abs_pos, 0);
			context_j = isl_basic_set_simplify(context_j);
			context_j = isl_basic_set_finalize(context_j);
			if (isl_basic_set_is_empty(context_j)) {
				isl_basic_set_free(context_j);
				continue;
			}
			lower = isl_basic_set_constraint(isl_basic_set_copy(bset),
							&bset->ineq[i]);
			upper = isl_basic_set_constraint(isl_basic_set_copy(bset),
							&bset->ineq[j]);
			if (!lower || !upper || !context_j)
				goto error;
			if (fn(lower, upper, context_j, user) < 0)
				break;
		}

		isl_basic_set_free(context_i);

		if (j < bset->n_ineq)
			break;
	}

	isl_basic_set_free(context);

	if (i < bset->n_ineq)
		return isl_stat_error;

	return isl_stat_ok;
error:
	isl_constraint_free(lower);
	isl_constraint_free(upper);
	isl_basic_set_free(context_i);
	isl_basic_set_free(context_j);
	isl_basic_set_free(context);
	return isl_stat_error;
}

/* For each pair of lower and upper bounds on the variable "pos"
 * of type "type", call "fn" with these lower and upper bounds and the
 * set of constraints on the remaining variables where these bounds
 * are active, i.e., (stricly) larger/smaller than the other lower/upper bounds.
 *
 * If the designated variable is equal to an affine combination of the
 * other variables then fn is called with both lower and upper
 * set to the corresponding equality.
 *
 * If there is no lower (or upper) bound, then NULL is passed
 * as the corresponding bound.
 *
 * We first check if the variable is involved in any equality.
 * If not, we count the number of lower and upper bounds and
 * act accordingly.
 */
isl_stat isl_basic_set_foreach_bound_pair(__isl_keep isl_basic_set *bset,
	enum isl_dim_type type, unsigned pos,
	isl_stat (*fn)(__isl_take isl_constraint *lower,
		  __isl_take isl_constraint *upper,
		  __isl_take isl_basic_set *bset, void *user), void *user)
{
	int i;
	isl_constraint *lower = NULL;
	isl_constraint *upper = NULL;
	isl_basic_set *context = NULL;
	unsigned abs_pos;
	int n_lower, n_upper;

	if (!bset)
		return isl_stat_error;
	isl_assert(bset->ctx, pos < isl_basic_set_dim(bset, type),
		return isl_stat_error);
	isl_assert(bset->ctx, type == isl_dim_param || type == isl_dim_set,
		return isl_stat_error);

	abs_pos = pos;
	if (type == isl_dim_set)
		abs_pos += isl_basic_set_dim(bset, isl_dim_param);

	for (i = 0; i < bset->n_eq; ++i) {
		if (isl_int_is_zero(bset->eq[i][1 + abs_pos]))
			continue;

		lower = isl_basic_set_constraint(isl_basic_set_copy(bset),
						&bset->eq[i]);
		upper = isl_constraint_copy(lower);
		context = isl_basic_set_remove_dims(isl_basic_set_copy(bset),
					type, pos, 1);
		if (!lower || !upper || !context)
			goto error;
		return fn(lower, upper, context, user);
	}

	n_lower = 0;
	n_upper = 0;
	for (i = 0; i < bset->n_ineq; ++i) {
		if (isl_int_is_pos(bset->ineq[i][1 + abs_pos]))
			n_lower++;
		else if (isl_int_is_neg(bset->ineq[i][1 + abs_pos]))
			n_upper++;
	}

	context = isl_basic_set_copy(bset);
	context = isl_basic_set_cow(context);
	if (!context)
		goto error;
	for (i = context->n_ineq - 1; i >= 0; --i)
		if (!isl_int_is_zero(context->ineq[i][1 + abs_pos]))
			isl_basic_set_drop_inequality(context, i);

	context = isl_basic_set_drop(context, type, pos, 1);
	if (!n_lower && !n_upper)
		return fn(NULL, NULL, context, user);
	if (!n_lower)
		return foreach_upper_bound(bset, type, abs_pos, context, n_upper,
						fn, user);
	if (!n_upper)
		return foreach_lower_bound(bset, type, abs_pos, context, n_lower,
						fn, user);
	return foreach_bound_pair(bset, type, abs_pos, context, n_lower, n_upper,
					fn, user);
error:
	isl_constraint_free(lower);
	isl_constraint_free(upper);
	isl_basic_set_free(context);
	return -1;
}

__isl_give isl_aff *isl_constraint_get_bound(
	__isl_keep isl_constraint *constraint, enum isl_dim_type type, int pos)
{
	isl_aff *aff;
	isl_ctx *ctx;

	if (!constraint)
		return NULL;
	ctx = isl_constraint_get_ctx(constraint);
	if (pos >= isl_constraint_dim(constraint, type))
		isl_die(ctx, isl_error_invalid,
			"index out of bounds", return NULL);
	if (isl_constraint_dim(constraint, isl_dim_in) != 0)
		isl_die(ctx, isl_error_invalid,
			"not a set constraint", return NULL);

	pos += offset(constraint, type);
	if (isl_int_is_zero(constraint->v->el[pos]))
		isl_die(ctx, isl_error_invalid,
			"constraint does not define a bound on given dimension",
			return NULL);

	aff = isl_aff_alloc(isl_local_space_copy(constraint->ls));
	if (!aff)
		return NULL;

	if (isl_int_is_neg(constraint->v->el[pos]))
		isl_seq_cpy(aff->v->el + 1, constraint->v->el, aff->v->size - 1);
	else
		isl_seq_neg(aff->v->el + 1, constraint->v->el, aff->v->size - 1);
	isl_int_set_si(aff->v->el[1 + pos], 0);
	isl_int_abs(aff->v->el[0], constraint->v->el[pos]);

	return aff;
}

/* For an inequality constraint
 *
 *	f >= 0
 *
 * or an equality constraint
 *
 *	f = 0
 *
 * return the affine expression f.
 */
__isl_give isl_aff *isl_constraint_get_aff(
	__isl_keep isl_constraint *constraint)
{
	isl_aff *aff;

	if (!constraint)
		return NULL;

	aff = isl_aff_alloc(isl_local_space_copy(constraint->ls));
	if (!aff)
		return NULL;

	isl_seq_cpy(aff->v->el + 1, constraint->v->el, aff->v->size - 1);
	isl_int_set_si(aff->v->el[0], 1);

	return aff;
}

/* Construct an inequality (eq = 0) or equality (eq = 1) constraint from "aff".
 * In particular, construct aff >= 0 or aff = 0.
 *
 * The denominator of "aff" can be ignored.
 */
static __isl_give isl_constraint *isl_constraint_alloc_aff(int eq,
	__isl_take isl_aff *aff)
{
	isl_local_space *ls;
	isl_vec *v;

	if (!aff)
		return NULL;
	ls = isl_aff_get_domain_local_space(aff);
	v = isl_vec_drop_els(isl_vec_copy(aff->v), 0, 1);
	isl_aff_free(aff);

	return isl_constraint_alloc_vec(eq, ls, v);
}

/* Construct an equality constraint equating the given affine expression
 * to zero.
 */
__isl_give isl_constraint *isl_equality_from_aff(__isl_take isl_aff *aff)
{
	return isl_constraint_alloc_aff(1, aff);
}

/* Construct an inequality constraint enforcing the given affine expression
 * to be non-negative.
 */
__isl_give isl_constraint *isl_inequality_from_aff(__isl_take isl_aff *aff)
{
	return isl_constraint_alloc_aff(0, aff);
}

/* Compare two isl_constraints.
 *
 * Return -1 if "c1" is "smaller" than "c2", 1 if "c1" is "greater"
 * than "c2" and 0 if they are equal.
 *
 * The order is fairly arbitrary.  We do consider constraints that only involve
 * earlier dimensions as "smaller".
 */
int isl_constraint_plain_cmp(__isl_keep isl_constraint *c1,
	__isl_keep isl_constraint *c2)
{
	int cmp;
	int last1, last2;

	if (c1 == c2)
		return 0;
	if (!c1)
		return -1;
	if (!c2)
		return 1;
	cmp = isl_local_space_cmp(c1->ls, c2->ls);
	if (cmp != 0)
		return cmp;

	last1 = isl_seq_last_non_zero(c1->v->el + 1, c1->v->size - 1);
	last2 = isl_seq_last_non_zero(c2->v->el + 1, c1->v->size - 1);
	if (last1 != last2)
		return last1 - last2;

	return isl_seq_cmp(c1->v->el, c2->v->el, c1->v->size);
}

/* Compare two constraints based on their final (non-zero) coefficients.
 * In particular, the constraint that involves later variables or
 * that has a larger coefficient for a shared latest variable
 * is considered "greater" than the other constraint.
 *
 * Return -1 if "c1" is "smaller" than "c2", 1 if "c1" is "greater"
 * than "c2" and 0 if they are equal.
 *
 * If the constraints live in different local spaces, then we cannot
 * really compare the constraints so we compare the local spaces instead.
 */
int isl_constraint_cmp_last_non_zero(__isl_keep isl_constraint *c1,
	__isl_keep isl_constraint *c2)
{
	int cmp;
	int last1, last2;

	if (c1 == c2)
		return 0;
	if (!c1)
		return -1;
	if (!c2)
		return 1;
	cmp = isl_local_space_cmp(c1->ls, c2->ls);
	if (cmp != 0)
		return cmp;

	last1 = isl_seq_last_non_zero(c1->v->el + 1, c1->v->size - 1);
	last2 = isl_seq_last_non_zero(c2->v->el + 1, c1->v->size - 1);
	if (last1 != last2)
		return last1 - last2;
	if (last1 == -1)
		return 0;
	return isl_int_abs_cmp(c1->v->el[1 + last1], c2->v->el[1 + last2]);
}
