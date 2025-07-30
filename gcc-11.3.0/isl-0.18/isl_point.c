#include <isl_map_private.h>
#include <isl_point_private.h>
#include <isl/set.h>
#include <isl/union_set.h>
#include <isl_sample.h>
#include <isl_scan.h>
#include <isl_seq.h>
#include <isl_space_private.h>
#include <isl_val_private.h>
#include <isl_vec_private.h>
#include <isl_output_private.h>
#include <isl/deprecated/point_int.h>

#include <set_to_map.c>

isl_ctx *isl_point_get_ctx(__isl_keep isl_point *pnt)
{
	return pnt ? isl_space_get_ctx(pnt->dim) : NULL;
}

__isl_give isl_space *isl_point_get_space(__isl_keep isl_point *pnt)
{
	return pnt ? isl_space_copy(pnt->dim) : NULL;
}

__isl_give isl_point *isl_point_alloc(__isl_take isl_space *dim,
	__isl_take isl_vec *vec)
{
	struct isl_point *pnt;

	if (!dim || !vec)
		goto error;

	if (vec->size > 1 + isl_space_dim(dim, isl_dim_all)) {
		vec = isl_vec_cow(vec);
		if (!vec)
			goto error;
		vec->size = 1 + isl_space_dim(dim, isl_dim_all);
	}

	pnt = isl_alloc_type(dim->ctx, struct isl_point);
	if (!pnt)
		goto error;

	pnt->ref = 1;
	pnt->dim = dim;
	pnt->vec = vec;

	return pnt;
error:
	isl_space_free(dim);
	isl_vec_free(vec);
	return NULL;
}

__isl_give isl_point *isl_point_zero(__isl_take isl_space *dim)
{
	isl_vec *vec;

	if (!dim)
		return NULL;
	vec = isl_vec_alloc(dim->ctx, 1 + isl_space_dim(dim, isl_dim_all));
	if (!vec)
		goto error;
	isl_int_set_si(vec->el[0], 1);
	isl_seq_clr(vec->el + 1, vec->size - 1);
	return isl_point_alloc(dim, vec);
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give isl_point *isl_point_dup(__isl_keep isl_point *pnt)
{
	struct isl_point *pnt2;

	if (!pnt)
		return NULL;
	pnt2 = isl_point_alloc(isl_space_copy(pnt->dim), isl_vec_copy(pnt->vec));
	return pnt2;
}

__isl_give isl_point *isl_point_cow(__isl_take isl_point *pnt)
{
	struct isl_point *pnt2;
	if (!pnt)
		return NULL;

	if (pnt->ref == 1)
		return pnt;

	pnt2 = isl_point_dup(pnt);
	isl_point_free(pnt);
	return pnt2;
}

__isl_give isl_point *isl_point_copy(__isl_keep isl_point *pnt)
{
	if (!pnt)
		return NULL;

	pnt->ref++;
	return pnt;
}

void isl_point_free(__isl_take isl_point *pnt)
{
	if (!pnt)
		return;

	if (--pnt->ref > 0)
		return;

	isl_space_free(pnt->dim);
	isl_vec_free(pnt->vec);
	free(pnt);
}

__isl_give isl_point *isl_point_void(__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;

	return isl_point_alloc(dim, isl_vec_alloc(dim->ctx, 0));
}

isl_bool isl_point_is_void(__isl_keep isl_point *pnt)
{
	if (!pnt)
		return isl_bool_error;

	return pnt->vec->size == 0;
}

int isl_point_get_coordinate(__isl_keep isl_point *pnt,
	enum isl_dim_type type, int pos, isl_int *v)
{
	if (!pnt || isl_point_is_void(pnt))
		return -1;

	if (pos < 0 || pos >= isl_space_dim(pnt->dim, type))
		isl_die(isl_point_get_ctx(pnt), isl_error_invalid,
			"position out of bounds", return -1);

	if (type == isl_dim_set)
		pos += isl_space_dim(pnt->dim, isl_dim_param);
	isl_int_set(*v, pnt->vec->el[1 + pos]);

	return 0;
}

/* Return the value of coordinate "pos" of type "type" of "pnt".
 */
__isl_give isl_val *isl_point_get_coordinate_val(__isl_keep isl_point *pnt,
	enum isl_dim_type type, int pos)
{
	isl_ctx *ctx;
	isl_val *v;

	if (!pnt)
		return NULL;

	ctx = isl_point_get_ctx(pnt);
	if (isl_point_is_void(pnt))
		isl_die(ctx, isl_error_invalid,
			"void point does not have coordinates", return NULL);
	if (pos < 0 || pos >= isl_space_dim(pnt->dim, type))
		isl_die(ctx, isl_error_invalid,
			"position out of bounds", return NULL);

	if (type == isl_dim_set)
		pos += isl_space_dim(pnt->dim, isl_dim_param);

	v = isl_val_rat_from_isl_int(ctx, pnt->vec->el[1 + pos],
						pnt->vec->el[0]);
	return isl_val_normalize(v);
}

__isl_give isl_point *isl_point_set_coordinate(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, isl_int v)
{
	if (!pnt || isl_point_is_void(pnt))
		return pnt;

	pnt = isl_point_cow(pnt);
	if (!pnt)
		return NULL;
	pnt->vec = isl_vec_cow(pnt->vec);
	if (!pnt->vec)
		goto error;

	if (type == isl_dim_set)
		pos += isl_space_dim(pnt->dim, isl_dim_param);

	isl_int_set(pnt->vec->el[1 + pos], v);

	return pnt;
error:
	isl_point_free(pnt);
	return NULL;
}

/* Replace coordinate "pos" of type "type" of "pnt" by "v".
 */
__isl_give isl_point *isl_point_set_coordinate_val(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, __isl_take isl_val *v)
{
	if (!pnt || !v)
		goto error;
	if (isl_point_is_void(pnt))
		isl_die(isl_point_get_ctx(pnt), isl_error_invalid,
			"void point does not have coordinates", goto error);
	if (pos < 0 || pos >= isl_space_dim(pnt->dim, type))
		isl_die(isl_point_get_ctx(pnt), isl_error_invalid,
			"position out of bounds", goto error);
	if (!isl_val_is_rat(v))
		isl_die(isl_point_get_ctx(pnt), isl_error_invalid,
			"expecting rational value", goto error);

	if (isl_int_eq(pnt->vec->el[1 + pos], v->n) &&
	    isl_int_eq(pnt->vec->el[0], v->d)) {
		isl_val_free(v);
		return pnt;
	}

	pnt = isl_point_cow(pnt);
	if (!pnt)
		goto error;
	pnt->vec = isl_vec_cow(pnt->vec);
	if (!pnt->vec)
		goto error;

	if (isl_int_eq(pnt->vec->el[0], v->d)) {
		isl_int_set(pnt->vec->el[1 + pos], v->n);
	} else if (isl_int_is_one(v->d)) {
		isl_int_mul(pnt->vec->el[1 + pos], pnt->vec->el[0], v->n);
	} else {
		isl_seq_scale(pnt->vec->el + 1,
				pnt->vec->el + 1, v->d, pnt->vec->size - 1);
		isl_int_mul(pnt->vec->el[1 + pos], pnt->vec->el[0], v->n);
		isl_int_mul(pnt->vec->el[0], pnt->vec->el[0], v->d);
		pnt->vec = isl_vec_normalize(pnt->vec);
		if (!pnt->vec)
			goto error;
	}

	isl_val_free(v);
	return pnt;
error:
	isl_val_free(v);
	isl_point_free(pnt);
	return NULL;
}

__isl_give isl_point *isl_point_add_ui(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, unsigned val)
{
	if (!pnt || isl_point_is_void(pnt))
		return pnt;

	pnt = isl_point_cow(pnt);
	if (!pnt)
		return NULL;
	pnt->vec = isl_vec_cow(pnt->vec);
	if (!pnt->vec)
		goto error;

	if (type == isl_dim_set)
		pos += isl_space_dim(pnt->dim, isl_dim_param);

	isl_int_add_ui(pnt->vec->el[1 + pos], pnt->vec->el[1 + pos], val);

	return pnt;
error:
	isl_point_free(pnt);
	return NULL;
}

__isl_give isl_point *isl_point_sub_ui(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, unsigned val)
{
	if (!pnt || isl_point_is_void(pnt))
		return pnt;

	pnt = isl_point_cow(pnt);
	if (!pnt)
		return NULL;
	pnt->vec = isl_vec_cow(pnt->vec);
	if (!pnt->vec)
		goto error;

	if (type == isl_dim_set)
		pos += isl_space_dim(pnt->dim, isl_dim_param);

	isl_int_sub_ui(pnt->vec->el[1 + pos], pnt->vec->el[1 + pos], val);

	return pnt;
error:
	isl_point_free(pnt);
	return NULL;
}

struct isl_foreach_point {
	struct isl_scan_callback callback;
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user);
	void *user;
	isl_space *dim;
};

static isl_stat foreach_point(struct isl_scan_callback *cb,
	__isl_take isl_vec *sample)
{
	struct isl_foreach_point *fp = (struct isl_foreach_point *)cb;
	isl_point *pnt;

	pnt = isl_point_alloc(isl_space_copy(fp->dim), sample);

	return fp->fn(pnt, fp->user);
}

isl_stat isl_set_foreach_point(__isl_keep isl_set *set,
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user), void *user)
{
	struct isl_foreach_point fp = { { &foreach_point }, fn, user };
	int i;

	if (!set)
		return isl_stat_error;

	fp.dim = isl_set_get_space(set);
	if (!fp.dim)
		return isl_stat_error;

	set = isl_set_copy(set);
	set = isl_set_cow(set);
	set = isl_set_make_disjoint(set);
	set = isl_set_compute_divs(set);
	if (!set)
		goto error;

	for (i = 0; i < set->n; ++i)
		if (isl_basic_set_scan(isl_basic_set_copy(set->p[i]),
					&fp.callback) < 0)
			goto error;

	isl_set_free(set);
	isl_space_free(fp.dim);

	return isl_stat_ok;
error:
	isl_set_free(set);
	isl_space_free(fp.dim);
	return isl_stat_error;
}

/* Return 1 if "bmap" contains the point "point".
 * "bmap" is assumed to have known divs.
 * The point is first extended with the divs and then passed
 * to basic_map_contains.
 */
isl_bool isl_basic_map_contains_point(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_point *point)
{
	int i;
	struct isl_vec *vec;
	unsigned dim;
	isl_bool contains;

	if (!bmap || !point)
		return isl_bool_error;
	isl_assert(bmap->ctx, isl_space_is_equal(bmap->dim, point->dim),
		return isl_bool_error);
	if (bmap->n_div == 0)
		return isl_basic_map_contains(bmap, point->vec);

	dim = isl_basic_map_total_dim(bmap) - bmap->n_div;
	vec = isl_vec_alloc(bmap->ctx, 1 + dim + bmap->n_div);
	if (!vec)
		return isl_bool_error;

	isl_seq_cpy(vec->el, point->vec->el, point->vec->size);
	for (i = 0; i < bmap->n_div; ++i) {
		isl_seq_inner_product(bmap->div[i] + 1, vec->el,
					1 + dim + i, &vec->el[1+dim+i]);
		isl_int_fdiv_q(vec->el[1+dim+i], vec->el[1+dim+i],
				bmap->div[i][0]);
	}

	contains = isl_basic_map_contains(bmap, vec);

	isl_vec_free(vec);
	return contains;
}

int isl_map_contains_point(__isl_keep isl_map *map, __isl_keep isl_point *point)
{
	int i;
	int found = 0;

	if (!map || !point)
		return -1;

	map = isl_map_copy(map);
	map = isl_map_compute_divs(map);
	if (!map)
		return -1;

	for (i = 0; i < map->n; ++i) {
		found = isl_basic_map_contains_point(map->p[i], point);
		if (found < 0)
			goto error;
		if (found)
			break;
	}
	isl_map_free(map);

	return found;
error:
	isl_map_free(map);
	return -1;
}

isl_bool isl_set_contains_point(__isl_keep isl_set *set,
	__isl_keep isl_point *point)
{
	return isl_map_contains_point(set_to_map(set), point);
}

__isl_give isl_basic_set *isl_basic_set_from_point(__isl_take isl_point *pnt)
{
	isl_basic_set *bset;
	isl_basic_set *model;

	if (!pnt)
		return NULL;

	model = isl_basic_set_empty(isl_space_copy(pnt->dim));
	bset = isl_basic_set_from_vec(isl_vec_copy(pnt->vec));
	bset = isl_basic_set_from_underlying_set(bset, model);
	isl_point_free(pnt);

	return bset;
}

__isl_give isl_set *isl_set_from_point(__isl_take isl_point *pnt)
{
	isl_basic_set *bset;
	bset = isl_basic_set_from_point(pnt);
	return isl_set_from_basic_set(bset);
}

/* Construct a union set, containing the single element "pnt".
 * If "pnt" is void, then return an empty union set.
 */
__isl_give isl_union_set *isl_union_set_from_point(__isl_take isl_point *pnt)
{
	if (!pnt)
		return NULL;
	if (isl_point_is_void(pnt)) {
		isl_space *space;

		space = isl_point_get_space(pnt);
		isl_point_free(pnt);
		return isl_union_set_empty(space);
	}

	return isl_union_set_from_set(isl_set_from_point(pnt));
}

__isl_give isl_basic_set *isl_basic_set_box_from_points(
	__isl_take isl_point *pnt1, __isl_take isl_point *pnt2)
{
	isl_basic_set *bset;
	unsigned total;
	int i;
	int k;
	isl_int t;

	isl_int_init(t);

	if (!pnt1 || !pnt2)
		goto error;

	isl_assert(pnt1->dim->ctx,
			isl_space_is_equal(pnt1->dim, pnt2->dim), goto error);

	if (isl_point_is_void(pnt1) && isl_point_is_void(pnt2)) {
		isl_space *dim = isl_space_copy(pnt1->dim);
		isl_point_free(pnt1);
		isl_point_free(pnt2);
		isl_int_clear(t);
		return isl_basic_set_empty(dim);
	}
	if (isl_point_is_void(pnt1)) {
		isl_point_free(pnt1);
		isl_int_clear(t);
		return isl_basic_set_from_point(pnt2);
	}
	if (isl_point_is_void(pnt2)) {
		isl_point_free(pnt2);
		isl_int_clear(t);
		return isl_basic_set_from_point(pnt1);
	}

	total = isl_space_dim(pnt1->dim, isl_dim_all);
	bset = isl_basic_set_alloc_space(isl_space_copy(pnt1->dim), 0, 0, 2 * total);

	for (i = 0; i < total; ++i) {
		isl_int_mul(t, pnt1->vec->el[1 + i], pnt2->vec->el[0]);
		isl_int_submul(t, pnt2->vec->el[1 + i], pnt1->vec->el[0]);

		k = isl_basic_set_alloc_inequality(bset);
		if (k < 0)
			goto error;
		isl_seq_clr(bset->ineq[k] + 1, total);
		if (isl_int_is_pos(t)) {
			isl_int_set_si(bset->ineq[k][1 + i], -1);
			isl_int_set(bset->ineq[k][0], pnt1->vec->el[1 + i]);
		} else {
			isl_int_set_si(bset->ineq[k][1 + i], 1);
			isl_int_neg(bset->ineq[k][0], pnt1->vec->el[1 + i]);
		}
		isl_int_fdiv_q(bset->ineq[k][0], bset->ineq[k][0], pnt1->vec->el[0]);

		k = isl_basic_set_alloc_inequality(bset);
		if (k < 0)
			goto error;
		isl_seq_clr(bset->ineq[k] + 1, total);
		if (isl_int_is_pos(t)) {
			isl_int_set_si(bset->ineq[k][1 + i], 1);
			isl_int_neg(bset->ineq[k][0], pnt2->vec->el[1 + i]);
		} else {
			isl_int_set_si(bset->ineq[k][1 + i], -1);
			isl_int_set(bset->ineq[k][0], pnt2->vec->el[1 + i]);
		}
		isl_int_fdiv_q(bset->ineq[k][0], bset->ineq[k][0], pnt2->vec->el[0]);
	}

	bset = isl_basic_set_finalize(bset);

	isl_point_free(pnt1);
	isl_point_free(pnt2);

	isl_int_clear(t);

	return bset;
error:
	isl_point_free(pnt1);
	isl_point_free(pnt2);
	isl_int_clear(t);
	return NULL;
}

__isl_give isl_set *isl_set_box_from_points(__isl_take isl_point *pnt1,
	__isl_take isl_point *pnt2)
{
	isl_basic_set *bset;
	bset = isl_basic_set_box_from_points(pnt1, pnt2);
	return isl_set_from_basic_set(bset);
}

/* Print the coordinate at position "pos" of the point "pnt".
 */
static __isl_give isl_printer *print_coordinate(__isl_take isl_printer *p,
	struct isl_print_space_data *data, unsigned pos)
{
	isl_point *pnt = data->user;

	p = isl_printer_print_isl_int(p, pnt->vec->el[1 + pos]);
	if (!isl_int_is_one(pnt->vec->el[0])) {
		p = isl_printer_print_str(p, "/");
		p = isl_printer_print_isl_int(p, pnt->vec->el[0]);
	}

	return p;
}

__isl_give isl_printer *isl_printer_print_point(
	__isl_take isl_printer *p, __isl_keep isl_point *pnt)
{
	struct isl_print_space_data data = { 0 };
	int i;
	unsigned nparam;
	unsigned dim;

	if (!pnt)
		return p;
	if (isl_point_is_void(pnt)) {
		p = isl_printer_print_str(p, "void");
		return p;
	}

	nparam = isl_space_dim(pnt->dim, isl_dim_param);
	dim = isl_space_dim(pnt->dim, isl_dim_set);
	if (nparam > 0) {
		p = isl_printer_print_str(p, "[");
		for (i = 0; i < nparam; ++i) {
			const char *name;
			if (i)
				p = isl_printer_print_str(p, ", ");
			name = isl_space_get_dim_name(pnt->dim, isl_dim_param, i);
			if (name) {
				p = isl_printer_print_str(p, name);
				p = isl_printer_print_str(p, " = ");
			}
			p = isl_printer_print_isl_int(p, pnt->vec->el[1 + i]);
			if (!isl_int_is_one(pnt->vec->el[0])) {
				p = isl_printer_print_str(p, "/");
				p = isl_printer_print_isl_int(p, pnt->vec->el[0]);
			}
		}
		p = isl_printer_print_str(p, "]");
		p = isl_printer_print_str(p, " -> ");
	}
	data.print_dim = &print_coordinate;
	data.user = pnt;
	p = isl_printer_print_str(p, "{ ");
	p = isl_print_space(pnt->dim, p, 0, &data);
	p = isl_printer_print_str(p, " }");
	return p;
}
