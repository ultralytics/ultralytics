/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_band_private.h>
#include <isl_schedule_private.h>

#undef BASE
#define BASE band

#include <isl_list_templ.c>

isl_ctx *isl_band_get_ctx(__isl_keep isl_band *band)
{
	return band ? isl_union_pw_multi_aff_get_ctx(band->pma) : NULL;
}

__isl_give isl_band *isl_band_alloc(isl_ctx *ctx)
{
	isl_band *band;

	band = isl_calloc_type(ctx, isl_band);
	if (!band)
		return NULL;

	band->ref = 1;

	return band;
}

/* Create a duplicate of the given band.  The duplicate refers
 * to the same schedule and parent as the input, but does not
 * increment their reference counts.
 */
__isl_give isl_band *isl_band_dup(__isl_keep isl_band *band)
{
	int i;
	isl_ctx *ctx;
	isl_band *dup;

	if (!band)
		return NULL;

	ctx = isl_band_get_ctx(band);
	dup = isl_band_alloc(ctx);
	if (!dup)
		return NULL;

	dup->n = band->n;
	dup->coincident = isl_alloc_array(ctx, int, band->n);
	if (band->n && !dup->coincident)
		goto error;

	for (i = 0; i < band->n; ++i)
		dup->coincident[i] = band->coincident[i];

	dup->pma = isl_union_pw_multi_aff_copy(band->pma);
	dup->schedule = band->schedule;
	dup->parent = band->parent;

	if (!dup->pma)
		goto error;

	return dup;
error:
	isl_band_free(dup);
	return NULL;
}

/* We not only increment the reference count of the band,
 * but also that of the schedule that contains this band.
 * This ensures that the schedule won't disappear while there
 * is still a reference to the band outside of the schedule.
 * There is no need to increment the reference count of the parent
 * band as the parent band is part of the same schedule.
 */
__isl_give isl_band *isl_band_copy(__isl_keep isl_band *band)
{
	if (!band)
		return NULL;

	band->ref++;
	band->schedule->ref++;
	return band;
}

/* If this is not the last reference to the band (the one from within the
 * schedule), then we also need to decrement the reference count of the
 * containing schedule as it was incremented in isl_band_copy.
 */
__isl_null isl_band *isl_band_free(__isl_take isl_band *band)
{
	if (!band)
		return NULL;

	if (--band->ref > 0) {
		isl_schedule_free(band->schedule);
		return NULL;
	}

	isl_union_pw_multi_aff_free(band->pma);
	isl_band_list_free(band->children);
	free(band->coincident);
	free(band);

	return NULL;
}

int isl_band_has_children(__isl_keep isl_band *band)
{
	if (!band)
		return -1;

	return band->children != NULL;
}

__isl_give isl_band_list *isl_band_get_children(
	__isl_keep isl_band *band)
{
	if (!band)
		return NULL;
	if (!band->children)
		isl_die(isl_band_get_ctx(band), isl_error_invalid,
			"band has no children", return NULL);
	return isl_band_list_dup(band->children);
}

int isl_band_n_member(__isl_keep isl_band *band)
{
	return band ? band->n : 0;
}

/* Is the given scheduling dimension coincident within the band and
 * with respect to the coincidence constraints.
 */
int isl_band_member_is_coincident(__isl_keep isl_band *band, int pos)
{
	if (!band)
		return -1;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_band_get_ctx(band), isl_error_invalid,
			"invalid member position", return -1);

	return band->coincident[pos];
}

/* Return the schedule that leads up to this band.
 */
__isl_give isl_union_map *isl_band_get_prefix_schedule(
	__isl_keep isl_band *band)
{
	isl_union_set *domain;
	isl_union_pw_multi_aff *prefix;
	isl_band *a;

	if (!band)
		return NULL;

	prefix = isl_union_pw_multi_aff_copy(band->pma);
	domain = isl_union_pw_multi_aff_domain(prefix);
	prefix = isl_union_pw_multi_aff_from_domain(domain);

	for (a = band->parent; a; a = a->parent) {
		isl_union_pw_multi_aff *partial;

		partial = isl_union_pw_multi_aff_copy(a->pma);
		prefix = isl_union_pw_multi_aff_flat_range_product(partial,
								   prefix);
	}

	return isl_union_map_from_union_pw_multi_aff(prefix);
}

/* Return the schedule of the band in isolation.
 */
__isl_give isl_union_pw_multi_aff *
isl_band_get_partial_schedule_union_pw_multi_aff(__isl_keep isl_band *band)
{
	return band ? isl_union_pw_multi_aff_copy(band->pma) : NULL;
}

/* Return the schedule of the band in isolation.
 */
__isl_give isl_union_map *isl_band_get_partial_schedule(
	__isl_keep isl_band *band)
{
	isl_union_pw_multi_aff *sched;

	sched = isl_band_get_partial_schedule_union_pw_multi_aff(band);
	return isl_union_map_from_union_pw_multi_aff(sched);
}

__isl_give isl_union_pw_multi_aff *
isl_band_get_suffix_schedule_union_pw_multi_aff(__isl_keep isl_band *band);

/* Return the schedule for the given band list.
 * For each band in the list, the schedule is composed of the partial
 * and suffix schedules of that band.
 */
__isl_give isl_union_pw_multi_aff *
isl_band_list_get_suffix_schedule_union_pw_multi_aff(
	__isl_keep isl_band_list *list)
{
	isl_ctx *ctx;
	int i, n;
	isl_space *space;
	isl_union_pw_multi_aff *suffix;

	if (!list)
		return NULL;

	ctx = isl_band_list_get_ctx(list);
	space = isl_space_alloc(ctx, 0, 0, 0);
	suffix = isl_union_pw_multi_aff_empty(space);
	n = isl_band_list_n_band(list);
	for (i = 0; i < n; ++i) {
		isl_band *el;
		isl_union_pw_multi_aff *partial;
		isl_union_pw_multi_aff *suffix_i;

		el = isl_band_list_get_band(list, i);
		partial = isl_band_get_partial_schedule_union_pw_multi_aff(el);
		suffix_i = isl_band_get_suffix_schedule_union_pw_multi_aff(el);
		suffix_i = isl_union_pw_multi_aff_flat_range_product(
				partial, suffix_i);
		suffix = isl_union_pw_multi_aff_union_add(suffix, suffix_i);

		isl_band_free(el);
	}

	return suffix;
}

/* Return the schedule for the given band list.
 * For each band in the list, the schedule is composed of the partial
 * and suffix schedules of that band.
 */
__isl_give isl_union_map *isl_band_list_get_suffix_schedule(
	__isl_keep isl_band_list *list)
{
	isl_union_pw_multi_aff *suffix;

	suffix = isl_band_list_get_suffix_schedule_union_pw_multi_aff(list);
	return isl_union_map_from_union_pw_multi_aff(suffix);
}

/* Return the schedule for the forest underneath the given band.
 */
__isl_give isl_union_pw_multi_aff *
isl_band_get_suffix_schedule_union_pw_multi_aff(__isl_keep isl_band *band)
{
	isl_union_pw_multi_aff *suffix;

	if (!band)
		return NULL;

	if (!isl_band_has_children(band)) {
		isl_union_set *domain;

		suffix = isl_union_pw_multi_aff_copy(band->pma);
		domain = isl_union_pw_multi_aff_domain(suffix);
		suffix = isl_union_pw_multi_aff_from_domain(domain);
	} else {
		isl_band_list *list;

		list = isl_band_get_children(band);
		suffix =
		    isl_band_list_get_suffix_schedule_union_pw_multi_aff(list);
		isl_band_list_free(list);
	}

	return suffix;
}

/* Return the schedule for the forest underneath the given band.
 */
__isl_give isl_union_map *isl_band_get_suffix_schedule(
	__isl_keep isl_band *band)
{
	isl_union_pw_multi_aff *suffix;

	suffix = isl_band_get_suffix_schedule_union_pw_multi_aff(band);
	return isl_union_map_from_union_pw_multi_aff(suffix);
}

/* Call "fn" on each band (recursively) in the list
 * in depth-first post-order.
 */
int isl_band_list_foreach_band(__isl_keep isl_band_list *list,
	int (*fn)(__isl_keep isl_band *band, void *user), void *user)
{
	int i, n;

	if (!list)
		return -1;

	n = isl_band_list_n_band(list);
	for (i = 0; i < n; ++i) {
		isl_band *band;
		int r = 0;

		band = isl_band_list_get_band(list, i);
		if (isl_band_has_children(band)) {
			isl_band_list *children;

			children = isl_band_get_children(band);
			r = isl_band_list_foreach_band(children, fn, user);
			isl_band_list_free(children);
		}

		if (!band)
			r = -1;
		if (r == 0)
			r = fn(band, user);

		isl_band_free(band);
		if (r)
			return r;
	}

	return 0;
}

/* Internal data used during the construction of the schedule
 * for the tile loops.
 *
 * sizes contains the tile sizes
 * scale is set if the tile loops should be scaled
 * tiled collects the result for a single statement
 * res collects the result for all statements
 */
struct isl_band_tile_data {
	isl_multi_val *sizes;
	isl_union_pw_multi_aff *res;
	isl_pw_multi_aff *tiled;
	int scale;
};

/* Given part of the schedule of a band, construct the corresponding
 * schedule for the tile loops based on the tile sizes in data->sizes
 * and add the result to data->tiled.
 *
 * If data->scale is set, then dimension i of the schedule will be
 * of the form
 *
 *	m_i * floor(s_i(x) / m_i)
 *
 * where s_i(x) refers to the original schedule and m_i is the tile size.
 * If data->scale is not set, then dimension i of the schedule will be
 * of the form
 *
 *	floor(s_i(x) / m_i)
 *
 */
static isl_stat multi_aff_tile(__isl_take isl_set *set,
	__isl_take isl_multi_aff *ma, void *user)
{
	struct isl_band_tile_data *data = user;
	isl_pw_multi_aff *pma;
	int i, n;
	isl_val *v;

	n = isl_multi_aff_dim(ma, isl_dim_out);

	for (i = 0; i < n; ++i) {
		isl_aff *aff;

		aff = isl_multi_aff_get_aff(ma, i);
		v = isl_multi_val_get_val(data->sizes, i);

		aff = isl_aff_scale_down_val(aff, isl_val_copy(v));
		aff = isl_aff_floor(aff);
		if (data->scale)
			aff = isl_aff_scale_val(aff, isl_val_copy(v));
		isl_val_free(v);

		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	pma = isl_pw_multi_aff_alloc(set, ma);
	data->tiled = isl_pw_multi_aff_union_add(data->tiled, pma);

	return isl_stat_ok;
}

/* Given part of the schedule of a band, construct the corresponding
 * schedule for the tile loops based on the tile sizes in data->sizes
 * and add the result to data->res.
 */
static isl_stat pw_multi_aff_tile(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_band_tile_data *data = user;

	data->tiled = isl_pw_multi_aff_empty(isl_pw_multi_aff_get_space(pma));

	if (isl_pw_multi_aff_foreach_piece(pma, &multi_aff_tile, data) < 0)
		goto error;

	isl_pw_multi_aff_free(pma);
	data->res = isl_union_pw_multi_aff_add_pw_multi_aff(data->res,
								data->tiled);

	return isl_stat_ok;
error:
	isl_pw_multi_aff_free(pma);
	isl_pw_multi_aff_free(data->tiled);
	return isl_stat_error;
}

/* Given the schedule of a band, construct the corresponding
 * schedule for the tile loops based on the given tile sizes
 * and return the result.
 */
static isl_union_pw_multi_aff *isl_union_pw_multi_aff_tile(
	__isl_take isl_union_pw_multi_aff *sched,
	__isl_keep isl_multi_val *sizes)
{
	isl_ctx *ctx;
	isl_space *space;
	struct isl_band_tile_data data = { sizes };

	ctx = isl_multi_val_get_ctx(sizes);

	space = isl_union_pw_multi_aff_get_space(sched);
	data.res = isl_union_pw_multi_aff_empty(space);
	data.scale = isl_options_get_tile_scale_tile_loops(ctx);

	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(sched,
						&pw_multi_aff_tile, &data) < 0)
		goto error;

	isl_union_pw_multi_aff_free(sched);
	return data.res;
error:
	isl_union_pw_multi_aff_free(sched);
	isl_union_pw_multi_aff_free(data.res);
	return NULL;
}

/* Extract the range space from "pma" and store it in *user.
 * All entries are expected to have the same range space, so we can
 * stop after extracting the range space from the first entry.
 */
static isl_stat extract_range_space(__isl_take isl_pw_multi_aff *pma,
	void *user)
{
	isl_space **space = user;

	*space = isl_space_range(isl_pw_multi_aff_get_space(pma));
	isl_pw_multi_aff_free(pma);

	return isl_stat_error;
}

/* Extract the range space of "band".  All entries in band->pma should
 * have the same range space.  Furthermore, band->pma should have at least
 * one entry.
 */
static __isl_give isl_space *band_get_range_space(__isl_keep isl_band *band)
{
	isl_space *space;

	if (!band)
		return NULL;

	space = NULL;
	isl_union_pw_multi_aff_foreach_pw_multi_aff(band->pma,
						&extract_range_space, &space);

	return space;
}

/* Construct and return an isl_multi_val in the given space, with as entries
 * the first elements of "v", padded with ones if the size of "v" is smaller
 * than the dimension of "space".
 */
static __isl_give isl_multi_val *multi_val_from_vec(__isl_take isl_space *space,
	__isl_take isl_vec *v)
{
	isl_ctx *ctx;
	isl_multi_val *mv;
	int i, n, size;

	if (!space || !v)
		goto error;

	ctx = isl_space_get_ctx(space);
	mv = isl_multi_val_zero(space);
	n = isl_multi_val_dim(mv, isl_dim_set);
	size = isl_vec_size(v);
	if (n < size)
		size = n;

	for (i = 0; i < size; ++i) {
		isl_val *val = isl_vec_get_element_val(v, i);
		mv = isl_multi_val_set_val(mv, i, val);
	}
	for (i = size; i < n; ++i)
		mv = isl_multi_val_set_val(mv, i, isl_val_one(ctx));

	isl_vec_free(v);
	return mv;
error:
	isl_space_free(space);
	isl_vec_free(v);
	return NULL;
}

/* Tile the given band using the specified tile sizes.
 * The given band is modified to refer to the tile loops and
 * a child band is created to refer to the point loops.
 * The children of this point loop band are the children
 * of the original band.
 *
 * If the scale tile loops option is set, then the tile loops
 * are scaled by the tile sizes.  If the shift point loops option is set,
 * then the point loops are shifted to start at zero.
 * In particular, these options affect the tile and point loop schedules
 * as follows
 *
 *	scale	shift	original	tile		point
 *
 *	0	0	i		floor(i/s)	i
 *	1	0	i		s * floor(i/s)	i
 *	0	1	i		floor(i/s)	i - s * floor(i/s)
 *	1	1	i		s * floor(i/s)	i - s * floor(i/s)
 */
int isl_band_tile(__isl_keep isl_band *band, __isl_take isl_vec *sizes)
{
	isl_ctx *ctx;
	isl_band *child;
	isl_band_list *list = NULL;
	isl_union_pw_multi_aff *sched = NULL, *child_sched = NULL;
	isl_space *space;
	isl_multi_val *mv_sizes;

	if (!band || !sizes)
		goto error;

	ctx = isl_vec_get_ctx(sizes);
	child = isl_band_dup(band);
	list = isl_band_list_alloc(ctx, 1);
	list = isl_band_list_add(list, child);
	if (!list)
		goto error;

	space = band_get_range_space(band);
	mv_sizes = multi_val_from_vec(space, isl_vec_copy(sizes));
	sched = isl_union_pw_multi_aff_copy(band->pma);
	sched = isl_union_pw_multi_aff_tile(sched, mv_sizes);

	child_sched = isl_union_pw_multi_aff_copy(child->pma);
	if (isl_options_get_tile_shift_point_loops(ctx)) {
		isl_union_pw_multi_aff *scaled;
		scaled = isl_union_pw_multi_aff_copy(sched);
		if (!isl_options_get_tile_scale_tile_loops(ctx))
			scaled = isl_union_pw_multi_aff_scale_multi_val(scaled,
						isl_multi_val_copy(mv_sizes));
		child_sched = isl_union_pw_multi_aff_sub(child_sched, scaled);
	}
	isl_multi_val_free(mv_sizes);
	if (!sched || !child_sched)
		goto error;

	child->children = band->children;
	band->children = list;
	child->parent = band;
	isl_union_pw_multi_aff_free(band->pma);
	band->pma = sched;
	isl_union_pw_multi_aff_free(child->pma);
	child->pma = child_sched;

	isl_vec_free(sizes);
	return 0;
error:
	isl_union_pw_multi_aff_free(sched);
	isl_union_pw_multi_aff_free(child_sched);
	isl_band_list_free(list);
	isl_vec_free(sizes);
	return -1;
}

/* Internal data structure used inside isl_union_pw_multi_aff_drop.
 *
 * "pos" is the position of the first dimension to drop.
 * "n" is the number of dimensions to drop.
 * "res" accumulates the result.
 */
struct isl_union_pw_multi_aff_drop_data {
	int pos;
	int n;
	isl_union_pw_multi_aff *res;
};

/* Drop the data->n output dimensions starting at data->pos from "pma"
 * and add the result to data->res.
 */
static isl_stat pw_multi_aff_drop(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_pw_multi_aff_drop_data *data = user;

	pma = isl_pw_multi_aff_drop_dims(pma, isl_dim_out, data->pos, data->n);

	data->res = isl_union_pw_multi_aff_add_pw_multi_aff(data->res, pma);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Drop the "n" output dimensions starting at "pos" from "sched".
 */
static isl_union_pw_multi_aff *isl_union_pw_multi_aff_drop(
	__isl_take isl_union_pw_multi_aff *sched, int pos, int n)
{
	isl_space *space;
	struct isl_union_pw_multi_aff_drop_data data = { pos, n };

	space = isl_union_pw_multi_aff_get_space(sched);
	data.res = isl_union_pw_multi_aff_empty(space);

	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(sched,
						&pw_multi_aff_drop, &data) < 0)
		data.res = isl_union_pw_multi_aff_free(data.res);

	isl_union_pw_multi_aff_free(sched);
	return data.res;
}

/* Drop the "n" dimensions starting at "pos" from "band".
 */
static int isl_band_drop(__isl_keep isl_band *band, int pos, int n)
{
	int i;
	isl_union_pw_multi_aff *sched;

	if (!band)
		return -1;
	if (n == 0)
		return 0;

	sched = isl_union_pw_multi_aff_copy(band->pma);
	sched = isl_union_pw_multi_aff_drop(sched, pos, n);
	if (!sched)
		return -1;

	isl_union_pw_multi_aff_free(band->pma);
	band->pma = sched;

	for (i = pos + n; i < band->n; ++i)
		band->coincident[i - n] = band->coincident[i];

	band->n -= n;

	return 0;
}

/* Split the given band into two nested bands, one with the first "pos"
 * dimensions of "band" and one with the remaining band->n - pos dimensions.
 */
int isl_band_split(__isl_keep isl_band *band, int pos)
{
	isl_ctx *ctx;
	isl_band *child;
	isl_band_list *list;

	if (!band)
		return -1;

	ctx = isl_band_get_ctx(band);

	if (pos < 0 || pos > band->n)
		isl_die(ctx, isl_error_invalid, "position out of bounds",
			return -1);

	child = isl_band_dup(band);
	if (isl_band_drop(child, 0, pos) < 0)
		child = isl_band_free(child);
	list = isl_band_list_alloc(ctx, 1);
	list = isl_band_list_add(list, child);
	if (!list)
		return -1;

	if (isl_band_drop(band, pos, band->n - pos) < 0) {
		isl_band_list_free(list);
		return -1;
	}

	child->children = band->children;
	band->children = list;
	child->parent = band;

	return 0;
}

__isl_give isl_printer *isl_printer_print_band(__isl_take isl_printer *p,
	__isl_keep isl_band *band)
{
	isl_union_map *prefix, *partial, *suffix;

	prefix = isl_band_get_prefix_schedule(band);
	partial = isl_band_get_partial_schedule(band);
	suffix = isl_band_get_suffix_schedule(band);

	p = isl_printer_print_str(p, "(");
	p = isl_printer_print_union_map(p, prefix);
	p = isl_printer_print_str(p, ",");
	p = isl_printer_print_union_map(p, partial);
	p = isl_printer_print_str(p, ",");
	p = isl_printer_print_union_map(p, suffix);
	p = isl_printer_print_str(p, ")");

	isl_union_map_free(prefix);
	isl_union_map_free(partial);
	isl_union_map_free(suffix);

	return p;
}
