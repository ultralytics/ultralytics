/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2013-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#define ISL_DIM_H
#include <isl_map_private.h>
#include <isl_union_map_private.h>
#include <isl/ctx.h>
#include <isl/hash.h>
#include <isl/aff.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl_space_private.h>
#include <isl/union_set.h>
#include <isl/deprecated/union_map_int.h>

#include <bset_from_bmap.c>
#include <set_to_map.c>
#include <set_from_map.c>

/* Return the number of parameters of "umap", where "type"
 * is required to be set to isl_dim_param.
 */
unsigned isl_union_map_dim(__isl_keep isl_union_map *umap,
	enum isl_dim_type type)
{
	if (!umap)
		return 0;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only reference parameters", return 0);

	return isl_space_dim(umap->dim, type);
}

/* Return the number of parameters of "uset", where "type"
 * is required to be set to isl_dim_param.
 */
unsigned isl_union_set_dim(__isl_keep isl_union_set *uset,
	enum isl_dim_type type)
{
	return isl_union_map_dim(uset, type);
}

/* Return the id of the specified dimension.
 */
__isl_give isl_id *isl_union_map_get_dim_id(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, unsigned pos)
{
	if (!umap)
		return NULL;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only reference parameters", return NULL);

	return isl_space_get_dim_id(umap->dim, type, pos);
}

/* Is this union set a parameter domain?
 */
isl_bool isl_union_set_is_params(__isl_keep isl_union_set *uset)
{
	isl_set *set;
	isl_bool params;

	if (!uset)
		return isl_bool_error;
	if (uset->table.n != 1)
		return isl_bool_false;

	set = isl_set_from_union_set(isl_union_set_copy(uset));
	params = isl_set_is_params(set);
	isl_set_free(set);
	return params;
}

static __isl_give isl_union_map *isl_union_map_alloc(
	__isl_take isl_space *space, int size)
{
	isl_union_map *umap;

	space = isl_space_params(space);
	if (!space)
		return NULL;

	umap = isl_calloc_type(space->ctx, isl_union_map);
	if (!umap) {
		isl_space_free(space);
		return NULL;
	}

	umap->ref = 1;
	umap->dim = space;
	if (isl_hash_table_init(space->ctx, &umap->table, size) < 0)
		return isl_union_map_free(umap);

	return umap;
}

__isl_give isl_union_map *isl_union_map_empty(__isl_take isl_space *dim)
{
	return isl_union_map_alloc(dim, 16);
}

__isl_give isl_union_set *isl_union_set_empty(__isl_take isl_space *dim)
{
	return isl_union_map_empty(dim);
}

isl_ctx *isl_union_map_get_ctx(__isl_keep isl_union_map *umap)
{
	return umap ? umap->dim->ctx : NULL;
}

isl_ctx *isl_union_set_get_ctx(__isl_keep isl_union_set *uset)
{
	return uset ? uset->dim->ctx : NULL;
}

__isl_give isl_space *isl_union_map_get_space(__isl_keep isl_union_map *umap)
{
	if (!umap)
		return NULL;
	return isl_space_copy(umap->dim);
}

/* Return the position of the parameter with the given name
 * in "umap".
 * Return -1 if no such dimension can be found.
 */
int isl_union_map_find_dim_by_name(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, const char *name)
{
	if (!umap)
		return -1;
	return isl_space_find_dim_by_name(umap->dim, type, name);
}

__isl_give isl_space *isl_union_set_get_space(__isl_keep isl_union_set *uset)
{
	return isl_union_map_get_space(uset);
}

static isl_stat free_umap_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_map_free(map);
	return isl_stat_ok;
}

static isl_stat add_map(__isl_take isl_map *map, void *user)
{
	isl_union_map **umap = (isl_union_map **)user;

	*umap = isl_union_map_add_map(*umap, map);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_dup(__isl_keep isl_union_map *umap)
{
	isl_union_map *dup;

	if (!umap)
		return NULL;

	dup = isl_union_map_empty(isl_space_copy(umap->dim));
	if (isl_union_map_foreach_map(umap, &add_map, &dup) < 0)
		goto error;
	return dup;
error:
	isl_union_map_free(dup);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_cow(__isl_take isl_union_map *umap)
{
	if (!umap)
		return NULL;

	if (umap->ref == 1)
		return umap;
	umap->ref--;
	return isl_union_map_dup(umap);
}

struct isl_union_align {
	isl_reordering *exp;
	isl_union_map *res;
};

static isl_stat align_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_reordering *exp;
	struct isl_union_align *data = user;

	exp = isl_reordering_extend_space(isl_reordering_copy(data->exp),
				    isl_map_get_space(map));

	data->res = isl_union_map_add_map(data->res,
					isl_map_realign(isl_map_copy(map), exp));

	return isl_stat_ok;
}

/* Align the parameters of umap along those of model.
 * The result has the parameters of model first, in the same order
 * as they appear in model, followed by any remaining parameters of
 * umap that do not appear in model.
 */
__isl_give isl_union_map *isl_union_map_align_params(
	__isl_take isl_union_map *umap, __isl_take isl_space *model)
{
	struct isl_union_align data = { NULL, NULL };

	if (!umap || !model)
		goto error;

	if (isl_space_match(umap->dim, isl_dim_param, model, isl_dim_param)) {
		isl_space_free(model);
		return umap;
	}

	model = isl_space_params(model);
	data.exp = isl_parameter_alignment_reordering(umap->dim, model);
	if (!data.exp)
		goto error;

	data.res = isl_union_map_alloc(isl_space_copy(data.exp->dim),
					umap->table.n);
	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
					&align_entry, &data) < 0)
		goto error;

	isl_reordering_free(data.exp);
	isl_union_map_free(umap);
	isl_space_free(model);
	return data.res;
error:
	isl_reordering_free(data.exp);
	isl_union_map_free(umap);
	isl_union_map_free(data.res);
	isl_space_free(model);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_align_params(
	__isl_take isl_union_set *uset, __isl_take isl_space *model)
{
	return isl_union_map_align_params(uset, model);
}

__isl_give isl_union_map *isl_union_map_union(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2)
{
	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	umap1 = isl_union_map_cow(umap1);

	if (!umap1 || !umap2)
		goto error;

	if (isl_union_map_foreach_map(umap2, &add_map, &umap1) < 0)
		goto error;

	isl_union_map_free(umap2);

	return umap1;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_union(__isl_take isl_union_set *uset1,
	__isl_take isl_union_set *uset2)
{
	return isl_union_map_union(uset1, uset2);
}

__isl_give isl_union_map *isl_union_map_copy(__isl_keep isl_union_map *umap)
{
	if (!umap)
		return NULL;

	umap->ref++;
	return umap;
}

__isl_give isl_union_set *isl_union_set_copy(__isl_keep isl_union_set *uset)
{
	return isl_union_map_copy(uset);
}

__isl_null isl_union_map *isl_union_map_free(__isl_take isl_union_map *umap)
{
	if (!umap)
		return NULL;

	if (--umap->ref > 0)
		return NULL;

	isl_hash_table_foreach(umap->dim->ctx, &umap->table,
			       &free_umap_entry, NULL);
	isl_hash_table_clear(&umap->table);
	isl_space_free(umap->dim);
	free(umap);
	return NULL;
}

__isl_null isl_union_set *isl_union_set_free(__isl_take isl_union_set *uset)
{
	return isl_union_map_free(uset);
}

static int has_dim(const void *entry, const void *val)
{
	isl_map *map = (isl_map *)entry;
	isl_space *dim = (isl_space *)val;

	return isl_space_is_equal(map->dim, dim);
}

__isl_give isl_union_map *isl_union_map_add_map(__isl_take isl_union_map *umap,
	__isl_take isl_map *map)
{
	uint32_t hash;
	struct isl_hash_table_entry *entry;

	if (!map || !umap)
		goto error;

	if (isl_map_plain_is_empty(map)) {
		isl_map_free(map);
		return umap;
	}

	if (!isl_space_match(map->dim, isl_dim_param, umap->dim, isl_dim_param)) {
		umap = isl_union_map_align_params(umap, isl_map_get_space(map));
		map = isl_map_align_params(map, isl_union_map_get_space(umap));
	}

	umap = isl_union_map_cow(umap);

	if (!map || !umap)
		goto error;

	hash = isl_space_get_hash(map->dim);
	entry = isl_hash_table_find(umap->dim->ctx, &umap->table, hash,
				    &has_dim, map->dim, 1);
	if (!entry)
		goto error;

	if (!entry->data)
		entry->data = map;
	else {
		entry->data = isl_map_union(entry->data, isl_map_copy(map));
		if (!entry->data)
			goto error;
		isl_map_free(map);
	}

	return umap;
error:
	isl_map_free(map);
	isl_union_map_free(umap);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_add_set(__isl_take isl_union_set *uset,
	__isl_take isl_set *set)
{
	return isl_union_map_add_map(uset, set_to_map(set));
}

__isl_give isl_union_map *isl_union_map_from_map(__isl_take isl_map *map)
{
	isl_space *dim;
	isl_union_map *umap;

	if (!map)
		return NULL;

	dim = isl_map_get_space(map);
	dim = isl_space_params(dim);
	umap = isl_union_map_empty(dim);
	umap = isl_union_map_add_map(umap, map);

	return umap;
}

__isl_give isl_union_set *isl_union_set_from_set(__isl_take isl_set *set)
{
	return isl_union_map_from_map(set_to_map(set));
}

__isl_give isl_union_map *isl_union_map_from_basic_map(
	__isl_take isl_basic_map *bmap)
{
	return isl_union_map_from_map(isl_map_from_basic_map(bmap));
}

__isl_give isl_union_set *isl_union_set_from_basic_set(
	__isl_take isl_basic_set *bset)
{
	return isl_union_map_from_basic_map(bset);
}

struct isl_union_map_foreach_data
{
	isl_stat (*fn)(__isl_take isl_map *map, void *user);
	void *user;
};

static isl_stat call_on_copy(void **entry, void *user)
{
	isl_map *map = *entry;
	struct isl_union_map_foreach_data *data;
	data = (struct isl_union_map_foreach_data *)user;

	return data->fn(isl_map_copy(map), data->user);
}

int isl_union_map_n_map(__isl_keep isl_union_map *umap)
{
	return umap ? umap->table.n : 0;
}

int isl_union_set_n_set(__isl_keep isl_union_set *uset)
{
	return uset ? uset->table.n : 0;
}

isl_stat isl_union_map_foreach_map(__isl_keep isl_union_map *umap,
	isl_stat (*fn)(__isl_take isl_map *map, void *user), void *user)
{
	struct isl_union_map_foreach_data data = { fn, user };

	if (!umap)
		return isl_stat_error;

	return isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				      &call_on_copy, &data);
}

static isl_stat copy_map(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_map **map_p = user;

	*map_p = isl_map_copy(map);

	return isl_stat_error;
}

__isl_give isl_map *isl_map_from_union_map(__isl_take isl_union_map *umap)
{
	isl_ctx *ctx;
	isl_map *map = NULL;

	if (!umap)
		return NULL;
	ctx = isl_union_map_get_ctx(umap);
	if (umap->table.n != 1)
		isl_die(ctx, isl_error_invalid,
			"union map needs to contain elements in exactly "
			"one space", goto error);

	isl_hash_table_foreach(ctx, &umap->table, &copy_map, &map);

	isl_union_map_free(umap);

	return map;
error:
	isl_union_map_free(umap);
	return NULL;
}

__isl_give isl_set *isl_set_from_union_set(__isl_take isl_union_set *uset)
{
	return isl_map_from_union_map(uset);
}

/* Extract the map in "umap" that lives in the given space (ignoring
 * parameters).
 */
__isl_give isl_map *isl_union_map_extract_map(__isl_keep isl_union_map *umap,
	__isl_take isl_space *space)
{
	uint32_t hash;
	struct isl_hash_table_entry *entry;

	space = isl_space_drop_dims(space, isl_dim_param,
					0, isl_space_dim(space, isl_dim_param));
	space = isl_space_align_params(space, isl_union_map_get_space(umap));
	if (!umap || !space)
		goto error;

	hash = isl_space_get_hash(space);
	entry = isl_hash_table_find(umap->dim->ctx, &umap->table, hash,
				    &has_dim, space, 0);
	if (!entry)
		return isl_map_empty(space);
	isl_space_free(space);
	return isl_map_copy(entry->data);
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_set *isl_union_set_extract_set(__isl_keep isl_union_set *uset,
	__isl_take isl_space *dim)
{
	return set_from_map(isl_union_map_extract_map(uset, dim));
}

/* Check if umap contains a map in the given space.
 */
__isl_give int isl_union_map_contains(__isl_keep isl_union_map *umap,
	__isl_keep isl_space *dim)
{
	uint32_t hash;
	struct isl_hash_table_entry *entry;

	if (!umap || !dim)
		return -1;

	hash = isl_space_get_hash(dim);
	entry = isl_hash_table_find(umap->dim->ctx, &umap->table, hash,
				    &has_dim, dim, 0);
	return !!entry;
}

__isl_give int isl_union_set_contains(__isl_keep isl_union_set *uset,
	__isl_keep isl_space *dim)
{
	return isl_union_map_contains(uset, dim);
}

isl_stat isl_union_set_foreach_set(__isl_keep isl_union_set *uset,
	isl_stat (*fn)(__isl_take isl_set *set, void *user), void *user)
{
	return isl_union_map_foreach_map(uset,
		(isl_stat(*)(__isl_take isl_map *, void*))fn, user);
}

struct isl_union_set_foreach_point_data {
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user);
	void *user;
};

static isl_stat foreach_point(__isl_take isl_set *set, void *user)
{
	struct isl_union_set_foreach_point_data *data = user;
	isl_stat r;

	r = isl_set_foreach_point(set, data->fn, data->user);
	isl_set_free(set);

	return r;
}

isl_stat isl_union_set_foreach_point(__isl_keep isl_union_set *uset,
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user), void *user)
{
	struct isl_union_set_foreach_point_data data = { fn, user };
	return isl_union_set_foreach_set(uset, &foreach_point, &data);
}

struct isl_union_map_gen_bin_data {
	isl_union_map *umap2;
	isl_union_map *res;
};

static isl_stat subtract_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_map *map = *entry;

	hash = isl_space_get_hash(map->dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, map->dim, 0);
	map = isl_map_copy(map);
	if (entry2) {
		int empty;
		map = isl_map_subtract(map, isl_map_copy(entry2->data));

		empty = isl_map_is_empty(map);
		if (empty < 0) {
			isl_map_free(map);
			return isl_stat_error;
		}
		if (empty) {
			isl_map_free(map);
			return isl_stat_ok;
		}
	}
	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

static __isl_give isl_union_map *gen_bin_op(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2, isl_stat (*fn)(void **, void *))
{
	struct isl_union_map_gen_bin_data data = { NULL, NULL };

	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	data.res = isl_union_map_alloc(isl_space_copy(umap1->dim),
				       umap1->table.n);
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   fn, &data) < 0)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return data.res;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_map_free(data.res);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_subtract(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return gen_bin_op(umap1, umap2, &subtract_entry);
}

__isl_give isl_union_set *isl_union_set_subtract(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return isl_union_map_subtract(uset1, uset2);
}

struct isl_union_map_gen_bin_set_data {
	isl_set *set;
	isl_union_map *res;
};

static isl_stat intersect_params_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_set_data *data = user;
	isl_map *map = *entry;
	int empty;

	map = isl_map_copy(map);
	map = isl_map_intersect_params(map, isl_set_copy(data->set));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

static __isl_give isl_union_map *gen_bin_set_op(__isl_take isl_union_map *umap,
	__isl_take isl_set *set, isl_stat (*fn)(void **, void *))
{
	struct isl_union_map_gen_bin_set_data data = { NULL, NULL };

	umap = isl_union_map_align_params(umap, isl_set_get_space(set));
	set = isl_set_align_params(set, isl_union_map_get_space(umap));

	if (!umap || !set)
		goto error;

	data.set = set;
	data.res = isl_union_map_alloc(isl_space_copy(umap->dim),
				       umap->table.n);
	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   fn, &data) < 0)
		goto error;

	isl_union_map_free(umap);
	isl_set_free(set);
	return data.res;
error:
	isl_union_map_free(umap);
	isl_set_free(set);
	isl_union_map_free(data.res);
	return NULL;
}

/* Intersect "umap" with the parameter domain "set".
 *
 * If "set" does not have any constraints, then we can return immediately.
 */
__isl_give isl_union_map *isl_union_map_intersect_params(
	__isl_take isl_union_map *umap, __isl_take isl_set *set)
{
	int is_universe;

	is_universe = isl_set_plain_is_universe(set);
	if (is_universe < 0)
		goto error;
	if (is_universe) {
		isl_set_free(set);
		return umap;
	}

	return gen_bin_set_op(umap, set, &intersect_params_entry);
error:
	isl_union_map_free(umap);
	isl_set_free(set);
	return NULL;
}

__isl_give isl_union_set *isl_union_set_intersect_params(
	__isl_take isl_union_set *uset, __isl_take isl_set *set)
{
	return isl_union_map_intersect_params(uset, set);
}

static __isl_give isl_union_map *union_map_intersect_params(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return isl_union_map_intersect_params(umap,
						isl_set_from_union_set(uset));
}

static __isl_give isl_union_map *union_map_gist_params(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return isl_union_map_gist_params(umap, isl_set_from_union_set(uset));
}

struct isl_union_map_match_bin_data {
	isl_union_map *umap2;
	isl_union_map *res;
	__isl_give isl_map *(*fn)(__isl_take isl_map*, __isl_take isl_map*);
};

static isl_stat match_bin_entry(void **entry, void *user)
{
	struct isl_union_map_match_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_map *map = *entry;
	int empty;

	hash = isl_space_get_hash(map->dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, map->dim, 0);
	if (!entry2)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = data->fn(map, isl_map_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

static __isl_give isl_union_map *match_bin_op(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2,
	__isl_give isl_map *(*fn)(__isl_take isl_map*, __isl_take isl_map*))
{
	struct isl_union_map_match_bin_data data = { NULL, NULL, fn };

	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	data.res = isl_union_map_alloc(isl_space_copy(umap1->dim),
				       umap1->table.n);
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &match_bin_entry, &data) < 0)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return data.res;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_map_free(data.res);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_intersect(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return match_bin_op(umap1, umap2, &isl_map_intersect);
}

/* Compute the intersection of the two union_sets.
 * As a special case, if exactly one of the two union_sets
 * is a parameter domain, then intersect the parameter domain
 * of the other one with this set.
 */
__isl_give isl_union_set *isl_union_set_intersect(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	int p1, p2;

	p1 = isl_union_set_is_params(uset1);
	p2 = isl_union_set_is_params(uset2);
	if (p1 < 0 || p2 < 0)
		goto error;
	if (!p1 && p2)
		return union_map_intersect_params(uset1, uset2);
	if (p1 && !p2)
		return union_map_intersect_params(uset2, uset1);
	return isl_union_map_intersect(uset1, uset2);
error:
	isl_union_set_free(uset1);
	isl_union_set_free(uset2);
	return NULL;
}

static isl_stat gist_params_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_set_data *data = user;
	isl_map *map = *entry;
	int empty;

	map = isl_map_copy(map);
	map = isl_map_gist_params(map, isl_set_copy(data->set));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_gist_params(
	__isl_take isl_union_map *umap, __isl_take isl_set *set)
{
	return gen_bin_set_op(umap, set, &gist_params_entry);
}

__isl_give isl_union_set *isl_union_set_gist_params(
	__isl_take isl_union_set *uset, __isl_take isl_set *set)
{
	return isl_union_map_gist_params(uset, set);
}

__isl_give isl_union_map *isl_union_map_gist(__isl_take isl_union_map *umap,
	__isl_take isl_union_map *context)
{
	return match_bin_op(umap, context, &isl_map_gist);
}

__isl_give isl_union_set *isl_union_set_gist(__isl_take isl_union_set *uset,
	__isl_take isl_union_set *context)
{
	if (isl_union_set_is_params(context))
		return union_map_gist_params(uset, context);
	return isl_union_map_gist(uset, context);
}

static __isl_give isl_map *lex_le_set(__isl_take isl_map *set1,
	__isl_take isl_map *set2)
{
	return isl_set_lex_le_set(set_from_map(set1), set_from_map(set2));
}

static __isl_give isl_map *lex_lt_set(__isl_take isl_map *set1,
	__isl_take isl_map *set2)
{
	return isl_set_lex_lt_set(set_from_map(set1), set_from_map(set2));
}

__isl_give isl_union_map *isl_union_set_lex_lt_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return match_bin_op(uset1, uset2, &lex_lt_set);
}

__isl_give isl_union_map *isl_union_set_lex_le_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return match_bin_op(uset1, uset2, &lex_le_set);
}

__isl_give isl_union_map *isl_union_set_lex_gt_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return isl_union_map_reverse(isl_union_set_lex_lt_union_set(uset2, uset1));
}

__isl_give isl_union_map *isl_union_set_lex_ge_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2)
{
	return isl_union_map_reverse(isl_union_set_lex_le_union_set(uset2, uset1));
}

__isl_give isl_union_map *isl_union_map_lex_gt_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return isl_union_map_reverse(isl_union_map_lex_lt_union_map(umap2, umap1));
}

__isl_give isl_union_map *isl_union_map_lex_ge_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return isl_union_map_reverse(isl_union_map_lex_le_union_map(umap2, umap1));
}

static isl_stat intersect_domain_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_space *dim;
	isl_map *map = *entry;
	isl_bool empty;

	dim = isl_map_get_space(map);
	dim = isl_space_domain(dim);
	hash = isl_space_get_hash(dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, dim, 0);
	isl_space_free(dim);
	if (!entry2)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = isl_map_intersect_domain(map, isl_set_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Intersect the domain of "umap" with "uset".
 * If "uset" is a parameters domain, then intersect the parameter
 * domain of "umap" with this set.
 */
__isl_give isl_union_map *isl_union_map_intersect_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	if (isl_union_set_is_params(uset))
		return union_map_intersect_params(umap, uset);
	return gen_bin_op(umap, uset, &intersect_domain_entry);
}

/* Remove the elements of data->umap2 from the domain of *entry
 * and add the result to data->res.
 */
static isl_stat subtract_domain_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_space *dim;
	isl_map *map = *entry;
	isl_bool empty;

	dim = isl_map_get_space(map);
	dim = isl_space_domain(dim);
	hash = isl_space_get_hash(dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, dim, 0);
	isl_space_free(dim);

	map = isl_map_copy(map);

	if (!entry2) {
		data->res = isl_union_map_add_map(data->res, map);
		return isl_stat_ok;
	}

	map = isl_map_subtract_domain(map, isl_set_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Remove the elements of "uset" from the domain of "umap".
 */
__isl_give isl_union_map *isl_union_map_subtract_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *dom)
{
	return gen_bin_op(umap, dom, &subtract_domain_entry);
}

/* Remove the elements of data->umap2 from the range of *entry
 * and add the result to data->res.
 */
static isl_stat subtract_range_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	isl_map *map = *entry;
	isl_bool empty;

	space = isl_map_get_space(map);
	space = isl_space_range(space);
	hash = isl_space_get_hash(space);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, space, 0);
	isl_space_free(space);

	map = isl_map_copy(map);

	if (!entry2) {
		data->res = isl_union_map_add_map(data->res, map);
		return isl_stat_ok;
	}

	map = isl_map_subtract_range(map, isl_set_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Remove the elements of "uset" from the range of "umap".
 */
__isl_give isl_union_map *isl_union_map_subtract_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *dom)
{
	return gen_bin_op(umap, dom, &subtract_range_entry);
}

static isl_stat gist_domain_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_space *dim;
	isl_map *map = *entry;
	isl_bool empty;

	dim = isl_map_get_space(map);
	dim = isl_space_domain(dim);
	hash = isl_space_get_hash(dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, dim, 0);
	isl_space_free(dim);
	if (!entry2)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = isl_map_gist_domain(map, isl_set_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Compute the gist of "umap" with respect to the domain "uset".
 * If "uset" is a parameters domain, then compute the gist
 * with respect to this parameter domain.
 */
__isl_give isl_union_map *isl_union_map_gist_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	if (isl_union_set_is_params(uset))
		return union_map_gist_params(umap, uset);
	return gen_bin_op(umap, uset, &gist_domain_entry);
}

static isl_stat gist_range_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	isl_map *map = *entry;
	isl_bool empty;

	space = isl_map_get_space(map);
	space = isl_space_range(space);
	hash = isl_space_get_hash(space);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, space, 0);
	isl_space_free(space);
	if (!entry2)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = isl_map_gist_range(map, isl_set_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Compute the gist of "umap" with respect to the range "uset".
 */
__isl_give isl_union_map *isl_union_map_gist_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return gen_bin_op(umap, uset, &gist_range_entry);
}

static isl_stat intersect_range_entry(void **entry, void *user)
{
	struct isl_union_map_gen_bin_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_space *dim;
	isl_map *map = *entry;
	isl_bool empty;

	dim = isl_map_get_space(map);
	dim = isl_space_range(dim);
	hash = isl_space_get_hash(dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, dim, 0);
	isl_space_free(dim);
	if (!entry2)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = isl_map_intersect_range(map, isl_set_copy(entry2->data));

	empty = isl_map_is_empty(map);
	if (empty < 0) {
		isl_map_free(map);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_intersect_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset)
{
	return gen_bin_op(umap, uset, &intersect_range_entry);
}

struct isl_union_map_bin_data {
	isl_union_map *umap2;
	isl_union_map *res;
	isl_map *map;
	isl_stat (*fn)(void **entry, void *user);
};

static isl_stat apply_range_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;
	isl_bool empty;

	if (!isl_space_tuple_is_equal(data->map->dim, isl_dim_out,
				 map2->dim, isl_dim_in))
		return isl_stat_ok;

	map2 = isl_map_apply_range(isl_map_copy(data->map), isl_map_copy(map2));

	empty = isl_map_is_empty(map2);
	if (empty < 0) {
		isl_map_free(map2);
		return isl_stat_error;
	}
	if (empty) {
		isl_map_free(map2);
		return isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

static isl_stat bin_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map = *entry;

	data->map = map;
	if (isl_hash_table_foreach(data->umap2->dim->ctx, &data->umap2->table,
				   data->fn, data) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

static __isl_give isl_union_map *bin_op(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2,
	isl_stat (*fn)(void **entry, void *user))
{
	struct isl_union_map_bin_data data = { NULL, NULL, NULL, fn };

	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	data.res = isl_union_map_alloc(isl_space_copy(umap1->dim),
				       umap1->table.n);
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &bin_entry, &data) < 0)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return data.res;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	isl_union_map_free(data.res);
	return NULL;
}

__isl_give isl_union_map *isl_union_map_apply_range(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &apply_range_entry);
}

__isl_give isl_union_map *isl_union_map_apply_domain(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	umap1 = isl_union_map_reverse(umap1);
	umap1 = isl_union_map_apply_range(umap1, umap2);
	return isl_union_map_reverse(umap1);
}

__isl_give isl_union_set *isl_union_set_apply(
	__isl_take isl_union_set *uset, __isl_take isl_union_map *umap)
{
	return isl_union_map_apply_range(uset, umap);
}

static isl_stat map_lex_lt_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	if (!isl_space_tuple_is_equal(data->map->dim, isl_dim_out,
				 map2->dim, isl_dim_out))
		return isl_stat_ok;

	map2 = isl_map_lex_lt_map(isl_map_copy(data->map), isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_lex_lt_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &map_lex_lt_entry);
}

static isl_stat map_lex_le_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	if (!isl_space_tuple_is_equal(data->map->dim, isl_dim_out,
				 map2->dim, isl_dim_out))
		return isl_stat_ok;

	map2 = isl_map_lex_le_map(isl_map_copy(data->map), isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_lex_le_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &map_lex_le_entry);
}

static isl_stat product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	map2 = isl_map_product(isl_map_copy(data->map), isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_product(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &product_entry);
}

static isl_stat set_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_set *set2 = *entry;

	set2 = isl_set_product(isl_set_copy(data->map), isl_set_copy(set2));

	data->res = isl_union_set_add_set(data->res, set2);

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_set_product(__isl_take isl_union_set *uset1,
	__isl_take isl_union_set *uset2)
{
	return bin_op(uset1, uset2, &set_product_entry);
}

static isl_stat domain_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	if (!isl_space_tuple_is_equal(data->map->dim, isl_dim_out,
				 map2->dim, isl_dim_out))
		return isl_stat_ok;

	map2 = isl_map_domain_product(isl_map_copy(data->map),
				     isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

/* Given two maps A -> B and C -> D, construct a map [A -> C] -> (B * D)
 */
__isl_give isl_union_map *isl_union_map_domain_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &domain_product_entry);
}

static isl_stat range_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	if (!isl_space_tuple_is_equal(data->map->dim, isl_dim_in,
				 map2->dim, isl_dim_in))
		return isl_stat_ok;

	map2 = isl_map_range_product(isl_map_copy(data->map),
				     isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_range_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &range_product_entry);
}

/* If data->map A -> B and "map2" C -> D have the same range space,
 * then add (A, C) -> (B * D) to data->res.
 */
static isl_stat flat_domain_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	if (!isl_space_tuple_is_equal(data->map->dim, isl_dim_out,
				 map2->dim, isl_dim_out))
		return isl_stat_ok;

	map2 = isl_map_flat_domain_product(isl_map_copy(data->map),
					  isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

/* Given two maps A -> B and C -> D, construct a map (A, C) -> (B * D).
 */
__isl_give isl_union_map *isl_union_map_flat_domain_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &flat_domain_product_entry);
}

static isl_stat flat_range_product_entry(void **entry, void *user)
{
	struct isl_union_map_bin_data *data = user;
	isl_map *map2 = *entry;

	if (!isl_space_tuple_is_equal(data->map->dim, isl_dim_in,
				 map2->dim, isl_dim_in))
		return isl_stat_ok;

	map2 = isl_map_flat_range_product(isl_map_copy(data->map),
					  isl_map_copy(map2));

	data->res = isl_union_map_add_map(data->res, map2);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_flat_range_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2)
{
	return bin_op(umap1, umap2, &flat_range_product_entry);
}

static __isl_give isl_union_set *cond_un_op(__isl_take isl_union_map *umap,
	isl_stat (*fn)(void **, void *))
{
	isl_union_set *res;

	if (!umap)
		return NULL;

	res = isl_union_map_alloc(isl_space_copy(umap->dim), umap->table.n);
	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table, fn, &res) < 0)
		goto error;

	isl_union_map_free(umap);
	return res;
error:
	isl_union_map_free(umap);
	isl_union_set_free(res);
	return NULL;
}

static isl_stat from_range_entry(void **entry, void *user)
{
	isl_map *set = *entry;
	isl_union_set **res = user;

	*res = isl_union_map_add_map(*res,
					isl_map_from_range(isl_set_copy(set)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_from_range(
	__isl_take isl_union_set *uset)
{
	return cond_un_op(uset, &from_range_entry);
}

__isl_give isl_union_map *isl_union_map_from_domain(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_reverse(isl_union_map_from_range(uset));
}

__isl_give isl_union_map *isl_union_map_from_domain_and_range(
	__isl_take isl_union_set *domain, __isl_take isl_union_set *range)
{
	return isl_union_map_apply_range(isl_union_map_from_domain(domain),
				         isl_union_map_from_range(range));
}

static __isl_give isl_union_map *un_op(__isl_take isl_union_map *umap,
	isl_stat (*fn)(void **, void *))
{
	umap = isl_union_map_cow(umap);
	if (!umap)
		return NULL;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table, fn, NULL) < 0)
		goto error;

	return umap;
error:
	isl_union_map_free(umap);
	return NULL;
}

static isl_stat affine_entry(void **entry, void *user)
{
	isl_map **map = (isl_map **)entry;

	*map = isl_map_from_basic_map(isl_map_affine_hull(*map));

	return *map ? isl_stat_ok : isl_stat_error;
}

__isl_give isl_union_map *isl_union_map_affine_hull(
	__isl_take isl_union_map *umap)
{
	return un_op(umap, &affine_entry);
}

__isl_give isl_union_set *isl_union_set_affine_hull(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_affine_hull(uset);
}

static isl_stat polyhedral_entry(void **entry, void *user)
{
	isl_map **map = (isl_map **)entry;

	*map = isl_map_from_basic_map(isl_map_polyhedral_hull(*map));

	return *map ? isl_stat_ok : isl_stat_error;
}

__isl_give isl_union_map *isl_union_map_polyhedral_hull(
	__isl_take isl_union_map *umap)
{
	return un_op(umap, &polyhedral_entry);
}

__isl_give isl_union_set *isl_union_set_polyhedral_hull(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_polyhedral_hull(uset);
}

static isl_stat simple_entry(void **entry, void *user)
{
	isl_map **map = (isl_map **)entry;

	*map = isl_map_from_basic_map(isl_map_simple_hull(*map));

	return *map ? isl_stat_ok : isl_stat_error;
}

__isl_give isl_union_map *isl_union_map_simple_hull(
	__isl_take isl_union_map *umap)
{
	return un_op(umap, &simple_entry);
}

__isl_give isl_union_set *isl_union_set_simple_hull(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_simple_hull(uset);
}

static isl_stat inplace_entry(void **entry, void *user)
{
	__isl_give isl_map *(*fn)(__isl_take isl_map *);
	isl_map **map = (isl_map **)entry;
	isl_map *copy;

	fn = *(__isl_give isl_map *(**)(__isl_take isl_map *)) user;
	copy = fn(isl_map_copy(*map));
	if (!copy)
		return isl_stat_error;

	isl_map_free(*map);
	*map = copy;

	return isl_stat_ok;
}

static __isl_give isl_union_map *inplace(__isl_take isl_union_map *umap,
	__isl_give isl_map *(*fn)(__isl_take isl_map *))
{
	if (!umap)
		return NULL;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				    &inplace_entry, &fn) < 0)
		goto error;

	return umap;
error:
	isl_union_map_free(umap);
	return NULL;
}

/* Remove redundant constraints in each of the basic maps of "umap".
 * Since removing redundant constraints does not change the meaning
 * or the space, the operation can be performed in-place.
 */
__isl_give isl_union_map *isl_union_map_remove_redundancies(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_remove_redundancies);
}

/* Remove redundant constraints in each of the basic sets of "uset".
 */
__isl_give isl_union_set *isl_union_set_remove_redundancies(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_remove_redundancies(uset);
}

__isl_give isl_union_map *isl_union_map_coalesce(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_coalesce);
}

__isl_give isl_union_set *isl_union_set_coalesce(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_coalesce(uset);
}

__isl_give isl_union_map *isl_union_map_detect_equalities(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_detect_equalities);
}

__isl_give isl_union_set *isl_union_set_detect_equalities(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_detect_equalities(uset);
}

__isl_give isl_union_map *isl_union_map_compute_divs(
	__isl_take isl_union_map *umap)
{
	return inplace(umap, &isl_map_compute_divs);
}

__isl_give isl_union_set *isl_union_set_compute_divs(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_compute_divs(uset);
}

static isl_stat lexmin_entry(void **entry, void *user)
{
	isl_map **map = (isl_map **)entry;

	*map = isl_map_lexmin(*map);

	return *map ? isl_stat_ok : isl_stat_error;
}

__isl_give isl_union_map *isl_union_map_lexmin(
	__isl_take isl_union_map *umap)
{
	return un_op(umap, &lexmin_entry);
}

__isl_give isl_union_set *isl_union_set_lexmin(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_lexmin(uset);
}

static isl_stat lexmax_entry(void **entry, void *user)
{
	isl_map **map = (isl_map **)entry;

	*map = isl_map_lexmax(*map);

	return *map ? isl_stat_ok : isl_stat_error;
}

__isl_give isl_union_map *isl_union_map_lexmax(
	__isl_take isl_union_map *umap)
{
	return un_op(umap, &lexmax_entry);
}

__isl_give isl_union_set *isl_union_set_lexmax(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_lexmax(uset);
}

static isl_stat universe_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	map = isl_map_universe(isl_map_get_space(map));
	*res = isl_union_map_add_map(*res, map);

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_universe(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &universe_entry);
}

__isl_give isl_union_set *isl_union_set_universe(__isl_take isl_union_set *uset)
{
	return isl_union_map_universe(uset);
}

static isl_stat reverse_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	*res = isl_union_map_add_map(*res, isl_map_reverse(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_reverse(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &reverse_entry);
}

static isl_stat params_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_set **res = user;

	*res = isl_union_set_add_set(*res, isl_map_params(isl_map_copy(map)));

	return isl_stat_ok;
}

/* Compute the parameter domain of the given union map.
 */
__isl_give isl_set *isl_union_map_params(__isl_take isl_union_map *umap)
{
	int empty;

	empty = isl_union_map_is_empty(umap);
	if (empty < 0)
		goto error;
	if (empty) {
		isl_space *space;
		space = isl_union_map_get_space(umap);
		isl_union_map_free(umap);
		return isl_set_empty(space);
	}
	return isl_set_from_union_set(cond_un_op(umap, &params_entry));
error:
	isl_union_map_free(umap);
	return NULL;
}

/* Compute the parameter domain of the given union set.
 */
__isl_give isl_set *isl_union_set_params(__isl_take isl_union_set *uset)
{
	return isl_union_map_params(uset);
}

static isl_stat domain_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_set **res = user;

	*res = isl_union_set_add_set(*res, isl_map_domain(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_map_domain(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &domain_entry);
}

static isl_stat range_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_set **res = user;

	*res = isl_union_set_add_set(*res, isl_map_range(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_map_range(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &range_entry);
}

static isl_stat domain_map_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_set **res = user;

	*res = isl_union_map_add_map(*res,
					isl_map_domain_map(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_domain_map(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &domain_map_entry);
}

/* Construct an isl_pw_multi_aff that maps "map" to its domain and
 * add the result to "res".
 */
static isl_stat domain_map_upma(__isl_take isl_map *map, void *user)
{
	isl_union_pw_multi_aff **res = user;
	isl_multi_aff *ma;
	isl_pw_multi_aff *pma;

	ma = isl_multi_aff_domain_map(isl_map_get_space(map));
	pma = isl_pw_multi_aff_alloc(isl_map_wrap(map), ma);
	*res = isl_union_pw_multi_aff_add_pw_multi_aff(*res, pma);

	return *res ? isl_stat_ok : isl_stat_error;

}

/* Return an isl_union_pw_multi_aff that maps a wrapped copy of "umap"
 * to its domain.
 */
__isl_give isl_union_pw_multi_aff *isl_union_map_domain_map_union_pw_multi_aff(
	__isl_take isl_union_map *umap)
{
	isl_union_pw_multi_aff *res;

	res = isl_union_pw_multi_aff_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &domain_map_upma, &res) < 0)
		res = isl_union_pw_multi_aff_free(res);

	isl_union_map_free(umap);
	return res;
}

static isl_stat range_map_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_set **res = user;

	*res = isl_union_map_add_map(*res,
					isl_map_range_map(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_range_map(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &range_map_entry);
}

/* Check if "set" is of the form A[B -> C].
 * If so, add A[B -> C] -> B to "res".
 */
static isl_stat wrapped_domain_map_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_set **res = user;
	int wrapping;

	wrapping = isl_set_is_wrapping(set);
	if (wrapping < 0)
		return isl_stat_error;
	if (!wrapping)
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
				isl_set_wrapped_domain_map(isl_set_copy(set)));

	return isl_stat_ok;
}

/* Given a collection of wrapped maps of the form A[B -> C],
 * return the collection of maps A[B -> C] -> B.
 */
__isl_give isl_union_map *isl_union_set_wrapped_domain_map(
	__isl_take isl_union_set *uset)
{
	return cond_un_op(uset, &wrapped_domain_map_entry);
}

static isl_stat deltas_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_set **res = user;

	if (!isl_space_tuple_is_equal(map->dim, isl_dim_in,
					map->dim, isl_dim_out))
		return isl_stat_ok;

	*res = isl_union_set_add_set(*res, isl_map_deltas(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_map_deltas(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &deltas_entry);
}

static isl_stat deltas_map_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_space_tuple_is_equal(map->dim, isl_dim_in,
					map->dim, isl_dim_out))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
				     isl_map_deltas_map(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_deltas_map(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &deltas_map_entry);
}

static isl_stat identity_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_map **res = user;

	*res = isl_union_map_add_map(*res, isl_set_identity(isl_set_copy(set)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_set_identity(__isl_take isl_union_set *uset)
{
	return cond_un_op(uset, &identity_entry);
}

/* Construct an identity isl_pw_multi_aff on "set" and add it to *res.
 */
static isl_stat identity_upma(__isl_take isl_set *set, void *user)
{
	isl_union_pw_multi_aff **res = user;
	isl_space *space;
	isl_pw_multi_aff *pma;

	space = isl_space_map_from_set(isl_set_get_space(set));
	pma = isl_pw_multi_aff_identity(space);
	pma = isl_pw_multi_aff_intersect_domain(pma, set);
	*res = isl_union_pw_multi_aff_add_pw_multi_aff(*res, pma);

	return *res ? isl_stat_ok : isl_stat_error;
}

/* Return an identity function on "uset" in the form
 * of an isl_union_pw_multi_aff.
 */
__isl_give isl_union_pw_multi_aff *isl_union_set_identity_union_pw_multi_aff(
	__isl_take isl_union_set *uset)
{
	isl_union_pw_multi_aff *res;

	res = isl_union_pw_multi_aff_empty(isl_union_set_get_space(uset));
	if (isl_union_set_foreach_set(uset, &identity_upma, &res) < 0)
		res = isl_union_pw_multi_aff_free(res);

	isl_union_set_free(uset);
	return res;
}

/* If "map" is of the form [A -> B] -> C, then add A -> C to "res".
 */
static isl_stat domain_factor_domain_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_domain_is_wrapping(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
			    isl_map_domain_factor_domain(isl_map_copy(map)));

	return *res ? isl_stat_ok : isl_stat_error;
}

/* For each map in "umap" of the form [A -> B] -> C,
 * construct the map A -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_domain_factor_domain(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &domain_factor_domain_entry);
}

/* If "map" is of the form [A -> B] -> C, then add B -> C to "res".
 */
static isl_stat domain_factor_range_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_domain_is_wrapping(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
				isl_map_domain_factor_range(isl_map_copy(map)));

	return *res ? isl_stat_ok : isl_stat_error;
}

/* For each map in "umap" of the form [A -> B] -> C,
 * construct the map B -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_domain_factor_range(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &domain_factor_range_entry);
}

/* If "map" is of the form A -> [B -> C], then add A -> B to "res".
 */
static isl_stat range_factor_domain_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_range_is_wrapping(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
				isl_map_range_factor_domain(isl_map_copy(map)));

	return *res ? isl_stat_ok : isl_stat_error;
}

/* For each map in "umap" of the form A -> [B -> C],
 * construct the map A -> B and collect the results.
 */
__isl_give isl_union_map *isl_union_map_range_factor_domain(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &range_factor_domain_entry);
}

/* If "map" is of the form A -> [B -> C], then add A -> C to "res".
 */
static isl_stat range_factor_range_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_range_is_wrapping(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
				isl_map_range_factor_range(isl_map_copy(map)));

	return *res ? isl_stat_ok : isl_stat_error;
}

/* For each map in "umap" of the form A -> [B -> C],
 * construct the map A -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_range_factor_range(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &range_factor_range_entry);
}

/* If "map" is of the form [A -> B] -> [C -> D], then add A -> C to "res".
 */
static isl_stat factor_domain_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_domain_is_wrapping(map) || !isl_map_range_is_wrapping(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
				isl_map_factor_domain(isl_map_copy(map)));

	return *res ? isl_stat_ok : isl_stat_error;
}

/* For each map in "umap" of the form [A -> B] -> [C -> D],
 * construct the map A -> C and collect the results.
 */
__isl_give isl_union_map *isl_union_map_factor_domain(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &factor_domain_entry);
}

/* If "map" is of the form [A -> B] -> [C -> D], then add B -> D to "res".
 */
static isl_stat factor_range_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_domain_is_wrapping(map) || !isl_map_range_is_wrapping(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res,
				isl_map_factor_range(isl_map_copy(map)));

	return *res ? isl_stat_ok : isl_stat_error;
}

/* For each map in "umap" of the form [A -> B] -> [C -> D],
 * construct the map B -> D and collect the results.
 */
__isl_give isl_union_map *isl_union_map_factor_range(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &factor_range_entry);
}

static isl_stat unwrap_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_set **res = user;

	if (!isl_set_is_wrapping(set))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res, isl_set_unwrap(isl_set_copy(set)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_set_unwrap(__isl_take isl_union_set *uset)
{
	return cond_un_op(uset, &unwrap_entry);
}

static isl_stat wrap_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_set **res = user;

	*res = isl_union_set_add_set(*res, isl_map_wrap(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_map_wrap(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &wrap_entry);
}

struct isl_union_map_is_subset_data {
	isl_union_map *umap2;
	isl_bool is_subset;
};

static isl_stat is_subset_entry(void **entry, void *user)
{
	struct isl_union_map_is_subset_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_map *map = *entry;

	hash = isl_space_get_hash(map->dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, map->dim, 0);
	if (!entry2) {
		int empty = isl_map_is_empty(map);
		if (empty < 0)
			return isl_stat_error;
		if (empty)
			return isl_stat_ok;
		data->is_subset = 0;
		return isl_stat_error;
	}

	data->is_subset = isl_map_is_subset(map, entry2->data);
	if (data->is_subset < 0 || !data->is_subset)
		return isl_stat_error;

	return isl_stat_ok;
}

isl_bool isl_union_map_is_subset(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	struct isl_union_map_is_subset_data data = { NULL, isl_bool_true };

	umap1 = isl_union_map_copy(umap1);
	umap2 = isl_union_map_copy(umap2);
	umap1 = isl_union_map_align_params(umap1, isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2, isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &is_subset_entry, &data) < 0 &&
	    data.is_subset)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);

	return data.is_subset;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return isl_bool_error;
}

isl_bool isl_union_set_is_subset(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_subset(uset1, uset2);
}

isl_bool isl_union_map_is_equal(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	isl_bool is_subset;

	if (!umap1 || !umap2)
		return isl_bool_error;
	is_subset = isl_union_map_is_subset(umap1, umap2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_union_map_is_subset(umap2, umap1);
	return is_subset;
}

isl_bool isl_union_set_is_equal(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_equal(uset1, uset2);
}

isl_bool isl_union_map_is_strict_subset(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	isl_bool is_subset;

	if (!umap1 || !umap2)
		return isl_bool_error;
	is_subset = isl_union_map_is_subset(umap1, umap2);
	if (is_subset != isl_bool_true)
		return is_subset;
	is_subset = isl_union_map_is_subset(umap2, umap1);
	if (is_subset == isl_bool_error)
		return is_subset;
	return !is_subset;
}

isl_bool isl_union_set_is_strict_subset(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_strict_subset(uset1, uset2);
}

/* Internal data structure for isl_union_map_is_disjoint.
 * umap2 is the union map with which we are comparing.
 * is_disjoint is initialized to 1 and is set to 0 as soon
 * as the union maps turn out not to be disjoint.
 */
struct isl_union_map_is_disjoint_data {
	isl_union_map *umap2;
	isl_bool is_disjoint;
};

/* Check if "map" is disjoint from data->umap2 and abort
 * the search if it is not.
 */
static isl_stat is_disjoint_entry(void **entry, void *user)
{
	struct isl_union_map_is_disjoint_data *data = user;
	uint32_t hash;
	struct isl_hash_table_entry *entry2;
	isl_map *map = *entry;

	hash = isl_space_get_hash(map->dim);
	entry2 = isl_hash_table_find(data->umap2->dim->ctx, &data->umap2->table,
				     hash, &has_dim, map->dim, 0);
	if (!entry2)
		return isl_stat_ok;

	data->is_disjoint = isl_map_is_disjoint(map, entry2->data);
	if (data->is_disjoint < 0 || !data->is_disjoint)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Are "umap1" and "umap2" disjoint?
 */
isl_bool isl_union_map_is_disjoint(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2)
{
	struct isl_union_map_is_disjoint_data data = { NULL, isl_bool_true };

	umap1 = isl_union_map_copy(umap1);
	umap2 = isl_union_map_copy(umap2);
	umap1 = isl_union_map_align_params(umap1,
						isl_union_map_get_space(umap2));
	umap2 = isl_union_map_align_params(umap2,
						isl_union_map_get_space(umap1));

	if (!umap1 || !umap2)
		goto error;

	data.umap2 = umap2;
	if (isl_hash_table_foreach(umap1->dim->ctx, &umap1->table,
				   &is_disjoint_entry, &data) < 0 &&
	    data.is_disjoint)
		goto error;

	isl_union_map_free(umap1);
	isl_union_map_free(umap2);

	return data.is_disjoint;
error:
	isl_union_map_free(umap1);
	isl_union_map_free(umap2);
	return isl_bool_error;
}

/* Are "uset1" and "uset2" disjoint?
 */
isl_bool isl_union_set_is_disjoint(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2)
{
	return isl_union_map_is_disjoint(uset1, uset2);
}

static isl_stat sample_entry(void **entry, void *user)
{
	isl_basic_map **sample = (isl_basic_map **)user;
	isl_map *map = *entry;

	*sample = isl_map_sample(isl_map_copy(map));
	if (!*sample)
		return isl_stat_error;
	if (!isl_basic_map_plain_is_empty(*sample))
		return isl_stat_error;
	return isl_stat_ok;
}

__isl_give isl_basic_map *isl_union_map_sample(__isl_take isl_union_map *umap)
{
	isl_basic_map *sample = NULL;

	if (!umap)
		return NULL;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   &sample_entry, &sample) < 0 &&
	    !sample)
		goto error;

	if (!sample)
		sample = isl_basic_map_empty(isl_union_map_get_space(umap));

	isl_union_map_free(umap);

	return sample;
error:
	isl_union_map_free(umap);
	return NULL;
}

__isl_give isl_basic_set *isl_union_set_sample(__isl_take isl_union_set *uset)
{
	return bset_from_bmap(isl_union_map_sample(uset));
}

/* Return an element in "uset" in the form of an isl_point.
 * Return a void isl_point if "uset" is empty.
 */
__isl_give isl_point *isl_union_set_sample_point(__isl_take isl_union_set *uset)
{
	return isl_basic_set_sample_point(isl_union_set_sample(uset));
}

struct isl_forall_data {
	isl_bool res;
	isl_bool (*fn)(__isl_keep isl_map *map);
};

static isl_stat forall_entry(void **entry, void *user)
{
	struct isl_forall_data *data = user;
	isl_map *map = *entry;

	data->res = data->fn(map);
	if (data->res < 0)
		return isl_stat_error;

	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

static isl_bool union_map_forall(__isl_keep isl_union_map *umap,
	isl_bool (*fn)(__isl_keep isl_map *map))
{
	struct isl_forall_data data = { isl_bool_true, fn };

	if (!umap)
		return isl_bool_error;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   &forall_entry, &data) < 0 && data.res)
		return isl_bool_error;

	return data.res;
}

struct isl_forall_user_data {
	isl_bool res;
	isl_bool (*fn)(__isl_keep isl_map *map, void *user);
	void *user;
};

static isl_stat forall_user_entry(void **entry, void *user)
{
	struct isl_forall_user_data *data = user;
	isl_map *map = *entry;

	data->res = data->fn(map, data->user);
	if (data->res < 0)
		return isl_stat_error;

	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Check if fn(map, user) returns true for all maps "map" in umap.
 */
static isl_bool union_map_forall_user(__isl_keep isl_union_map *umap,
	isl_bool (*fn)(__isl_keep isl_map *map, void *user), void *user)
{
	struct isl_forall_user_data data = { isl_bool_true, fn, user };

	if (!umap)
		return isl_bool_error;

	if (isl_hash_table_foreach(umap->dim->ctx, &umap->table,
				   &forall_user_entry, &data) < 0 && data.res)
		return isl_bool_error;

	return data.res;
}

isl_bool isl_union_map_is_empty(__isl_keep isl_union_map *umap)
{
	return union_map_forall(umap, &isl_map_is_empty);
}

isl_bool isl_union_set_is_empty(__isl_keep isl_union_set *uset)
{
	return isl_union_map_is_empty(uset);
}

static isl_bool is_subset_of_identity(__isl_keep isl_map *map)
{
	isl_bool is_subset;
	isl_space *dim;
	isl_map *id;

	if (!map)
		return isl_bool_error;

	if (!isl_space_tuple_is_equal(map->dim, isl_dim_in,
					map->dim, isl_dim_out))
		return isl_bool_false;

	dim = isl_map_get_space(map);
	id = isl_map_identity(dim);

	is_subset = isl_map_is_subset(map, id);

	isl_map_free(id);

	return is_subset;
}

/* Given an isl_union_map that consists of a single map, check
 * if it is single-valued.
 */
static isl_bool single_map_is_single_valued(__isl_keep isl_union_map *umap)
{
	isl_map *map;
	isl_bool sv;

	umap = isl_union_map_copy(umap);
	map = isl_map_from_union_map(umap);
	sv = isl_map_is_single_valued(map);
	isl_map_free(map);

	return sv;
}

/* Internal data structure for single_valued_on_domain.
 *
 * "umap" is the union map to be tested.
 * "sv" is set to 1 as long as "umap" may still be single-valued.
 */
struct isl_union_map_is_sv_data {
	isl_union_map *umap;
	isl_bool sv;
};

/* Check if the data->umap is single-valued on "set".
 *
 * If data->umap consists of a single map on "set", then test it
 * as an isl_map.
 *
 * Otherwise, compute
 *
 *	M \circ M^-1
 *
 * check if the result is a subset of the identity mapping and
 * store the result in data->sv.
 *
 * Terminate as soon as data->umap has been determined not to
 * be single-valued.
 */
static isl_stat single_valued_on_domain(__isl_take isl_set *set, void *user)
{
	struct isl_union_map_is_sv_data *data = user;
	isl_union_map *umap, *test;

	umap = isl_union_map_copy(data->umap);
	umap = isl_union_map_intersect_domain(umap,
						isl_union_set_from_set(set));

	if (isl_union_map_n_map(umap) == 1) {
		data->sv = single_map_is_single_valued(umap);
		isl_union_map_free(umap);
	} else {
		test = isl_union_map_reverse(isl_union_map_copy(umap));
		test = isl_union_map_apply_range(test, umap);

		data->sv = union_map_forall(test, &is_subset_of_identity);

		isl_union_map_free(test);
	}

	if (data->sv < 0 || !data->sv)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Check if the given map is single-valued.
 *
 * If the union map consists of a single map, then test it as an isl_map.
 * Otherwise, check if the union map is single-valued on each of its
 * domain spaces.
 */
isl_bool isl_union_map_is_single_valued(__isl_keep isl_union_map *umap)
{
	isl_union_map *universe;
	isl_union_set *domain;
	struct isl_union_map_is_sv_data data;

	if (isl_union_map_n_map(umap) == 1)
		return single_map_is_single_valued(umap);

	universe = isl_union_map_universe(isl_union_map_copy(umap));
	domain = isl_union_map_domain(universe);

	data.sv = isl_bool_true;
	data.umap = umap;
	if (isl_union_set_foreach_set(domain,
			    &single_valued_on_domain, &data) < 0 && data.sv)
		data.sv = isl_bool_error;
	isl_union_set_free(domain);

	return data.sv;
}

isl_bool isl_union_map_is_injective(__isl_keep isl_union_map *umap)
{
	isl_bool in;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_reverse(umap);
	in = isl_union_map_is_single_valued(umap);
	isl_union_map_free(umap);

	return in;
}

/* Is "map" obviously not an identity relation because
 * it maps elements from one space to another space?
 * Update *non_identity accordingly.
 *
 * In particular, if the domain and range spaces are the same,
 * then the map is not considered to obviously not be an identity relation.
 * Otherwise, the map is considered to obviously not be an identity relation
 * if it is is non-empty.
 *
 * If "map" is determined to obviously not be an identity relation,
 * then the search is aborted.
 */
static isl_stat map_plain_is_not_identity(__isl_take isl_map *map, void *user)
{
	isl_bool *non_identity = user;
	isl_bool equal;
	isl_space *space;

	space = isl_map_get_space(map);
	equal = isl_space_tuple_is_equal(space, isl_dim_in, space, isl_dim_out);
	if (equal >= 0 && !equal)
		*non_identity = isl_bool_not(isl_map_is_empty(map));
	else
		*non_identity = isl_bool_not(equal);
	isl_space_free(space);
	isl_map_free(map);

	if (*non_identity < 0 || *non_identity)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Is "umap" obviously not an identity relation because
 * it maps elements from one space to another space?
 *
 * As soon as a map has been found that maps elements to a different space,
 * non_identity is changed and the search is aborted.
 */
static isl_bool isl_union_map_plain_is_not_identity(
	__isl_keep isl_union_map *umap)
{
	isl_bool non_identity;

	non_identity = isl_bool_false;
	if (isl_union_map_foreach_map(umap, &map_plain_is_not_identity,
					&non_identity) < 0 &&
	    non_identity == isl_bool_false)
		return isl_bool_error;

	return non_identity;
}

/* Does "map" only map elements to themselves?
 * Update *identity accordingly.
 *
 * If "map" is determined not to be an identity relation,
 * then the search is aborted.
 */
static isl_stat map_is_identity(__isl_take isl_map *map, void *user)
{
	isl_bool *identity = user;

	*identity = isl_map_is_identity(map);
	isl_map_free(map);

	if (*identity < 0 || !*identity)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Does "umap" only map elements to themselves?
 *
 * First check if there are any maps that map elements to different spaces.
 * If not, then check that all the maps (between identical spaces)
 * are identity relations.
 */
isl_bool isl_union_map_is_identity(__isl_keep isl_union_map *umap)
{
	isl_bool non_identity;
	isl_bool identity;

	non_identity = isl_union_map_plain_is_not_identity(umap);
	if (non_identity < 0 || non_identity)
		return isl_bool_not(non_identity);

	identity = isl_bool_true;
	if (isl_union_map_foreach_map(umap, &map_is_identity, &identity) < 0 &&
	    identity == isl_bool_true)
		return isl_bool_error;

	return identity;
}

/* Represents a map that has a fixed value (v) for one of its
 * range dimensions.
 * The map in this structure is not reference counted, so it
 * is only valid while the isl_union_map from which it was
 * obtained is still alive.
 */
struct isl_fixed_map {
	isl_int v;
	isl_map *map;
};

static struct isl_fixed_map *alloc_isl_fixed_map_array(isl_ctx *ctx,
	int n)
{
	int i;
	struct isl_fixed_map *v;

	v = isl_calloc_array(ctx, struct isl_fixed_map, n);
	if (!v)
		return NULL;
	for (i = 0; i < n; ++i)
		isl_int_init(v[i].v);
	return v;
}

static void free_isl_fixed_map_array(struct isl_fixed_map *v, int n)
{
	int i;

	if (!v)
		return;
	for (i = 0; i < n; ++i)
		isl_int_clear(v[i].v);
	free(v);
}

/* Compare the "v" field of two isl_fixed_map structs.
 */
static int qsort_fixed_map_cmp(const void *p1, const void *p2)
{
	const struct isl_fixed_map *e1 = (const struct isl_fixed_map *) p1;
	const struct isl_fixed_map *e2 = (const struct isl_fixed_map *) p2;

	return isl_int_cmp(e1->v, e2->v);
}

/* Internal data structure used while checking whether all maps
 * in a union_map have a fixed value for a given output dimension.
 * v is the list of maps, with the fixed value for the dimension
 * n is the number of maps considered so far
 * pos is the output dimension under investigation
 */
struct isl_fixed_dim_data {
	struct isl_fixed_map *v;
	int n;
	int pos;
};

static isl_bool fixed_at_pos(__isl_keep isl_map *map, void *user)
{
	struct isl_fixed_dim_data *data = user;

	data->v[data->n].map = map;
	return isl_map_plain_is_fixed(map, isl_dim_out, data->pos,
				      &data->v[data->n++].v);
}

static isl_bool plain_injective_on_range(__isl_take isl_union_map *umap,
	int first, int n_range);

/* Given a list of the maps, with their fixed values at output dimension "pos",
 * check whether the ranges of the maps form an obvious partition.
 *
 * We first sort the maps according to their fixed values.
 * If all maps have a different value, then we know the ranges form
 * a partition.
 * Otherwise, we collect the maps with the same fixed value and
 * check whether each such collection is obviously injective
 * based on later dimensions.
 */
static int separates(struct isl_fixed_map *v, int n,
	__isl_take isl_space *dim, int pos, int n_range)
{
	int i;

	if (!v)
		goto error;

	qsort(v, n, sizeof(*v), &qsort_fixed_map_cmp);

	for (i = 0; i + 1 < n; ++i) {
		int j, k;
		isl_union_map *part;
		int injective;

		for (j = i + 1; j < n; ++j)
			if (isl_int_ne(v[i].v, v[j].v))
				break;

		if (j == i + 1)
			continue;

		part = isl_union_map_alloc(isl_space_copy(dim), j - i);
		for (k = i; k < j; ++k)
			part = isl_union_map_add_map(part,
						     isl_map_copy(v[k].map));

		injective = plain_injective_on_range(part, pos + 1, n_range);
		if (injective < 0)
			goto error;
		if (!injective)
			break;

		i = j - 1;
	}

	isl_space_free(dim);
	free_isl_fixed_map_array(v, n);
	return i + 1 >= n;
error:
	isl_space_free(dim);
	free_isl_fixed_map_array(v, n);
	return -1;
}

/* Check whether the maps in umap have obviously distinct ranges.
 * In particular, check for an output dimension in the range
 * [first,n_range) for which all maps have a fixed value
 * and then check if these values, possibly along with fixed values
 * at later dimensions, entail distinct ranges.
 */
static isl_bool plain_injective_on_range(__isl_take isl_union_map *umap,
	int first, int n_range)
{
	isl_ctx *ctx;
	int n;
	struct isl_fixed_dim_data data = { NULL };

	ctx = isl_union_map_get_ctx(umap);

	n = isl_union_map_n_map(umap);
	if (!umap)
		goto error;

	if (n <= 1) {
		isl_union_map_free(umap);
		return isl_bool_true;
	}

	if (first >= n_range) {
		isl_union_map_free(umap);
		return isl_bool_false;
	}

	data.v = alloc_isl_fixed_map_array(ctx, n);
	if (!data.v)
		goto error;

	for (data.pos = first; data.pos < n_range; ++data.pos) {
		isl_bool fixed;
		int injective;
		isl_space *dim;

		data.n = 0;
		fixed = union_map_forall_user(umap, &fixed_at_pos, &data);
		if (fixed < 0)
			goto error;
		if (!fixed)
			continue;
		dim = isl_union_map_get_space(umap);
		injective = separates(data.v, n, dim, data.pos, n_range);
		isl_union_map_free(umap);
		return injective;
	}

	free_isl_fixed_map_array(data.v, n);
	isl_union_map_free(umap);

	return isl_bool_false;
error:
	free_isl_fixed_map_array(data.v, n);
	isl_union_map_free(umap);
	return isl_bool_error;
}

/* Check whether the maps in umap that map to subsets of "ran"
 * have obviously distinct ranges.
 */
static isl_bool plain_injective_on_range_wrap(__isl_keep isl_set *ran,
	void *user)
{
	isl_union_map *umap = user;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_intersect_range(umap,
			isl_union_set_from_set(isl_set_copy(ran)));
	return plain_injective_on_range(umap, 0, isl_set_dim(ran, isl_dim_set));
}

/* Check if the given union_map is obviously injective.
 *
 * In particular, we first check if all individual maps are obviously
 * injective and then check if all the ranges of these maps are
 * obviously disjoint.
 */
isl_bool isl_union_map_plain_is_injective(__isl_keep isl_union_map *umap)
{
	isl_bool in;
	isl_union_map *univ;
	isl_union_set *ran;

	in = union_map_forall(umap, &isl_map_plain_is_injective);
	if (in < 0)
		return isl_bool_error;
	if (!in)
		return isl_bool_false;

	univ = isl_union_map_universe(isl_union_map_copy(umap));
	ran = isl_union_map_range(univ);

	in = union_map_forall_user(ran, &plain_injective_on_range_wrap, umap);

	isl_union_set_free(ran);

	return in;
}

isl_bool isl_union_map_is_bijective(__isl_keep isl_union_map *umap)
{
	isl_bool sv;

	sv = isl_union_map_is_single_valued(umap);
	if (sv < 0 || !sv)
		return sv;

	return isl_union_map_is_injective(umap);
}

static isl_stat zip_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_can_zip(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res, isl_map_zip(isl_map_copy(map)));

	return isl_stat_ok;
}

__isl_give isl_union_map *isl_union_map_zip(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &zip_entry);
}

static isl_stat uncurry_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_can_uncurry(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res, isl_map_uncurry(isl_map_copy(map)));

	return isl_stat_ok;
}

/* Given a union map, take the maps of the form A -> (B -> C) and
 * return the union of the corresponding maps (A -> B) -> C.
 */
__isl_give isl_union_map *isl_union_map_uncurry(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &uncurry_entry);
}

static isl_stat curry_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_can_curry(map))
		return isl_stat_ok;

	*res = isl_union_map_add_map(*res, isl_map_curry(isl_map_copy(map)));

	return isl_stat_ok;
}

/* Given a union map, take the maps of the form (A -> B) -> C and
 * return the union of the corresponding maps A -> (B -> C).
 */
__isl_give isl_union_map *isl_union_map_curry(__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &curry_entry);
}

/* If *entry is of the form A -> ((B -> C) -> D), then apply
 * isl_map_range_curry to it and add the result to *res.
 */
static isl_stat range_curry_entry(void **entry, void *user)
{
	isl_map *map = *entry;
	isl_union_map **res = user;

	if (!isl_map_can_range_curry(map))
		return isl_stat_ok;

	map = isl_map_range_curry(isl_map_copy(map));
	*res = isl_union_map_add_map(*res, map);

	return isl_stat_ok;
}

/* Given a union map, take the maps of the form A -> ((B -> C) -> D) and
 * return the union of the corresponding maps A -> (B -> (C -> D)).
 */
__isl_give isl_union_map *isl_union_map_range_curry(
	__isl_take isl_union_map *umap)
{
	return cond_un_op(umap, &range_curry_entry);
}

static isl_stat lift_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_set **res = user;

	*res = isl_union_set_add_set(*res, isl_set_lift(isl_set_copy(set)));

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_set_lift(__isl_take isl_union_set *uset)
{
	return cond_un_op(uset, &lift_entry);
}

static isl_stat coefficients_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_set **res = user;

	set = isl_set_copy(set);
	set = isl_set_from_basic_set(isl_set_coefficients(set));
	*res = isl_union_set_add_set(*res, set);

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_set_coefficients(
	__isl_take isl_union_set *uset)
{
	isl_ctx *ctx;
	isl_space *dim;
	isl_union_set *res;

	if (!uset)
		return NULL;

	ctx = isl_union_set_get_ctx(uset);
	dim = isl_space_set_alloc(ctx, 0, 0);
	res = isl_union_map_alloc(dim, uset->table.n);
	if (isl_hash_table_foreach(uset->dim->ctx, &uset->table,
				   &coefficients_entry, &res) < 0)
		goto error;

	isl_union_set_free(uset);
	return res;
error:
	isl_union_set_free(uset);
	isl_union_set_free(res);
	return NULL;
}

static isl_stat solutions_entry(void **entry, void *user)
{
	isl_set *set = *entry;
	isl_union_set **res = user;

	set = isl_set_copy(set);
	set = isl_set_from_basic_set(isl_set_solutions(set));
	if (!*res)
		*res = isl_union_set_from_set(set);
	else
		*res = isl_union_set_add_set(*res, set);

	if (!*res)
		return isl_stat_error;

	return isl_stat_ok;
}

__isl_give isl_union_set *isl_union_set_solutions(
	__isl_take isl_union_set *uset)
{
	isl_union_set *res = NULL;

	if (!uset)
		return NULL;

	if (uset->table.n == 0) {
		res = isl_union_set_empty(isl_union_set_get_space(uset));
		isl_union_set_free(uset);
		return res;
	}

	if (isl_hash_table_foreach(uset->dim->ctx, &uset->table,
				   &solutions_entry, &res) < 0)
		goto error;

	isl_union_set_free(uset);
	return res;
error:
	isl_union_set_free(uset);
	isl_union_set_free(res);
	return NULL;
}

/* Is the domain space of "map" equal to "space"?
 */
static int domain_match(__isl_keep isl_map *map, __isl_keep isl_space *space)
{
	return isl_space_tuple_is_equal(map->dim, isl_dim_in,
					space, isl_dim_out);
}

/* Is the range space of "map" equal to "space"?
 */
static int range_match(__isl_keep isl_map *map, __isl_keep isl_space *space)
{
	return isl_space_tuple_is_equal(map->dim, isl_dim_out,
					space, isl_dim_out);
}

/* Is the set space of "map" equal to "space"?
 */
static int set_match(__isl_keep isl_map *map, __isl_keep isl_space *space)
{
	return isl_space_tuple_is_equal(map->dim, isl_dim_set,
					space, isl_dim_out);
}

/* Internal data structure for preimage_pw_multi_aff.
 *
 * "pma" is the function under which the preimage should be taken.
 * "space" is the space of "pma".
 * "res" collects the results.
 * "fn" computes the preimage for a given map.
 * "match" returns true if "fn" can be called.
 */
struct isl_union_map_preimage_data {
	isl_space *space;
	isl_pw_multi_aff *pma;
	isl_union_map *res;
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space);
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_pw_multi_aff *pma);
};

/* Call data->fn to compute the preimage of the domain or range of *entry
 * under the function represented by data->pma, provided the domain/range
 * space of *entry matches the target space of data->pma
 * (as given by data->match), and add the result to data->res.
 */
static isl_stat preimage_entry(void **entry, void *user)
{
	int m;
	isl_map *map = *entry;
	struct isl_union_map_preimage_data *data = user;
	isl_bool empty;

	m = data->match(map, data->space);
	if (m < 0)
		return isl_stat_error;
	if (!m)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = data->fn(map, isl_pw_multi_aff_copy(data->pma));

	empty = isl_map_is_empty(map);
	if (empty < 0 || empty) {
		isl_map_free(map);
		return empty < 0 ? isl_stat_error : isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Compute the preimage of the domain or range of "umap" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the domain or range of "umap".
 * The function "fn" performs the actual preimage computation on a map,
 * while "match" determines to which maps the function should be applied.
 */
static __isl_give isl_union_map *preimage_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma,
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space),
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_pw_multi_aff *pma))
{
	isl_ctx *ctx;
	isl_space *space;
	struct isl_union_map_preimage_data data;

	umap = isl_union_map_align_params(umap,
					    isl_pw_multi_aff_get_space(pma));
	pma = isl_pw_multi_aff_align_params(pma, isl_union_map_get_space(umap));

	if (!umap || !pma)
		goto error;

	ctx = isl_union_map_get_ctx(umap);
	space = isl_union_map_get_space(umap);
	data.space = isl_pw_multi_aff_get_space(pma);
	data.pma = pma;
	data.res = isl_union_map_alloc(space, umap->table.n);
	data.match = match;
	data.fn = fn;
	if (isl_hash_table_foreach(ctx, &umap->table, &preimage_entry,
					&data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_space_free(data.space);
	isl_union_map_free(umap);
	isl_pw_multi_aff_free(pma);
	return data.res;
error:
	isl_union_map_free(umap);
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to the target space of "pma",
 * except that the domain has been replaced by the domain space of "pma".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma)
{
	return preimage_pw_multi_aff(umap, pma, &domain_match,
					&isl_map_preimage_domain_pw_multi_aff);
}

/* Compute the preimage of the range of "umap" under the function
 * represented by "pma".
 * In other words, plug in "pma" in the range of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with range space equal to the target space of "pma",
 * except that the range has been replaced by the domain space of "pma".
 */
__isl_give isl_union_map *isl_union_map_preimage_range_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma)
{
	return preimage_pw_multi_aff(umap, pma, &range_match,
					&isl_map_preimage_range_pw_multi_aff);
}

/* Compute the preimage of "uset" under the function represented by "pma".
 * In other words, plug in "pma" in "uset".
 * The result contains sets that live in the same spaces as the sets of "uset"
 * with space equal to the target space of "pma",
 * except that the space has been replaced by the domain space of "pma".
 */
__isl_give isl_union_set *isl_union_set_preimage_pw_multi_aff(
	__isl_take isl_union_set *uset, __isl_take isl_pw_multi_aff *pma)
{
	return preimage_pw_multi_aff(uset, pma, &set_match,
					&isl_set_preimage_pw_multi_aff);
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to the target space of "ma",
 * except that the domain has been replaced by the domain space of "ma".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_aff *ma)
{
	return isl_union_map_preimage_domain_pw_multi_aff(umap,
					isl_pw_multi_aff_from_multi_aff(ma));
}

/* Compute the preimage of the range of "umap" under the function
 * represented by "ma".
 * In other words, plug in "ma" in the range of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with range space equal to the target space of "ma",
 * except that the range has been replaced by the domain space of "ma".
 */
__isl_give isl_union_map *isl_union_map_preimage_range_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_aff *ma)
{
	return isl_union_map_preimage_range_pw_multi_aff(umap,
					isl_pw_multi_aff_from_multi_aff(ma));
}

/* Compute the preimage of "uset" under the function represented by "ma".
 * In other words, plug in "ma" in "uset".
 * The result contains sets that live in the same spaces as the sets of "uset"
 * with space equal to the target space of "ma",
 * except that the space has been replaced by the domain space of "ma".
 */
__isl_give isl_union_map *isl_union_set_preimage_multi_aff(
	__isl_take isl_union_set *uset, __isl_take isl_multi_aff *ma)
{
	return isl_union_set_preimage_pw_multi_aff(uset,
					isl_pw_multi_aff_from_multi_aff(ma));
}

/* Internal data structure for preimage_multi_pw_aff.
 *
 * "mpa" is the function under which the preimage should be taken.
 * "space" is the space of "mpa".
 * "res" collects the results.
 * "fn" computes the preimage for a given map.
 * "match" returns true if "fn" can be called.
 */
struct isl_union_map_preimage_mpa_data {
	isl_space *space;
	isl_multi_pw_aff *mpa;
	isl_union_map *res;
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space);
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_multi_pw_aff *mpa);
};

/* Call data->fn to compute the preimage of the domain or range of *entry
 * under the function represented by data->mpa, provided the domain/range
 * space of *entry matches the target space of data->mpa
 * (as given by data->match), and add the result to data->res.
 */
static isl_stat preimage_mpa_entry(void **entry, void *user)
{
	int m;
	isl_map *map = *entry;
	struct isl_union_map_preimage_mpa_data *data = user;
	isl_bool empty;

	m = data->match(map, data->space);
	if (m < 0)
		return isl_stat_error;
	if (!m)
		return isl_stat_ok;

	map = isl_map_copy(map);
	map = data->fn(map, isl_multi_pw_aff_copy(data->mpa));

	empty = isl_map_is_empty(map);
	if (empty < 0 || empty) {
		isl_map_free(map);
		return empty < 0 ? isl_stat_error : isl_stat_ok;
	}

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Compute the preimage of the domain or range of "umap" under the function
 * represented by "mpa".
 * In other words, plug in "mpa" in the domain or range of "umap".
 * The function "fn" performs the actual preimage computation on a map,
 * while "match" determines to which maps the function should be applied.
 */
static __isl_give isl_union_map *preimage_multi_pw_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_pw_aff *mpa,
	int (*match)(__isl_keep isl_map *map, __isl_keep isl_space *space),
	__isl_give isl_map *(*fn)(__isl_take isl_map *map,
		__isl_take isl_multi_pw_aff *mpa))
{
	isl_ctx *ctx;
	isl_space *space;
	struct isl_union_map_preimage_mpa_data data;

	umap = isl_union_map_align_params(umap,
					    isl_multi_pw_aff_get_space(mpa));
	mpa = isl_multi_pw_aff_align_params(mpa, isl_union_map_get_space(umap));

	if (!umap || !mpa)
		goto error;

	ctx = isl_union_map_get_ctx(umap);
	space = isl_union_map_get_space(umap);
	data.space = isl_multi_pw_aff_get_space(mpa);
	data.mpa = mpa;
	data.res = isl_union_map_alloc(space, umap->table.n);
	data.match = match;
	data.fn = fn;
	if (isl_hash_table_foreach(ctx, &umap->table, &preimage_mpa_entry,
					&data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_space_free(data.space);
	isl_union_map_free(umap);
	isl_multi_pw_aff_free(mpa);
	return data.res;
error:
	isl_union_map_free(umap);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "mpa".
 * In other words, plug in "mpa" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to the target space of "mpa",
 * except that the domain has been replaced by the domain space of "mpa".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_multi_pw_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_pw_aff *mpa)
{
	return preimage_multi_pw_aff(umap, mpa, &domain_match,
					&isl_map_preimage_domain_multi_pw_aff);
}

/* Internal data structure for preimage_upma.
 *
 * "umap" is the map of which the preimage should be computed.
 * "res" collects the results.
 * "fn" computes the preimage for a given piecewise multi-affine function.
 */
struct isl_union_map_preimage_upma_data {
	isl_union_map *umap;
	isl_union_map *res;
	__isl_give isl_union_map *(*fn)(__isl_take isl_union_map *umap,
		__isl_take isl_pw_multi_aff *pma);
};

/* Call data->fn to compute the preimage of the domain or range of data->umap
 * under the function represented by pma and add the result to data->res.
 */
static isl_stat preimage_upma(__isl_take isl_pw_multi_aff *pma, void *user)
{
	struct isl_union_map_preimage_upma_data *data = user;
	isl_union_map *umap;

	umap = isl_union_map_copy(data->umap);
	umap = data->fn(umap, pma);
	data->res = isl_union_map_union(data->res, umap);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Compute the preimage of the domain or range of "umap" under the function
 * represented by "upma".
 * In other words, plug in "upma" in the domain or range of "umap".
 * The function "fn" performs the actual preimage computation
 * on a piecewise multi-affine function.
 */
static __isl_give isl_union_map *preimage_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma,
	__isl_give isl_union_map *(*fn)(__isl_take isl_union_map *umap,
		__isl_take isl_pw_multi_aff *pma))
{
	struct isl_union_map_preimage_upma_data data;

	data.umap = umap;
	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	data.fn = fn;
	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma,
						    &preimage_upma, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_union_map_free(umap);
	isl_union_pw_multi_aff_free(upma);

	return data.res;
}

/* Compute the preimage of the domain of "umap" under the function
 * represented by "upma".
 * In other words, plug in "upma" in the domain of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with domain space equal to one of the target spaces of "upma",
 * except that the domain has been replaced by one of the the domain spaces that
 * corresponds to that target space of "upma".
 */
__isl_give isl_union_map *isl_union_map_preimage_domain_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma)
{
	return preimage_union_pw_multi_aff(umap, upma,
				&isl_union_map_preimage_domain_pw_multi_aff);
}

/* Compute the preimage of the range of "umap" under the function
 * represented by "upma".
 * In other words, plug in "upma" in the range of "umap".
 * The result contains maps that live in the same spaces as the maps of "umap"
 * with range space equal to one of the target spaces of "upma",
 * except that the range has been replaced by one of the the domain spaces that
 * corresponds to that target space of "upma".
 */
__isl_give isl_union_map *isl_union_map_preimage_range_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma)
{
	return preimage_union_pw_multi_aff(umap, upma,
				&isl_union_map_preimage_range_pw_multi_aff);
}

/* Compute the preimage of "uset" under the function represented by "upma".
 * In other words, plug in "upma" in the range of "uset".
 * The result contains sets that live in the same spaces as the sets of "uset"
 * with space equal to one of the target spaces of "upma",
 * except that the space has been replaced by one of the the domain spaces that
 * corresponds to that target space of "upma".
 */
__isl_give isl_union_set *isl_union_set_preimage_union_pw_multi_aff(
	__isl_take isl_union_set *uset,
	__isl_take isl_union_pw_multi_aff *upma)
{
	return preimage_union_pw_multi_aff(uset, upma,
					&isl_union_set_preimage_pw_multi_aff);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of *entry.
 */
static isl_stat reset_user(void **entry, void *user)
{
	isl_map **map = (isl_map **)entry;

	*map = isl_map_reset_user(*map);

	return *map ? isl_stat_ok : isl_stat_error;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the spaces of "umap".
 */
__isl_give isl_union_map *isl_union_map_reset_user(
	__isl_take isl_union_map *umap)
{
	umap = isl_union_map_cow(umap);
	if (!umap)
		return NULL;
	umap->dim = isl_space_reset_user(umap->dim);
	if (!umap->dim)
		return isl_union_map_free(umap);
	umap = un_op(umap, &reset_user);

	return umap;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the spaces of "uset".
 */
__isl_give isl_union_set *isl_union_set_reset_user(
	__isl_take isl_union_set *uset)
{
	return isl_union_map_reset_user(uset);
}

/* Internal data structure for isl_union_map_project_out.
 * "type", "first" and "n" are the arguments for the isl_map_project_out
 * call.
 * "res" collects the results.
 */
struct isl_union_map_project_out_data {
	enum isl_dim_type type;
	unsigned first;
	unsigned n;

	isl_union_map *res;
};

/* Turn the data->n dimensions of type data->type, starting at data->first
 * into existentially quantified variables and add the result to data->res.
 */
static isl_stat project_out(__isl_take isl_map *map, void *user)
{
	struct isl_union_map_project_out_data *data = user;

	map = isl_map_project_out(map, data->type, data->first, data->n);
	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Turn the "n" dimensions of type "type", starting at "first"
 * into existentially quantified variables.
 * Since the space of an isl_union_map only contains parameters,
 * type is required to be equal to isl_dim_param.
 */
__isl_give isl_union_map *isl_union_map_project_out(
	__isl_take isl_union_map *umap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_space *space;
	struct isl_union_map_project_out_data data = { type, first, n };

	if (!umap)
		return NULL;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only project out parameters",
			return isl_union_map_free(umap));

	space = isl_union_map_get_space(umap);
	space = isl_space_drop_dims(space, type, first, n);
	data.res = isl_union_map_empty(space);
	if (isl_union_map_foreach_map(umap, &project_out, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_union_map_free(umap);

	return data.res;
}

/* Turn the "n" dimensions of type "type", starting at "first"
 * into existentially quantified variables.
 * Since the space of an isl_union_set only contains parameters,
 * "type" is required to be equal to isl_dim_param.
 */
__isl_give isl_union_set *isl_union_set_project_out(
	__isl_take isl_union_set *uset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_union_map_project_out(uset, type, first, n);
}

/* Internal data structure for isl_union_map_involves_dims.
 * "first" and "n" are the arguments for the isl_map_involves_dims calls.
 */
struct isl_union_map_involves_dims_data {
	unsigned first;
	unsigned n;
};

/* Does "map" _not_ involve the data->n parameters starting at data->first?
 */
static isl_bool map_excludes(__isl_keep isl_map *map, void *user)
{
	struct isl_union_map_involves_dims_data *data = user;
	isl_bool involves;

	involves = isl_map_involves_dims(map,
					isl_dim_param, data->first, data->n);
	if (involves < 0)
		return isl_bool_error;
	return !involves;
}

/* Does "umap" involve any of the n parameters starting at first?
 * "type" is required to be set to isl_dim_param.
 *
 * "umap" involves any of those parameters if any of its maps
 * involve the parameters.  In other words, "umap" does not
 * involve any of the parameters if all its maps to not
 * involve the parameters.
 */
isl_bool isl_union_map_involves_dims(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	struct isl_union_map_involves_dims_data data = { first, n };
	isl_bool excludes;

	if (type != isl_dim_param)
		isl_die(isl_union_map_get_ctx(umap), isl_error_invalid,
			"can only reference parameters", return isl_bool_error);

	excludes = union_map_forall_user(umap, &map_excludes, &data);

	if (excludes < 0)
		return isl_bool_error;

	return !excludes;
}

/* Internal data structure for isl_union_map_reset_range_space.
 * "range" is the space from which to set the range space.
 * "res" collects the results.
 */
struct isl_union_map_reset_range_space_data {
	isl_space *range;
	isl_union_map *res;
};

/* Replace the range space of "map" by the range space of data->range and
 * add the result to data->res.
 */
static isl_stat reset_range_space(__isl_take isl_map *map, void *user)
{
	struct isl_union_map_reset_range_space_data *data = user;
	isl_space *space;

	space = isl_map_get_space(map);
	space = isl_space_domain(space);
	space = isl_space_extend_domain_with_range(space,
						isl_space_copy(data->range));
	map = isl_map_reset_space(map, space);
	data->res = isl_union_map_add_map(data->res, map);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Replace the range space of all the maps in "umap" by
 * the range space of "space".
 *
 * This assumes that all maps have the same output dimension.
 * This function should therefore not be made publicly available.
 *
 * Since the spaces of the maps change, so do their hash value.
 * We therefore need to create a new isl_union_map.
 */
__isl_give isl_union_map *isl_union_map_reset_range_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space)
{
	struct isl_union_map_reset_range_space_data data = { space };

	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &reset_range_space, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_space_free(space);
	isl_union_map_free(umap);
	return data.res;
}

/* Internal data structure for isl_union_map_order_at_multi_union_pw_aff.
 * "mupa" is the function from which the isl_multi_pw_affs are extracted.
 * "order" is applied to the extracted isl_multi_pw_affs that correspond
 * to the domain and the range of each map.
 * "res" collects the results.
 */
struct isl_union_order_at_data {
	isl_multi_union_pw_aff *mupa;
	__isl_give isl_map *(*order)(__isl_take isl_multi_pw_aff *mpa1,
		__isl_take isl_multi_pw_aff *mpa2);
	isl_union_map *res;
};

/* Intersect "map" with the result of applying data->order to
 * the functions in data->mupa that apply to the domain and the range
 * of "map" and add the result to data->res.
 */
static isl_stat order_at(__isl_take isl_map *map, void *user)
{
	struct isl_union_order_at_data *data = user;
	isl_space *space;
	isl_multi_pw_aff *mpa1, *mpa2;
	isl_map *order;

	space = isl_space_domain(isl_map_get_space(map));
	mpa1 = isl_multi_union_pw_aff_extract_multi_pw_aff(data->mupa, space);
	space = isl_space_range(isl_map_get_space(map));
	mpa2 = isl_multi_union_pw_aff_extract_multi_pw_aff(data->mupa, space);
	order = data->order(mpa1, mpa2);
	map = isl_map_intersect(map, order);
	data->res = isl_union_map_add_map(data->res, map);

	return data->res ? isl_stat_ok : isl_stat_error;
}

/* Intersect each map in "umap" with the result of calling "order"
 * on the functions is "mupa" that apply to the domain and the range
 * of the map.
 */
static __isl_give isl_union_map *isl_union_map_order_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_union_pw_aff *mupa,
	__isl_give isl_map *(*order)(__isl_take isl_multi_pw_aff *mpa1,
		__isl_take isl_multi_pw_aff *mpa2))
{
	struct isl_union_order_at_data data;

	umap = isl_union_map_align_params(umap,
				isl_multi_union_pw_aff_get_space(mupa));
	mupa = isl_multi_union_pw_aff_align_params(mupa,
				isl_union_map_get_space(umap));
	data.mupa = mupa;
	data.order = order;
	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &order_at, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_multi_union_pw_aff_free(mupa);
	isl_union_map_free(umap);
	return data.res;
}

/* Return the subset of "umap" where the domain and the range
 * have equal "mupa" values.
 */
__isl_give isl_union_map *isl_union_map_eq_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa)
{
	return isl_union_map_order_at_multi_union_pw_aff(umap, mupa,
						&isl_multi_pw_aff_eq_map);
}

/* Return the subset of "umap" where the domain has a lexicographically
 * smaller "mupa" value than the range.
 */
__isl_give isl_union_map *isl_union_map_lex_lt_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa)
{
	return isl_union_map_order_at_multi_union_pw_aff(umap, mupa,
						&isl_multi_pw_aff_lex_lt_map);
}

/* Return the subset of "umap" where the domain has a lexicographically
 * greater "mupa" value than the range.
 */
__isl_give isl_union_map *isl_union_map_lex_gt_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa)
{
	return isl_union_map_order_at_multi_union_pw_aff(umap, mupa,
						&isl_multi_pw_aff_lex_gt_map);
}

/* Return the union of the elements in the list "list".
 */
__isl_give isl_union_set *isl_union_set_list_union(
	__isl_take isl_union_set_list *list)
{
	int i, n;
	isl_ctx *ctx;
	isl_space *space;
	isl_union_set *res;

	if (!list)
		return NULL;

	ctx = isl_union_set_list_get_ctx(list);
	space = isl_space_params_alloc(ctx, 0);
	res = isl_union_set_empty(space);

	n = isl_union_set_list_n_union_set(list);
	for (i = 0; i < n; ++i) {
		isl_union_set *uset_i;

		uset_i = isl_union_set_list_get_union_set(list, i);
		res = isl_union_set_union(res, uset_i);
	}

	isl_union_set_list_free(list);
	return res;
}

/* Update *hash with the hash value of "map".
 */
static isl_stat add_hash(__isl_take isl_map *map, void *user)
{
	uint32_t *hash = user;
	uint32_t map_hash;

	map_hash = isl_map_get_hash(map);
	isl_hash_hash(*hash, map_hash);

	isl_map_free(map);
	return isl_stat_ok;
}

/* Return a hash value that digests "umap".
 */
uint32_t isl_union_map_get_hash(__isl_keep isl_union_map *umap)
{
	uint32_t hash;

	if (!umap)
		return 0;

	hash = isl_hash_init();
	if (isl_union_map_foreach_map(umap, &add_hash, &hash) < 0)
		return 0;

	return hash;
}

/* Return a hash value that digests "uset".
 */
uint32_t isl_union_set_get_hash(__isl_keep isl_union_set *uset)
{
	return isl_union_map_get_hash(uset);
}
