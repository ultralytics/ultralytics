/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010-2011 INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 */

#include <isl_map_private.h>
#include <isl_space_private.h>
#include <isl_dim_map.h>
#include <isl_reordering.h>

struct isl_dim_map_entry {
	int pos;
	int sgn;
};

/* Maps dst positions to src positions */
struct isl_dim_map {
	unsigned len;
	struct isl_dim_map_entry m[1];
};

__isl_give isl_dim_map *isl_dim_map_alloc(isl_ctx *ctx, unsigned len)
{
	int i;
	struct isl_dim_map *dim_map;
	dim_map = isl_alloc(ctx, struct isl_dim_map,
	    sizeof(struct isl_dim_map) + len * sizeof(struct isl_dim_map_entry));
	if (!dim_map)
		return NULL;
	dim_map->len = 1 + len;
	dim_map->m[0].pos = 0;
	dim_map->m[0].sgn = 1;
	for (i = 0; i < len; ++i)
		dim_map->m[1 + i].sgn = 0;
	return dim_map;
}

void isl_dim_map_range(__isl_keep isl_dim_map *dim_map,
	unsigned dst_pos, unsigned dst_stride,
	unsigned src_pos, unsigned src_stride,
	unsigned n, int sign)
{
	int i;

	if (!dim_map)
		return;

	for (i = 0; i < n; ++i) {
		unsigned d = 1 + dst_pos + dst_stride * i;
		unsigned s = 1 + src_pos + src_stride * i;
		dim_map->m[d].pos = s;
		dim_map->m[d].sgn = sign;
	}
}

void isl_dim_map_dim_range(__isl_keep isl_dim_map *dim_map,
	__isl_keep isl_space *dim, enum isl_dim_type type,
	unsigned first, unsigned n, unsigned dst_pos)
{
	int i;
	unsigned src_pos;

	if (!dim_map || !dim)
		return;
	
	src_pos = 1 + isl_space_offset(dim, type);
	for (i = 0; i < n; ++i) {
		dim_map->m[1 + dst_pos + i].pos = src_pos + first + i;
		dim_map->m[1 + dst_pos + i].sgn = 1;
	}
}

void isl_dim_map_dim(__isl_keep isl_dim_map *dim_map, __isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned dst_pos)
{
	isl_dim_map_dim_range(dim_map, dim, type,
			      0, isl_space_dim(dim, type), dst_pos);
}

void isl_dim_map_div(__isl_keep isl_dim_map *dim_map,
	__isl_keep isl_basic_map *bmap, unsigned dst_pos)
{
	int i;
	unsigned src_pos;

	if (!dim_map || !bmap)
		return;
	
	src_pos = 1 + isl_space_dim(bmap->dim, isl_dim_all);
	for (i = 0; i < bmap->n_div; ++i) {
		dim_map->m[1 + dst_pos + i].pos = src_pos + i;
		dim_map->m[1 + dst_pos + i].sgn = 1;
	}
}

void isl_dim_map_dump(struct isl_dim_map *dim_map)
{
	int i;

	for (i = 0; i < dim_map->len; ++i)
		fprintf(stderr, "%d -> %d * %d; ", i,
			dim_map->m[i].sgn, dim_map->m[i].pos);
	fprintf(stderr, "\n");
}

static void copy_constraint_dim_map(isl_int *dst, isl_int *src,
					struct isl_dim_map *dim_map)
{
	int i;

	for (i = 0; i < dim_map->len; ++i) {
		if (dim_map->m[i].sgn == 0)
			isl_int_set_si(dst[i], 0);
		else if (dim_map->m[i].sgn > 0)
			isl_int_set(dst[i], src[dim_map->m[i].pos]);
		else
			isl_int_neg(dst[i], src[dim_map->m[i].pos]);
	}
}

static void copy_div_dim_map(isl_int *dst, isl_int *src,
					struct isl_dim_map *dim_map)
{
	isl_int_set(dst[0], src[0]);
	copy_constraint_dim_map(dst+1, src+1, dim_map);
}

__isl_give isl_basic_map *isl_basic_map_add_constraints_dim_map(
	__isl_take isl_basic_map *dst, __isl_take isl_basic_map *src,
	__isl_take isl_dim_map *dim_map)
{
	int i;

	if (!src || !dst || !dim_map)
		goto error;

	for (i = 0; i < src->n_eq; ++i) {
		int i1 = isl_basic_map_alloc_equality(dst);
		if (i1 < 0)
			goto error;
		copy_constraint_dim_map(dst->eq[i1], src->eq[i], dim_map);
	}

	for (i = 0; i < src->n_ineq; ++i) {
		int i1 = isl_basic_map_alloc_inequality(dst);
		if (i1 < 0)
			goto error;
		copy_constraint_dim_map(dst->ineq[i1], src->ineq[i], dim_map);
	}

	for (i = 0; i < src->n_div; ++i) {
		int i1 = isl_basic_map_alloc_div(dst);
		if (i1 < 0)
			goto error;
		copy_div_dim_map(dst->div[i1], src->div[i], dim_map);
	}

	free(dim_map);
	isl_basic_map_free(src);

	return dst;
error:
	free(dim_map);
	isl_basic_map_free(src);
	isl_basic_map_free(dst);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_add_constraints_dim_map(
	__isl_take isl_basic_set *dst, __isl_take isl_basic_set *src,
	__isl_take isl_dim_map *dim_map)
{
	return isl_basic_map_add_constraints_dim_map(dst, src, dim_map);
}

/* Extend the given dim_map with mappings for the divs in bmap.
 */
__isl_give isl_dim_map *isl_dim_map_extend(__isl_keep isl_dim_map *dim_map,
	__isl_keep isl_basic_map *bmap)
{
	int i;
	struct isl_dim_map *res;
	int offset;

	offset = isl_basic_map_offset(bmap, isl_dim_div);

	res = isl_dim_map_alloc(bmap->ctx, dim_map->len - 1 + bmap->n_div);
	if (!res)
		return NULL;

	for (i = 0; i < dim_map->len; ++i)
		res->m[i] = dim_map->m[i];
	for (i = 0; i < bmap->n_div; ++i) {
		res->m[dim_map->len + i].pos = offset + i;
		res->m[dim_map->len + i].sgn = 1;
	}

	return res;
}

/* Extract a dim_map from a reordering.
 * We essentially need to reverse the mapping, and add an offset
 * of 1 for the constant term.
 */
__isl_give isl_dim_map *isl_dim_map_from_reordering(
	__isl_keep isl_reordering *exp)
{
	int i;
	isl_ctx *ctx;
	struct isl_dim_map *dim_map;

	if (!exp)
		return NULL;

	ctx = isl_space_get_ctx(exp->dim);
	dim_map = isl_dim_map_alloc(ctx, isl_space_dim(exp->dim, isl_dim_all));
	if (!dim_map)
		return NULL;

	for (i = 0; i < exp->len; ++i) {
		dim_map->m[1 + exp->pos[i]].pos = 1 + i;
		dim_map->m[1 + exp->pos[i]].sgn = 1;
	}

	return dim_map;
}
