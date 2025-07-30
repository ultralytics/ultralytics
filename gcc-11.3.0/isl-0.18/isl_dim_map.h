#ifndef ISL_DIM_MAP_H
#define ISL_DIM_MAP_H

#include <isl/ctx.h>
#include <isl/space.h>
#include <isl/map.h>
#include <isl_reordering.h>

struct isl_dim_map;
typedef struct isl_dim_map isl_dim_map;

__isl_give isl_dim_map *isl_dim_map_alloc(isl_ctx *ctx, unsigned len);
void isl_dim_map_range(__isl_keep isl_dim_map *dim_map,
	unsigned dst_pos, unsigned dst_stride,
	unsigned src_pos, unsigned src_stride,
	unsigned n, int sign);
void isl_dim_map_dim_range(__isl_keep isl_dim_map *dim_map,
	isl_space *dim, enum isl_dim_type type,
	unsigned first, unsigned n, unsigned dst_pos);
void isl_dim_map_dim(__isl_keep isl_dim_map *dim_map, __isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned dst_pos);
void isl_dim_map_div(__isl_keep isl_dim_map *dim_map,
	__isl_keep isl_basic_map *bmap, unsigned dst_pos);
__isl_give isl_basic_set *isl_basic_set_add_constraints_dim_map(
	__isl_take isl_basic_set *dst, __isl_take isl_basic_set *src,
	__isl_take isl_dim_map *dim_map);
__isl_give isl_basic_map *isl_basic_map_add_constraints_dim_map(
	__isl_take isl_basic_map *dst, __isl_take isl_basic_map *src,
	__isl_take isl_dim_map *dim_map);

__isl_give isl_dim_map *isl_dim_map_extend(__isl_keep isl_dim_map *dim_map,
	__isl_keep isl_basic_map *bmap);

__isl_give isl_dim_map *isl_dim_map_from_reordering(
	__isl_keep isl_reordering *exp);

#endif
