/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_BLK_H
#define ISL_BLK_H

#include <isl_int.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_blk {
	size_t size;
	isl_int *data;
};

#define ISL_BLK_CACHE_SIZE	20

struct isl_ctx;

struct isl_blk isl_blk_alloc(struct isl_ctx *ctx, size_t n);
struct isl_blk isl_blk_empty(void);
int isl_blk_is_error(struct isl_blk block);
struct isl_blk isl_blk_extend(struct isl_ctx *ctx, struct isl_blk block,
				size_t new_n);
void isl_blk_free(struct isl_ctx *ctx, struct isl_blk block);
void isl_blk_clear_cache(struct isl_ctx *ctx);

#if defined(__cplusplus)
}
#endif

#endif
