/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <isl_blk.h>
#include <isl_ctx_private.h>

/* The maximal number of cache misses before first element is evicted */
#define ISL_BLK_MAX_MISS	100

struct isl_blk isl_blk_empty()
{
	struct isl_blk block;
	block.size = 0;
	block.data = NULL;
	return block;
}

static int isl_blk_is_empty(struct isl_blk block)
{
	return block.size == 0 && block.data == NULL;
}

static struct isl_blk isl_blk_error()
{
	struct isl_blk block;
	block.size = -1;
	block.data = NULL;
	return block;
}

int isl_blk_is_error(struct isl_blk block)
{
	return block.size == -1 && block.data == NULL;
}

static void isl_blk_free_force(struct isl_ctx *ctx, struct isl_blk block)
{
	int i;

	for (i = 0; i < block.size; ++i)
		isl_int_clear(block.data[i]);
	free(block.data);
}

static struct isl_blk extend(struct isl_ctx *ctx, struct isl_blk block,
				size_t new_n)
{
	int i;
	isl_int *p;

	if (block.size >= new_n)
		return block;

	p = isl_realloc_array(ctx, block.data, isl_int, new_n);
	if (!p) {
		isl_blk_free_force(ctx, block);
		return isl_blk_error();
	}
	block.data = p;

	for (i = block.size; i < new_n; ++i)
		isl_int_init(block.data[i]);
	block.size = new_n;

	return block;
}

struct isl_blk isl_blk_alloc(struct isl_ctx *ctx, size_t n)
{
	int i;
	struct isl_blk block;

	block = isl_blk_empty();
	if (n && ctx->n_cached) {
		int best = 0;
		for (i = 1; ctx->cache[best].size != n && i < ctx->n_cached; ++i) {
			if (ctx->cache[best].size < n) {
				if (ctx->cache[i].size > ctx->cache[best].size)
					best = i;
			} else if (ctx->cache[i].size >= n &&
				   ctx->cache[i].size < ctx->cache[best].size)
					best = i;
		}
		if (ctx->cache[best].size < 2 * n + 100) {
			block = ctx->cache[best];
			if (--ctx->n_cached != best)
				ctx->cache[best] = ctx->cache[ctx->n_cached];
			if (best == 0)
				ctx->n_miss = 0;
		} else if (ctx->n_miss++ >= ISL_BLK_MAX_MISS) {
			isl_blk_free_force(ctx, ctx->cache[0]);
			if (--ctx->n_cached != 0)
				ctx->cache[0] = ctx->cache[ctx->n_cached];
			ctx->n_miss = 0;
		}
	}

	return extend(ctx, block, n);
}

struct isl_blk isl_blk_extend(struct isl_ctx *ctx, struct isl_blk block,
				size_t new_n)
{
	if (isl_blk_is_empty(block))
		return isl_blk_alloc(ctx, new_n);

	return extend(ctx, block, new_n);
}

void isl_blk_free(struct isl_ctx *ctx, struct isl_blk block)
{
	if (isl_blk_is_empty(block) || isl_blk_is_error(block))
		return;

	if (ctx->n_cached < ISL_BLK_CACHE_SIZE)
		ctx->cache[ctx->n_cached++] = block;
	else
		isl_blk_free_force(ctx, block);
}

void isl_blk_clear_cache(struct isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ctx->n_cached; ++i)
		isl_blk_free_force(ctx, ctx->cache[i]);
	ctx->n_cached = 0;
}
