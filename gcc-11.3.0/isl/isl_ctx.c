/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <isl_ctx_private.h>
#include <isl/vec.h>
#include <isl_options_private.h>

#define __isl_calloc(type,size)		((type *)calloc(1, size))
#define __isl_calloc_type(type)		__isl_calloc(type,sizeof(type))

/* Return the negation of "b", where the negation of isl_bool_error
 * is isl_bool_error again.
 */
isl_bool isl_bool_not(isl_bool b)
{
	return b < 0 ? isl_bool_error : !b;
}

/* Check that the result of an allocation ("p") is not NULL and
 * complain if it is.
 * The only exception is when allocation size ("size") is equal to zero.
 */
static void *check_non_null(isl_ctx *ctx, void *p, size_t size)
{
	if (p || size == 0)
		return p;
	isl_die(ctx, isl_error_alloc, "allocation failure", return NULL);
}

/* Prepare for performing the next "operation" in the context.
 * Return 0 if we are allowed to perform this operation and
 * return -1 if we should abort the computation.
 *
 * In particular, we should stop if the user has explicitly aborted
 * the computation or if the maximal number of operations has been exceeded.
 */
int isl_ctx_next_operation(isl_ctx *ctx)
{
	if (!ctx)
		return -1;
	if (ctx->abort) {
		isl_ctx_set_error(ctx, isl_error_abort);
		return -1;
	}
	if (ctx->max_operations && ctx->operations >= ctx->max_operations)
		isl_die(ctx, isl_error_quota,
			"maximal number of operations exceeded", return -1);
	ctx->operations++;
	return 0;
}

/* Call malloc and complain if it fails.
 * If ctx is NULL, then return NULL.
 */
void *isl_malloc_or_die(isl_ctx *ctx, size_t size)
{
	if (isl_ctx_next_operation(ctx) < 0)
		return NULL;
	return ctx ? check_non_null(ctx, malloc(size), size) : NULL;
}

/* Call calloc and complain if it fails.
 * If ctx is NULL, then return NULL.
 */
void *isl_calloc_or_die(isl_ctx *ctx, size_t nmemb, size_t size)
{
	if (isl_ctx_next_operation(ctx) < 0)
		return NULL;
	return ctx ? check_non_null(ctx, calloc(nmemb, size), nmemb) : NULL;
}

/* Call realloc and complain if it fails.
 * If ctx is NULL, then return NULL.
 */
void *isl_realloc_or_die(isl_ctx *ctx, void *ptr, size_t size)
{
	if (isl_ctx_next_operation(ctx) < 0)
		return NULL;
	return ctx ? check_non_null(ctx, realloc(ptr, size), size) : NULL;
}

void isl_handle_error(isl_ctx *ctx, enum isl_error error, const char *msg,
	const char *file, int line)
{
	if (!ctx)
		return;

	isl_ctx_set_error(ctx, error);

	switch (ctx->opt->on_error) {
	case ISL_ON_ERROR_WARN:
		fprintf(stderr, "%s:%d: %s\n", file, line, msg);
		return;
	case ISL_ON_ERROR_CONTINUE:
		return;
	case ISL_ON_ERROR_ABORT:
		fprintf(stderr, "%s:%d: %s\n", file, line, msg);
		abort();
		return;
	}
}

static struct isl_options *find_nested_options(struct isl_args *args,
	void *opt, struct isl_args *wanted)
{
	int i;
	struct isl_options *options;

	if (args == wanted)
		return opt;

	for (i = 0; args->args[i].type != isl_arg_end; ++i) {
		struct isl_arg *arg = &args->args[i];
		void *child;

		if (arg->type != isl_arg_child)
			continue;

		if (arg->offset == (size_t) -1)
			child = opt;
		else
			child = *(void **)(((char *)opt) + arg->offset);

		options = find_nested_options(arg->u.child.child,
						child, wanted);
		if (options)
			return options;
	}

	return NULL;
}

static struct isl_options *find_nested_isl_options(struct isl_args *args,
	void *opt)
{
	return find_nested_options(args, opt, &isl_options_args);
}

void *isl_ctx_peek_options(isl_ctx *ctx, struct isl_args *args)
{
	if (!ctx)
		return NULL;
	if (args == &isl_options_args)
		return ctx->opt;
	return find_nested_options(ctx->user_args, ctx->user_opt, args);
}

isl_ctx *isl_ctx_alloc_with_options(struct isl_args *args, void *user_opt)
{
	struct isl_ctx *ctx = NULL;
	struct isl_options *opt = NULL;
	int opt_allocated = 0;

	if (!user_opt)
		return NULL;

	opt = find_nested_isl_options(args, user_opt);
	if (!opt) {
		opt = isl_options_new_with_defaults();
		if (!opt)
			goto error;
		opt_allocated = 1;
	}

	ctx = __isl_calloc_type(struct isl_ctx);
	if (!ctx)
		goto error;

	if (isl_hash_table_init(ctx, &ctx->id_table, 0))
		goto error;

	ctx->stats = isl_calloc_type(ctx, struct isl_stats);
	if (!ctx->stats)
		goto error;

	ctx->user_args = args;
	ctx->user_opt = user_opt;
	ctx->opt_allocated = opt_allocated;
	ctx->opt = opt;
	ctx->ref = 0;

	isl_int_init(ctx->zero);
	isl_int_set_si(ctx->zero, 0);

	isl_int_init(ctx->one);
	isl_int_set_si(ctx->one, 1);

	isl_int_init(ctx->two);
	isl_int_set_si(ctx->two, 2);

	isl_int_init(ctx->negone);
	isl_int_set_si(ctx->negone, -1);

	isl_int_init(ctx->normalize_gcd);

	ctx->n_cached = 0;
	ctx->n_miss = 0;

	ctx->error = isl_error_none;

	ctx->operations = 0;
	isl_ctx_set_max_operations(ctx, ctx->opt->max_operations);

	return ctx;
error:
	isl_args_free(args, user_opt);
	if (opt_allocated)
		isl_options_free(opt);
	free(ctx);
	return NULL;
}

struct isl_ctx *isl_ctx_alloc()
{
	struct isl_options *opt;

	opt = isl_options_new_with_defaults();

	return isl_ctx_alloc_with_options(&isl_options_args, opt);
}

void isl_ctx_ref(struct isl_ctx *ctx)
{
	ctx->ref++;
}

void isl_ctx_deref(struct isl_ctx *ctx)
{
	isl_assert(ctx, ctx->ref > 0, return);
	ctx->ref--;
}

/* Print statistics on usage.
 */
static void print_stats(isl_ctx *ctx)
{
	fprintf(stderr, "operations: %lu\n", ctx->operations);
}

void isl_ctx_free(struct isl_ctx *ctx)
{
	if (!ctx)
		return;
	if (ctx->ref != 0)
		isl_die(ctx, isl_error_invalid,
			"isl_ctx freed, but some objects still reference it",
			return);

	if (ctx->opt->print_stats)
		print_stats(ctx);

	isl_hash_table_clear(&ctx->id_table);
	isl_blk_clear_cache(ctx);
	isl_int_clear(ctx->zero);
	isl_int_clear(ctx->one);
	isl_int_clear(ctx->two);
	isl_int_clear(ctx->negone);
	isl_int_clear(ctx->normalize_gcd);
	isl_args_free(ctx->user_args, ctx->user_opt);
	if (ctx->opt_allocated)
		isl_options_free(ctx->opt);
	free(ctx->stats);
	free(ctx);
}

struct isl_options *isl_ctx_options(isl_ctx *ctx)
{
	if (!ctx)
		return NULL;
	return ctx->opt;
}

enum isl_error isl_ctx_last_error(isl_ctx *ctx)
{
	return ctx->error;
}

void isl_ctx_reset_error(isl_ctx *ctx)
{
	ctx->error = isl_error_none;
}

void isl_ctx_set_error(isl_ctx *ctx, enum isl_error error)
{
	if (ctx)
		ctx->error = error;
}

void isl_ctx_abort(isl_ctx *ctx)
{
	if (ctx)
		ctx->abort = 1;
}

void isl_ctx_resume(isl_ctx *ctx)
{
	if (ctx)
		ctx->abort = 0;
}

int isl_ctx_aborted(isl_ctx *ctx)
{
	return ctx ? ctx->abort : -1;
}

int isl_ctx_parse_options(isl_ctx *ctx, int argc, char **argv, unsigned flags)
{
	if (!ctx)
		return -1;
	return isl_args_parse(ctx->user_args, argc, argv, ctx->user_opt, flags);
}

/* Set the maximal number of iterations of "ctx" to "max_operations".
 */
void isl_ctx_set_max_operations(isl_ctx *ctx, unsigned long max_operations)
{
	if (!ctx)
		return;
	ctx->max_operations = max_operations;
}

/* Return the maximal number of iterations of "ctx".
 */
unsigned long isl_ctx_get_max_operations(isl_ctx *ctx)
{
	return ctx ? ctx->max_operations : 0;
}

/* Reset the number of operations performed by "ctx".
 */
void isl_ctx_reset_operations(isl_ctx *ctx)
{
	if (!ctx)
		return;
	ctx->operations = 0;
}
