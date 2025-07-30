/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_CTX_H
#define ISL_CTX_H

#include <stdio.h>
#include <stdlib.h>

#include <isl/arg.h>

#ifndef __isl_give
#define __isl_give
#endif
#ifndef __isl_take
#define __isl_take
#endif
#ifndef __isl_keep
#define __isl_keep
#endif
#ifndef __isl_null
#define __isl_null
#endif
#ifndef __isl_export
#define __isl_export
#endif
#ifndef __isl_overload
#define __isl_overload
#endif
#ifndef __isl_constructor
#define __isl_constructor
#endif
#ifndef __isl_subclass
#define __isl_subclass(super)
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* Nearly all isa functions require a struct isl_ctx allocated using
 * isl_ctx_alloc.  This ctx contains (or will contain) options that
 * control the behavior of the library and some caches.
 *
 * An object allocated within a given ctx should never be used inside
 * another ctx.  Functions for moving objects from one ctx to another
 * will be added as the need arises.
 *
 * A given context should only be used inside a single thread.
 * A global context for synchronization between different threads
 * as well as functions for moving a context to a different thread
 * will be added as the need arises.
 *
 * If anything goes wrong (out of memory, failed assertion), then
 * the library will currently simply abort.  This will be made
 * configurable in the future.
 * Users of the library should expect functions that return
 * a pointer to a structure, to return NULL, indicating failure.
 * Any function accepting a pointer to a structure will treat
 * a NULL argument as a failure, resulting in the function freeing
 * the remaining structures (if any) and returning NULL itself
 * (in case of pointer return type).
 * The only exception is the isl_ctx argument, which should never be NULL.
 */
struct isl_stats {
	long	gbr_solved_lps;
};
enum isl_error {
	isl_error_none = 0,
	isl_error_abort,
	isl_error_alloc,
	isl_error_unknown,
	isl_error_internal,
	isl_error_invalid,
	isl_error_quota,
	isl_error_unsupported
};
typedef enum {
	isl_stat_error = -1,
	isl_stat_ok = 0
} isl_stat;
typedef enum {
	isl_bool_error = -1,
	isl_bool_false = 0,
	isl_bool_true = 1
} isl_bool;
isl_bool isl_bool_not(isl_bool b);
struct isl_ctx;
typedef struct isl_ctx isl_ctx;

/* Some helper macros */

#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)
#define ISL_DEPRECATED	__attribute__((__deprecated__))
#else
#define ISL_DEPRECATED
#endif

#define ISL_FL_INIT(l, f)   (l) = (f)               /* Specific flags location. */
#define ISL_FL_SET(l, f)    ((l) |= (f))
#define ISL_FL_CLR(l, f)    ((l) &= ~(f))
#define ISL_FL_ISSET(l, f)  (!!((l) & (f)))

#define ISL_F_INIT(p, f)    ISL_FL_INIT((p)->flags, f)  /* Structure element flags. */
#define ISL_F_SET(p, f)     ISL_FL_SET((p)->flags, f)
#define ISL_F_CLR(p, f)     ISL_FL_CLR((p)->flags, f)
#define ISL_F_ISSET(p, f)   ISL_FL_ISSET((p)->flags, f)

void *isl_malloc_or_die(isl_ctx *ctx, size_t size);
void *isl_calloc_or_die(isl_ctx *ctx, size_t nmemb, size_t size);
void *isl_realloc_or_die(isl_ctx *ctx, void *ptr, size_t size);

#define isl_alloc(ctx,type,size)	((type *)isl_malloc_or_die(ctx, size))
#define isl_calloc(ctx,type,size)	((type *)isl_calloc_or_die(ctx,\
								    1, size))
#define isl_realloc(ctx,ptr,type,size)	((type *)isl_realloc_or_die(ctx,\
								    ptr, size))
#define isl_alloc_type(ctx,type)	isl_alloc(ctx,type,sizeof(type))
#define isl_calloc_type(ctx,type)	isl_calloc(ctx,type,sizeof(type))
#define isl_realloc_type(ctx,ptr,type)	isl_realloc(ctx,ptr,type,sizeof(type))
#define isl_alloc_array(ctx,type,n)	isl_alloc(ctx,type,(n)*sizeof(type))
#define isl_calloc_array(ctx,type,n)	((type *)isl_calloc_or_die(ctx,\
							    n, sizeof(type)))
#define isl_realloc_array(ctx,ptr,type,n) \
				    isl_realloc(ctx,ptr,type,(n)*sizeof(type))

#define isl_die(ctx,errno,msg,code)					\
	do {								\
		isl_handle_error(ctx, errno, msg, __FILE__, __LINE__);	\
		code;							\
	} while (0)

void isl_handle_error(isl_ctx *ctx, enum isl_error error, const char *msg,
	const char *file, int line);

#define isl_assert4(ctx,test,code,errno)				\
	do {								\
		if (test)						\
			break;						\
		isl_die(ctx, errno, "Assertion \"" #test "\" failed", code);	\
	} while (0)
#define isl_assert(ctx,test,code)					\
	isl_assert4(ctx,test,code,isl_error_unknown)

#define isl_min(a,b)			((a < b) ? (a) : (b))

/* struct isl_ctx functions */

struct isl_options *isl_ctx_options(isl_ctx *ctx);

isl_ctx *isl_ctx_alloc_with_options(struct isl_args *args,
	__isl_take void *opt);
isl_ctx *isl_ctx_alloc(void);
void *isl_ctx_peek_options(isl_ctx *ctx, struct isl_args *args);
int isl_ctx_parse_options(isl_ctx *ctx, int argc, char **argv, unsigned flags);
void isl_ctx_ref(struct isl_ctx *ctx);
void isl_ctx_deref(struct isl_ctx *ctx);
void isl_ctx_free(isl_ctx *ctx);

void isl_ctx_abort(isl_ctx *ctx);
void isl_ctx_resume(isl_ctx *ctx);
int isl_ctx_aborted(isl_ctx *ctx);

void isl_ctx_set_max_operations(isl_ctx *ctx, unsigned long max_operations);
unsigned long isl_ctx_get_max_operations(isl_ctx *ctx);
void isl_ctx_reset_operations(isl_ctx *ctx);

#define ISL_ARG_CTX_DECL(prefix,st,args)				\
st *isl_ctx_peek_ ## prefix(isl_ctx *ctx);

#define ISL_ARG_CTX_DEF(prefix,st,args)					\
st *isl_ctx_peek_ ## prefix(isl_ctx *ctx)				\
{									\
	return (st *)isl_ctx_peek_options(ctx, &(args));		\
}

#define ISL_CTX_GET_INT_DEF(prefix,st,args,field)			\
int prefix ## _get_ ## field(isl_ctx *ctx)				\
{									\
	st *options;							\
	options = isl_ctx_peek_ ## prefix(ctx);				\
	if (!options)							\
		isl_die(ctx, isl_error_invalid,				\
			"isl_ctx does not reference " #prefix,		\
			return -1);					\
	return options->field;						\
}

#define ISL_CTX_SET_INT_DEF(prefix,st,args,field)			\
isl_stat prefix ## _set_ ## field(isl_ctx *ctx, int val)		\
{									\
	st *options;							\
	options = isl_ctx_peek_ ## prefix(ctx);				\
	if (!options)							\
		isl_die(ctx, isl_error_invalid,				\
			"isl_ctx does not reference " #prefix,		\
			return isl_stat_error);				\
	options->field = val;						\
	return isl_stat_ok;						\
}

#define ISL_CTX_GET_STR_DEF(prefix,st,args,field)			\
const char *prefix ## _get_ ## field(isl_ctx *ctx)			\
{									\
	st *options;							\
	options = isl_ctx_peek_ ## prefix(ctx);				\
	if (!options)							\
		isl_die(ctx, isl_error_invalid,				\
			"isl_ctx does not reference " #prefix,		\
			return NULL);					\
	return options->field;						\
}

#define ISL_CTX_SET_STR_DEF(prefix,st,args,field)			\
isl_stat prefix ## _set_ ## field(isl_ctx *ctx, const char *val)	\
{									\
	st *options;							\
	options = isl_ctx_peek_ ## prefix(ctx);				\
	if (!options)							\
		isl_die(ctx, isl_error_invalid,				\
			"isl_ctx does not reference " #prefix,		\
			return isl_stat_error);				\
	if (!val)							\
		return isl_stat_error;					\
	free(options->field);						\
	options->field = strdup(val);					\
	if (!options->field)						\
		return isl_stat_error;					\
	return isl_stat_ok;						\
}

#define ISL_CTX_GET_BOOL_DEF(prefix,st,args,field)			\
	ISL_CTX_GET_INT_DEF(prefix,st,args,field)

#define ISL_CTX_SET_BOOL_DEF(prefix,st,args,field)			\
	ISL_CTX_SET_INT_DEF(prefix,st,args,field)

#define ISL_CTX_GET_CHOICE_DEF(prefix,st,args,field)			\
	ISL_CTX_GET_INT_DEF(prefix,st,args,field)

#define ISL_CTX_SET_CHOICE_DEF(prefix,st,args,field)			\
	ISL_CTX_SET_INT_DEF(prefix,st,args,field)

enum isl_error isl_ctx_last_error(isl_ctx *ctx);
void isl_ctx_reset_error(isl_ctx *ctx);
void isl_ctx_set_error(isl_ctx *ctx, enum isl_error error);

#if defined(__cplusplus)
}
#endif

#endif
