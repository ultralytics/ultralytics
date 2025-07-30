#include <isl/ctx.h>
#include <isl_blk.h>

struct isl_ctx {
	int			ref;

	struct isl_stats	*stats;

	int			 opt_allocated;
	struct isl_options	*opt;
	void			*user_opt;
	struct isl_args		*user_args;

	isl_int			zero;
	isl_int			one;
	isl_int			two;
	isl_int			negone;

	isl_int			normalize_gcd;

	int			n_cached;
	int			n_miss;
	struct isl_blk		cache[ISL_BLK_CACHE_SIZE];
	struct isl_hash_table	id_table;

	enum isl_error		error;

	int			abort;

	unsigned long		operations;
	unsigned long		max_operations;
};

int isl_ctx_next_operation(isl_ctx *ctx);
