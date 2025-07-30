#ifndef ISL_SPACE_PRIVATE
#define ISL_SPACE_PRIVATE

#include <isl/space.h>
#include <isl/hash.h>
#include <isl/id.h>

struct isl_name;
struct isl_space {
	int ref;

	struct isl_ctx *ctx;

	unsigned nparam;
	unsigned n_in;		/* zero for sets */
	unsigned n_out;		/* dim for sets */

	isl_id *tuple_id[2];
	isl_space *nested[2];

	unsigned n_id;
	isl_id **ids;
};

__isl_give isl_space *isl_space_cow(__isl_take isl_space *dim);

__isl_give isl_space *isl_space_underlying(__isl_take isl_space *dim,
	unsigned n_div);

uint32_t isl_space_get_hash(__isl_keep isl_space *dim);
uint32_t isl_space_get_domain_hash(__isl_keep isl_space *space);

isl_bool isl_space_is_domain_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_is_range_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
int isl_space_compatible_internal(__isl_keep isl_space *dim1,
	__isl_keep isl_space *dim2);

__isl_give isl_space *isl_space_as_set_space(__isl_take isl_space *dim);

unsigned isl_space_offset(__isl_keep isl_space *dim, enum isl_dim_type type);

int isl_space_may_be_set(__isl_keep isl_space *dim);
int isl_space_is_named_or_nested(__isl_keep isl_space *dim, enum isl_dim_type type);
int isl_space_has_named_params(__isl_keep isl_space *dim);
__isl_give isl_space *isl_space_reset(__isl_take isl_space *dim,
	enum isl_dim_type type);
__isl_give isl_space *isl_space_flatten(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_flatten_domain(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_flatten_range(__isl_take isl_space *dim);

__isl_give isl_space *isl_space_replace(__isl_take isl_space *dst,
	enum isl_dim_type type, __isl_keep isl_space *src);

__isl_give isl_space *isl_space_lift(__isl_take isl_space *dim, unsigned n_local);

__isl_give isl_space *isl_space_extend_domain_with_range(
	__isl_take isl_space *domain, __isl_take isl_space *model);

int isl_space_cmp(__isl_keep isl_space *space1, __isl_keep isl_space *space2);

#endif
