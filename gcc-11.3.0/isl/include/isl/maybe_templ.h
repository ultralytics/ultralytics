#include <isl/ctx.h>
#include <isl/maybe.h>

/* A structure that possibly contains a pointer to an object of type ISL_TYPE.
 * The pointer in "value" is only valid if "valid" is isl_bool_true.
 * Otherwise, "value" is set to NULL.
 */
struct ISL_MAYBE(ISL_TYPE) {
	isl_bool	valid;
	ISL_TYPE	*value;
};
typedef struct ISL_MAYBE(ISL_TYPE) ISL_MAYBE(ISL_TYPE);
