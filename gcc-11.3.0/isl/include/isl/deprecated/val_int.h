#ifndef ISL_DEPRECATED_VAL_INT_H
#define ISL_DEPRECATED_VAL_INT_H

#include <isl/deprecated/int.h>
#include <isl/val.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_val *isl_val_int_from_isl_int(isl_ctx *ctx, isl_int n);
int isl_val_get_num_isl_int(__isl_keep isl_val *v, isl_int *n);

#if defined(__cplusplus)
}
#endif

#endif
