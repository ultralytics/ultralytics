#ifndef ISL_VAL_GMP_H
#define ISL_VAL_GMP_H

#include <gmp.h>
#include <isl/val.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_val *isl_val_int_from_gmp(isl_ctx *ctx, mpz_t z);
__isl_give isl_val *isl_val_from_gmp(isl_ctx *ctx,
	const mpz_t n, const mpz_t d);
int isl_val_get_num_gmp(__isl_keep isl_val *v, mpz_t z);
int isl_val_get_den_gmp(__isl_keep isl_val *v, mpz_t z);

#if defined(__cplusplus)
}
#endif

#endif
