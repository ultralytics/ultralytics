#ifndef ISL_VAL_H
#define ISL_VAL_H

#include <isl/ctx.h>
#include <isl/list.h>
#include <isl/multi.h>
#include <isl/printer.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_val;
typedef struct isl_val isl_val;

ISL_DECLARE_LIST(val)

struct __isl_export isl_multi_val;
typedef struct isl_multi_val isl_multi_val;

ISL_DECLARE_MULTI(val)
ISL_DECLARE_MULTI_NEG(val)
ISL_DECLARE_MULTI_DIMS(val)
ISL_DECLARE_MULTI_WITH_DOMAIN(val)

__isl_export
__isl_give isl_val *isl_val_zero(isl_ctx *ctx);
__isl_export
__isl_give isl_val *isl_val_one(isl_ctx *ctx);
__isl_export
__isl_give isl_val *isl_val_negone(isl_ctx *ctx);
__isl_export
__isl_give isl_val *isl_val_nan(isl_ctx *ctx);
__isl_export
__isl_give isl_val *isl_val_infty(isl_ctx *ctx);
__isl_export
__isl_give isl_val *isl_val_neginfty(isl_ctx *ctx);
__isl_constructor
__isl_give isl_val *isl_val_int_from_si(isl_ctx *ctx, long i);
__isl_give isl_val *isl_val_int_from_ui(isl_ctx *ctx, unsigned long u);
__isl_give isl_val *isl_val_int_from_chunks(isl_ctx *ctx, size_t n,
	size_t size, const void *chunks);

__isl_give isl_val *isl_val_copy(__isl_keep isl_val *v);
__isl_null isl_val *isl_val_free(__isl_take isl_val *v);

isl_ctx *isl_val_get_ctx(__isl_keep isl_val *val);
uint32_t isl_val_get_hash(__isl_keep isl_val *val);
long isl_val_get_num_si(__isl_keep isl_val *v);
long isl_val_get_den_si(__isl_keep isl_val *v);
__isl_give isl_val *isl_val_get_den_val(__isl_keep isl_val *v);
double isl_val_get_d(__isl_keep isl_val *v);
size_t isl_val_n_abs_num_chunks(__isl_keep isl_val *v, size_t size);
int isl_val_get_abs_num_chunks(__isl_keep isl_val *v, size_t size,
	void *chunks);

__isl_give isl_val *isl_val_set_si(__isl_take isl_val *v, long i);

__isl_export
__isl_give isl_val *isl_val_abs(__isl_take isl_val *v);
__isl_export
__isl_give isl_val *isl_val_neg(__isl_take isl_val *v);
__isl_export
__isl_give isl_val *isl_val_inv(__isl_take isl_val *v);
__isl_export
__isl_give isl_val *isl_val_floor(__isl_take isl_val *v);
__isl_export
__isl_give isl_val *isl_val_ceil(__isl_take isl_val *v);
__isl_export
__isl_give isl_val *isl_val_trunc(__isl_take isl_val *v);
__isl_give isl_val *isl_val_2exp(__isl_take isl_val *v);
__isl_export
__isl_give isl_val *isl_val_min(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_export
__isl_give isl_val *isl_val_max(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_export
__isl_give isl_val *isl_val_add(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_give isl_val *isl_val_add_ui(__isl_take isl_val *v1, unsigned long v2);
__isl_export
__isl_give isl_val *isl_val_sub(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_give isl_val *isl_val_sub_ui(__isl_take isl_val *v1, unsigned long v2);
__isl_export
__isl_give isl_val *isl_val_mul(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_give isl_val *isl_val_mul_ui(__isl_take isl_val *v1, unsigned long v2);
__isl_export
__isl_give isl_val *isl_val_div(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_export
__isl_give isl_val *isl_val_mod(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_export
__isl_give isl_val *isl_val_gcd(__isl_take isl_val *v1, __isl_take isl_val *v2);
__isl_give isl_val *isl_val_gcdext(__isl_take isl_val *v1,
	__isl_take isl_val *v2, __isl_give isl_val **x, __isl_give isl_val **y);

__isl_export
int isl_val_sgn(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_zero(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_one(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_negone(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_nonneg(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_nonpos(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_pos(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_neg(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_int(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_rat(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_nan(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_infty(__isl_keep isl_val *v);
__isl_export
isl_bool isl_val_is_neginfty(__isl_keep isl_val *v);

__isl_export
int isl_val_cmp_si(__isl_keep isl_val *v, long i);

__isl_export
isl_bool isl_val_lt(__isl_keep isl_val *v1, __isl_keep isl_val *v2);
__isl_export
isl_bool isl_val_le(__isl_keep isl_val *v1, __isl_keep isl_val *v2);
__isl_export
isl_bool isl_val_gt(__isl_keep isl_val *v1, __isl_keep isl_val *v2);
__isl_export
isl_bool isl_val_ge(__isl_keep isl_val *v1, __isl_keep isl_val *v2);
__isl_export
isl_bool isl_val_eq(__isl_keep isl_val *v1, __isl_keep isl_val *v2);
__isl_export
isl_bool isl_val_ne(__isl_keep isl_val *v1, __isl_keep isl_val *v2);
__isl_export
isl_bool isl_val_abs_eq(__isl_keep isl_val *v1, __isl_keep isl_val *v2);

__isl_export
isl_bool isl_val_is_divisible_by(__isl_keep isl_val *v1,
	__isl_keep isl_val *v2);

__isl_constructor
__isl_give isl_val *isl_val_read_from_str(isl_ctx *ctx, const char *str);
__isl_give isl_printer *isl_printer_print_val(__isl_take isl_printer *p,
	__isl_keep isl_val *v);
void isl_val_dump(__isl_keep isl_val *v);
__isl_give char *isl_val_to_str(__isl_keep isl_val *v);

__isl_give isl_multi_val *isl_multi_val_add_val(__isl_take isl_multi_val *mv,
	__isl_take isl_val *v);
__isl_give isl_multi_val *isl_multi_val_mod_val(__isl_take isl_multi_val *mv,
	__isl_take isl_val *v);

__isl_give isl_multi_val *isl_multi_val_read_from_str(isl_ctx *ctx,
	const char *str);
__isl_give isl_printer *isl_printer_print_multi_val(__isl_take isl_printer *p,
	__isl_keep isl_multi_val *mv);
void isl_multi_val_dump(__isl_keep isl_multi_val *mv);
__isl_give char *isl_multi_val_to_str(__isl_keep isl_multi_val *mv);

#if defined(__cplusplus)
}
#endif

#endif
