/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl_ctx_private.h>
#include <isl_seq.h>
#include <isl_val_private.h>
#include <isl_vec_private.h>
#include <isl/deprecated/vec_int.h>

isl_ctx *isl_vec_get_ctx(__isl_keep isl_vec *vec)
{
	return vec ? vec->ctx : NULL;
}

/* Return a hash value that digests "vec".
 */
uint32_t isl_vec_get_hash(__isl_keep isl_vec *vec)
{
	if (!vec)
		return 0;

	return isl_seq_get_hash(vec->el, vec->size);
}

struct isl_vec *isl_vec_alloc(struct isl_ctx *ctx, unsigned size)
{
	struct isl_vec *vec;

	vec = isl_alloc_type(ctx, struct isl_vec);
	if (!vec)
		return NULL;

	vec->block = isl_blk_alloc(ctx, size);
	if (isl_blk_is_error(vec->block))
		goto error;

	vec->ctx = ctx;
	isl_ctx_ref(ctx);
	vec->ref = 1;
	vec->size = size;
	vec->el = vec->block.data;

	return vec;
error:
	isl_blk_free(ctx, vec->block);
	free(vec);
	return NULL;
}

__isl_give isl_vec *isl_vec_extend(__isl_take isl_vec *vec, unsigned size)
{
	if (!vec)
		return NULL;
	if (size <= vec->size)
		return vec;

	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;

	vec->block = isl_blk_extend(vec->ctx, vec->block, size);
	if (!vec->block.data)
		goto error;

	vec->size = size;
	vec->el = vec->block.data;

	return vec;
error:
	isl_vec_free(vec);
	return NULL;
}

/* Apply the expansion specified by "exp" to the "n" elements starting at "pos".
 * "expanded" it the number of elements that need to replace those "n"
 * elements.  The entries in "exp" have increasing values between
 * 0 and "expanded".
 */
__isl_give isl_vec *isl_vec_expand(__isl_take isl_vec *vec, int pos, int n,
	int *exp, int expanded)
{
	int i, j;
	int old_size, extra;

	if (!vec)
		return NULL;
	if (expanded < n)
		isl_die(isl_vec_get_ctx(vec), isl_error_invalid,
			"not an expansion", isl_vec_free(vec));
	if (expanded == n)
		return vec;
	if (pos < 0 || n < 0 || pos + n > vec->size)
		isl_die(isl_vec_get_ctx(vec), isl_error_invalid,
			"position out of bounds", return isl_vec_free(vec));

	old_size = vec->size;
	extra = expanded - n;
	vec = isl_vec_extend(vec, old_size + extra);
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;

	for (i = old_size - 1; i >= pos + n; --i)
		isl_int_set(vec->el[i + extra], vec->el[i]);

	j = n - 1;
	for (i = expanded - 1; i >= 0; --i) {
		if (j >= 0 && exp[j] == i) {
			if (i != j)
				isl_int_swap(vec->el[pos + i],
					     vec->el[pos + j]);
			j--;
		} else {
			isl_int_set_si(vec->el[pos + i], 0);
		}
	}

	return vec;
}

__isl_give isl_vec *isl_vec_zero_extend(__isl_take isl_vec *vec, unsigned size)
{
	int extra;

	if (!vec)
		return NULL;
	if (size <= vec->size)
		return vec;

	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;

	extra = size - vec->size;
	vec = isl_vec_extend(vec, size);
	if (!vec)
		return NULL;

	isl_seq_clr(vec->el + size - extra, extra);

	return vec;
}

/* Return a vector containing the elements of "vec1" followed by
 * those of "vec2".
 */
__isl_give isl_vec *isl_vec_concat(__isl_take isl_vec *vec1,
	__isl_take isl_vec *vec2)
{
	if (!vec1 || !vec2)
		goto error;

	if (vec2->size == 0) {
		isl_vec_free(vec2);
		return vec1;
	}

	if (vec1->size == 0) {
		isl_vec_free(vec1);
		return vec2;
	}

	vec1 = isl_vec_extend(vec1, vec1->size + vec2->size);
	if (!vec1)
		goto error;

	isl_seq_cpy(vec1->el + vec1->size - vec2->size, vec2->el, vec2->size);

	isl_vec_free(vec2);
	return vec1;
error:
	isl_vec_free(vec1);
	isl_vec_free(vec2);
	return NULL;
}

struct isl_vec *isl_vec_copy(struct isl_vec *vec)
{
	if (!vec)
		return NULL;

	vec->ref++;
	return vec;
}

struct isl_vec *isl_vec_dup(struct isl_vec *vec)
{
	struct isl_vec *vec2;

	if (!vec)
		return NULL;
	vec2 = isl_vec_alloc(vec->ctx, vec->size);
	if (!vec2)
		return NULL;
	isl_seq_cpy(vec2->el, vec->el, vec->size);
	return vec2;
}

struct isl_vec *isl_vec_cow(struct isl_vec *vec)
{
	struct isl_vec *vec2;
	if (!vec)
		return NULL;

	if (vec->ref == 1)
		return vec;

	vec2 = isl_vec_dup(vec);
	isl_vec_free(vec);
	return vec2;
}

__isl_null isl_vec *isl_vec_free(__isl_take isl_vec *vec)
{
	if (!vec)
		return NULL;

	if (--vec->ref > 0)
		return NULL;

	isl_ctx_deref(vec->ctx);
	isl_blk_free(vec->ctx, vec->block);
	free(vec);

	return NULL;
}

int isl_vec_size(__isl_keep isl_vec *vec)
{
	return vec ? vec->size : -1;
}

int isl_vec_get_element(__isl_keep isl_vec *vec, int pos, isl_int *v)
{
	if (!vec)
		return -1;

	if (pos < 0 || pos >= vec->size)
		isl_die(vec->ctx, isl_error_invalid, "position out of range",
			return -1);
	isl_int_set(*v, vec->el[pos]);
	return 0;
}

/* Extract the element at position "pos" of "vec".
 */
__isl_give isl_val *isl_vec_get_element_val(__isl_keep isl_vec *vec, int pos)
{
	isl_ctx *ctx;

	if (!vec)
		return NULL;
	ctx = isl_vec_get_ctx(vec);
	if (pos < 0 || pos >= vec->size)
		isl_die(ctx, isl_error_invalid, "position out of range",
			return NULL);
	return isl_val_int_from_isl_int(ctx, vec->el[pos]);
}

__isl_give isl_vec *isl_vec_set_element(__isl_take isl_vec *vec,
	int pos, isl_int v)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;
	if (pos < 0 || pos >= vec->size)
		isl_die(vec->ctx, isl_error_invalid, "position out of range",
			goto error);
	isl_int_set(vec->el[pos], v);
	return vec;
error:
	isl_vec_free(vec);
	return NULL;
}

__isl_give isl_vec *isl_vec_set_element_si(__isl_take isl_vec *vec,
	int pos, int v)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;
	if (pos < 0 || pos >= vec->size)
		isl_die(vec->ctx, isl_error_invalid, "position out of range",
			goto error);
	isl_int_set_si(vec->el[pos], v);
	return vec;
error:
	isl_vec_free(vec);
	return NULL;
}

/* Replace the element at position "pos" of "vec" by "v".
 */
__isl_give isl_vec *isl_vec_set_element_val(__isl_take isl_vec *vec,
	int pos, __isl_take isl_val *v)
{
	if (!v)
		return isl_vec_free(vec);
	if (!isl_val_is_int(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting integer value", goto error);
	vec = isl_vec_set_element(vec, pos, v->n);
	isl_val_free(v);
	return vec;
error:
	isl_val_free(v);
	return isl_vec_free(vec);
}

/* Compare the elements of "vec1" and "vec2" at position "pos".
 */
int isl_vec_cmp_element(__isl_keep isl_vec *vec1, __isl_keep isl_vec *vec2,
	int pos)
{
	if (!vec1 || !vec2)
		return 0;
	if (pos < 0 || pos >= vec1->size || pos >= vec2->size)
		isl_die(isl_vec_get_ctx(vec1), isl_error_invalid,
			"position out of range", return 0);
	return isl_int_cmp(vec1->el[pos], vec2->el[pos]);
}

isl_bool isl_vec_is_equal(__isl_keep isl_vec *vec1, __isl_keep isl_vec *vec2)
{
	if (!vec1 || !vec2)
		return isl_bool_error;

	if (vec1->size != vec2->size)
		return isl_bool_false;

	return isl_seq_eq(vec1->el, vec2->el, vec1->size);
}

__isl_give isl_printer *isl_printer_print_vec(__isl_take isl_printer *printer,
	__isl_keep isl_vec *vec)
{
	int i;

	if (!printer || !vec)
		goto error;

	printer = isl_printer_print_str(printer, "[");
	for (i = 0; i < vec->size; ++i) {
		if (i)
			printer = isl_printer_print_str(printer, ",");
		printer = isl_printer_print_isl_int(printer, vec->el[i]);
	}
	printer = isl_printer_print_str(printer, "]");

	return printer;
error:
	isl_printer_free(printer);
	return NULL;
}

void isl_vec_dump(struct isl_vec *vec)
{
	isl_printer *printer;

	if (!vec)
		return;

	printer = isl_printer_to_file(vec->ctx, stderr);
	printer = isl_printer_print_vec(printer, vec);
	printer = isl_printer_end_line(printer);

	isl_printer_free(printer);
}

__isl_give isl_vec *isl_vec_set(__isl_take isl_vec *vec, isl_int v)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;
	isl_seq_set(vec->el, v, vec->size);
	return vec;
}

__isl_give isl_vec *isl_vec_set_si(__isl_take isl_vec *vec, int v)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;
	isl_seq_set_si(vec->el, v, vec->size);
	return vec;
}

/* Replace all elements of "vec" by "v".
 */
__isl_give isl_vec *isl_vec_set_val(__isl_take isl_vec *vec,
	__isl_take isl_val *v)
{
	vec = isl_vec_cow(vec);
	if (!vec || !v)
		goto error;
	if (!isl_val_is_int(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting integer value", goto error);
	isl_seq_set(vec->el, v->n, vec->size);
	isl_val_free(v);
	return vec;
error:
	isl_vec_free(vec);
	isl_val_free(v);
	return NULL;
}

__isl_give isl_vec *isl_vec_clr(__isl_take isl_vec *vec)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;
	isl_seq_clr(vec->el, vec->size);
	return vec;
}

void isl_vec_lcm(struct isl_vec *vec, isl_int *lcm)
{
	isl_seq_lcm(vec->block.data, vec->size, lcm);
}

/* Given a rational vector, with the denominator in the first element
 * of the vector, round up all coordinates.
 */
struct isl_vec *isl_vec_ceil(struct isl_vec *vec)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;

	isl_seq_cdiv_q(vec->el + 1, vec->el + 1, vec->el[0], vec->size - 1);

	isl_int_set_si(vec->el[0], 1);

	return vec;
}

struct isl_vec *isl_vec_normalize(struct isl_vec *vec)
{
	if (!vec)
		return NULL;
	isl_seq_normalize(vec->ctx, vec->el, vec->size);
	return vec;
}

__isl_give isl_vec *isl_vec_neg(__isl_take isl_vec *vec)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;
	isl_seq_neg(vec->el, vec->el, vec->size);
	return vec;
}

__isl_give isl_vec *isl_vec_scale(__isl_take isl_vec *vec, isl_int m)
{
	if (isl_int_is_one(m))
		return vec;
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;
	isl_seq_scale(vec->el, vec->el, m, vec->size);
	return vec;
}

/* Reduce the elements of "vec" modulo "m".
 */
__isl_give isl_vec *isl_vec_fdiv_r(__isl_take isl_vec *vec, isl_int m)
{
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;

	isl_seq_fdiv_r(vec->el, vec->el, m, vec->size);

	return vec;
}

__isl_give isl_vec *isl_vec_add(__isl_take isl_vec *vec1,
	__isl_take isl_vec *vec2)
{
	vec1 = isl_vec_cow(vec1);
	if (!vec1 || !vec2)
		goto error;

	isl_assert(vec1->ctx, vec1->size == vec2->size, goto error);

	isl_seq_combine(vec1->el, vec1->ctx->one, vec1->el,
			vec1->ctx->one, vec2->el, vec1->size);
	
	isl_vec_free(vec2);
	return vec1;
error:
	isl_vec_free(vec1);
	isl_vec_free(vec2);
	return NULL;
}

static int qsort_int_cmp(const void *p1, const void *p2)
{
	const isl_int *i1 = (const isl_int *) p1;
	const isl_int *i2 = (const isl_int *) p2;

	return isl_int_cmp(*i1, *i2);
}

__isl_give isl_vec *isl_vec_sort(__isl_take isl_vec *vec)
{
	if (!vec)
		return NULL;
	
	qsort(vec->el, vec->size, sizeof(*vec->el), &qsort_int_cmp);

	return vec;
}

__isl_give isl_vec *isl_vec_drop_els(__isl_take isl_vec *vec,
	unsigned pos, unsigned n)
{
	if (n == 0)
		return vec;
	vec = isl_vec_cow(vec);
	if (!vec)
		return NULL;

	if (pos + n > vec->size)
		isl_die(vec->ctx, isl_error_invalid,
			"range out of bounds", goto error);

	if (pos + n != vec->size)
		isl_seq_cpy(vec->el + pos, vec->el + pos + n,
			    vec->size - pos - n);

	vec->size -= n;
	
	return vec;
error:
	isl_vec_free(vec);
	return NULL;
}

__isl_give isl_vec *isl_vec_insert_els(__isl_take isl_vec *vec,
	unsigned pos, unsigned n)
{
	isl_vec *ext = NULL;

	if (n == 0)
		return vec;
	if (!vec)
		return NULL;

	if (pos > vec->size)
		isl_die(vec->ctx, isl_error_invalid,
			"position out of bounds", goto error);

	ext =  isl_vec_alloc(vec->ctx, vec->size + n);
	if (!ext)
		goto error;

	isl_seq_cpy(ext->el, vec->el, pos);
	isl_seq_cpy(ext->el + pos + n, vec->el + pos, vec->size - pos);

	isl_vec_free(vec);
	return ext;
error:
	isl_vec_free(vec);
	isl_vec_free(ext);
	return NULL;
}

__isl_give isl_vec *isl_vec_insert_zero_els(__isl_take isl_vec *vec,
	unsigned pos, unsigned n)
{
	vec = isl_vec_insert_els(vec, pos, n);
	if (!vec)
		return NULL;

	isl_seq_clr(vec->el + pos, n);

	return vec;
}

/* Move the "n" elements starting as "src_pos" of "vec"
 * to "dst_pos".  The elements originally at "dst_pos" are moved
 * up or down depending on whether "dst_pos" is smaller or greater
 * than "src_pos".
 */
__isl_give isl_vec *isl_vec_move_els(__isl_take isl_vec *vec,
	unsigned dst_pos, unsigned src_pos, unsigned n)
{
	isl_vec *res;

	if (!vec)
		return NULL;

	if (src_pos + n > vec->size)
		isl_die(vec->ctx, isl_error_invalid,
			"source range out of bounds", return isl_vec_free(vec));
	if (dst_pos + n > vec->size)
		isl_die(vec->ctx, isl_error_invalid,
			"destination range out of bounds",
			return isl_vec_free(vec));

	if (n == 0 || dst_pos == src_pos)
		return vec;

	res = isl_vec_alloc(vec->ctx, vec->size);
	if (!res)
		return isl_vec_free(vec);

	if (dst_pos < src_pos) {
		isl_seq_cpy(res->el, vec->el, dst_pos);
		isl_seq_cpy(res->el + dst_pos, vec->el + src_pos, n);
		isl_seq_cpy(res->el + dst_pos + n,
			    vec->el + dst_pos, src_pos - dst_pos);
		isl_seq_cpy(res->el + src_pos + n,
			    vec->el + src_pos + n, res->size - src_pos - n);
	} else {
		isl_seq_cpy(res->el, vec->el, src_pos);
		isl_seq_cpy(res->el + src_pos,
			    vec->el + src_pos + n, dst_pos - src_pos);
		isl_seq_cpy(res->el + dst_pos, vec->el + src_pos, n);
		isl_seq_cpy(res->el + dst_pos + n,
			    vec->el + dst_pos + n, res->size - dst_pos - n);
	}

	isl_vec_free(vec);
	return res;
}
