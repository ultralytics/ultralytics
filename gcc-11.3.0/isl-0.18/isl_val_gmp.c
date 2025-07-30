#include <string.h>
#include <isl/val_gmp.h>
#include <isl_val_private.h>

/* Return a reference to an isl_val representing the integer "z".
 */
__isl_give isl_val *isl_val_int_from_gmp(isl_ctx *ctx, mpz_t z)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set(v->n, z);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Return a reference to an isl_val representing the rational value "n"/"d".
 */
__isl_give isl_val *isl_val_from_gmp(isl_ctx *ctx, const mpz_t n, const mpz_t d)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	isl_int_set(v->n, n);
	isl_int_set(v->d, d);

	return isl_val_normalize(v);
}

/* Extract the numerator of a rational value "v" in "z".
 *
 * If "v" is not a rational value, then the result is undefined.
 */
int isl_val_get_num_gmp(__isl_keep isl_val *v, mpz_t z)
{
	if (!v)
		return -1;
	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return -1);
	mpz_set(z, v->n);
	return 0;
}

/* Extract the denominator of a rational value "v" in "z".
 *
 * If "v" is not a rational value, then the result is undefined.
 */
int isl_val_get_den_gmp(__isl_keep isl_val *v, mpz_t z)
{
	if (!v)
		return -1;
	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return -1);
	mpz_set(z, v->d);
	return 0;
}

/* Return a reference to an isl_val representing the unsigned
 * integer value stored in the "n" chunks of size "size" at "chunks".
 * The least significant chunk is assumed to be stored first.
 */
__isl_give isl_val *isl_val_int_from_chunks(isl_ctx *ctx, size_t n,
	size_t size, const void *chunks)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	mpz_import(v->n, n, -1, size, 0, 0, chunks);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Return the number of chunks of size "size" required to
 * store the absolute value of the numerator of "v".
 */
size_t isl_val_n_abs_num_chunks(__isl_keep isl_val *v, size_t size)
{
	if (!v)
		return 0;

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return 0);

	size *= 8;
	return (mpz_sizeinbase(v->n, 2) + size - 1) / size;
}

/* Store a representation of the absolute value of the numerator of "v"
 * in terms of chunks of size "size" at "chunks".
 * The least significant chunk is stored first.
 * The number of chunks in the result can be obtained by calling
 * isl_val_n_abs_num_chunks.  The user is responsible for allocating
 * enough memory to store the results.
 *
 * In the special case of a zero value, isl_val_n_abs_num_chunks will
 * return one, while mpz_export will not fill in any chunks.  We therefore
 * do it ourselves.
 */
int isl_val_get_abs_num_chunks(__isl_keep isl_val *v, size_t size,
	void *chunks)
{
	if (!v || !chunks)
		return -1;

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return -1);

	mpz_export(chunks, NULL, -1, size, 0, 0, v->n);
	if (isl_val_is_zero(v))
		memset(chunks, 0, size);

	return 0;
}
