#include <isl_int.h>

uint32_t isl_imath_hash(mp_int v, uint32_t hash)
{
	unsigned const char *data = (unsigned char *)v->digits;
	unsigned const char *end = data + v->used * sizeof(v->digits[0]);

	if (v->sign == 1)
		isl_hash_byte(hash, 0xFF);
	for (; data < end; ++data)
		isl_hash_byte(hash, *data);
	return hash;
}

/* Try a standard conversion that fits into a long.
 */
int isl_imath_fits_slong_p(mp_int op)
{
	long out;
	mp_result res = mp_int_to_int(op, &out);
	return res == MP_OK;
}

/* Try a standard conversion that fits into an unsigned long.
 */
int isl_imath_fits_ulong_p(mp_int op)
{
	unsigned long out;
	mp_result res = mp_int_to_uint(op, &out);
	return res == MP_OK;
}

void isl_imath_addmul_ui(mp_int rop, mp_int op1, unsigned long op2)
{
	mpz_t temp;
	mp_int_init(&temp);

	mp_int_set_uvalue(&temp, op2);
	mp_int_mul(op1, &temp, &temp);
	mp_int_add(rop, &temp, rop);

	mp_int_clear(&temp);
}

void isl_imath_submul_ui(mp_int rop, mp_int op1, unsigned long op2)
{
	mpz_t temp;
	mp_int_init(&temp);

	mp_int_set_uvalue(&temp, op2);
	mp_int_mul(op1, &temp, &temp);
	mp_int_sub(rop, &temp, rop);

	mp_int_clear(&temp);
}
