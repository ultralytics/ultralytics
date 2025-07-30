/*
 * Copyright 2015 INRIA Paris-Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Michael Kruse, INRIA Paris-Rocquencourt,
 * Domaine de Voluceau, Rocquenqourt, B.P. 105,
 * 78153 Le Chesnay Cedex France
 */

#include <limits.h>
#include <assert.h>
#include <isl_imath.h>

/* This constant is not defined in limits.h, but IMath uses it */
#define ULONG_MIN 0ul

/* Test the IMath internals assumed by the imath implementation of isl_int.
 *
 * In particular, we test the ranges of IMath-defined types.
 *
 * Also, isl uses the existence and function of imath's struct
 * fields. The digits are stored with less significant digits at lower array
 * indices. Where they are stored (on the heap or in the field 'single') does
 * not matter.
 */
int test_imath_internals()
{
	mpz_t val;
	mp_result retval;

	assert(sizeof(mp_small) == sizeof(long));
	assert(MP_SMALL_MIN == LONG_MIN);
	assert(MP_SMALL_MAX == LONG_MAX);

	assert(sizeof(mp_usmall) == sizeof(unsigned long));
	assert(MP_USMALL_MIN == ULONG_MIN);
	assert(MP_USMALL_MAX == ULONG_MAX);

	retval = mp_int_init_value(&val, 0);
	assert(retval == MP_OK);
	assert(val.alloc >= val.used);
	assert(val.used == 1);
	assert(val.sign == MP_ZPOS);
	assert(val.digits[0] == 0);

	retval = mp_int_set_value(&val, -1);
	assert(retval == MP_OK);
	assert(val.alloc >= val.used);
	assert(val.used == 1);
	assert(val.sign == MP_NEG);
	assert(val.digits[0] == 1);

	retval = mp_int_set_value(&val, 1);
	assert(retval == MP_OK);
	assert(val.alloc >= val.used);
	assert(val.used == 1);
	assert(val.sign == MP_ZPOS);
	assert(val.digits[0] == 1);

	retval = mp_int_mul_pow2(&val, sizeof(mp_digit) * CHAR_BIT, &val);
	assert(retval == MP_OK);
	assert(val.alloc >= val.used);
	assert(val.used == 2);
	assert(val.sign == MP_ZPOS);
	assert(val.digits[0] == 0);
	assert(val.digits[1] == 1);

	mp_int_clear(&val);
	return 0;
}

int main()
{
	if (test_imath_internals() < 0)
		return -1;

	return 0;
}
