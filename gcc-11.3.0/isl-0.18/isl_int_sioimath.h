/*
 * Copyright 2015 INRIA Paris-Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Michael Kruse, INRIA Paris-Rocquencourt,
 * Domaine de Voluceau, Rocquenqourt, B.P. 105,
 * 78153 Le Chesnay Cedex France
 */
#ifndef ISL_INT_SIOIMATH_H
#define ISL_INT_SIOIMATH_H

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include <isl_imath.h>
#include <isl/hash.h>

#define ARRAY_SIZE(array) (sizeof(array)/sizeof(*array))

/* Visual Studio before VS2015 does not support the inline keyword when
 * compiling in C mode because it was introduced in C99 which it does not
 * officially support.  Instead, it has a proprietary extension using __inline.
 */
#if defined(_MSC_VER) && (_MSC_VER < 1900)
#define inline __inline
#endif

/* The type to represent integers optimized for small values. It is either a
 * pointer to an mp_int ( = mpz_t*; big representation) or an int32_t (small
 * represenation) with a discriminator at the least significant bit. In big
 * representation it will be always zero because of heap alignment. It is set
 * to 1 for small representation and use the 32 most significant bits for the
 * int32_t.
 *
 * Structure on 64 bit machines, with 8-byte aligment (3 bits):
 *
 * Big representation:
 * MSB                                                          LSB
 * |------------------------------------------------------------000
 * |                            mpz_t*                            |
 * |                           != NULL                            |
 *
 * Small representation:
 * MSB                           32                             LSB
 * |------------------------------|00000000000000000000000000000001
 * |          int32_t             |
 * |  2147483647 ... -2147483647  |
 *                                                                ^
 *                                                                |
 *                                                        discriminator bit
 *
 * On 32 bit machines isl_sioimath type is blown up to 8 bytes, i.e.
 * isl_sioimath is guaranteed to be at least 8 bytes. This is to ensure the
 * int32_t can be hidden in that type without data loss. In the future we might
 * optimize this to use 31 hidden bits in a 32 bit pointer. We may also use 63
 * bits on 64 bit machines, but this comes with the cost of additional overflow
 * checks because there is no standardized 128 bit integer we could expand to.
 *
 * We use native integer types and avoid union structures to avoid assumptions
 * on the machine's endianness.
 *
 * This implementation makes the following assumptions:
 * - long can represent any int32_t
 * - mp_small is signed long
 * - mp_usmall is unsigned long
 * - adresses returned by malloc are aligned to 2-byte boundaries (leastmost
 *   bit is zero)
 */
#if UINT64_MAX > UINTPTR_MAX
typedef uint64_t isl_sioimath;
#else
typedef uintptr_t isl_sioimath;
#endif

/* The negation of the smallest possible number in int32_t, INT32_MIN
 * (0x80000000u, -2147483648), cannot be represented in an int32_t, therefore
 * every operation that may produce this value needs to special-case it.
 * The operations are:
 * abs(INT32_MIN)
 * -INT32_MIN   (negation)
 * -1 * INT32_MIN (multiplication)
 * INT32_MIN/-1 (any division: divexact, fdiv, cdiv, tdiv)
 * To avoid checking these cases, we exclude INT32_MIN from small
 * representation.
 */
#define ISL_SIOIMATH_SMALL_MIN (-INT32_MAX)

/* Largest possible number in small representation */
#define ISL_SIOIMATH_SMALL_MAX INT32_MAX

/* Used for function parameters the function modifies. */
typedef isl_sioimath *isl_sioimath_ptr;

/* Used for function parameters that are read-only. */
typedef isl_sioimath isl_sioimath_src;

/* Return whether the argument is stored in small representation.
 */
inline int isl_sioimath_is_small(isl_sioimath val)
{
	return val & 0x00000001;
}

/* Return whether the argument is stored in big representation.
 */
inline int isl_sioimath_is_big(isl_sioimath val)
{
	return !isl_sioimath_is_small(val);
}

/* Get the number of an isl_int in small representation. Result is undefined if
 * val is not stored in that format.
 */
inline int32_t isl_sioimath_get_small(isl_sioimath val)
{
	return val >> 32;
}

/* Get the number of an in isl_int in big representation. Result is undefined if
 * val is not stored in that format.
 */
inline mp_int isl_sioimath_get_big(isl_sioimath val)
{
	return (mp_int)(uintptr_t) val;
}

/* Return 1 if val is stored in small representation and store its value to
 * small. We rely on the compiler to optimize the isl_sioimath_get_small such
 * that the shift is moved into the branch that executes in case of small
 * representation. If there is no such branch, then a single shift is still
 * cheaper than introducing branching code.
 */
inline int isl_sioimath_decode_small(isl_sioimath val, int32_t *small)
{
	*small = isl_sioimath_get_small(val);
	return isl_sioimath_is_small(val);
}

/* Return 1 if val is stored in big representation and store its value to big.
 */
inline int isl_sioimath_decode_big(isl_sioimath val, mp_int *big)
{
	*big = isl_sioimath_get_big(val);
	return isl_sioimath_is_big(val);
}

/* Encode a small representation into an isl_int.
 */
inline isl_sioimath isl_sioimath_encode_small(int32_t val)
{
	return ((isl_sioimath) val) << 32 | 0x00000001;
}

/* Encode a big representation.
 */
inline isl_sioimath isl_sioimath_encode_big(mp_int val)
{
	return (isl_sioimath)(uintptr_t) val;
}

/* A common situation is to call an IMath function with at least one argument
 * that is currently in small representation or an integer parameter, i.e. a big
 * representation of the same number is required. Promoting the original
 * argument comes with multiple problems, such as modifying a read-only
 * argument, the responsibility of deallocation and the execution cost. Instead,
 * we make a copy by 'faking' the IMath internal structure.
 *
 * We reserve the maximum number of required digits on the stack to avoid heap
 * allocations.
 *
 * mp_digit can be uint32_t or uint16_t. This code must work for little and big
 * endian digits. The structure for an uint64_t argument and 32-bit mp_digits is
 * sketched below.
 *
 * |----------------------------|
 *            uint64_t
 *
 * |-------------||-------------|
 *    mp_digit        mp_digit
 *    digits[1]       digits[0]
 * Most sig digit  Least sig digit
 */
typedef struct {
	mpz_t big;
	mp_digit digits[(sizeof(uintmax_t) + sizeof(mp_digit) - 1) /
	                sizeof(mp_digit)];
} isl_sioimath_scratchspace_t;

/* Convert a native integer to IMath's digit representation. A native integer
 * might be big- or little endian, but IMath always stores the least significant
 * digit in the lowest array indices.  memcpy therefore is not possible.
 *
 * We also have to consider that long and mp_digit can be of different sizes,
 * depending on the compiler (LP64, LLP64) and IMath's USE_64BIT_WORDS. This
 * macro should work for all of them.
 *
 * "used" is set to the number of written digits. It must be minimal (IMath
 * checks zeroness using the used field), but always at least one.  Also note
 * that the result of num>>(sizeof(num)*CHAR_BIT) is undefined.
 */
#define ISL_SIOIMATH_TO_DIGITS(num, digits, used)                              \
	do {                                                                   \
		int i = 0;                                                     \
		do {                                                           \
			(digits)[i] =                                          \
			    ((num) >> (sizeof(mp_digit) * CHAR_BIT * i));      \
			i += 1;                                                \
			if (i >= (sizeof(num) + sizeof(mp_digit) - 1) /        \
			             sizeof(mp_digit))                         \
				break;                                         \
			if (((num) >> (sizeof(mp_digit) * CHAR_BIT * i)) == 0) \
				break;                                         \
		} while (1);                                                   \
		(used) = i;                                                    \
	} while (0)

inline void isl_siomath_uint32_to_digits(uint32_t num, mp_digit *digits,
	mp_size *used)
{
	ISL_SIOIMATH_TO_DIGITS(num, digits, *used);
}

inline void isl_siomath_ulong_to_digits(unsigned long num, mp_digit *digits,
	mp_size *used)
{
	ISL_SIOIMATH_TO_DIGITS(num, digits, *used);
}

inline void isl_siomath_uint64_to_digits(uint64_t num, mp_digit *digits,
	mp_size *used)
{
	ISL_SIOIMATH_TO_DIGITS(num, digits, *used);
}

/* Get the IMath representation of an isl_int without modifying it.
 * For the case it is not in big representation yet, pass some scratch space we
 * can use to store the big representation in.
 * In order to avoid requiring init and free on the scratch space, we directly
 * modify the internal representation.
 *
 * The name derives from its indented use: getting the big representation of an
 * input (src) argument.
 */
inline mp_int isl_sioimath_bigarg_src(isl_sioimath arg,
	isl_sioimath_scratchspace_t *scratch)
{
	mp_int big;
	int32_t small;
	uint32_t num;

	if (isl_sioimath_decode_big(arg, &big))
		return big;

	small = isl_sioimath_get_small(arg);
	scratch->big.digits = scratch->digits;
	scratch->big.alloc = ARRAY_SIZE(scratch->digits);
	if (small >= 0) {
		scratch->big.sign = MP_ZPOS;
		num = small;
	} else {
		scratch->big.sign = MP_NEG;
		num = -small;
	}

	isl_siomath_uint32_to_digits(num, scratch->digits, &scratch->big.used);
	return &scratch->big;
}

/* Create a temporary IMath mp_int for a signed long.
 */
inline mp_int isl_sioimath_siarg_src(signed long arg,
	isl_sioimath_scratchspace_t *scratch)
{
	unsigned long num;

	scratch->big.digits = scratch->digits;
	scratch->big.alloc = ARRAY_SIZE(scratch->digits);
	if (arg >= 0) {
		scratch->big.sign = MP_ZPOS;
		num = arg;
	} else {
		scratch->big.sign = MP_NEG;
		num = (arg == LONG_MIN) ? ((unsigned long) LONG_MAX) + 1 : -arg;
	}

	isl_siomath_ulong_to_digits(num, scratch->digits, &scratch->big.used);
	return &scratch->big;
}

/* Create a temporary IMath mp_int for an int64_t.
 */
inline mp_int isl_sioimath_si64arg_src(int64_t arg,
	isl_sioimath_scratchspace_t *scratch)
{
	uint64_t num;

	scratch->big.digits = scratch->digits;
	scratch->big.alloc = ARRAY_SIZE(scratch->digits);
	if (arg >= 0) {
		scratch->big.sign = MP_ZPOS;
		num = arg;
	} else {
		scratch->big.sign = MP_NEG;
		num = (arg == INT64_MIN) ? ((uint64_t) INT64_MAX) + 1 : -arg;
	}

	isl_siomath_uint64_to_digits(num, scratch->digits, &scratch->big.used);
	return &scratch->big;
}

/* Create a temporary IMath mp_int for an unsigned long.
 */
inline mp_int isl_sioimath_uiarg_src(unsigned long arg,
	isl_sioimath_scratchspace_t *scratch)
{
	scratch->big.digits = scratch->digits;
	scratch->big.alloc = ARRAY_SIZE(scratch->digits);
	scratch->big.sign = MP_ZPOS;

	isl_siomath_ulong_to_digits(arg, scratch->digits, &scratch->big.used);
	return &scratch->big;
}

/* Ensure big representation. Does not preserve the current number.
 * Callers may use the fact that the value _is_ preserved if the presentation
 * was big before.
 */
inline mp_int isl_sioimath_reinit_big(isl_sioimath_ptr ptr)
{
	if (isl_sioimath_is_small(*ptr))
		*ptr = isl_sioimath_encode_big(mp_int_alloc());
	return isl_sioimath_get_big(*ptr);
}

/* Set ptr to a number in small representation.
 */
inline void isl_sioimath_set_small(isl_sioimath_ptr ptr, int32_t val)
{
	if (isl_sioimath_is_big(*ptr))
		mp_int_free(isl_sioimath_get_big(*ptr));
	*ptr = isl_sioimath_encode_small(val);
}

/* Set ptr to val, choosing small representation if possible.
 */
inline void isl_sioimath_set_int32(isl_sioimath_ptr ptr, int32_t val)
{
	if (ISL_SIOIMATH_SMALL_MIN <= val && val <= ISL_SIOIMATH_SMALL_MAX) {
		isl_sioimath_set_small(ptr, val);
		return;
	}

	mp_int_init_value(isl_sioimath_reinit_big(ptr), val);
}

/* Assign an int64_t number using small representation if possible.
 */
inline void isl_sioimath_set_int64(isl_sioimath_ptr ptr, int64_t val)
{
	if (ISL_SIOIMATH_SMALL_MIN <= val && val <= ISL_SIOIMATH_SMALL_MAX) {
		isl_sioimath_set_small(ptr, val);
		return;
	}

	isl_sioimath_scratchspace_t scratch;
	mp_int_copy(isl_sioimath_si64arg_src(val, &scratch),
	    isl_sioimath_reinit_big(ptr));
}

/* Convert to big representation while preserving the current number.
 */
inline void isl_sioimath_promote(isl_sioimath_ptr dst)
{
	int32_t small;

	if (isl_sioimath_is_big(*dst))
		return;

	small = isl_sioimath_get_small(*dst);
	mp_int_set_value(isl_sioimath_reinit_big(dst), small);
}

/* Convert to small representation while preserving the current number. Does
 * nothing if dst doesn't fit small representation.
 */
inline void isl_sioimath_try_demote(isl_sioimath_ptr dst)
{
	mp_small small;

	if (isl_sioimath_is_small(*dst))
		return;

	if (mp_int_to_int(isl_sioimath_get_big(*dst), &small) != MP_OK)
		return;

	if (ISL_SIOIMATH_SMALL_MIN <= small && small <= ISL_SIOIMATH_SMALL_MAX)
		isl_sioimath_set_small(dst, small);
}

/* Initialize an isl_int. The implicit value is 0 in small representation.
 */
inline void isl_sioimath_init(isl_sioimath_ptr dst)
{
	*dst = isl_sioimath_encode_small(0);
}

/* Free the resources taken by an isl_int.
 */
inline void isl_sioimath_clear(isl_sioimath_ptr dst)
{
	if (isl_sioimath_is_small(*dst))
		return;

	mp_int_free(isl_sioimath_get_big(*dst));
}

/* Copy the value of one isl_int to another.
 */
inline void isl_sioimath_set(isl_sioimath_ptr dst, isl_sioimath_src val)
{
	if (isl_sioimath_is_small(val)) {
		isl_sioimath_set_small(dst, isl_sioimath_get_small(val));
		return;
	}

	mp_int_copy(isl_sioimath_get_big(val), isl_sioimath_reinit_big(dst));
}

/* Store a signed long into an isl_int.
 */
inline void isl_sioimath_set_si(isl_sioimath_ptr dst, long val)
{
	if (ISL_SIOIMATH_SMALL_MIN <= val && val <= ISL_SIOIMATH_SMALL_MAX) {
		isl_sioimath_set_small(dst, val);
		return;
	}

	mp_int_set_value(isl_sioimath_reinit_big(dst), val);
}

/* Store an unsigned long into an isl_int.
 */
inline void isl_sioimath_set_ui(isl_sioimath_ptr dst, unsigned long val)
{
	if (val <= ISL_SIOIMATH_SMALL_MAX) {
		isl_sioimath_set_small(dst, val);
		return;
	}

	mp_int_set_uvalue(isl_sioimath_reinit_big(dst), val);
}

/* Return whether a number can be represented by a signed long.
 */
inline int isl_sioimath_fits_slong(isl_sioimath_src val)
{
	mp_small dummy;

	if (isl_sioimath_is_small(val))
		return 1;

	return mp_int_to_int(isl_sioimath_get_big(val), &dummy) == MP_OK;
}

/* Return a number as signed long. Result is undefined if the number cannot be
 * represented as long.
 */
inline long isl_sioimath_get_si(isl_sioimath_src val)
{
	mp_small result;

	if (isl_sioimath_is_small(val))
		return isl_sioimath_get_small(val);

	mp_int_to_int(isl_sioimath_get_big(val), &result);
	return result;
}

/* Return whether a number can be represented as unsigned long.
 */
inline int isl_sioimath_fits_ulong(isl_sioimath_src val)
{
	mp_usmall dummy;

	if (isl_sioimath_is_small(val))
		return isl_sioimath_get_small(val) >= 0;

	return mp_int_to_uint(isl_sioimath_get_big(val), &dummy) == MP_OK;
}

/* Return a number as unsigned long. Result is undefined if the number cannot be
 * represented as unsigned long.
 */
inline unsigned long isl_sioimath_get_ui(isl_sioimath_src val)
{
	mp_usmall result;

	if (isl_sioimath_is_small(val))
		return isl_sioimath_get_small(val);

	mp_int_to_uint(isl_sioimath_get_big(val), &result);
	return result;
}

/* Return a number as floating point value.
 */
inline double isl_sioimath_get_d(isl_sioimath_src val)
{
	mp_int big;
	double result = 0;
	int i;

	if (isl_sioimath_is_small(val))
		return isl_sioimath_get_small(val);

	big = isl_sioimath_get_big(val);
	for (i = 0; i < big->used; ++i)
		result = result * (double) ((uintmax_t) MP_DIGIT_MAX + 1) +
		         (double) big->digits[i];

	if (big->sign == MP_NEG)
		result = -result;

	return result;
}

/* Format a number as decimal string.
 *
 * The largest possible string from small representation is 12 characters
 * ("-2147483647").
 */
inline char *isl_sioimath_get_str(isl_sioimath_src val)
{
	char *result;

	if (isl_sioimath_is_small(val)) {
		result = malloc(12);
		snprintf(result, 12, "%" PRIi32, isl_sioimath_get_small(val));
		return result;
	}

	return impz_get_str(NULL, 10, isl_sioimath_get_big(val));
}

/* Return the absolute value.
 */
inline void isl_sioimath_abs(isl_sioimath_ptr dst, isl_sioimath_src arg)
{
	if (isl_sioimath_is_small(arg)) {
		isl_sioimath_set_small(dst, labs(isl_sioimath_get_small(arg)));
		return;
	}

	mp_int_abs(isl_sioimath_get_big(arg), isl_sioimath_reinit_big(dst));
}

/* Return the negation of a number.
 */
inline void isl_sioimath_neg(isl_sioimath_ptr dst, isl_sioimath_src arg)
{
	if (isl_sioimath_is_small(arg)) {
		isl_sioimath_set_small(dst, -isl_sioimath_get_small(arg));
		return;
	}

	mp_int_neg(isl_sioimath_get_big(arg), isl_sioimath_reinit_big(dst));
}

/* Swap two isl_ints.
 *
 * isl_sioimath can be copied bytewise; nothing depends on its address. It can
 * also be stored in a CPU register.
 */
inline void isl_sioimath_swap(isl_sioimath_ptr lhs, isl_sioimath_ptr rhs)
{
	isl_sioimath tmp = *lhs;
	*lhs = *rhs;
	*rhs = tmp;
}

/* Add an unsigned long to the number.
 *
 * On LP64 unsigned long exceeds the range of an int64_t, therefore we check in
 * advance whether small representation possibly overflows.
 */
inline void isl_sioimath_add_ui(isl_sioimath_ptr dst, isl_sioimath lhs,
	unsigned long rhs)
{
	int32_t smalllhs;
	isl_sioimath_scratchspace_t lhsscratch;

	if (isl_sioimath_decode_small(lhs, &smalllhs) &&
	    (rhs <= (uint64_t) INT64_MAX - (uint64_t) ISL_SIOIMATH_SMALL_MAX)) {
		isl_sioimath_set_int64(dst, (int64_t) smalllhs + rhs);
		return;
	}

	impz_add_ui(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &lhsscratch), rhs);
	isl_sioimath_try_demote(dst);
}

/* Subtract an unsigned long.
 *
 * On LP64 unsigned long exceeds the range of an int64_t.  If
 * ISL_SIOIMATH_SMALL_MIN-rhs>=INT64_MIN we can do the calculation using int64_t
 * without risking an overflow.
 */
inline void isl_sioimath_sub_ui(isl_sioimath_ptr dst, isl_sioimath lhs,
				unsigned long rhs)
{
	int32_t smalllhs;
	isl_sioimath_scratchspace_t lhsscratch;

	if (isl_sioimath_decode_small(lhs, &smalllhs) &&
	    (rhs < (uint64_t) INT64_MIN - (uint64_t) ISL_SIOIMATH_SMALL_MIN)) {
		isl_sioimath_set_int64(dst, (int64_t) smalllhs - rhs);
		return;
	}

	impz_sub_ui(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &lhsscratch), rhs);
	isl_sioimath_try_demote(dst);
}

/* Sum of two isl_ints.
 */
inline void isl_sioimath_add(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;
	int32_t smalllhs, smallrhs;

	if (isl_sioimath_decode_small(lhs, &smalllhs) &&
	    isl_sioimath_decode_small(rhs, &smallrhs)) {
		isl_sioimath_set_int64(
		    dst, (int64_t) smalllhs + (int64_t) smallrhs);
		return;
	}

	mp_int_add(isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_bigarg_src(rhs, &scratchrhs),
	    isl_sioimath_reinit_big(dst));
	isl_sioimath_try_demote(dst);
}

/* Subtract two isl_ints.
 */
inline void isl_sioimath_sub(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;
	int32_t smalllhs, smallrhs;

	if (isl_sioimath_decode_small(lhs, &smalllhs) &&
	    isl_sioimath_decode_small(rhs, &smallrhs)) {
		isl_sioimath_set_int64(
		    dst, (int64_t) smalllhs - (int64_t) smallrhs);
		return;
	}

	mp_int_sub(isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_bigarg_src(rhs, &scratchrhs),
	    isl_sioimath_reinit_big(dst));
	isl_sioimath_try_demote(dst);
}

/* Multiply two isl_ints.
 */
inline void isl_sioimath_mul(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;
	int32_t smalllhs, smallrhs;

	if (isl_sioimath_decode_small(lhs, &smalllhs) &&
	    isl_sioimath_decode_small(rhs, &smallrhs)) {
		isl_sioimath_set_int64(
		    dst, (int64_t) smalllhs * (int64_t) smallrhs);
		return;
	}

	mp_int_mul(isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_bigarg_src(rhs, &scratchrhs),
	    isl_sioimath_reinit_big(dst));
	isl_sioimath_try_demote(dst);
}

/* Shift lhs by rhs bits to the left and store the result in dst. Effectively,
 * this operation computes 'lhs * 2^rhs'.
 */
inline void isl_sioimath_mul_2exp(isl_sioimath_ptr dst, isl_sioimath lhs,
	unsigned long rhs)
{
	isl_sioimath_scratchspace_t scratchlhs;
	int32_t smalllhs;

	if (isl_sioimath_decode_small(lhs, &smalllhs) && (rhs <= 32ul)) {
		isl_sioimath_set_int64(dst, ((int64_t) smalllhs) << rhs);
		return;
	}

	mp_int_mul_pow2(isl_sioimath_bigarg_src(lhs, &scratchlhs), rhs,
	    isl_sioimath_reinit_big(dst));
}

/* Multiply an isl_int and a signed long.
 */
inline void isl_sioimath_mul_si(isl_sioimath_ptr dst, isl_sioimath lhs,
	signed long rhs)
{
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;
	int32_t smalllhs;

	if (isl_sioimath_decode_small(lhs, &smalllhs) && (rhs > LONG_MIN) &&
	    (labs(rhs) <= UINT32_MAX)) {
		isl_sioimath_set_int64(dst, (int64_t) smalllhs * (int64_t) rhs);
		return;
	}

	mp_int_mul(isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_siarg_src(rhs, &scratchrhs),
	    isl_sioimath_reinit_big(dst));
	isl_sioimath_try_demote(dst);
}

/* Multiply an isl_int and an unsigned long.
 */
inline void isl_sioimath_mul_ui(isl_sioimath_ptr dst, isl_sioimath lhs,
	unsigned long rhs)
{
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;
	int32_t smalllhs;

	if (isl_sioimath_decode_small(lhs, &smalllhs) && (rhs <= UINT32_MAX)) {
		isl_sioimath_set_int64(dst, (int64_t) smalllhs * (int64_t) rhs);
		return;
	}

	mp_int_mul(isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_uiarg_src(rhs, &scratchrhs),
	    isl_sioimath_reinit_big(dst));
	isl_sioimath_try_demote(dst);
}

/* Compute the power of an isl_int to an unsigned long.
 * Always let IMath do it; the result is unlikely to be small except in some
 * special cases.
 * Note: 0^0 == 1
 */
inline void isl_sioimath_pow_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs)
{
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;
	int32_t smalllhs;

	switch (rhs) {
	case 0:
		isl_sioimath_set_small(dst, 1);
		return;
	case 1:
		isl_sioimath_set(dst, lhs);
		return;
	case 2:
		isl_sioimath_mul(dst, lhs, lhs);
		return;
	}

	if (isl_sioimath_decode_small(lhs, &smalllhs)) {
		switch (smalllhs) {
		case 0:
			isl_sioimath_set_small(dst, 0);
			return;
		case 1:
			isl_sioimath_set_small(dst, 1);
			return;
		case 2:
			isl_sioimath_set_small(dst, 1);
			isl_sioimath_mul_2exp(dst, *dst, rhs);
			return;
		default:
			if ((MP_SMALL_MIN <= rhs) && (rhs <= MP_SMALL_MAX)) {
				mp_int_expt_value(smalllhs, rhs,
				    isl_sioimath_reinit_big(dst));
				isl_sioimath_try_demote(dst);
				return;
			}
		}
	}

	mp_int_expt_full(isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_uiarg_src(rhs, &scratchrhs),
	    isl_sioimath_reinit_big(dst));
	isl_sioimath_try_demote(dst);
}

/* Fused multiply-add.
 */
inline void isl_sioimath_addmul(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath tmp;
	isl_sioimath_init(&tmp);
	isl_sioimath_mul(&tmp, lhs, rhs);
	isl_sioimath_add(dst, *dst, tmp);
	isl_sioimath_clear(&tmp);
}

/* Fused multiply-add with an unsigned long.
 */
inline void isl_sioimath_addmul_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs)
{
	isl_sioimath tmp;
	isl_sioimath_init(&tmp);
	isl_sioimath_mul_ui(&tmp, lhs, rhs);
	isl_sioimath_add(dst, *dst, tmp);
	isl_sioimath_clear(&tmp);
}

/* Fused multiply-subtract.
 */
inline void isl_sioimath_submul(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath tmp;
	isl_sioimath_init(&tmp);
	isl_sioimath_mul(&tmp, lhs, rhs);
	isl_sioimath_sub(dst, *dst, tmp);
	isl_sioimath_clear(&tmp);
}

/* Fused multiply-add with an unsigned long.
 */
inline void isl_sioimath_submul_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs)
{
	isl_sioimath tmp;
	isl_sioimath_init(&tmp);
	isl_sioimath_mul_ui(&tmp, lhs, rhs);
	isl_sioimath_sub(dst, *dst, tmp);
	isl_sioimath_clear(&tmp);
}

void isl_sioimath_gcd(isl_sioimath_ptr dst, isl_sioimath_src lhs,
		      isl_sioimath_src rhs);
void isl_sioimath_lcm(isl_sioimath_ptr dst, isl_sioimath_src lhs,
		      isl_sioimath_src rhs);

/* Divide lhs by rhs, rounding to zero (Truncate).
 */
inline void isl_sioimath_tdiv_q(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t lhssmall, rhssmall;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall)) {
		isl_sioimath_set_small(dst, lhssmall / rhssmall);
		return;
	}

	mp_int_div(isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_bigarg_src(rhs, &rhsscratch),
	    isl_sioimath_reinit_big(dst), NULL);
	isl_sioimath_try_demote(dst);
	return;
}

/* Divide lhs by an unsigned long rhs, rounding to zero (Truncate).
 */
inline void isl_sioimath_tdiv_q_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t lhssmall;

	if (isl_sioimath_is_small(lhs) && (rhs <= (unsigned long) INT32_MAX)) {
		lhssmall = isl_sioimath_get_small(lhs);
		isl_sioimath_set_small(dst, lhssmall / (int32_t) rhs);
		return;
	}

	if (rhs <= MP_SMALL_MAX) {
		mp_int_div_value(isl_sioimath_bigarg_src(lhs, &lhsscratch), rhs,
		    isl_sioimath_reinit_big(dst), NULL);
		isl_sioimath_try_demote(dst);
		return;
	}

	mp_int_div(isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_uiarg_src(rhs, &rhsscratch),
	    isl_sioimath_reinit_big(dst), NULL);
	isl_sioimath_try_demote(dst);
}

/* Divide lhs by rhs, rounding to positive infinity (Ceil).
 */
inline void isl_sioimath_cdiv_q(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	int32_t lhssmall, rhssmall;
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t q;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall)) {
		if ((lhssmall >= 0) && (rhssmall >= 0))
			q = ((int64_t) lhssmall + (int64_t) rhssmall - 1) /
			    rhssmall;
		else if ((lhssmall < 0) && (rhssmall < 0))
			q = ((int64_t) lhssmall + (int64_t) rhssmall + 1) /
			    rhssmall;
		else
			q = lhssmall / rhssmall;
		isl_sioimath_set_small(dst, q);
		return;
	}

	impz_cdiv_q(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_bigarg_src(rhs, &rhsscratch));
	isl_sioimath_try_demote(dst);
}

/* Divide lhs by rhs, rounding to negative infinity (Floor).
 */
inline void isl_sioimath_fdiv_q(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t lhssmall, rhssmall;
	int32_t q;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall)) {
		if ((lhssmall < 0) && (rhssmall >= 0))
			q = ((int64_t) lhssmall - ((int64_t) rhssmall - 1)) /
			    rhssmall;
		else if ((lhssmall >= 0) && (rhssmall < 0))
			q = ((int64_t) lhssmall - ((int64_t) rhssmall + 1)) /
			    rhssmall;
		else
			q = lhssmall / rhssmall;
		isl_sioimath_set_small(dst, q);
		return;
	}

	impz_fdiv_q(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_bigarg_src(rhs, &rhsscratch));
	isl_sioimath_try_demote(dst);
}

/* Compute the division of lhs by a rhs of type unsigned long, rounding towards
 * negative infinity (Floor).
 */
inline void isl_sioimath_fdiv_q_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t lhssmall, q;

	if (isl_sioimath_decode_small(lhs, &lhssmall) && (rhs <= INT32_MAX)) {
		if (lhssmall >= 0)
			q = (uint32_t) lhssmall / rhs;
		else
			q = ((int64_t) lhssmall - ((int64_t) rhs - 1)) /
			    (int64_t) rhs;
		isl_sioimath_set_small(dst, q);
		return;
	}

	impz_fdiv_q(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_uiarg_src(rhs, &rhsscratch));
	isl_sioimath_try_demote(dst);
}

/* Get the remainder of: lhs divided by rhs rounded towards negative infinite
 * (Floor).
 */
inline void isl_sioimath_fdiv_r(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int64_t lhssmall, rhssmall;
	int32_t r;

	if (isl_sioimath_is_small(lhs) && isl_sioimath_is_small(rhs)) {
		lhssmall = isl_sioimath_get_small(lhs);
		rhssmall = isl_sioimath_get_small(rhs);
		r = (rhssmall + lhssmall % rhssmall) % rhssmall;
		isl_sioimath_set_small(dst, r);
		return;
	}

	impz_fdiv_r(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_bigarg_src(rhs, &rhsscratch));
	isl_sioimath_try_demote(dst);
}

void isl_sioimath_read(isl_sioimath_ptr dst, const char *str);

/* Return:
 *   +1 for a positive number
 *   -1 for a negative number
 *    0 if the number is zero
 */
inline int isl_sioimath_sgn(isl_sioimath_src arg)
{
	int32_t small;

	if (isl_sioimath_decode_small(arg, &small))
		return (small > 0) - (small < 0);

	return mp_int_compare_zero(isl_sioimath_get_big(arg));
}

/* Return:
 *   +1 if lhs > rhs
 *   -1 if lhs < rhs
 *    0 if lhs = rhs
 */
inline int isl_sioimath_cmp(isl_sioimath_src lhs, isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t lhssmall, rhssmall;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall))
		return (lhssmall > rhssmall) - (lhssmall < rhssmall);

	if (isl_sioimath_decode_small(rhs, &rhssmall))
		return mp_int_compare_value(
		    isl_sioimath_bigarg_src(lhs, &lhsscratch), rhssmall);

	if (isl_sioimath_decode_small(lhs, &lhssmall))
		return -mp_int_compare_value(
		           isl_sioimath_bigarg_src(rhs, &rhsscratch), lhssmall);

	return mp_int_compare(
	    isl_sioimath_get_big(lhs), isl_sioimath_get_big(rhs));
}

/* As isl_sioimath_cmp, but with signed long rhs.
 */
inline int isl_sioimath_cmp_si(isl_sioimath_src lhs, signed long rhs)
{
	int32_t lhssmall;

	if (isl_sioimath_decode_small(lhs, &lhssmall))
		return (lhssmall > rhs) - (lhssmall < rhs);

	return mp_int_compare_value(isl_sioimath_get_big(lhs), rhs);
}

/* Return:
 *   +1 if |lhs| > |rhs|
 *   -1 if |lhs| < |rhs|
 *    0 if |lhs| = |rhs|
 */
inline int isl_sioimath_abs_cmp(isl_sioimath_src lhs, isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t lhssmall, rhssmall;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall)) {
		lhssmall = labs(lhssmall);
		rhssmall = labs(rhssmall);
		return (lhssmall > rhssmall) - (lhssmall < rhssmall);
	}

	return mp_int_compare_unsigned(
	    isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_bigarg_src(rhs, &rhsscratch));
}

/* Return whether lhs is divisible by rhs.
 */
inline int isl_sioimath_is_divisible_by(isl_sioimath_src lhs,
					isl_sioimath_src rhs)
{
	isl_sioimath_scratchspace_t lhsscratch, rhsscratch;
	int32_t lhssmall, rhssmall;
	mpz_t rem;
	int cmp;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall))
		return lhssmall % rhssmall == 0;

	if (isl_sioimath_decode_small(rhs, &rhssmall))
		return mp_int_divisible_value(
		    isl_sioimath_bigarg_src(lhs, &lhsscratch), rhssmall);

	mp_int_init(&rem);
	mp_int_div(isl_sioimath_bigarg_src(lhs, &lhsscratch),
	    isl_sioimath_bigarg_src(rhs, &rhsscratch), NULL, &rem);
	cmp = mp_int_compare_zero(&rem);
	mp_int_clear(&rem);
	return cmp == 0;
}

/* Return a hash code of an isl_sioimath.
 * The hash code for a number in small and big representation must be identical
 * on the same machine because small representation if not obligatory if fits.
 */
inline uint32_t isl_sioimath_hash(isl_sioimath_src arg, uint32_t hash)
{
	int32_t small;
	int i;
	uint32_t num;
	mp_digit digits[(sizeof(uint32_t) + sizeof(mp_digit) - 1) /
	                sizeof(mp_digit)];
	mp_size used;
	const unsigned char *digitdata = (const unsigned char *) &digits;

	if (isl_sioimath_decode_small(arg, &small)) {
		if (small < 0)
			isl_hash_byte(hash, 0xFF);
		num = labs(small);

		isl_siomath_uint32_to_digits(num, digits, &used);
		for (i = 0; i < used * sizeof(mp_digit); i += 1)
			isl_hash_byte(hash, digitdata[i]);
		return hash;
	}

	return isl_imath_hash(isl_sioimath_get_big(arg), hash);
}

/* Return the number of digits in a number of the given base or more, i.e. the
 * string length without sign and null terminator.
 *
 * Current implementation for small representation returns the maximal number
 * of binary digits in that representation, which can be much larger than the
 * smallest possible solution.
 */
inline size_t isl_sioimath_sizeinbase(isl_sioimath_src arg, int base)
{
	int32_t small;

	if (isl_sioimath_decode_small(arg, &small))
		return sizeof(int32_t) * CHAR_BIT - 1;

	return impz_sizeinbase(isl_sioimath_get_big(arg), base);
}

void isl_sioimath_print(FILE *out, isl_sioimath_src i, int width);
void isl_sioimath_dump(isl_sioimath_src arg);

typedef isl_sioimath isl_int[1];
#define isl_int_init(i)			isl_sioimath_init((i))
#define isl_int_clear(i)		isl_sioimath_clear((i))

#define isl_int_set(r, i)		isl_sioimath_set((r), *(i))
#define isl_int_set_si(r, i)		isl_sioimath_set_si((r), i)
#define isl_int_set_ui(r, i)		isl_sioimath_set_ui((r), i)
#define isl_int_fits_slong(r)		isl_sioimath_fits_slong(*(r))
#define isl_int_get_si(r)		isl_sioimath_get_si(*(r))
#define isl_int_fits_ulong(r)		isl_sioimath_fits_ulong(*(r))
#define isl_int_get_ui(r)		isl_sioimath_get_ui(*(r))
#define isl_int_get_d(r)		isl_sioimath_get_d(*(r))
#define isl_int_get_str(r)		isl_sioimath_get_str(*(r))
#define isl_int_abs(r, i)		isl_sioimath_abs((r), *(i))
#define isl_int_neg(r, i)		isl_sioimath_neg((r), *(i))
#define isl_int_swap(i, j)		isl_sioimath_swap((i), (j))
#define isl_int_swap_or_set(i, j)	isl_sioimath_swap((i), (j))
#define isl_int_add_ui(r, i, j)		isl_sioimath_add_ui((r), *(i), j)
#define isl_int_sub_ui(r, i, j)		isl_sioimath_sub_ui((r), *(i), j)

#define isl_int_add(r, i, j)		isl_sioimath_add((r), *(i), *(j))
#define isl_int_sub(r, i, j)		isl_sioimath_sub((r), *(i), *(j))
#define isl_int_mul(r, i, j)		isl_sioimath_mul((r), *(i), *(j))
#define isl_int_mul_2exp(r, i, j)	isl_sioimath_mul_2exp((r), *(i), j)
#define isl_int_mul_si(r, i, j)		isl_sioimath_mul_si((r), *(i), j)
#define isl_int_mul_ui(r, i, j)		isl_sioimath_mul_ui((r), *(i), j)
#define isl_int_pow_ui(r, i, j)		isl_sioimath_pow_ui((r), *(i), j)
#define isl_int_addmul(r, i, j)		isl_sioimath_addmul((r), *(i), *(j))
#define isl_int_addmul_ui(r, i, j)	isl_sioimath_addmul_ui((r), *(i), j)
#define isl_int_submul(r, i, j)		isl_sioimath_submul((r), *(i), *(j))
#define isl_int_submul_ui(r, i, j)	isl_sioimath_submul_ui((r), *(i), j)

#define isl_int_gcd(r, i, j)		isl_sioimath_gcd((r), *(i), *(j))
#define isl_int_lcm(r, i, j)		isl_sioimath_lcm((r), *(i), *(j))
#define isl_int_divexact(r, i, j)	isl_sioimath_tdiv_q((r), *(i), *(j))
#define isl_int_divexact_ui(r, i, j)	isl_sioimath_tdiv_q_ui((r), *(i), j)
#define isl_int_tdiv_q(r, i, j)		isl_sioimath_tdiv_q((r), *(i), *(j))
#define isl_int_cdiv_q(r, i, j)		isl_sioimath_cdiv_q((r), *(i), *(j))
#define isl_int_fdiv_q(r, i, j)		isl_sioimath_fdiv_q((r), *(i), *(j))
#define isl_int_fdiv_r(r, i, j)		isl_sioimath_fdiv_r((r), *(i), *(j))
#define isl_int_fdiv_q_ui(r, i, j)	isl_sioimath_fdiv_q_ui((r), *(i), j)

#define isl_int_read(r, s)		isl_sioimath_read((r), s)
#define isl_int_sgn(i)			isl_sioimath_sgn(*(i))
#define isl_int_cmp(i, j)		isl_sioimath_cmp(*(i), *(j))
#define isl_int_cmp_si(i, si)		isl_sioimath_cmp_si(*(i), si)
#define isl_int_eq(i, j)		(isl_sioimath_cmp(*(i), *(j)) == 0)
#define isl_int_ne(i, j)		(isl_sioimath_cmp(*(i), *(j)) != 0)
#define isl_int_lt(i, j)		(isl_sioimath_cmp(*(i), *(j)) < 0)
#define isl_int_le(i, j)		(isl_sioimath_cmp(*(i), *(j)) <= 0)
#define isl_int_gt(i, j)		(isl_sioimath_cmp(*(i), *(j)) > 0)
#define isl_int_ge(i, j)		(isl_sioimath_cmp(*(i), *(j)) >= 0)
#define isl_int_abs_cmp(i, j)		isl_sioimath_abs_cmp(*(i), *(j))
#define isl_int_abs_eq(i, j)		(isl_sioimath_abs_cmp(*(i), *(j)) == 0)
#define isl_int_abs_ne(i, j)		(isl_sioimath_abs_cmp(*(i), *(j)) != 0)
#define isl_int_abs_lt(i, j)		(isl_sioimath_abs_cmp(*(i), *(j)) < 0)
#define isl_int_abs_gt(i, j)		(isl_sioimath_abs_cmp(*(i), *(j)) > 0)
#define isl_int_abs_ge(i, j)		(isl_sioimath_abs_cmp(*(i), *(j)) >= 0)
#define isl_int_is_divisible_by(i, j)	isl_sioimath_is_divisible_by(*(i), *(j))

#define isl_int_hash(v, h)		isl_sioimath_hash(*(v), h)
#define isl_int_free_str(s)		free(s)
#define isl_int_print(out, i, width)	isl_sioimath_print(out, *(i), width)

#endif /* ISL_INT_SIOIMATH_H */
