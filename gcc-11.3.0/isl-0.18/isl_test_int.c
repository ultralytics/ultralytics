/*
 * Copyright 2015 INRIA Paris-Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Michael Kruse, INRIA Paris-Rocquencourt,
 * Domaine de Voluceau, Rocquenqourt, B.P. 105,
 * 78153 Le Chesnay Cedex France
 */

#include <assert.h>
#include <stdio.h>
#include <isl_int.h>

#define ARRAY_SIZE(array) (sizeof(array)/sizeof(*array))

#ifdef USE_SMALL_INT_OPT
/* Test whether small and big representation of the same number have the same
 * hash.
 */
static void int_test_hash(isl_int val)
{
	uint32_t demotedhash, promotedhash;
	isl_int demoted, promoted;

	isl_int_init(demoted);
	isl_int_set(demoted, val);

	isl_int_init(promoted);
	isl_int_set(promoted, val);

	isl_sioimath_try_demote(demoted);
	isl_sioimath_promote(promoted);

	assert(isl_int_eq(demoted, promoted));

	demotedhash = isl_int_hash(demoted, 0);
	promotedhash = isl_int_hash(promoted, 0);
	assert(demotedhash == promotedhash);

	isl_int_clear(demoted);
	isl_int_clear(promoted);
}

struct {
	void (*fn)(isl_int);
	char *val;
} int_single_value_tests[] = {
	{ &int_test_hash, "0" },
	{ &int_test_hash, "1" },
	{ &int_test_hash, "-1" },
	{ &int_test_hash, "23" },
	{ &int_test_hash, "-23" },
	{ &int_test_hash, "107" },
	{ &int_test_hash, "32768" },
	{ &int_test_hash, "2147483647" },
	{ &int_test_hash, "-2147483647" },
	{ &int_test_hash, "2147483648" },
	{ &int_test_hash, "-2147483648" },
};

static void int_test_single_value()
{
	int i;

	for (i = 0; i < ARRAY_SIZE(int_single_value_tests); i += 1) {
		isl_int val;

		isl_int_init(val);
		isl_int_read(val, int_single_value_tests[i].val);

		(*int_single_value_tests[i].fn)(val);

		isl_int_clear(val);
	}
}

static void invoke_alternate_representations_2args(char *arg1, char *arg2,
	void (*fn)(isl_int, isl_int))
{
	int j;
	isl_int int1, int2;

	isl_int_init(int1);
	isl_int_init(int2);

	for (j = 0; j < 4; ++j) {
		isl_int_read(int1, arg1);
		isl_int_read(int2, arg2);

		if (j & 1)
			isl_sioimath_promote(int1);
		else
			isl_sioimath_try_demote(int1);

		if (j & 2)
			isl_sioimath_promote(int2);
		else
			isl_sioimath_try_demote(int2);

		(*fn)(int1, int2);
	}

	isl_int_clear(int1);
	isl_int_clear(int2);
}

static void invoke_alternate_representations_3args(char *arg1, char *arg2,
	char *arg3, void (*fn)(isl_int, isl_int, isl_int))
{
	int j;
	isl_int int1, int2, int3;

	isl_int_init(int1);
	isl_int_init(int2);
	isl_int_init(int3);

	for (j = 0; j < 8; ++j) {
		isl_int_read(int1, arg1);
		isl_int_read(int2, arg2);
		isl_int_read(int3, arg3);

		if (j & 1)
			isl_sioimath_promote(int1);
		else
			isl_sioimath_try_demote(int1);

		if (j & 2)
			isl_sioimath_promote(int2);
		else
			isl_sioimath_try_demote(int2);

		if (j & 4)
			isl_sioimath_promote(int3);
		else
			isl_sioimath_try_demote(int3);

		(*fn)(int1, int2, int3);
	}

	isl_int_clear(int1);
	isl_int_clear(int2);
	isl_int_clear(int3);
}
#else  /* USE_SMALL_INT_OPT */

static void int_test_single_value()
{
}

static void invoke_alternate_representations_2args(char *arg1, char *arg2,
	void (*fn)(isl_int, isl_int))
{
	isl_int int1, int2;

	isl_int_init(int1);
	isl_int_init(int2);

	isl_int_read(int1, arg1);
	isl_int_read(int2, arg2);

	(*fn)(int1, int2);

	isl_int_clear(int1);
	isl_int_clear(int2);
}

static void invoke_alternate_representations_3args(char *arg1, char *arg2,
	char *arg3, void (*fn)(isl_int, isl_int, isl_int))
{
	isl_int int1, int2, int3;

	isl_int_init(int1);
	isl_int_init(int2);
	isl_int_init(int3);

	isl_int_read(int1, arg1);
	isl_int_read(int2, arg2);
	isl_int_read(int3, arg3);

	(*fn)(int1, int2, int3);

	isl_int_clear(int1);
	isl_int_clear(int2);
	isl_int_clear(int3);
}
#endif /* USE_SMALL_INT_OPT */

static void int_test_neg(isl_int expected, isl_int arg)
{
	isl_int result;
	isl_int_init(result);

	isl_int_neg(result, arg);
	assert(isl_int_eq(result, expected));

	isl_int_neg(result, expected);
	assert(isl_int_eq(result, arg));

	isl_int_clear(result);
}

static void int_test_abs(isl_int expected, isl_int arg)
{
	isl_int result;
	isl_int_init(result);

	isl_int_abs(result, arg);
	assert(isl_int_eq(result, expected));

	isl_int_clear(result);
}

struct {
	void (*fn)(isl_int, isl_int);
	char *expected, *arg;
} int_unary_tests[] = {
	{ &int_test_neg, "0", "0" },
	{ &int_test_neg, "-1", "1" },
	{ &int_test_neg, "-2147483647", "2147483647" },
	{ &int_test_neg, "-2147483648", "2147483648" },
	{ &int_test_neg, "-9223372036854775807", "9223372036854775807" },
	{ &int_test_neg, "-9223372036854775808", "9223372036854775808" },

	{ &int_test_abs, "0", "0" },
	{ &int_test_abs, "1", "1" },
	{ &int_test_abs, "1", "-1" },
	{ &int_test_abs, "2147483647", "2147483647" },
	{ &int_test_abs, "2147483648", "-2147483648" },
	{ &int_test_abs, "9223372036854775807", "9223372036854775807" },
	{ &int_test_abs, "9223372036854775808", "-9223372036854775808" },
};

static void int_test_divexact(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	unsigned long rhsulong;

	if (isl_int_sgn(rhs) == 0)
		return;

	isl_int_init(result);

	isl_int_divexact(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_tdiv_q(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_fdiv_q(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_cdiv_q(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	if (isl_int_fits_ulong(rhs)) {
		rhsulong = isl_int_get_ui(rhs);

		isl_int_divexact_ui(result, lhs, rhsulong);
		assert(isl_int_eq(expected, result));

		isl_int_fdiv_q_ui(result, lhs, rhsulong);
		assert(isl_int_eq(expected, result));
	}

	isl_int_clear(result);
}

static void int_test_mul(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_mul(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	if (isl_int_fits_ulong(rhs)) {
		unsigned long rhsulong = isl_int_get_ui(rhs);

		isl_int_mul_ui(result, lhs, rhsulong);
		assert(isl_int_eq(expected, result));
	}

	if (isl_int_fits_slong(rhs)) {
		unsigned long rhsslong = isl_int_get_si(rhs);

		isl_int_mul_si(result, lhs, rhsslong);
		assert(isl_int_eq(expected, result));
	}

	isl_int_clear(result);
}

/* Use a triple that satisfies 'product = factor1 * factor2' to check the
 * operations mul, divexact, tdiv, fdiv and cdiv.
 */
static void int_test_product(isl_int product, isl_int factor1, isl_int factor2)
{
	int_test_divexact(factor1, product, factor2);
	int_test_divexact(factor2, product, factor1);

	int_test_mul(product, factor1, factor2);
	int_test_mul(product, factor2, factor1);
}

static void int_test_add(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_add(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_clear(result);
}

static void int_test_sub(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_sub(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_clear(result);
}

/* Use a triple that satisfies 'sum = term1 + term2' to check the operations add
 * and sub.
 */
static void int_test_sum(isl_int sum, isl_int term1, isl_int term2)
{
	int_test_sub(term1, sum, term2);
	int_test_sub(term2, sum, term1);

	int_test_add(sum, term1, term2);
	int_test_add(sum, term2, term1);
}

static void int_test_fdiv(isl_int expected, isl_int lhs, isl_int rhs)
{
	unsigned long rhsulong;
	isl_int result;
	isl_int_init(result);

	isl_int_fdiv_q(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	if (isl_int_fits_ulong(rhs)) {
		rhsulong = isl_int_get_ui(rhs);

		isl_int_fdiv_q_ui(result, lhs, rhsulong);
		assert(isl_int_eq(expected, result));
	}

	isl_int_clear(result);
}

static void int_test_cdiv(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_cdiv_q(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_clear(result);
}

static void int_test_tdiv(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_tdiv_q(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_clear(result);
}

static void int_test_fdiv_r(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_fdiv_r(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_clear(result);
}

static void int_test_gcd(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_gcd(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_gcd(result, rhs, lhs);
	assert(isl_int_eq(expected, result));

	isl_int_clear(result);
}

static void int_test_lcm(isl_int expected, isl_int lhs, isl_int rhs)
{
	isl_int result;
	isl_int_init(result);

	isl_int_lcm(result, lhs, rhs);
	assert(isl_int_eq(expected, result));

	isl_int_lcm(result, rhs, lhs);
	assert(isl_int_eq(expected, result));

	isl_int_clear(result);
}

static int sgn(int val)
{
	if (val > 0)
		return 1;
	if (val < 0)
		return -1;
	return 0;
}

static void int_test_cmp(int exp, isl_int lhs, isl_int rhs)
{
	long rhslong;

	assert(exp == sgn(isl_int_cmp(lhs, rhs)));

	if (isl_int_fits_slong(rhs)) {
		rhslong = isl_int_get_si(rhs);
		assert(exp == sgn(isl_int_cmp_si(lhs, rhslong)));
	}
}

/* Test the comparison relations over two numbers.
 * expected is the sign (1, 0 or -1) of 'lhs - rhs'.
 */
static void int_test_cmps(isl_int expected, isl_int lhs, isl_int rhs)
{
	int exp;
	isl_int diff;

	exp = isl_int_get_si(expected);

	isl_int_init(diff);
	isl_int_sub(diff, lhs, rhs);
	assert(exp == isl_int_sgn(diff));
	isl_int_clear(diff);

	int_test_cmp(exp, lhs, rhs);
	int_test_cmp(-exp, rhs, lhs);
}

static void int_test_abs_cmp(isl_int expected, isl_int lhs, isl_int rhs)
{
	int exp;

	exp = isl_int_get_si(expected);
	assert(exp == sgn(isl_int_abs_cmp(lhs, rhs)));
	assert(-exp == sgn(isl_int_abs_cmp(rhs, lhs)));
}

struct {
	void (*fn)(isl_int, isl_int, isl_int);
	char *expected, *lhs, *rhs;
} int_binary_tests[] = {
	{ &int_test_sum, "0", "0", "0" },
	{ &int_test_sum, "1", "1", "0" },
	{ &int_test_sum, "2", "1", "1" },
	{ &int_test_sum, "-1", "0", "-1" },
	{ &int_test_sum, "-2", "-1", "-1" },

	{ &int_test_sum, "2147483647", "1073741823", "1073741824" },
	{ &int_test_sum, "-2147483648", "-1073741824", "-1073741824" },

	{ &int_test_sum, "2147483648", "2147483647", "1" },
	{ &int_test_sum, "-2147483648", "-2147483647", "-1" },

	{ &int_test_product, "0", "0", "0" },
	{ &int_test_product, "0", "0", "1" },
	{ &int_test_product, "1", "1", "1" },

	{ &int_test_product, "6", "2", "3" },
	{ &int_test_product, "-6", "2", "-3" },
	{ &int_test_product, "-6", "-2", "3" },
	{ &int_test_product, "6", "-2", "-3" },

	{ &int_test_product, "2147483648", "65536", "32768" },
	{ &int_test_product, "-2147483648", "65536", "-32768" },

	{ &int_test_product,
	  "4611686014132420609", "2147483647", "2147483647" },
	{ &int_test_product,
	  "-4611686014132420609", "-2147483647", "2147483647" },

	{ &int_test_product,
	  "4611686016279904256", "2147483647", "2147483648" },
	{ &int_test_product,
	  "-4611686016279904256", "-2147483647", "2147483648" },
	{ &int_test_product,
	  "-4611686016279904256", "2147483647", "-2147483648" },
	{ &int_test_product,
	  "4611686016279904256", "-2147483647", "-2147483648" },

	{ &int_test_product, "85070591730234615847396907784232501249",
	  "9223372036854775807", "9223372036854775807" },
	{ &int_test_product, "-85070591730234615847396907784232501249",
	  "-9223372036854775807", "9223372036854775807" },

	{ &int_test_product, "85070591730234615856620279821087277056",
	  "9223372036854775807", "9223372036854775808" },
	{ &int_test_product, "-85070591730234615856620279821087277056",
	  "-9223372036854775807", "9223372036854775808" },
	{ &int_test_product, "-85070591730234615856620279821087277056",
	  "9223372036854775807", "-9223372036854775808" },
	{ &int_test_product, "85070591730234615856620279821087277056",
	  "-9223372036854775807", "-9223372036854775808" },

	{ &int_test_product, "340282366920938463426481119284349108225",
	  "18446744073709551615", "18446744073709551615" },
	{ &int_test_product, "-340282366920938463426481119284349108225",
	  "-18446744073709551615", "18446744073709551615" },

	{ &int_test_product, "340282366920938463444927863358058659840",
	  "18446744073709551615", "18446744073709551616" },
	{ &int_test_product, "-340282366920938463444927863358058659840",
	  "-18446744073709551615", "18446744073709551616" },
	{ &int_test_product, "-340282366920938463444927863358058659840",
	  "18446744073709551615", "-18446744073709551616" },
	{ &int_test_product, "340282366920938463444927863358058659840",
	  "-18446744073709551615", "-18446744073709551616" },

	{ &int_test_fdiv, "0", "1", "2" },
	{ &int_test_fdiv_r, "1", "1", "3" },
	{ &int_test_fdiv, "-1", "-1", "2" },
	{ &int_test_fdiv_r, "2", "-1", "3" },
	{ &int_test_fdiv, "-1", "1", "-2" },
	{ &int_test_fdiv_r, "-2", "1", "-3" },
	{ &int_test_fdiv, "0", "-1", "-2" },
	{ &int_test_fdiv_r, "-1", "-1", "-3" },

	{ &int_test_cdiv, "1", "1", "2" },
	{ &int_test_cdiv, "0", "-1", "2" },
	{ &int_test_cdiv, "0", "1", "-2" },
	{ &int_test_cdiv, "1", "-1", "-2" },

	{ &int_test_tdiv, "0", "1", "2" },
	{ &int_test_tdiv, "0", "-1", "2" },
	{ &int_test_tdiv, "0", "1", "-2" },
	{ &int_test_tdiv, "0", "-1", "-2" },

	{ &int_test_gcd, "0", "0", "0" },
	{ &int_test_lcm, "0", "0", "0" },
	{ &int_test_gcd, "7", "0", "7" },
	{ &int_test_lcm, "0", "0", "7" },
	{ &int_test_gcd, "1", "1", "1" },
	{ &int_test_lcm, "1", "1", "1" },
	{ &int_test_gcd, "1", "1", "-1" },
	{ &int_test_lcm, "1", "1", "-1" },
	{ &int_test_gcd, "1", "-1", "-1" },
	{ &int_test_lcm, "1", "-1", "-1" },
	{ &int_test_gcd, "3", "6", "9" },
	{ &int_test_lcm, "18", "6", "9" },
	{ &int_test_gcd, "1", "14", "2147483647" },
	{ &int_test_lcm, "15032385529", "7", "2147483647" },
	{ &int_test_gcd, "2", "6", "-2147483648" },
	{ &int_test_lcm, "6442450944", "6", "-2147483648" },
	{ &int_test_gcd, "1", "6", "9223372036854775807" },
	{ &int_test_lcm, "55340232221128654842", "6", "9223372036854775807" },
	{ &int_test_gcd, "2", "6", "-9223372036854775808" },
	{ &int_test_lcm, "27670116110564327424", "6", "-9223372036854775808" },
	{ &int_test_gcd, "1", "18446744073709551616", "18446744073709551615" },
	{ &int_test_lcm, "340282366920938463444927863358058659840",
	  "18446744073709551616", "18446744073709551615" },

	{ &int_test_cmps, "0", "0", "0" },
	{ &int_test_abs_cmp, "0", "0", "0" },
	{ &int_test_cmps, "1", "1", "0" },
	{ &int_test_abs_cmp, "1", "1", "0" },
	{ &int_test_cmps, "-1", "-1", "0" },
	{ &int_test_abs_cmp, "1", "-1", "0" },
	{ &int_test_cmps, "-1", "-1", "1" },
	{ &int_test_abs_cmp, "0", "-1", "1" },

	{ &int_test_cmps, "-1", "5", "2147483647" },
	{ &int_test_abs_cmp, "-1", "5", "2147483647" },
	{ &int_test_cmps, "1", "5", "-2147483648" },
	{ &int_test_abs_cmp, "-1", "5", "-2147483648" },
	{ &int_test_cmps, "-1", "5", "9223372036854775807" },
	{ &int_test_abs_cmp, "-1", "5", "9223372036854775807" },
	{ &int_test_cmps, "1", "5", "-9223372036854775809" },
	{ &int_test_abs_cmp, "-1", "5", "-9223372036854775809" },
};

/* Tests the isl_int_* function to give the expected results. Tests are
 * grouped by the number of arguments they take.
 *
 * If small integer optimization is enabled, we also test whether the results
 * are the same in small and big representation.
 */
int main()
{
	int i;

	int_test_single_value();

	for (i = 0; i < ARRAY_SIZE(int_unary_tests); i += 1) {
		invoke_alternate_representations_2args(
		    int_unary_tests[i].expected, int_unary_tests[i].arg,
		    int_unary_tests[i].fn);
	}

	for (i = 0; i < ARRAY_SIZE(int_binary_tests); i += 1) {
		invoke_alternate_representations_3args(
		    int_binary_tests[i].expected, int_binary_tests[i].lhs,
		    int_binary_tests[i].rhs, int_binary_tests[i].fn);
	}

	return 0;
}
