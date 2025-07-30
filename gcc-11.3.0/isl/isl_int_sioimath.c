#include <stdlib.h>
#include <string.h>

#include <isl_int.h>

extern int isl_sioimath_decode(isl_sioimath val, int32_t *small, mp_int *big);
extern int isl_sioimath_decode_big(isl_sioimath val, mp_int *big);
extern int isl_sioimath_decode_small(isl_sioimath val, int32_t *small);

extern isl_sioimath isl_sioimath_encode_small(int32_t val);
extern isl_sioimath isl_sioimath_encode_big(mp_int val);
extern int isl_sioimath_is_small(isl_sioimath val);
extern int isl_sioimath_is_big(isl_sioimath val);
extern int32_t isl_sioimath_get_small(isl_sioimath val);
extern mp_int isl_sioimath_get_big(isl_sioimath val);

extern void isl_siomath_uint32_to_digits(uint32_t num, mp_digit *digits,
	mp_size *used);
extern void isl_siomath_ulong_to_digits(unsigned long num, mp_digit *digits,
	mp_size *used);
extern void isl_siomath_uint64_to_digits(uint64_t num, mp_digit *digits,
	mp_size *used);

extern mp_int isl_sioimath_bigarg_src(isl_sioimath arg,
	isl_sioimath_scratchspace_t *scratch);
extern mp_int isl_sioimath_siarg_src(signed long arg,
	isl_sioimath_scratchspace_t *scratch);
extern mp_int isl_sioimath_si64arg_src(int64_t arg,
	isl_sioimath_scratchspace_t *scratch);
extern mp_int isl_sioimath_uiarg_src(unsigned long arg,
	isl_sioimath_scratchspace_t *scratch);
extern mp_int isl_sioimath_reinit_big(isl_sioimath_ptr ptr);
extern void isl_sioimath_set_small(isl_sioimath_ptr ptr, int32_t val);
extern void isl_sioimath_set_int32(isl_sioimath_ptr ptr, int32_t val);
extern void isl_sioimath_set_int64(isl_sioimath_ptr ptr, int64_t val);
extern void isl_sioimath_promote(isl_sioimath_ptr dst);
extern void isl_sioimath_try_demote(isl_sioimath_ptr dst);

extern void isl_sioimath_init(isl_sioimath_ptr dst);
extern void isl_sioimath_clear(isl_sioimath_ptr dst);
extern void isl_sioimath_set(isl_sioimath_ptr dst, isl_sioimath_src val);
extern void isl_sioimath_set_si(isl_sioimath_ptr dst, long val);
extern void isl_sioimath_set_ui(isl_sioimath_ptr dst, unsigned long val);
extern int isl_sioimath_fits_slong(isl_sioimath_src val);
extern long isl_sioimath_get_si(isl_sioimath_src val);
extern int isl_sioimath_fits_ulong(isl_sioimath_src val);
extern unsigned long isl_sioimath_get_ui(isl_sioimath_src val);
extern double isl_sioimath_get_d(isl_sioimath_src val);
extern char *isl_sioimath_get_str(isl_sioimath_src val);
extern void isl_sioimath_abs(isl_sioimath_ptr dst, isl_sioimath_src arg);
extern void isl_sioimath_neg(isl_sioimath_ptr dst, isl_sioimath_src arg);
extern void isl_sioimath_swap(isl_sioimath_ptr lhs, isl_sioimath_ptr rhs);
extern void isl_sioimath_add_ui(isl_sioimath_ptr dst, isl_sioimath lhs,
	unsigned long rhs);
extern void isl_sioimath_sub_ui(isl_sioimath_ptr dst, isl_sioimath lhs,
	unsigned long rhs);

extern void isl_sioimath_add(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_sub(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_mul(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_mul_2exp(isl_sioimath_ptr dst, isl_sioimath lhs,
	unsigned long rhs);
extern void isl_sioimath_mul_si(isl_sioimath_ptr dst, isl_sioimath lhs,
	signed long rhs);
extern void isl_sioimath_mul_ui(isl_sioimath_ptr dst, isl_sioimath lhs,
	unsigned long rhs);
extern void isl_sioimath_pow_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs);
extern void isl_sioimath_addmul(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_addmul_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs);
extern void isl_sioimath_submul(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_submul_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs);

/* Implements the Euclidean algorithm to compute the greatest common divisor of
 * two values in small representation.
 */
static uint32_t isl_sioimath_smallgcd(int32_t lhs, int32_t rhs)
{
	uint32_t dividend, divisor, remainder;

	dividend = labs(lhs);
	divisor = labs(rhs);
	while (divisor) {
		remainder = dividend % divisor;
		dividend = divisor;
		divisor = remainder;
	}

	return dividend;
}

/* Compute the greatest common divisor.
 *
 * Per GMP convention, gcd(0,0)==0 and otherwise always positive.
 */
void isl_sioimath_gcd(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	int32_t lhssmall, rhssmall;
	uint32_t smallgcd;
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall)) {
		smallgcd = isl_sioimath_smallgcd(lhssmall, rhssmall);
		isl_sioimath_set_small(dst, smallgcd);
		return;
	}

	impz_gcd(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_bigarg_src(rhs, &scratchrhs));
	isl_sioimath_try_demote(dst);
}

/* Compute the lowest common multiple of two numbers.
 */
void isl_sioimath_lcm(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs)
{
	int32_t lhssmall, rhssmall;
	uint32_t smallgcd;
	uint64_t multiple;
	isl_sioimath_scratchspace_t scratchlhs, scratchrhs;

	if (isl_sioimath_decode_small(lhs, &lhssmall) &&
	    isl_sioimath_decode_small(rhs, &rhssmall)) {
		if (lhssmall == 0 || rhssmall == 0) {
			isl_sioimath_set_small(dst, 0);
			return;
		}
		smallgcd = isl_sioimath_smallgcd(lhssmall, rhssmall);
		multiple = (uint64_t) abs(lhssmall) * (uint64_t) abs(rhssmall);
		isl_sioimath_set_int64(dst, multiple / smallgcd);
		return;
	}

	impz_lcm(isl_sioimath_reinit_big(dst),
	    isl_sioimath_bigarg_src(lhs, &scratchlhs),
	    isl_sioimath_bigarg_src(rhs, &scratchrhs));
	isl_sioimath_try_demote(dst);
}

extern void isl_sioimath_tdiv_q(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_tdiv_q_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs);
extern void isl_sioimath_cdiv_q(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_fdiv_q(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);
extern void isl_sioimath_fdiv_q_ui(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	unsigned long rhs);
extern void isl_sioimath_fdiv_r(isl_sioimath_ptr dst, isl_sioimath_src lhs,
	isl_sioimath_src rhs);

/* Parse a number from a string.
 * If it has less than 10 characters then it will fit into the small
 * representation (i.e. strlen("2147483647")). Otherwise, let IMath parse it.
 */
void isl_sioimath_read(isl_sioimath_ptr dst, const char *str)
{
	int32_t small;

	if (strlen(str) < 10) {
		small = strtol(str, NULL, 10);
		isl_sioimath_set_small(dst, small);
		return;
	}

	mp_int_read_string(isl_sioimath_reinit_big(dst), 10, str);
	isl_sioimath_try_demote(dst);
}

extern int isl_sioimath_sgn(isl_sioimath_src arg);
extern int isl_sioimath_cmp(isl_sioimath_src lhs, isl_sioimath_src rhs);
extern int isl_sioimath_cmp_si(isl_sioimath_src lhs, signed long rhs);
extern int isl_sioimath_abs_cmp(isl_sioimath_src lhs, isl_sioimath_src rhs);
extern int isl_sioimath_is_divisible_by(isl_sioimath_src lhs,
	isl_sioimath_src rhs);

extern uint32_t isl_sioimath_hash(isl_sioimath_src arg, uint32_t hash);
extern size_t isl_sioimath_sizeinbase(isl_sioimath_src arg, int base);
extern void isl_sioimath_print(FILE *out, isl_sioimath_src i, int width);

/* Print an isl_int to FILE*. Adds space padding to the left until at least
 * width characters are printed.
 */
void isl_sioimath_print(FILE *out, isl_sioimath_src i, int width)
{
	size_t len;
	int32_t small;
	mp_int big;
	char *buf;

	if (isl_sioimath_decode_small(i, &small)) {
		fprintf(out, "%*" PRIi32, width, small);
		return;
	}

	big = isl_sioimath_get_big(i);
	len = mp_int_string_len(big, 10);
	buf = malloc(len);
	mp_int_to_string(big, 10, buf, len);
	fprintf(out, "%*s", width, buf);
	free(buf);
}

/* Print a number to stdout. Meant for debugging.
 */
void isl_sioimath_dump(isl_sioimath_src arg)
{
	isl_sioimath_print(stdout, arg, 0);
}
