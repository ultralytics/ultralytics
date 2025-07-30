/* mpn_get_str -- Convert {UP,USIZE} to a base BASE string in STR.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE, EXCEPT mpn_get_str, ARE INTERNAL WITH A MUTABLE
   INTERFACE.  IT IS ONLY SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN
   FACT, IT IS ALMOST GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE
   GNU MP RELEASE.

Copyright 1991-1994, 1996, 2000-2002, 2004, 2006-2008, 2011, 2012 Free Software
Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

/* Conversion of U {up,un} to a string in base b.  Internally, we convert to
   base B = b^m, the largest power of b that fits a limb.  Basic algorithms:

  A) Divide U repeatedly by B, generating a quotient and remainder, until the
     quotient becomes zero.  The remainders hold the converted digits.  Digits
     come out from right to left.  (Used in mpn_sb_get_str.)

  B) Divide U by b^g, for g such that 1/b <= U/b^g < 1, generating a fraction.
     Then develop digits by multiplying the fraction repeatedly by b.  Digits
     come out from left to right.  (Currently not used herein, except for in
     code for converting single limbs to individual digits.)

  C) Compute B^1, B^2, B^4, ..., B^s, for s such that B^s is just above
     sqrt(U).  Then divide U by B^s, generating quotient and remainder.
     Recursively convert the quotient, then the remainder, using the
     precomputed powers.  Digits come out from left to right.  (Used in
     mpn_dc_get_str.)

  When using algorithm C, algorithm B might be suitable for basecase code,
  since the required b^g power will be readily accessible.

  Optimization ideas:
  1. The recursive function of (C) could use less temporary memory.  The powtab
     allocation could be trimmed with some computation, and the tmp area could
     be reduced, or perhaps eliminated if up is reused for both quotient and
     remainder (it is currently used just for remainder).
  2. Store the powers of (C) in normalized form, with the normalization count.
     Quotients will usually need to be left-shifted before each divide, and
     remainders will either need to be left-shifted of right-shifted.
  3. In the code for developing digits from a single limb, we could avoid using
     a full umul_ppmm except for the first (or first few) digits, provided base
     is even.  Subsequent digits can be developed using plain multiplication.
     (This saves on register-starved machines (read x86) and on all machines
     that generate the upper product half using a separate instruction (alpha,
     powerpc, IA-64) or lacks such support altogether (sparc64, hppa64).
  4. Separate mpn_dc_get_str basecase code from code for small conversions. The
     former code will have the exact right power readily available in the
     powtab parameter for dividing the current number into a fraction.  Convert
     that using algorithm B.
  5. Completely avoid division.  Compute the inverses of the powers now in
     powtab instead of the actual powers.
  6. Decrease powtab allocation for even bases.  E.g. for base 10 we could save
     about 30% (1-log(5)/log(10)).

  Basic structure of (C):
    mpn_get_str:
      if POW2_P (n)
	...
      else
	if (un < GET_STR_PRECOMPUTE_THRESHOLD)
	  mpn_sb_get_str (str, base, up, un);
	else
	  precompute_power_tables
	  mpn_dc_get_str

    mpn_dc_get_str:
	mpn_tdiv_qr
	if (qn < GET_STR_DC_THRESHOLD)
	  mpn_sb_get_str
	else
	  mpn_dc_get_str
	if (rn < GET_STR_DC_THRESHOLD)
	  mpn_sb_get_str
	else
	  mpn_dc_get_str


  The reason for the two threshold values is the cost of
  precompute_power_tables.  GET_STR_PRECOMPUTE_THRESHOLD will be considerably
  larger than GET_STR_PRECOMPUTE_THRESHOLD.  */


/* The x86s and m68020 have a quotient and remainder "div" instruction and
   gcc recognises an adjacent "/" and "%" can be combined using that.
   Elsewhere "/" and "%" are either separate instructions, or separate
   libgcc calls (which unfortunately gcc as of version 3.0 doesn't combine).
   A multiply and subtract should be faster than a "%" in those cases.  */
#if HAVE_HOST_CPU_FAMILY_x86            \
  || HAVE_HOST_CPU_m68020               \
  || HAVE_HOST_CPU_m68030               \
  || HAVE_HOST_CPU_m68040               \
  || HAVE_HOST_CPU_m68060               \
  || HAVE_HOST_CPU_m68360 /* CPU32 */
#define udiv_qrnd_unnorm(q,r,n,d)       \
  do {                                  \
    mp_limb_t  __q = (n) / (d);         \
    mp_limb_t  __r = (n) % (d);         \
    (q) = __q;                          \
    (r) = __r;                          \
  } while (0)
#else
#define udiv_qrnd_unnorm(q,r,n,d)       \
  do {                                  \
    mp_limb_t  __q = (n) / (d);         \
    mp_limb_t  __r = (n) - __q*(d);     \
    (q) = __q;                          \
    (r) = __r;                          \
  } while (0)
#endif


/* Convert {up,un} to a string in base base, and put the result in str.
   Generate len characters, possibly padding with zeros to the left.  If len is
   zero, generate as many characters as required.  Return a pointer immediately
   after the last digit of the result string.  Complexity is O(un^2); intended
   for small conversions.  */
static unsigned char *
mpn_sb_get_str (unsigned char *str, size_t len,
		mp_ptr up, mp_size_t un, int base)
{
  mp_limb_t rl, ul;
  unsigned char *s;
  size_t l;
  /* Allocate memory for largest possible string, given that we only get here
     for operands with un < GET_STR_PRECOMPUTE_THRESHOLD and that the smallest
     base is 3.  7/11 is an approximation to 1/log2(3).  */
#if TUNE_PROGRAM_BUILD
#define BUF_ALLOC (GET_STR_THRESHOLD_LIMIT * GMP_LIMB_BITS * 7 / 11)
#else
#define BUF_ALLOC (GET_STR_PRECOMPUTE_THRESHOLD * GMP_LIMB_BITS * 7 / 11)
#endif
  unsigned char buf[BUF_ALLOC];
#if TUNE_PROGRAM_BUILD
  mp_limb_t rp[GET_STR_THRESHOLD_LIMIT];
#else
  mp_limb_t rp[GET_STR_PRECOMPUTE_THRESHOLD];
#endif

  if (base == 10)
    {
      /* Special case code for base==10 so that the compiler has a chance to
	 optimize things.  */

      MPN_COPY (rp + 1, up, un);

      s = buf + BUF_ALLOC;
      while (un > 1)
	{
	  int i;
	  mp_limb_t frac, digit;
	  MPN_DIVREM_OR_PREINV_DIVREM_1 (rp, (mp_size_t) 1, rp + 1, un,
					 MP_BASES_BIG_BASE_10,
					 MP_BASES_BIG_BASE_INVERTED_10,
					 MP_BASES_NORMALIZATION_STEPS_10);
	  un -= rp[un] == 0;
	  frac = (rp[0] + 1) << GMP_NAIL_BITS;
	  s -= MP_BASES_CHARS_PER_LIMB_10;
#if HAVE_HOST_CPU_FAMILY_x86
	  /* The code below turns out to be a bit slower for x86 using gcc.
	     Use plain code.  */
	  i = MP_BASES_CHARS_PER_LIMB_10;
	  do
	    {
	      umul_ppmm (digit, frac, frac, 10);
	      *s++ = digit;
	    }
	  while (--i);
#else
	  /* Use the fact that 10 in binary is 1010, with the lowest bit 0.
	     After a few umul_ppmm, we will have accumulated enough low zeros
	     to use a plain multiply.  */
	  if (MP_BASES_NORMALIZATION_STEPS_10 == 0)
	    {
	      umul_ppmm (digit, frac, frac, 10);
	      *s++ = digit;
	    }
	  if (MP_BASES_NORMALIZATION_STEPS_10 <= 1)
	    {
	      umul_ppmm (digit, frac, frac, 10);
	      *s++ = digit;
	    }
	  if (MP_BASES_NORMALIZATION_STEPS_10 <= 2)
	    {
	      umul_ppmm (digit, frac, frac, 10);
	      *s++ = digit;
	    }
	  if (MP_BASES_NORMALIZATION_STEPS_10 <= 3)
	    {
	      umul_ppmm (digit, frac, frac, 10);
	      *s++ = digit;
	    }
	  i = (MP_BASES_CHARS_PER_LIMB_10 - ((MP_BASES_NORMALIZATION_STEPS_10 < 4)
					     ? (4-MP_BASES_NORMALIZATION_STEPS_10)
					     : 0));
	  frac = (frac + 0xf) >> 4;
	  do
	    {
	      frac *= 10;
	      digit = frac >> (GMP_LIMB_BITS - 4);
	      *s++ = digit;
	      frac &= (~(mp_limb_t) 0) >> 4;
	    }
	  while (--i);
#endif
	  s -= MP_BASES_CHARS_PER_LIMB_10;
	}

      ul = rp[1];
      while (ul != 0)
	{
	  udiv_qrnd_unnorm (ul, rl, ul, 10);
	  *--s = rl;
	}
    }
  else /* not base 10 */
    {
      unsigned chars_per_limb;
      mp_limb_t big_base, big_base_inverted;
      unsigned normalization_steps;

      chars_per_limb = mp_bases[base].chars_per_limb;
      big_base = mp_bases[base].big_base;
      big_base_inverted = mp_bases[base].big_base_inverted;
      count_leading_zeros (normalization_steps, big_base);

      MPN_COPY (rp + 1, up, un);

      s = buf + BUF_ALLOC;
      while (un > 1)
	{
	  int i;
	  mp_limb_t frac;
	  MPN_DIVREM_OR_PREINV_DIVREM_1 (rp, (mp_size_t) 1, rp + 1, un,
					 big_base, big_base_inverted,
					 normalization_steps);
	  un -= rp[un] == 0;
	  frac = (rp[0] + 1) << GMP_NAIL_BITS;
	  s -= chars_per_limb;
	  i = chars_per_limb;
	  do
	    {
	      mp_limb_t digit;
	      umul_ppmm (digit, frac, frac, base);
	      *s++ = digit;
	    }
	  while (--i);
	  s -= chars_per_limb;
	}

      ul = rp[1];
      while (ul != 0)
	{
	  udiv_qrnd_unnorm (ul, rl, ul, base);
	  *--s = rl;
	}
    }

  l = buf + BUF_ALLOC - s;
  while (l < len)
    {
      *str++ = 0;
      len--;
    }
  while (l != 0)
    {
      *str++ = *s++;
      l--;
    }
  return str;
}


/* Convert {UP,UN} to a string with a base as represented in POWTAB, and put
   the string in STR.  Generate LEN characters, possibly padding with zeros to
   the left.  If LEN is zero, generate as many characters as required.
   Return a pointer immediately after the last digit of the result string.
   This uses divide-and-conquer and is intended for large conversions.  */
static unsigned char *
mpn_dc_get_str (unsigned char *str, size_t len,
		mp_ptr up, mp_size_t un,
		const powers_t *powtab, mp_ptr tmp)
{
  if (BELOW_THRESHOLD (un, GET_STR_DC_THRESHOLD))
    {
      if (un != 0)
	str = mpn_sb_get_str (str, len, up, un, powtab->base);
      else
	{
	  while (len != 0)
	    {
	      *str++ = 0;
	      len--;
	    }
	}
    }
  else
    {
      mp_ptr pwp, qp, rp;
      mp_size_t pwn, qn;
      mp_size_t sn;

      pwp = powtab->p;
      pwn = powtab->n;
      sn = powtab->shift;

      if (un < pwn + sn || (un == pwn + sn && mpn_cmp (up + sn, pwp, un - sn) < 0))
	{
	  str = mpn_dc_get_str (str, len, up, un, powtab - 1, tmp);
	}
      else
	{
	  qp = tmp;		/* (un - pwn + 1) limbs for qp */
	  rp = up;		/* pwn limbs for rp; overwrite up area */

	  mpn_tdiv_qr (qp, rp + sn, 0L, up + sn, un - sn, pwp, pwn);
	  qn = un - sn - pwn; qn += qp[qn] != 0;		/* quotient size */

	  ASSERT (qn < pwn + sn || (qn == pwn + sn && mpn_cmp (qp + sn, pwp, pwn) < 0));

	  if (len != 0)
	    len = len - powtab->digits_in_base;

	  str = mpn_dc_get_str (str, len, qp, qn, powtab - 1, tmp + qn);
	  str = mpn_dc_get_str (str, powtab->digits_in_base, rp, pwn + sn, powtab - 1, tmp);
	}
    }
  return str;
}


/* There are no leading zeros on the digits generated at str, but that's not
   currently a documented feature.  The current mpz_out_str and mpz_get_str
   rely on it.  */

size_t
mpn_get_str (unsigned char *str, int base, mp_ptr up, mp_size_t un)
{
  mp_ptr powtab_mem, powtab_mem_ptr;
  mp_limb_t big_base;
  size_t digits_in_base;
  powers_t powtab[GMP_LIMB_BITS];
  int pi;
  mp_size_t n;
  mp_ptr p, t;
  size_t out_len;
  mp_ptr tmp;
  TMP_DECL;

  /* Special case zero, as the code below doesn't handle it.  */
  if (un == 0)
    {
      str[0] = 0;
      return 1;
    }

  if (POW2_P (base))
    {
      /* The base is a power of 2.  Convert from most significant end.  */
      mp_limb_t n1, n0;
      int bits_per_digit = mp_bases[base].big_base;
      int cnt;
      int bit_pos;
      mp_size_t i;
      unsigned char *s = str;
      mp_bitcnt_t bits;

      n1 = up[un - 1];
      count_leading_zeros (cnt, n1);

      /* BIT_POS should be R when input ends in least significant nibble,
	 R + bits_per_digit * n when input ends in nth least significant
	 nibble. */

      bits = (mp_bitcnt_t) GMP_NUMB_BITS * un - cnt + GMP_NAIL_BITS;
      cnt = bits % bits_per_digit;
      if (cnt != 0)
	bits += bits_per_digit - cnt;
      bit_pos = bits - (mp_bitcnt_t) (un - 1) * GMP_NUMB_BITS;

      /* Fast loop for bit output.  */
      i = un - 1;
      for (;;)
	{
	  bit_pos -= bits_per_digit;
	  while (bit_pos >= 0)
	    {
	      *s++ = (n1 >> bit_pos) & ((1 << bits_per_digit) - 1);
	      bit_pos -= bits_per_digit;
	    }
	  i--;
	  if (i < 0)
	    break;
	  n0 = (n1 << -bit_pos) & ((1 << bits_per_digit) - 1);
	  n1 = up[i];
	  bit_pos += GMP_NUMB_BITS;
	  *s++ = n0 | (n1 >> bit_pos);
	}

      return s - str;
    }

  /* General case.  The base is not a power of 2.  */

  if (BELOW_THRESHOLD (un, GET_STR_PRECOMPUTE_THRESHOLD))
    return mpn_sb_get_str (str, (size_t) 0, up, un, base) - str;

  TMP_MARK;

  /* Allocate one large block for the powers of big_base.  */
  powtab_mem = TMP_BALLOC_LIMBS (mpn_dc_get_str_powtab_alloc (un));
  powtab_mem_ptr = powtab_mem;

  /* Compute a table of powers, were the largest power is >= sqrt(U).  */

  big_base = mp_bases[base].big_base;
  digits_in_base = mp_bases[base].chars_per_limb;

  {
    mp_size_t n_pows, xn, pn, exptab[GMP_LIMB_BITS], bexp;
    mp_limb_t cy;
    mp_size_t shift;
    size_t ndig;

    DIGITS_IN_BASE_PER_LIMB (ndig, un, base);
    xn = 1 + ndig / mp_bases[base].chars_per_limb; /* FIXME: scalar integer division */

    n_pows = 0;
    for (pn = xn; pn != 1; pn = (pn + 1) >> 1)
      {
	exptab[n_pows] = pn;
	n_pows++;
      }
    exptab[n_pows] = 1;

    powtab[0].p = &big_base;
    powtab[0].n = 1;
    powtab[0].digits_in_base = digits_in_base;
    powtab[0].base = base;
    powtab[0].shift = 0;

    powtab[1].p = powtab_mem_ptr;  powtab_mem_ptr += 2;
    powtab[1].p[0] = big_base;
    powtab[1].n = 1;
    powtab[1].digits_in_base = digits_in_base;
    powtab[1].base = base;
    powtab[1].shift = 0;

    n = 1;
    p = &big_base;
    bexp = 1;
    shift = 0;
    for (pi = 2; pi < n_pows; pi++)
      {
	t = powtab_mem_ptr;
	powtab_mem_ptr += 2 * n + 2;

	ASSERT_ALWAYS (powtab_mem_ptr < powtab_mem + mpn_dc_get_str_powtab_alloc (un));

	mpn_sqr (t, p, n);

	digits_in_base *= 2;
	n *= 2;  n -= t[n - 1] == 0;
	bexp *= 2;

	if (bexp + 1 < exptab[n_pows - pi])
	  {
	    digits_in_base += mp_bases[base].chars_per_limb;
	    cy = mpn_mul_1 (t, t, n, big_base);
	    t[n] = cy;
	    n += cy != 0;
	    bexp += 1;
	  }
	shift *= 2;
	/* Strip low zero limbs.  */
	while (t[0] == 0)
	  {
	    t++;
	    n--;
	    shift++;
	  }
	p = t;
	powtab[pi].p = p;
	powtab[pi].n = n;
	powtab[pi].digits_in_base = digits_in_base;
	powtab[pi].base = base;
	powtab[pi].shift = shift;
      }

    for (pi = 1; pi < n_pows; pi++)
      {
	t = powtab[pi].p;
	n = powtab[pi].n;
	cy = mpn_mul_1 (t, t, n, big_base);
	t[n] = cy;
	n += cy != 0;
	if (t[0] == 0)
	  {
	    powtab[pi].p = t + 1;
	    n--;
	    powtab[pi].shift++;
	  }
	powtab[pi].n = n;
	powtab[pi].digits_in_base += mp_bases[base].chars_per_limb;
      }

#if 0
    { int i;
      printf ("Computed table values for base=%d, un=%d, xn=%d:\n", base, un, xn);
      for (i = 0; i < n_pows; i++)
	printf ("%2d: %10ld %10ld %11ld %ld\n", i, exptab[n_pows-i], powtab[i].n, powtab[i].digits_in_base, powtab[i].shift);
    }
#endif
  }

  /* Using our precomputed powers, now in powtab[], convert our number.  */
  tmp = TMP_BALLOC_LIMBS (mpn_dc_get_str_itch (un));
  out_len = mpn_dc_get_str (str, 0, up, un, powtab + (pi - 1), tmp) - str;
  TMP_FREE;

  return out_len;
}
