/* mpn_set_str (mp_ptr res_ptr, const char *str, size_t str_len, int base) --
   Convert a STR_LEN long base BASE byte string pointed to by STR to a limb
   vector pointed to by RES_PTR.  Return the number of limbs in RES_PTR.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE, EXCEPT mpn_set_str, ARE INTERNAL WITH A MUTABLE
   INTERFACE.  IT IS ONLY SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN
   FACT, IT IS ALMOST GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE
   GNU MP RELEASE.

Copyright 1991-1994, 1996, 2000-2002, 2004, 2006-2008, 2012, 2013 Free
Software Foundation, Inc.

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


/* TODO:

      Perhaps do not compute the highest power?
      Instead, multiply twice by the 2nd highest power:

	       _______
	      |_______|  hp
	      |_______|  pow
       _______________
      |_______________|  final result


	       _______
	      |_______|  hp
		  |___|  pow[-1]
	   ___________
	  |___________|  intermediate result
		  |___|  pow[-1]
       _______________
      |_______________|  final result

      Generalizing that idea, perhaps we should make powtab contain successive
      cubes, not squares.
*/

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

mp_size_t
mpn_set_str (mp_ptr rp, const unsigned char *str, size_t str_len, int base)
{
  if (POW2_P (base))
    {
      /* The base is a power of 2.  Read the input string from least to most
	 significant character/digit.  */

      const unsigned char *s;
      int next_bitpos;
      mp_limb_t res_digit;
      mp_size_t size;
      int bits_per_indigit = mp_bases[base].big_base;

      size = 0;
      res_digit = 0;
      next_bitpos = 0;

      for (s = str + str_len - 1; s >= str; s--)
	{
	  int inp_digit = *s;

	  res_digit |= ((mp_limb_t) inp_digit << next_bitpos) & GMP_NUMB_MASK;
	  next_bitpos += bits_per_indigit;
	  if (next_bitpos >= GMP_NUMB_BITS)
	    {
	      rp[size++] = res_digit;
	      next_bitpos -= GMP_NUMB_BITS;
	      res_digit = inp_digit >> (bits_per_indigit - next_bitpos);
	    }
	}

      if (res_digit != 0)
	rp[size++] = res_digit;
      return size;
    }

  if (BELOW_THRESHOLD (str_len, SET_STR_PRECOMPUTE_THRESHOLD))
    return mpn_bc_set_str (rp, str, str_len, base);
  else
    {
      mp_ptr powtab_mem, tp;
      powers_t powtab[GMP_LIMB_BITS];
      int chars_per_limb;
      mp_size_t size;
      mp_size_t un;
      TMP_DECL;

      TMP_MARK;

      chars_per_limb = mp_bases[base].chars_per_limb;

      un = str_len / chars_per_limb + 1;

      /* Allocate one large block for the powers of big_base.  */
      powtab_mem = TMP_BALLOC_LIMBS (mpn_dc_set_str_powtab_alloc (un));

      mpn_set_str_compute_powtab (powtab, powtab_mem, un, base);

      tp = TMP_BALLOC_LIMBS (mpn_dc_set_str_itch (un));
      size = mpn_dc_set_str (rp, str, str_len, powtab, tp);

      TMP_FREE;
      return size;
    }
}

void
mpn_set_str_compute_powtab (powers_t *powtab, mp_ptr powtab_mem, mp_size_t un, int base)
{
  mp_ptr powtab_mem_ptr;
  long i, pi;
  mp_size_t n;
  mp_ptr p, t;
  mp_limb_t big_base;
  int chars_per_limb;
  size_t digits_in_base;
  mp_size_t shift;

  powtab_mem_ptr = powtab_mem;

  chars_per_limb = mp_bases[base].chars_per_limb;
  big_base = mp_bases[base].big_base;

  p = powtab_mem_ptr;
  powtab_mem_ptr += 1;

  digits_in_base = chars_per_limb;

  p[0] = big_base;
  n = 1;

  count_leading_zeros (i, un - 1);
  i = GMP_LIMB_BITS - 1 - i;

  powtab[i].p = p;
  powtab[i].n = n;
  powtab[i].digits_in_base = digits_in_base;
  powtab[i].base = base;
  powtab[i].shift = 0;

  shift = 0;
  for (pi = i - 1; pi >= 0; pi--)
    {
      t = powtab_mem_ptr;
      powtab_mem_ptr += 2 * n;

      ASSERT_ALWAYS (powtab_mem_ptr < powtab_mem + mpn_dc_set_str_powtab_alloc (un));

      mpn_sqr (t, p, n);
      n = 2 * n - 1; n += t[n] != 0;
      digits_in_base *= 2;
#if 1
      if ((((un - 1) >> pi) & 2) == 0)
	{
	  mpn_divexact_1 (t, t, n, big_base);
	  n -= t[n - 1] == 0;
	  digits_in_base -= chars_per_limb;
	}
#else
      if (CLEVER_CONDITION_1 ())
	{
	  /* perform adjustment operation of previous */
	  cy = mpn_mul_1 (p, p, n, big_base);
	}
      if (CLEVER_CONDITION_2 ())
	{
	  /* perform adjustment operation of new */
	  cy = mpn_mul_1 (t, t, n, big_base);
	}
#endif
      shift *= 2;
      /* Strip low zero limbs, but be careful to keep the result divisible by
	 big_base.  */
      while (t[0] == 0 && (t[1] & ((big_base & -big_base) - 1)) == 0)
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
}

mp_size_t
mpn_dc_set_str (mp_ptr rp, const unsigned char *str, size_t str_len,
		const powers_t *powtab, mp_ptr tp)
{
  size_t len_lo, len_hi;
  mp_limb_t cy;
  mp_size_t ln, hn, n, sn;

  len_lo = powtab->digits_in_base;

  if (str_len <= len_lo)
    {
      if (BELOW_THRESHOLD (str_len, SET_STR_DC_THRESHOLD))
	return mpn_bc_set_str (rp, str, str_len, powtab->base);
      else
	return mpn_dc_set_str (rp, str, str_len, powtab + 1, tp);
    }

  len_hi = str_len - len_lo;
  ASSERT (len_lo >= len_hi);

  if (BELOW_THRESHOLD (len_hi, SET_STR_DC_THRESHOLD))
    hn = mpn_bc_set_str (tp, str, len_hi, powtab->base);
  else
    hn = mpn_dc_set_str (tp, str, len_hi, powtab + 1, rp);

  sn = powtab->shift;

  if (hn == 0)
    {
      /* Zero +1 limb here, to avoid reading an allocated but uninitialised
	 limb in mpn_incr_u below.  */
      MPN_ZERO (rp, powtab->n + sn + 1);
    }
  else
    {
      if (powtab->n > hn)
	mpn_mul (rp + sn, powtab->p, powtab->n, tp, hn);
      else
	mpn_mul (rp + sn, tp, hn, powtab->p, powtab->n);
      MPN_ZERO (rp, sn);
    }

  str = str + str_len - len_lo;
  if (BELOW_THRESHOLD (len_lo, SET_STR_DC_THRESHOLD))
    ln = mpn_bc_set_str (tp, str, len_lo, powtab->base);
  else
    ln = mpn_dc_set_str (tp, str, len_lo, powtab + 1, tp + powtab->n + sn + 1);

  if (ln != 0)
    {
      cy = mpn_add_n (rp, rp, tp, ln);
      mpn_incr_u (rp + ln, cy);
    }
  n = hn + powtab->n + sn;
  return n - (rp[n - 1] == 0);
}

mp_size_t
mpn_bc_set_str (mp_ptr rp, const unsigned char *str, size_t str_len, int base)
{
  mp_size_t size;
  size_t i;
  long j;
  mp_limb_t cy_limb;

  mp_limb_t big_base;
  int chars_per_limb;
  mp_limb_t res_digit;

  ASSERT (base >= 2);
  ASSERT (base < numberof (mp_bases));
  ASSERT (str_len >= 1);

  big_base = mp_bases[base].big_base;
  chars_per_limb = mp_bases[base].chars_per_limb;

  size = 0;
  for (i = chars_per_limb; i < str_len; i += chars_per_limb)
    {
      res_digit = *str++;
      if (base == 10)
	{ /* This is a common case.
	     Help the compiler to avoid multiplication.  */
	  for (j = MP_BASES_CHARS_PER_LIMB_10 - 1; j != 0; j--)
	    res_digit = res_digit * 10 + *str++;
	}
      else
	{
	  for (j = chars_per_limb - 1; j != 0; j--)
	    res_digit = res_digit * base + *str++;
	}

      if (size == 0)
	{
	  if (res_digit != 0)
	    {
	      rp[0] = res_digit;
	      size = 1;
	    }
	}
      else
	{
#if HAVE_NATIVE_mpn_mul_1c
	  cy_limb = mpn_mul_1c (rp, rp, size, big_base, res_digit);
#else
	  cy_limb = mpn_mul_1 (rp, rp, size, big_base);
	  cy_limb += mpn_add_1 (rp, rp, size, res_digit);
#endif
	  if (cy_limb != 0)
	    rp[size++] = cy_limb;
	}
    }

  big_base = base;
  res_digit = *str++;
  if (base == 10)
    { /* This is a common case.
	 Help the compiler to avoid multiplication.  */
      for (j = str_len - (i - MP_BASES_CHARS_PER_LIMB_10) - 1; j > 0; j--)
	{
	  res_digit = res_digit * 10 + *str++;
	  big_base *= 10;
	}
    }
  else
    {
      for (j = str_len - (i - chars_per_limb) - 1; j > 0; j--)
	{
	  res_digit = res_digit * base + *str++;
	  big_base *= base;
	}
    }

  if (size == 0)
    {
      if (res_digit != 0)
	{
	  rp[0] = res_digit;
	  size = 1;
	}
    }
  else
    {
#if HAVE_NATIVE_mpn_mul_1c
      cy_limb = mpn_mul_1c (rp, rp, size, big_base, res_digit);
#else
      cy_limb = mpn_mul_1 (rp, rp, size, big_base);
      cy_limb += mpn_add_1 (rp, rp, size, res_digit);
#endif
      if (cy_limb != 0)
	rp[size++] = cy_limb;
    }
  return size;
}
