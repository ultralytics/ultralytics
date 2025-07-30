/* Test mpn_get_d.

Copyright 2002-2004 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include "config.h"

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


#ifndef _GMP_IEEE_FLOATS
#define _GMP_IEEE_FLOATS 0
#endif


/* Exercise various 2^n values, with various exponents and positive and
   negative.  */
void
check_onebit (void)
{
  static const int bit_table[] = {
    0, 1, 2, 3,
    GMP_NUMB_BITS - 2, GMP_NUMB_BITS - 1,
    GMP_NUMB_BITS,
    GMP_NUMB_BITS + 1, GMP_NUMB_BITS + 2,
    2 * GMP_NUMB_BITS - 2, 2 * GMP_NUMB_BITS - 1,
    2 * GMP_NUMB_BITS,
    2 * GMP_NUMB_BITS + 1, 2 * GMP_NUMB_BITS + 2,
    3 * GMP_NUMB_BITS - 2, 3 * GMP_NUMB_BITS - 1,
    3 * GMP_NUMB_BITS,
    3 * GMP_NUMB_BITS + 1, 3 * GMP_NUMB_BITS + 2,
    4 * GMP_NUMB_BITS - 2, 4 * GMP_NUMB_BITS - 1,
    4 * GMP_NUMB_BITS,
    4 * GMP_NUMB_BITS + 1, 4 * GMP_NUMB_BITS + 2,
    5 * GMP_NUMB_BITS - 2, 5 * GMP_NUMB_BITS - 1,
    5 * GMP_NUMB_BITS,
    5 * GMP_NUMB_BITS + 1, 5 * GMP_NUMB_BITS + 2,
    6 * GMP_NUMB_BITS - 2, 6 * GMP_NUMB_BITS - 1,
    6 * GMP_NUMB_BITS,
    6 * GMP_NUMB_BITS + 1, 6 * GMP_NUMB_BITS + 2,
  };
  static const int exp_table[] = {
    0, -100, -10, -1, 1, 10, 100,
  };

  /* FIXME: It'd be better to base this on the float format. */
#if defined (__vax) || defined (__vax__)
  int     limit = 127;  /* vax fp numbers have limited range */
#else
  int     limit = 511;
#endif

  int        bit_i, exp_i, i;
  double     got, want;
  mp_size_t  nsize, sign;
  long       bit, exp, want_bit;
  mp_limb_t  np[20];

  for (bit_i = 0; bit_i < numberof (bit_table); bit_i++)
    {
      bit = bit_table[bit_i];

      nsize = BITS_TO_LIMBS (bit+1);
      refmpn_zero (np, nsize);
      np[bit/GMP_NUMB_BITS] = CNST_LIMB(1) << (bit % GMP_NUMB_BITS);

      for (exp_i = 0; exp_i < numberof (exp_table); exp_i++)
        {
          exp = exp_table[exp_i];

          want_bit = bit + exp;
          if (want_bit >= limit || want_bit <= -limit)
            continue;

          want = 1.0;
          for (i = 0; i < want_bit; i++)
            want *= 2.0;
          for (i = 0; i > want_bit; i--)
            want *= 0.5;

          for (sign = 0; sign >= -1; sign--, want = -want)
            {
              got = mpn_get_d (np, nsize, sign, exp);
              if (got != want)
                {
                  printf    ("mpn_get_d wrong on 2^n\n");
                  printf    ("   bit      %ld\n", bit);
                  printf    ("   exp      %ld\n", exp);
                  printf    ("   want_bit %ld\n", want_bit);
                  printf    ("   sign     %ld\n", (long) sign);
                  mpn_trace ("   n        ", np, nsize);
                  printf    ("   nsize    %ld\n", (long) nsize);
                  d_trace   ("   want     ", want);
                  d_trace   ("   got      ", got);
                  abort();
                }
            }
        }
    }
}


/* Exercise values 2^n+1, while such a value fits the mantissa of a double. */
void
check_twobit (void)
{
  int        i, mant_bits;
  double     got, want;
  mp_size_t  nsize, sign;
  mp_ptr     np;

  mant_bits = tests_dbl_mant_bits ();
  if (mant_bits == 0)
    return;

  np = refmpn_malloc_limbs (BITS_TO_LIMBS (mant_bits));
  want = 3.0;
  for (i = 1; i < mant_bits; i++)
    {
      nsize = BITS_TO_LIMBS (i+1);
      refmpn_zero (np, nsize);
      np[i/GMP_NUMB_BITS] = CNST_LIMB(1) << (i % GMP_NUMB_BITS);
      np[0] |= 1;

      for (sign = 0; sign >= -1; sign--)
        {
          got = mpn_get_d (np, nsize, sign, 0);
          if (got != want)
            {
              printf    ("mpn_get_d wrong on 2^%d + 1\n", i);
              printf    ("   sign     %ld\n", (long) sign);
              mpn_trace ("   n        ", np, nsize);
              printf    ("   nsize    %ld\n", (long) nsize);
              d_trace   ("   want     ", want);
              d_trace   ("   got      ", got);
              abort();
            }
          want = -want;
        }

      want = 2.0 * want - 1.0;
    }

  free (np);
}


/* Expect large negative exponents to underflow to 0.0.
   Some systems might have hardware traps for such an underflow (though
   usually it's not the default), so watch out for SIGFPE. */
void
check_underflow (void)
{
  static const long exp_table[] = {
    -999999L, LONG_MIN,
  };
  static const mp_limb_t  np[1] = { 1 };

  static long exp;
  mp_size_t  nsize, sign;
  double     got;
  int        exp_i;

  nsize = numberof (np);

  if (tests_setjmp_sigfpe() == 0)
    {
      for (exp_i = 0; exp_i < numberof (exp_table); exp_i++)
        {
          exp = exp_table[exp_i];

          for (sign = 0; sign >= -1; sign--)
            {
              got = mpn_get_d (np, nsize, sign, exp);
              if (got != 0.0)
                {
                  printf  ("mpn_get_d wrong, didn't get 0.0 on underflow\n");
                  printf  ("  nsize    %ld\n", (long) nsize);
                  printf  ("  exp      %ld\n", exp);
                  printf  ("  sign     %ld\n", (long) sign);
                  d_trace ("  got      ", got);
                  abort ();
                }
            }
        }
    }
  else
    {
      printf ("Warning, underflow to zero tests skipped due to SIGFPE (exp=%ld)\n", exp);
    }
  tests_sigfpe_done ();
}


/* Expect large values to result in +/-inf, on IEEE systems. */
void
check_inf (void)
{
  static const long exp_table[] = {
    999999L, LONG_MAX,
  };
  static const mp_limb_t  np[4] = { 1, 1, 1, 1 };
  long       exp;
  mp_size_t  nsize, sign, got_sign;
  double     got;
  int        exp_i;

  if (! _GMP_IEEE_FLOATS)
    return;

  for (nsize = 1; nsize <= numberof (np); nsize++)
    {
      for (exp_i = 0; exp_i < numberof (exp_table); exp_i++)
        {
          exp = exp_table[exp_i];

          for (sign = 0; sign >= -1; sign--)
            {
              got = mpn_get_d (np, nsize, sign, exp);
              got_sign = (got >= 0 ? 0 : -1);
              if (! tests_isinf (got))
                {
                  printf  ("mpn_get_d wrong, didn't get infinity\n");
                bad:
                  printf  ("  nsize    %ld\n", (long) nsize);
                  printf  ("  exp      %ld\n", exp);
                  printf  ("  sign     %ld\n", (long) sign);
                  d_trace ("  got      ", got);
                  printf  ("  got sign %ld\n", (long) got_sign);
                  abort ();
                }
              if (got_sign != sign)
                {
                  printf  ("mpn_get_d wrong sign on infinity\n");
                  goto bad;
                }
            }
        }
    }
}

/* Check values 2^n approaching and into IEEE denorm range.
   Some systems might not support denorms, or might have traps setup, so
   watch out for SIGFPE.  */
void
check_ieee_denorm (void)
{
  static long exp;
  mp_limb_t  n = 1;
  long       i;
  mp_size_t  sign;
  double     want, got;

  if (! _GMP_IEEE_FLOATS)
    return;

  if (tests_setjmp_sigfpe() == 0)
    {
      exp = -1020;
      want = 1.0;
      for (i = 0; i > exp; i--)
        want *= 0.5;

      for ( ; exp > -1500 && want != 0.0; exp--)
        {
          for (sign = 0; sign >= -1; sign--)
            {
              got = mpn_get_d (&n, (mp_size_t) 1, sign, exp);
              if (got != want)
                {
                  printf  ("mpn_get_d wrong on denorm\n");
                  printf  ("  n=1\n");
                  printf  ("  exp   %ld\n", exp);
                  printf  ("  sign  %ld\n", (long) sign);
                  d_trace ("  got   ", got);
                  d_trace ("  want  ", want);
                  abort ();
                }
              want = -want;
            }
          want *= 0.5;
          FORCE_DOUBLE (want);
        }
    }
  else
    {
      printf ("Warning, IEEE denorm tests skipped due to SIGFPE (exp=%ld)\n", exp);
    }
  tests_sigfpe_done ();
}


/* Check values 2^n approaching exponent overflow.
   Some systems might trap on overflow, so watch out for SIGFPE.  */
void
check_ieee_overflow (void)
{
  static long exp;
  mp_limb_t  n = 1;
  long       i;
  mp_size_t  sign;
  double     want, got;

  if (! _GMP_IEEE_FLOATS)
    return;

  if (tests_setjmp_sigfpe() == 0)
    {
      exp = 1010;
      want = 1.0;
      for (i = 0; i < exp; i++)
        want *= 2.0;

      for ( ; exp < 1050; exp++)
        {
          for (sign = 0; sign >= -1; sign--)
            {
              got = mpn_get_d (&n, (mp_size_t) 1, sign, exp);
              if (got != want)
                {
                  printf  ("mpn_get_d wrong on overflow\n");
                  printf  ("  n=1\n");
                  printf  ("  exp   %ld\n", exp);
                  printf  ("  sign  %ld\n", (long) sign);
                  d_trace ("  got   ", got);
                  d_trace ("  want  ", want);
                  abort ();
                }
              want = -want;
            }
          want *= 2.0;
          FORCE_DOUBLE (want);
        }
    }
  else
    {
      printf ("Warning, IEEE overflow tests skipped due to SIGFPE (exp=%ld)\n", exp);
    }
  tests_sigfpe_done ();
}


/* ARM gcc 2.95.4 was seen generating bad code for ulong->double
   conversions, resulting in for instance 0x81c25113 incorrectly converted.
   This test exercises that value, to see mpn_get_d has avoided the
   problem.  */
void
check_0x81c25113 (void)
{
#if GMP_NUMB_BITS >= 32
  double     want = 2176995603.0;
  double     got;
  mp_limb_t  np[4];
  mp_size_t  nsize;
  long       exp;

  if (tests_dbl_mant_bits() < 32)
    return;

  for (nsize = 1; nsize <= numberof (np); nsize++)
    {
      refmpn_zero (np, nsize-1);
      np[nsize-1] = CNST_LIMB(0x81c25113);
      exp = - (nsize-1) * GMP_NUMB_BITS;
      got = mpn_get_d (np, nsize, (mp_size_t) 0, exp);
      if (got != want)
        {
          printf  ("mpn_get_d wrong on 2176995603 (0x81c25113)\n");
          printf  ("  nsize  %ld\n", (long) nsize);
          printf  ("  exp    %ld\n", exp);
          d_trace ("  got    ", got);
          d_trace ("  want   ", want);
          abort ();
        }
    }
#endif
}


void
check_rand (void)
{
  gmp_randstate_ptr rands = RANDS;
  int            rep, i;
  unsigned long  mant_bits;
  long           exp, exp_min, exp_max;
  double         got, want, d;
  mp_size_t      nalloc, nsize, sign;
  mp_limb_t      nhigh_mask;
  mp_ptr         np;

  mant_bits = tests_dbl_mant_bits ();
  if (mant_bits == 0)
    return;

  /* Allow for vax D format with exponent 127 to -128 only.
     FIXME: Do something to probe for a valid exponent range.  */
  exp_min = -100 - mant_bits;
  exp_max =  100 - mant_bits;

  /* space for mant_bits */
  nalloc = BITS_TO_LIMBS (mant_bits);
  np = refmpn_malloc_limbs (nalloc);
  nhigh_mask = MP_LIMB_T_MAX
    >> (GMP_NAIL_BITS + nalloc * GMP_NUMB_BITS - mant_bits);

  for (rep = 0; rep < 200; rep++)
    {
      /* random exp_min to exp_max, inclusive */
      exp = exp_min + (long) gmp_urandomm_ui (rands, exp_max - exp_min + 1);

      /* mant_bits worth of random at np */
      if (rep & 1)
        mpn_random (np, nalloc);
      else
        mpn_random2 (np, nalloc);
      nsize = nalloc;
      np[nsize-1] &= nhigh_mask;
      MPN_NORMALIZE (np, nsize);
      if (nsize == 0)
        continue;

      sign = (mp_size_t) gmp_urandomb_ui (rands, 1L) - 1;

      /* want = {np,nsize}, converting one bit at a time */
      want = 0.0;
      for (i = 0, d = 1.0; i < mant_bits; i++, d *= 2.0)
        if (np[i/GMP_NUMB_BITS] & (CNST_LIMB(1) << (i%GMP_NUMB_BITS)))
          want += d;
      if (sign < 0)
        want = -want;

      /* want = want * 2^exp */
      for (i = 0; i < exp; i++)
        want *= 2.0;
      for (i = 0; i > exp; i--)
        want *= 0.5;

      got = mpn_get_d (np, nsize, sign, exp);

      if (got != want)
        {
          printf    ("mpn_get_d wrong on random data\n");
          printf    ("   sign     %ld\n", (long) sign);
          mpn_trace ("   n        ", np, nsize);
          printf    ("   nsize    %ld\n", (long) nsize);
          printf    ("   exp      %ld\n", exp);
          d_trace   ("   want     ", want);
          d_trace   ("   got      ", got);
          abort();
        }
    }

  free (np);
}


int
main (void)
{
  tests_start ();
  mp_trace_base = -16;

  check_onebit ();
  check_twobit ();
  check_inf ();
  check_underflow ();
  check_ieee_denorm ();
  check_ieee_overflow ();
  check_0x81c25113 ();
#if ! (defined (__vax) || defined (__vax__))
  check_rand ();
#endif

  tests_end ();
  exit (0);
}
