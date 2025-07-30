/* mpz_divexact_gcd -- exact division optimized for GCDs.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE AND ARE ALMOST CERTAIN TO
   BE SUBJECT TO INCOMPATIBLE CHANGES IN FUTURE GNU MP RELEASES.

Copyright 2000, 2005, 2011, 2012 Free Software Foundation, Inc.

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


/* Set q to a/d, expecting d to be from a GCD and therefore usually small.

   The distribution of GCDs of random numbers can be found in Knuth volume 2
   section 4.5.2 theorem D.

            GCD     chance
             1       60.8%
            2^k      20.2%     (1<=k<32)
           3*2^k      9.0%     (1<=k<32)
           other     10.1%

   Only the low limb is examined for optimizations, since GCDs bigger than
   2^32 (or 2^64) will occur very infrequently.

   Future: This could change to an mpn_divexact_gcd, possibly partly
   inlined, if/when the relevant mpq functions change to an mpn based
   implementation.  */


#if GMP_NUMB_BITS % 2 == 0
static void
mpz_divexact_by3 (mpz_ptr q, mpz_srcptr a)
{
  mp_size_t  size = SIZ(a);
  mp_size_t  abs_size = ABS(size);
  mp_ptr     qp;

  qp = MPZ_REALLOC (q, abs_size);

  mpn_bdiv_dbm1 (qp, PTR(a), abs_size, GMP_NUMB_MASK / 3);

  abs_size -= (qp[abs_size-1] == 0);
  SIZ(q) = (size>0 ? abs_size : -abs_size);
}
#endif

#if GMP_NUMB_BITS % 4 == 0
static void
mpz_divexact_by5 (mpz_ptr q, mpz_srcptr a)
{
  mp_size_t  size = SIZ(a);
  mp_size_t  abs_size = ABS(size);
  mp_ptr     qp;

  qp = MPZ_REALLOC (q, abs_size);

  mpn_bdiv_dbm1 (qp, PTR(a), abs_size, GMP_NUMB_MASK / 5);

  abs_size -= (qp[abs_size-1] == 0);
  SIZ(q) = (size>0 ? abs_size : -abs_size);
}
#endif

static void
mpz_divexact_limb (mpz_ptr q, mpz_srcptr a, mp_limb_t d)
{
  mp_size_t  size = SIZ(a);
  mp_size_t  abs_size = ABS(size);
  mp_ptr     qp;

  qp = MPZ_REALLOC (q, abs_size);

  mpn_divexact_1 (qp, PTR(a), abs_size, d);

  abs_size -= (qp[abs_size-1] == 0);
  SIZ(q) = (size>0 ? abs_size : -abs_size);
}

void
mpz_divexact_gcd (mpz_ptr q, mpz_srcptr a, mpz_srcptr d)
{
  ASSERT (mpz_sgn (d) > 0);

  if (SIZ(a) == 0)
    {
      SIZ(q) = 0;
      return;
    }

  if (SIZ(d) == 1)
    {
      mp_limb_t  dl = PTR(d)[0];
      int        twos;

      if ((dl & 1) == 0)
	{
	  count_trailing_zeros (twos, dl);
	  dl >>= twos;
	  mpz_tdiv_q_2exp (q, a, twos);
	  a = q;
	}

      if (dl == 1)
	{
	  if (q != a)
	    mpz_set (q, a);
	  return;
	}
#if GMP_NUMB_BITS % 2 == 0
      if (dl == 3)
	{
	  mpz_divexact_by3 (q, a);
	  return;
	}
#endif
#if GMP_NUMB_BITS % 4 == 0
      if (dl == 5)
	{
	  mpz_divexact_by5 (q, a);
	  return;
	}
#endif

      mpz_divexact_limb (q, a, dl);
      return;
    }

  mpz_divexact (q, a, d);
}
