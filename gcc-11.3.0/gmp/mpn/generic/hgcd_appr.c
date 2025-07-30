/* hgcd_appr.c.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2011, 2012 Free Software Foundation, Inc.

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

/* Identical to mpn_hgcd_itch. FIXME: Do we really need to add
   HGCD_THRESHOLD at the end? */
mp_size_t
mpn_hgcd_appr_itch (mp_size_t n)
{
  if (BELOW_THRESHOLD (n, HGCD_APPR_THRESHOLD))
    return n;
  else
    {
      unsigned k;
      int count;
      mp_size_t nscaled;

      /* Get the recursion depth. */
      nscaled = (n - 1) / (HGCD_APPR_THRESHOLD - 1);
      count_leading_zeros (count, nscaled);
      k = GMP_LIMB_BITS - count;

      return 20 * ((n+3) / 4) + 22 * k + HGCD_THRESHOLD;
    }
}

/* Destroys inputs. */
int
mpn_hgcd_appr (mp_ptr ap, mp_ptr bp, mp_size_t n,
	       struct hgcd_matrix *M, mp_ptr tp)
{
  mp_size_t s;
  int success = 0;

  ASSERT (n > 0);

  ASSERT ((ap[n-1] | bp[n-1]) != 0);

  if (n <= 2)
    /* Implies s = n. A fairly uninteresting case but exercised by the
       random inputs of the testsuite. */
    return 0;

  ASSERT ((n+1)/2 - 1 < M->alloc);

  /* We aim for reduction of to GMP_NUMB_BITS * s bits. But each time
     we discard some of the least significant limbs, we must keep one
     additional bit to account for the truncation error. We maintain
     the GMP_NUMB_BITS * s - extra_bits as the current target size. */

  s = n/2 + 1;
  if (BELOW_THRESHOLD (n, HGCD_APPR_THRESHOLD))
    {
      unsigned extra_bits = 0;

      while (n > 2)
	{
	  mp_size_t nn;

	  ASSERT (n > s);
	  ASSERT (n <= 2*s);

	  nn = mpn_hgcd_step (n, ap, bp, s, M, tp);
	  if (!nn)
	    break;

	  n = nn;
	  success = 1;

	  /* We can truncate and discard the lower p bits whenever nbits <=
	     2*sbits - p. To account for the truncation error, we must
	     adjust

	     sbits <-- sbits + 1 - p,

	     rather than just sbits <-- sbits - p. This adjustment makes
	     the produced matrix slightly smaller than it could be. */

	  if (GMP_NUMB_BITS * (n + 1) + 2 * extra_bits <= 2*GMP_NUMB_BITS * s)
	    {
	      mp_size_t p = (GMP_NUMB_BITS * (2*s - n) - 2*extra_bits) / GMP_NUMB_BITS;

	      if (extra_bits == 0)
		{
		  /* We cross a limb boundary and bump s. We can't do that
		     if the result is that it makes makes min(U, V)
		     smaller than 2^{GMP_NUMB_BITS} s. */
		  if (s + 1 == n
		      || mpn_zero_p (ap + s + 1, n - s - 1)
		      || mpn_zero_p (bp + s + 1, n - s - 1))
		    continue;

		  extra_bits = GMP_NUMB_BITS - 1;
		  s++;
		}
	      else
		{
		  extra_bits--;
		}

	      /* Drop the p least significant limbs */
	      ap += p; bp += p; n -= p; s -= p;
	    }
	}

      ASSERT (s > 0);

      if (extra_bits > 0)
	{
	  /* We can get here only of we have dropped at least one of the least
	     significant bits, so we can decrement ap and bp. We can then shift
	     left extra bits using mpn_rshift. */
	  /* NOTE: In the unlikely case that n is large, it would be preferable
	     to do an initial subdiv step to reduce the size before shifting,
	     but that would mean duplicating mpn_gcd_subdiv_step with a bit
	     count rather than a limb count. */
	  ap--; bp--;
	  ap[0] = mpn_rshift (ap+1, ap+1, n, GMP_NUMB_BITS - extra_bits);
	  bp[0] = mpn_rshift (bp+1, bp+1, n, GMP_NUMB_BITS - extra_bits);
	  n += (ap[n] | bp[n]) > 0;

	  ASSERT (success);

	  while (n > 2)
	    {
	      mp_size_t nn;

	      ASSERT (n > s);
	      ASSERT (n <= 2*s);

	      nn = mpn_hgcd_step (n, ap, bp, s, M, tp);

	      if (!nn)
		return 1;

	      n = nn;
	    }
	}

      if (n == 2)
	{
	  struct hgcd_matrix1 M1;
	  ASSERT (s == 1);

	  if (mpn_hgcd2 (ap[1], ap[0], bp[1], bp[0], &M1))
	    {
	      /* Multiply M <- M * M1 */
	      mpn_hgcd_matrix_mul_1 (M, &M1, tp);
	      success = 1;
	    }
	}
      return success;
    }
  else
    {
      mp_size_t n2 = (3*n)/4 + 1;
      mp_size_t p = n/2;
      mp_size_t nn;

      nn = mpn_hgcd_reduce (M, ap, bp, n, p, tp);
      if (nn)
	{
	  n = nn;
	  /* FIXME: Discard some of the low limbs immediately? */
	  success = 1;
	}

      while (n > n2)
	{
	  mp_size_t nn;

	  /* Needs n + 1 storage */
	  nn = mpn_hgcd_step (n, ap, bp, s, M, tp);
	  if (!nn)
	    return success;

	  n = nn;
	  success = 1;
	}
      if (n > s + 2)
	{
	  struct hgcd_matrix M1;
	  mp_size_t scratch;

	  p = 2*s - n + 1;
	  scratch = MPN_HGCD_MATRIX_INIT_ITCH (n-p);

	  mpn_hgcd_matrix_init(&M1, n - p, tp);
	  if (mpn_hgcd_appr (ap + p, bp + p, n - p, &M1, tp + scratch))
	    {
	      /* We always have max(M) > 2^{-(GMP_NUMB_BITS + 1)} max(M1) */
	      ASSERT (M->n + 2 >= M1.n);

	      /* Furthermore, assume M ends with a quotient (1, q; 0, 1),
		 then either q or q + 1 is a correct quotient, and M1 will
		 start with either (1, 0; 1, 1) or (2, 1; 1, 1). This
		 rules out the case that the size of M * M1 is much
		 smaller than the expected M->n + M1->n. */

	      ASSERT (M->n + M1.n < M->alloc);

	      /* We need a bound for of M->n + M1.n. Let n be the original
		 input size. Then

		 ceil(n/2) - 1 >= size of product >= M.n + M1.n - 2

		 and it follows that

		 M.n + M1.n <= ceil(n/2) + 1

		 Then 3*(M.n + M1.n) + 5 <= 3 * ceil(n/2) + 8 is the
		 amount of needed scratch space. */
	      mpn_hgcd_matrix_mul (M, &M1, tp + scratch);
	      return 1;
	    }
	}

      for(;;)
	{
	  mp_size_t nn;

	  ASSERT (n > s);
	  ASSERT (n <= 2*s);

	  nn = mpn_hgcd_step (n, ap, bp, s, M, tp);

	  if (!nn)
	    return success;

	  n = nn;
	  success = 1;
	}
    }
}
