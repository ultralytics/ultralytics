/* mpz_gcdext(g, s, t, a, b) -- Set G to gcd(a, b), and S and T such that
   g = as + bt.

Copyright 1991, 1993-1997, 2000, 2001, 2005, 2011, 2012 Free Software
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

#include <stdio.h> /* for NULL */
#include "gmp.h"
#include "gmp-impl.h"

void
mpz_gcdext (mpz_ptr g, mpz_ptr s, mpz_ptr t, mpz_srcptr a, mpz_srcptr b)
{
  mp_size_t asize, bsize;
  mp_ptr tmp_ap, tmp_bp;
  mp_size_t gsize, ssize, tmp_ssize;
  mp_ptr gp, tmp_gp, tmp_sp;
  TMP_DECL;

  /* mpn_gcdext requires that Usize >= Vsize.  Therefore, we often
     have to swap U and V.  The computed cofactor will be the
     "smallest" one, which is faster to produce.  The wanted one will
     be computed here; this is needed anyway when both are requested.  */

  asize = ABSIZ (a);
  bsize = ABSIZ (b);

  if (asize < bsize)
    {
      MPZ_SRCPTR_SWAP (a, b);
      MP_SIZE_T_SWAP (asize, bsize);
      MPZ_PTR_SWAP (s, t);
    }

  if (bsize == 0)
    {
      /* g = |a|, s = sgn(a), t = 0. */
      ssize = SIZ (a) >= 0 ? (asize != 0) : -1;

      gp = MPZ_REALLOC (g, asize);
      MPN_COPY (gp, PTR (a), asize);
      SIZ (g) = asize;

      if (t != NULL)
	SIZ (t) = 0;
      if (s != NULL)
	{
	  SIZ (s) = ssize;
	  PTR (s)[0] = 1;
	}
      return;
    }

  TMP_MARK;

  TMP_ALLOC_LIMBS_2 (tmp_ap, asize, tmp_bp, bsize);
  MPN_COPY (tmp_ap, PTR (a), asize);
  MPN_COPY (tmp_bp, PTR (b), bsize);

  TMP_ALLOC_LIMBS_2 (tmp_gp, bsize, tmp_sp, bsize + 1);

  gsize = mpn_gcdext (tmp_gp, tmp_sp, &tmp_ssize, tmp_ap, asize, tmp_bp, bsize);

  ssize = ABS (tmp_ssize);
  tmp_ssize = SIZ (a) >= 0 ? tmp_ssize : -tmp_ssize;

  if (t != NULL)
    {
      mpz_t x;
      __mpz_struct gtmp, stmp;

      PTR (&gtmp) = tmp_gp;
      SIZ (&gtmp) = gsize;

      PTR (&stmp) = tmp_sp;
      SIZ (&stmp) = tmp_ssize;

      MPZ_TMP_INIT (x, ssize + asize + 1);
      mpz_mul (x, &stmp, a);
      mpz_sub (x, &gtmp, x);
      mpz_divexact (t, x, b);
    }

  if (s != NULL)
    {
      mp_ptr sp;

      sp = MPZ_REALLOC (s, ssize);
      MPN_COPY (sp, tmp_sp, ssize);
      SIZ (s) = tmp_ssize;
    }

  gp = MPZ_REALLOC (g, gsize);
  MPN_COPY (gp, tmp_gp, gsize);
  SIZ (g) = gsize;

  TMP_FREE;
}
