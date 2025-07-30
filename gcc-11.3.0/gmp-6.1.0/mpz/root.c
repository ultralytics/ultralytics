/* mpz_root(root, u, nth) --  Set ROOT to floor(U^(1/nth)).
   Return an indication if the result is exact.

Copyright 1999-2003, 2005, 2012 Free Software Foundation, Inc.

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

#include <stdio.h>		/* for NULL */
#include "gmp.h"
#include "gmp-impl.h"

int
mpz_root (mpz_ptr root, mpz_srcptr u, unsigned long int nth)
{
  mp_ptr rootp, up;
  mp_size_t us, un, rootn, remn;
  TMP_DECL;

  us = SIZ(u);

  /* even roots of negatives provoke an exception */
  if (UNLIKELY (us < 0 && (nth & 1) == 0))
    SQRT_OF_NEGATIVE;

  /* root extraction interpreted as c^(1/nth) means a zeroth root should
     provoke a divide by zero, do this even if c==0 */
  if (UNLIKELY (nth == 0))
    DIVIDE_BY_ZERO;

  if (us == 0)
    {
      if (root != NULL)
	SIZ(root) = 0;
      return 1;			/* exact result */
    }

  un = ABS (us);
  rootn = (un - 1) / nth + 1;

  TMP_MARK;

  /* FIXME: Perhaps disallow root == NULL */
  if (root != NULL && u != root)
    rootp = MPZ_REALLOC (root, rootn);
  else
    rootp = TMP_ALLOC_LIMBS (rootn);

  up = PTR(u);

  if (nth == 1)
    {
      MPN_COPY (rootp, up, un);
      remn = 0;
    }
  else
    {
      remn = mpn_rootrem (rootp, NULL, up, un, (mp_limb_t) nth);
    }

  if (root != NULL)
    {
      SIZ(root) = us >= 0 ? rootn : -rootn;
      if (u == root)
	MPN_COPY (up, rootp, rootn);
    }

  TMP_FREE;
  return remn == 0;
}
