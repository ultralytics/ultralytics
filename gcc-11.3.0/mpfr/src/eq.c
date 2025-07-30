/* mpfr_eq -- Compare two floats up to a specified bit #.

Copyright 1999, 2001, 2003-2004, 2006-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */


#include "mpfr-impl.h"

/* return non-zero if the first n_bits bits of u, v are equal,
   0 otherwise */
int
mpfr_eq (mpfr_srcptr u, mpfr_srcptr v, unsigned long int n_bits)
{
  mpfr_limb_srcptr up, vp;
  mp_size_t usize, vsize, size, i;
  mpfr_exp_t uexp, vexp;
  int k;

  if (MPFR_ARE_SINGULAR(u, v))
    {
      if (MPFR_IS_NAN(u) || MPFR_IS_NAN(v))
        return 0; /* non equal */
      else if (MPFR_IS_INF(u) && MPFR_IS_INF(v))
        return (MPFR_SIGN(u) == MPFR_SIGN(v));
      else if (MPFR_IS_ZERO(u) && MPFR_IS_ZERO(v))
        return 1;
      else
        return 0;
    }

  /* 1. Are the signs different?  */
  if (MPFR_SIGN(u) != MPFR_SIGN(v))
    return 0;

  uexp = MPFR_GET_EXP (u);
  vexp = MPFR_GET_EXP (v);

  /* 2. Are the exponents different?  */
  if (uexp != vexp)
    return 0; /* no bit agree */

  usize = MPFR_LIMB_SIZE (u);
  vsize = MPFR_LIMB_SIZE (v);

  if (vsize > usize) /* exchange u and v */
    {
      up = MPFR_MANT(v);
      vp = MPFR_MANT(u);
      size = vsize;
      vsize = usize;
      usize = size;
    }
  else
    {
      up = MPFR_MANT(u);
      vp = MPFR_MANT(v);
    }

  /* now usize >= vsize */
  MPFR_ASSERTD(usize >= vsize);

  if (usize > vsize)
    {
      if ((unsigned long) vsize * GMP_NUMB_BITS < n_bits)
        {
          /* check if low min(PREC(u), n_bits) - (vsize * GMP_NUMB_BITS)
             bits from u are non-zero */
          unsigned long remains = n_bits - (vsize * GMP_NUMB_BITS);
          k = usize - vsize - 1;
          while (k >= 0 && remains >= GMP_NUMB_BITS && !up[k])
            {
              k--;
              remains -= GMP_NUMB_BITS;
            }
          /* now either k < 0: all low bits from u are zero
                 or remains < GMP_NUMB_BITS: check high bits from up[k]
                 or up[k] <> 0: different */
          if (k >= 0 && (((remains < GMP_NUMB_BITS) &&
                          (up[k] >> (GMP_NUMB_BITS - remains))) ||
                         (remains >= GMP_NUMB_BITS && up[k])))
            return 0;           /* surely too different */
        }
      size = vsize;
    }
  else
    {
      size = usize;
    }

  /* now size = min (usize, vsize) */

  /* If size is too large wrt n_bits, reduce it to look only at the
     high n_bits bits.
     Otherwise, if n_bits > size * GMP_NUMB_BITS, reduce n_bits to
     size * GMP_NUMB_BITS, since the extra low bits of one of the
     operands have already been check above. */
  if ((unsigned long) size > 1 + (n_bits - 1) / GMP_NUMB_BITS)
    size = 1 + (n_bits - 1) / GMP_NUMB_BITS;
  else if (n_bits > (unsigned long) size * GMP_NUMB_BITS)
    n_bits = size * GMP_NUMB_BITS;

  up += usize - size;
  vp += vsize - size;

  for (i = size - 1; i > 0 && n_bits >= GMP_NUMB_BITS; i--)
    {
      if (up[i] != vp[i])
        return 0;
      n_bits -= GMP_NUMB_BITS;
    }

  /* now either i=0 or n_bits<GMP_NUMB_BITS */

  /* since n_bits <= size * GMP_NUMB_BITS before the above for-loop,
     we have the invariant n_bits <= (i+1) * GMP_NUMB_BITS, thus
     we always have n_bits <= GMP_NUMB_BITS here */
  MPFR_ASSERTD(n_bits <= GMP_NUMB_BITS);

  if (n_bits & (GMP_NUMB_BITS - 1))
    return (up[i] >> (GMP_NUMB_BITS - (n_bits & (GMP_NUMB_BITS - 1))) ==
            vp[i] >> (GMP_NUMB_BITS - (n_bits & (GMP_NUMB_BITS - 1))));
  else
    return (up[i] == vp[i]);
}
