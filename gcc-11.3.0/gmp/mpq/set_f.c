/* mpq_set_f -- set an mpq from an mpf.

Copyright 2000-2002 Free Software Foundation, Inc.

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


void
mpq_set_f (mpq_ptr q, mpf_srcptr f)
{
  mp_size_t  fexp = EXP(f);
  mp_ptr     fptr = PTR(f);
  mp_size_t  fsize = SIZ(f);
  mp_size_t  abs_fsize = ABS(fsize);
  mp_limb_t  flow;

  if (fsize == 0)
    {
      /* set q=0 */
      SIZ(NUM(q)) = 0;
      SIZ(DEN(q)) = 1;
      PTR(DEN(q))[0] = 1;
      return;
    }

  /* strip low zero limbs from f */
  flow = *fptr;
  MPN_STRIP_LOW_ZEROS_NOT_ZERO (fptr, abs_fsize, flow);

  if (fexp >= abs_fsize)
    {
      /* radix point is to the right of the limbs, no denominator */
      mp_ptr  num_ptr;

      num_ptr = MPZ_NEWALLOC (mpq_numref (q), fexp);
      MPN_ZERO (num_ptr, fexp - abs_fsize);
      MPN_COPY (num_ptr + fexp - abs_fsize, fptr, abs_fsize);

      SIZ(NUM(q)) = fsize >= 0 ? fexp : -fexp;
      SIZ(DEN(q)) = 1;
      PTR(DEN(q))[0] = 1;
    }
  else
    {
      /* radix point is within or to the left of the limbs, use denominator */
      mp_ptr     num_ptr, den_ptr;
      mp_size_t  den_size;

      den_size = abs_fsize - fexp;
      num_ptr = MPZ_NEWALLOC (mpq_numref (q), abs_fsize);
      den_ptr = MPZ_NEWALLOC (mpq_denref (q), den_size+1);

      if (flow & 1)
        {
          /* no powers of two to strip from numerator */

          MPN_COPY (num_ptr, fptr, abs_fsize);
          MPN_ZERO (den_ptr, den_size);
          den_ptr[den_size] = 1;
        }
      else
        {
          /* right shift numerator, adjust denominator accordingly */
          int  shift;

          den_size--;
          count_trailing_zeros (shift, flow);

          mpn_rshift (num_ptr, fptr, abs_fsize, shift);
          abs_fsize -= (num_ptr[abs_fsize-1] == 0);

          MPN_ZERO (den_ptr, den_size);
          den_ptr[den_size] = GMP_LIMB_HIGHBIT >> (shift-1);
        }

      SIZ(NUM(q)) = fsize >= 0 ? abs_fsize : -abs_fsize;
      SIZ(DEN(q)) = den_size + 1;
    }
}
