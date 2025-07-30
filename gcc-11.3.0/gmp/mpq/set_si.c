/* mpq_set_si(dest,ulong_num,ulong_den) -- Set DEST to the rational number
   ULONG_NUM/ULONG_DEN.

Copyright 1991, 1994, 1995, 2001, 2003 Free Software Foundation, Inc.

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

void
mpq_set_si (mpq_t dest, signed long int num, unsigned long int den)
{
  unsigned long int abs_num;

  if (GMP_NUMB_BITS < BITS_PER_ULONG)
    {
      if (num == 0)  /* Canonicalize 0/d to 0/1.  */
        den = 1;
      mpz_set_si (mpq_numref (dest), num);
      mpz_set_ui (mpq_denref (dest), den);
      return;
    }

  abs_num = ABS_CAST (unsigned long, num);

  if (num == 0)
    {
      /* Canonicalize 0/d to 0/1.  */
      den = 1;
      SIZ(NUM(dest)) = 0;
    }
  else
    {
      PTR(NUM(dest))[0] = abs_num;
      SIZ(NUM(dest)) = num > 0 ? 1 : -1;
    }

  PTR(DEN(dest))[0] = den;
  SIZ(DEN(dest)) = (den != 0);
}
