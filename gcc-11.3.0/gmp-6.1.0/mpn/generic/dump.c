/* THIS IS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS NOT SAFE TO
   CALL THIS FUNCTION DIRECTLY.  IN FACT, IT IS ALMOST GUARANTEED THAT THIS
   FUNCTION WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.


Copyright 1996, 2000-2002, 2005 Free Software Foundation, Inc.

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

#include <stdio.h>
#include "gmp.h"
#include "gmp-impl.h"

#if GMP_NUMB_BITS % 4 == 0
void
mpn_dump (mp_srcptr ptr, mp_size_t n)
{
  MPN_NORMALIZE (ptr, n);

  if (n == 0)
    printf ("0\n");
  else
    {
      n--;
#if _LONG_LONG_LIMB
      if ((ptr[n] >> GMP_LIMB_BITS / 2) != 0)
	{
	  printf ("%lX", (unsigned long) (ptr[n] >> GMP_LIMB_BITS / 2));
	  printf ("%0*lX", (GMP_LIMB_BITS / 2 / 4), (unsigned long) ptr[n]);
	}
      else
#endif
	printf ("%lX", (unsigned long) ptr[n]);

      while (n)
	{
	  n--;
#if _LONG_LONG_LIMB
	  printf ("%0*lX", (GMP_NUMB_BITS - GMP_LIMB_BITS / 2) / 4,
		  (unsigned long) (ptr[n] >> GMP_LIMB_BITS / 2));
	  printf ("%0*lX", GMP_LIMB_BITS / 2 / 4, (unsigned long) ptr[n]);
#else
	  printf ("%0*lX", GMP_NUMB_BITS / 4, (unsigned long) ptr[n]);
#endif
	}
      printf ("\n");
    }
}

#else

static void
mpn_recdump (mp_ptr p, mp_size_t n)
{
  mp_limb_t lo;
  if (n != 0)
    {
      lo = p[0] & 0xf;
      mpn_rshift (p, p, n, 4);
      mpn_recdump (p, n);
      printf ("%lX", lo);
    }
}

void
mpn_dump (mp_srcptr p, mp_size_t n)
{
  mp_ptr tp;
  TMP_DECL;
  TMP_MARK;
  tp = TMP_ALLOC_LIMBS (n);
  MPN_COPY (tp, p, n);
  TMP_FREE;
}

#endif
