/* __gmp_extract_double -- convert from double to array of mp_limb_t.

Copyright 1996, 1999-2002, 2006, 2012 Free Software Foundation, Inc.

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

#ifdef XDEBUG
#undef _GMP_IEEE_FLOATS
#endif

#ifndef _GMP_IEEE_FLOATS
#define _GMP_IEEE_FLOATS 0
#endif

/* Extract a non-negative double in d.  */

int
__gmp_extract_double (mp_ptr rp, double d)
{
  long exp;
  unsigned sc;
#ifdef _LONG_LONG_LIMB
#define BITS_PER_PART 64	/* somewhat bogus */
  unsigned long long int manl;
#else
#define BITS_PER_PART GMP_LIMB_BITS
  unsigned long int manh, manl;
#endif

  /* BUGS

     1. Should handle Inf and NaN in IEEE specific code.
     2. Handle Inf and NaN also in default code, to avoid hangs.
     3. Generalize to handle all GMP_LIMB_BITS >= 32.
     4. This lits is incomplete and misspelled.
   */

  ASSERT (d >= 0.0);

  if (d == 0.0)
    {
      MPN_ZERO (rp, LIMBS_PER_DOUBLE);
      return 0;
    }

#if _GMP_IEEE_FLOATS
  {
#if defined (__alpha) && __GNUC__ == 2 && __GNUC_MINOR__ == 8
    /* Work around alpha-specific bug in GCC 2.8.x.  */
    volatile
#endif
    union ieee_double_extract x;
    x.d = d;
    exp = x.s.exp;
#if BITS_PER_PART == 64		/* generalize this to BITS_PER_PART > BITS_IN_MANTISSA */
    manl = (((mp_limb_t) 1 << 63)
	    | ((mp_limb_t) x.s.manh << 43) | ((mp_limb_t) x.s.manl << 11));
    if (exp == 0)
      {
	/* Denormalized number.  Don't try to be clever about this,
	   since it is not an important case to make fast.  */
	exp = 1;
	do
	  {
	    manl = manl << 1;
	    exp--;
	  }
	while ((manl & GMP_LIMB_HIGHBIT) == 0);
      }
#endif
#if BITS_PER_PART == 32
    manh = ((mp_limb_t) 1 << 31) | (x.s.manh << 11) | (x.s.manl >> 21);
    manl = x.s.manl << 11;
    if (exp == 0)
      {
	/* Denormalized number.  Don't try to be clever about this,
	   since it is not an important case to make fast.  */
	exp = 1;
	do
	  {
	    manh = (manh << 1) | (manl >> 31);
	    manl = manl << 1;
	    exp--;
	  }
	while ((manh & GMP_LIMB_HIGHBIT) == 0);
      }
#endif
#if BITS_PER_PART != 32 && BITS_PER_PART != 64
  You need to generalize the code above to handle this.
#endif
    exp -= 1022;		/* Remove IEEE bias.  */
  }
#else
  {
    /* Unknown (or known to be non-IEEE) double format.  */
    exp = 0;
    if (d >= 1.0)
      {
	ASSERT_ALWAYS (d * 0.5 != d);

	while (d >= 32768.0)
	  {
	    d *= (1.0 / 65536.0);
	    exp += 16;
	  }
	while (d >= 1.0)
	  {
	    d *= 0.5;
	    exp += 1;
	  }
      }
    else if (d < 0.5)
      {
	while (d < (1.0 / 65536.0))
	  {
	    d *=  65536.0;
	    exp -= 16;
	  }
	while (d < 0.5)
	  {
	    d *= 2.0;
	    exp -= 1;
	  }
      }

    d *= (4.0 * ((unsigned long int) 1 << (BITS_PER_PART - 2)));
#if BITS_PER_PART == 64
    manl = d;
#endif
#if BITS_PER_PART == 32
    manh = d;
    manl = (d - manh) * (4.0 * ((unsigned long int) 1 << (BITS_PER_PART - 2)));
#endif
  }
#endif /* IEEE */

  sc = (unsigned) (exp + 64 * GMP_NUMB_BITS) % GMP_NUMB_BITS;

  /* We add something here to get rounding right.  */
  exp = (exp + 64 * GMP_NUMB_BITS) / GMP_NUMB_BITS - 64 * GMP_NUMB_BITS / GMP_NUMB_BITS + 1;

#if BITS_PER_PART == 64 && LIMBS_PER_DOUBLE == 2
#if GMP_NAIL_BITS == 0
  if (sc != 0)
    {
      rp[1] = manl >> (GMP_LIMB_BITS - sc);
      rp[0] = manl << sc;
    }
  else
    {
      rp[1] = manl;
      rp[0] = 0;
      exp--;
    }
#else
  if (sc > GMP_NAIL_BITS)
    {
      rp[1] = manl >> (GMP_LIMB_BITS - sc);
      rp[0] = (manl << (sc - GMP_NAIL_BITS)) & GMP_NUMB_MASK;
    }
  else
    {
      if (sc == 0)
	{
	  rp[1] = manl >> GMP_NAIL_BITS;
	  rp[0] = (manl << GMP_NUMB_BITS - GMP_NAIL_BITS) & GMP_NUMB_MASK;
	  exp--;
	}
      else
	{
	  rp[1] = manl >> (GMP_LIMB_BITS - sc);
	  rp[0] = (manl >> (GMP_NAIL_BITS - sc)) & GMP_NUMB_MASK;
	}
    }
#endif
#endif

#if BITS_PER_PART == 64 && LIMBS_PER_DOUBLE == 3
  if (sc > GMP_NAIL_BITS)
    {
      rp[2] = manl >> (GMP_LIMB_BITS - sc);
      rp[1] = (manl << sc - GMP_NAIL_BITS) & GMP_NUMB_MASK;
      if (sc >= 2 * GMP_NAIL_BITS)
	rp[0] = 0;
      else
	rp[0] = (manl << GMP_NUMB_BITS - GMP_NAIL_BITS + sc) & GMP_NUMB_MASK;
    }
  else
    {
      if (sc == 0)
	{
	  rp[2] = manl >> GMP_NAIL_BITS;
	  rp[1] = (manl << GMP_NUMB_BITS - GMP_NAIL_BITS) & GMP_NUMB_MASK;
	  rp[0] = 0;
	  exp--;
	}
      else
	{
	  rp[2] = manl >> (GMP_LIMB_BITS - sc);
	  rp[1] = (manl >> GMP_NAIL_BITS - sc) & GMP_NUMB_MASK;
	  rp[0] = (manl << GMP_NUMB_BITS - GMP_NAIL_BITS + sc) & GMP_NUMB_MASK;
	}
    }
#endif

#if BITS_PER_PART == 32 && LIMBS_PER_DOUBLE == 3
#if GMP_NAIL_BITS == 0
  if (sc != 0)
    {
      rp[2] = manh >> (GMP_LIMB_BITS - sc);
      rp[1] = (manh << sc) | (manl >> (GMP_LIMB_BITS - sc));
      rp[0] = manl << sc;
    }
  else
    {
      rp[2] = manh;
      rp[1] = manl;
      rp[0] = 0;
      exp--;
    }
#else
  if (sc > GMP_NAIL_BITS)
    {
      rp[2] = (manh >> (GMP_LIMB_BITS - sc));
      rp[1] = ((manh << (sc - GMP_NAIL_BITS)) |
	       (manl >> (GMP_LIMB_BITS - sc + GMP_NAIL_BITS))) & GMP_NUMB_MASK;
      if (sc >= 2 * GMP_NAIL_BITS)
	rp[0] = (manl << sc - 2 * GMP_NAIL_BITS) & GMP_NUMB_MASK;
      else
	rp[0] = manl >> (2 * GMP_NAIL_BITS - sc) & GMP_NUMB_MASK;
    }
  else
    {
      if (sc == 0)
	{
	  rp[2] = manh >> GMP_NAIL_BITS;
	  rp[1] = ((manh << GMP_NUMB_BITS - GMP_NAIL_BITS) | (manl >> 2 * GMP_NAIL_BITS)) & GMP_NUMB_MASK;
	  rp[0] = (manl << GMP_NUMB_BITS - 2 * GMP_NAIL_BITS) & GMP_NUMB_MASK;
	  exp--;
	}
      else
	{
	  rp[2] = (manh >> (GMP_LIMB_BITS - sc));
	  rp[1] = (manh >> (GMP_NAIL_BITS - sc)) & GMP_NUMB_MASK;
	  rp[0] = ((manh << (GMP_NUMB_BITS - GMP_NAIL_BITS + sc))
		   | (manl >> (GMP_LIMB_BITS - (GMP_NUMB_BITS - GMP_NAIL_BITS + sc)))) & GMP_NUMB_MASK;
	}
    }
#endif
#endif

#if BITS_PER_PART == 32 && LIMBS_PER_DOUBLE > 3
  if (sc == 0)
    {
      int i;

      for (i = LIMBS_PER_DOUBLE - 1; i >= 0; i--)
	{
	  rp[i] = manh >> (BITS_PER_ULONG - GMP_NUMB_BITS);
	  manh = ((manh << GMP_NUMB_BITS)
		  | (manl >> (BITS_PER_ULONG - GMP_NUMB_BITS)));
	  manl = manl << GMP_NUMB_BITS;
	}
      exp--;
    }
  else
    {
      int i;

      rp[LIMBS_PER_DOUBLE - 1] = (manh >> (GMP_LIMB_BITS - sc));
      manh = (manh << sc) | (manl >> (GMP_LIMB_BITS - sc));
      manl = (manl << sc);
      for (i = LIMBS_PER_DOUBLE - 2; i >= 0; i--)
	{
	  rp[i] = manh >> (BITS_PER_ULONG - GMP_NUMB_BITS);
	  manh = ((manh << GMP_NUMB_BITS)
		  | (manl >> (BITS_PER_ULONG - GMP_NUMB_BITS)));
	  manl = manl << GMP_NUMB_BITS;
	}
  }
#endif

  return exp;
}
