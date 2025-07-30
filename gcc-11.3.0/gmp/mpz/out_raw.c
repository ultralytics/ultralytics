/* mpz_out_raw -- write an mpz_t in raw format.

Copyright 2001, 2002 Free Software Foundation, Inc.

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
#include "longlong.h"


/* HTON_LIMB_STORE takes a normal host byte order limb and stores it as
   network byte order (ie. big endian). */

#if HAVE_LIMB_BIG_ENDIAN
#define HTON_LIMB_STORE(dst, limb)  do { *(dst) = (limb); } while (0)
#endif

#if HAVE_LIMB_LITTLE_ENDIAN
#define HTON_LIMB_STORE(dst, limb)  BSWAP_LIMB_STORE (dst, limb)
#endif

#ifndef HTON_LIMB_STORE
#define HTON_LIMB_STORE(dst, limb)                                      \
  do {                                                                  \
    mp_limb_t  __limb = (limb);                                         \
    char      *__p = (char *) (dst);                                    \
    int        __i;                                                     \
    for (__i = 0; __i < GMP_LIMB_BYTES; __i++)                       \
      __p[__i] = (char) (__limb >> ((GMP_LIMB_BYTES-1 - __i) * 8));  \
  } while (0)
#endif


size_t
mpz_out_raw (FILE *fp, mpz_srcptr x)
{
  mp_size_t   xsize, abs_xsize, bytes, i;
  mp_srcptr   xp;
  char        *tp, *bp;
  mp_limb_t   xlimb;
  int         zeros;
  size_t      tsize, ssize;

  xsize = SIZ(x);
  abs_xsize = ABS (xsize);
  bytes = (abs_xsize * GMP_NUMB_BITS + 7) / 8;
  tsize = ROUND_UP_MULTIPLE ((unsigned) 4, GMP_LIMB_BYTES) + bytes;

  tp = __GMP_ALLOCATE_FUNC_TYPE (tsize, char);
  bp = tp + ROUND_UP_MULTIPLE ((unsigned) 4, GMP_LIMB_BYTES);

  if (bytes != 0)
    {
      bp += bytes;
      xp = PTR (x);
      i = abs_xsize;

      if (GMP_NAIL_BITS == 0)
	{
	  /* reverse limb order, and byte swap if necessary */
#ifdef _CRAY
	  _Pragma ("_CRI ivdep");
#endif
	  do
	    {
	      bp -= GMP_LIMB_BYTES;
	      xlimb = *xp;
	      HTON_LIMB_STORE ((mp_ptr) bp, xlimb);
	      xp++;
	    }
	  while (--i > 0);

	  /* strip high zero bytes (without fetching from bp) */
	  count_leading_zeros (zeros, xlimb);
	  zeros /= 8;
	  bp += zeros;
	  bytes -= zeros;
	}
      else
	{
	  mp_limb_t  new_xlimb;
	  int        bits;
	  ASSERT_CODE (char *bp_orig = bp - bytes);

	  ASSERT_ALWAYS (GMP_NUMB_BITS >= 8);

	  bits = 0;
	  xlimb = 0;
	  for (;;)
	    {
	      while (bits >= 8)
		{
		  ASSERT (bp > bp_orig);
		  *--bp = xlimb & 0xFF;
		  xlimb >>= 8;
		  bits -= 8;
		}

	      if (i == 0)
		break;

	      new_xlimb = *xp++;
	      i--;
	      ASSERT (bp > bp_orig);
	      *--bp = (xlimb | (new_xlimb << bits)) & 0xFF;
	      xlimb = new_xlimb >> (8 - bits);
	      bits += GMP_NUMB_BITS - 8;
	    }

	  if (bits != 0)
	    {
	      ASSERT (bp > bp_orig);
	      *--bp = xlimb;
	    }

	  ASSERT (bp == bp_orig);
	  while (*bp == 0)
	    {
	      bp++;
	      bytes--;
	    }
	}
    }

  /* total bytes to be written */
  ssize = 4 + bytes;

  /* twos complement negative for the size value */
  bytes = (xsize >= 0 ? bytes : -bytes);

  /* so we don't rely on sign extension in ">>" */
  ASSERT_ALWAYS (sizeof (bytes) >= 4);

  bp[-4] = bytes >> 24;
  bp[-3] = bytes >> 16;
  bp[-2] = bytes >> 8;
  bp[-1] = bytes;
  bp -= 4;

  if (fp == 0)
    fp = stdout;
  if (fwrite (bp, ssize, 1, fp) != 1)
    ssize = 0;

  (*__gmp_free_func) (tp, tsize);
  return ssize;
}
