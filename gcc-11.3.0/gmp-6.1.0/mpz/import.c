/* mpz_import -- set mpz from word data.

Copyright 2002, 2012 Free Software Foundation, Inc.

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



#if HAVE_LIMB_BIG_ENDIAN
#define HOST_ENDIAN     1
#endif
#if HAVE_LIMB_LITTLE_ENDIAN
#define HOST_ENDIAN     (-1)
#endif
#ifndef HOST_ENDIAN
static const mp_limb_t  endian_test = (CNST_LIMB(1) << (GMP_LIMB_BITS-7)) - 1;
#define HOST_ENDIAN     (* (signed char *) &endian_test)
#endif


void
mpz_import (mpz_ptr z, size_t count, int order,
	    size_t size, int endian, size_t nail, const void *data)
{
  mp_size_t  zsize;
  mp_ptr     zp;

  ASSERT (order == 1 || order == -1);
  ASSERT (endian == 1 || endian == 0 || endian == -1);
  ASSERT (nail <= 8*size);

  zsize = BITS_TO_LIMBS (count * (8*size - nail));
  zp = MPZ_NEWALLOC (z, zsize);

  if (endian == 0)
    endian = HOST_ENDIAN;

  /* Can't use these special cases with nails currently, since they don't
     mask out the nail bits in the input data.  */
  if (nail == 0 && GMP_NAIL_BITS == 0)
    {
      unsigned  align = ((char *) data - (char *) NULL) % sizeof (mp_limb_t);

      if (order == -1
	  && size == sizeof (mp_limb_t)
	  && endian == HOST_ENDIAN
	  && align == 0)
	{
	  MPN_COPY (zp, (mp_srcptr) data, (mp_size_t) count);
	  goto done;
	}

      if (order == -1
	  && size == sizeof (mp_limb_t)
	  && endian == - HOST_ENDIAN
	  && align == 0)
	{
	  MPN_BSWAP (zp, (mp_srcptr) data, (mp_size_t) count);
	  goto done;
	}

      if (order == 1
	  && size == sizeof (mp_limb_t)
	  && endian == HOST_ENDIAN
	  && align == 0)
	{
	  MPN_REVERSE (zp, (mp_srcptr) data, (mp_size_t) count);
	  goto done;
	}
    }

  {
    mp_limb_t      limb, byte, wbitsmask;
    size_t         i, j, numb, wbytes;
    mp_size_t      woffset;
    unsigned char  *dp;
    int            lbits, wbits;

    numb = size * 8 - nail;

    /* whole bytes to process */
    wbytes = numb / 8;

    /* partial byte to process */
    wbits = numb % 8;
    wbitsmask = (CNST_LIMB(1) << wbits) - 1;

    /* offset to get to the next word after processing wbytes and wbits */
    woffset = (numb + 7) / 8;
    woffset = (endian >= 0 ? woffset : -woffset)
      + (order < 0 ? size : - (mp_size_t) size);

    /* least significant byte */
    dp = (unsigned char *) data
      + (order >= 0 ? (count-1)*size : 0) + (endian >= 0 ? size-1 : 0);

#define ACCUMULATE(N)                                   \
    do {                                                \
      ASSERT (lbits < GMP_NUMB_BITS);                   \
      ASSERT (limb <= (CNST_LIMB(1) << lbits) - 1);     \
                                                        \
      limb |= (mp_limb_t) byte << lbits;                \
      lbits += (N);                                     \
      if (lbits >= GMP_NUMB_BITS)                       \
        {                                               \
          *zp++ = limb & GMP_NUMB_MASK;                 \
          lbits -= GMP_NUMB_BITS;                       \
          ASSERT (lbits < (N));                         \
          limb = byte >> ((N) - lbits);                 \
        }                                               \
    } while (0)

    limb = 0;
    lbits = 0;
    for (i = 0; i < count; i++)
      {
	for (j = 0; j < wbytes; j++)
	  {
	    byte = *dp;
	    dp -= endian;
	    ACCUMULATE (8);
	  }
	if (wbits != 0)
	  {
	    byte = *dp & wbitsmask;
	    dp -= endian;
	    ACCUMULATE (wbits);
	  }
	dp += woffset;
      }

    if (lbits != 0)
      {
	ASSERT (lbits <= GMP_NUMB_BITS);
	ASSERT_LIMB (limb);
	*zp++ = limb;
      }

    ASSERT (zp == PTR(z) + zsize);

    /* low byte of word after most significant */
    ASSERT (dp == (unsigned char *) data
	    + (order < 0 ? count*size : - (mp_size_t) size)
	    + (endian >= 0 ? (mp_size_t) size - 1 : 0));

  }

 done:
  zp = PTR(z);
  MPN_NORMALIZE (zp, zsize);
  SIZ(z) = zsize;
}
