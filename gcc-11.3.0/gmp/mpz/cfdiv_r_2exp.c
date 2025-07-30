/* mpz_cdiv_r_2exp, mpz_fdiv_r_2exp -- remainder from mpz divided by 2^n.

Copyright 2001, 2002, 2004, 2012 Free Software Foundation, Inc.

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


/* Bit mask of "n" least significant bits of a limb. */
#define LOW_MASK(n)   ((CNST_LIMB(1) << (n)) - 1)


/* dir==1 for ceil, dir==-1 for floor */

static void __gmpz_cfdiv_r_2exp (REGPARM_3_1 (mpz_ptr, mpz_srcptr, mp_bitcnt_t, int)) REGPARM_ATTR (1);
#define cfdiv_r_2exp(w,u,cnt,dir)  __gmpz_cfdiv_r_2exp (REGPARM_3_1 (w, u, cnt, dir))

REGPARM_ATTR (1) static void
cfdiv_r_2exp (mpz_ptr w, mpz_srcptr u, mp_bitcnt_t cnt, int dir)
{
  mp_size_t  usize, abs_usize, limb_cnt, i;
  mp_srcptr  up;
  mp_ptr     wp;
  mp_limb_t  high;

  usize = SIZ(u);
  if (usize == 0)
    {
      SIZ(w) = 0;
      return;
    }

  limb_cnt = cnt / GMP_NUMB_BITS;
  cnt %= GMP_NUMB_BITS;
  abs_usize = ABS (usize);

  /* MPZ_REALLOC(w) below is only when w!=u, so we can fetch PTR(u) here
     nice and early */
  up = PTR(u);

  if ((usize ^ dir) < 0)
    {
      /* Round towards zero, means just truncate */

      if (w == u)
	{
	  /* if already smaller than limb_cnt then do nothing */
	  if (abs_usize <= limb_cnt)
	    return;
	  wp = (mp_ptr) up;
	}
      else
	{
	  i = MIN (abs_usize, limb_cnt+1);
	  wp = MPZ_NEWALLOC (w, i);
	  MPN_COPY (wp, up, i);

	  /* if smaller than limb_cnt then only the copy is needed */
	  if (abs_usize <= limb_cnt)
	    {
	      SIZ(w) = usize;
	      return;
	    }
	}
    }
  else
    {
      /* Round away from zero, means twos complement if non-zero */

      /* if u!=0 and smaller than divisor, then must negate */
      if (abs_usize <= limb_cnt)
	goto negate;

      /* if non-zero low limb, then must negate */
      for (i = 0; i < limb_cnt; i++)
	if (up[i] != 0)
	  goto negate;

      /* if non-zero partial limb, then must negate */
      if ((up[limb_cnt] & LOW_MASK (cnt)) != 0)
	goto negate;

      /* otherwise low bits of u are zero, so that's the result */
      SIZ(w) = 0;
      return;

    negate:
      /* twos complement negation to get 2**cnt-u */

      wp = MPZ_REALLOC (w, limb_cnt+1);
      up = PTR(u);

      /* Ones complement */
      i = MIN (abs_usize, limb_cnt+1);
      ASSERT_CARRY (mpn_neg (wp, up, i));
      for ( ; i <= limb_cnt; i++)
	wp[i] = GMP_NUMB_MAX;

      usize = -usize;
    }

  /* Mask the high limb */
  high = wp[limb_cnt];
  high &= LOW_MASK (cnt);
  wp[limb_cnt] = high;

  /* Strip any consequent high zeros */
  while (high == 0)
    {
      limb_cnt--;
      if (limb_cnt < 0)
	{
	  SIZ(w) = 0;
	  return;
	}
      high = wp[limb_cnt];
    }

  limb_cnt++;
  SIZ(w) = (usize >= 0 ? limb_cnt : -limb_cnt);
}


void
mpz_cdiv_r_2exp (mpz_ptr w, mpz_srcptr u, mp_bitcnt_t cnt)
{
  cfdiv_r_2exp (w, u, cnt, 1);
}

void
mpz_fdiv_r_2exp (mpz_ptr w, mpz_srcptr u, mp_bitcnt_t cnt)
{
  cfdiv_r_2exp (w, u, cnt, -1);
}
