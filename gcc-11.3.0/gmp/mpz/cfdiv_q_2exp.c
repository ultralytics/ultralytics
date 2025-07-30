/* mpz_cdiv_q_2exp, mpz_fdiv_q_2exp -- quotient from mpz divided by 2^n.

Copyright 1991, 1993, 1994, 1996, 1998, 1999, 2001, 2002, 2004, 2012,
2015 Free Software Foundation, Inc.

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


/* dir==1 for ceil, dir==-1 for floor */

static void __gmpz_cfdiv_q_2exp (REGPARM_3_1 (mpz_ptr, mpz_srcptr, mp_bitcnt_t, int)) REGPARM_ATTR (1);
#define cfdiv_q_2exp(w,u,cnt,dir)  __gmpz_cfdiv_q_2exp (REGPARM_3_1 (w,u,cnt,dir))

REGPARM_ATTR (1) static void
cfdiv_q_2exp (mpz_ptr w, mpz_srcptr u, mp_bitcnt_t cnt, int dir)
{
  mp_size_t  wsize, usize, abs_usize, limb_cnt, i;
  mp_srcptr  up;
  mp_ptr     wp;
  mp_limb_t  round, rmask;

  usize = SIZ (u);
  abs_usize = ABS (usize);
  limb_cnt = cnt / GMP_NUMB_BITS;
  wsize = abs_usize - limb_cnt;
  if (wsize <= 0)
    {
      /* u < 2**cnt, so result 1, 0 or -1 according to rounding */
      PTR(w)[0] = 1;
      SIZ(w) = (usize == 0 || (usize ^ dir) < 0 ? 0 : dir);
      return;
    }

  /* +1 limb to allow for mpn_add_1 below */
  wp = MPZ_REALLOC (w, wsize+1);

  /* Check for rounding if direction matches u sign.
     Set round if we're skipping non-zero limbs.  */
  up = PTR(u);
  round = 0;
  rmask = ((usize ^ dir) >= 0 ? MP_LIMB_T_MAX : 0);
  if (rmask != 0)
    for (i = 0; i < limb_cnt && round == 0; i++)
      round = up[i];

  cnt %= GMP_NUMB_BITS;
  if (cnt != 0)
    {
      round |= rmask & mpn_rshift (wp, up + limb_cnt, wsize, cnt);
      wsize -= (wp[wsize - 1] == 0);
    }
  else
    MPN_COPY_INCR (wp, up + limb_cnt, wsize);

  if (round != 0)
    {
      if (wsize != 0)
	{
	  mp_limb_t cy;
	  cy = mpn_add_1 (wp, wp, wsize, CNST_LIMB(1));
	  wp[wsize] = cy;
	  wsize += cy;
	}
      else
	{
	  /* We shifted something to zero.  */
	  wp[0] = 1;
	  wsize = 1;
	}
    }
  SIZ(w) = (usize >= 0 ? wsize : -wsize);
}


void
mpz_cdiv_q_2exp (mpz_ptr w, mpz_srcptr u, mp_bitcnt_t cnt)
{
  cfdiv_q_2exp (w, u, cnt, 1);
}

void
mpz_fdiv_q_2exp (mpz_ptr w, mpz_srcptr u, mp_bitcnt_t cnt)
{
  cfdiv_q_2exp (w, u, cnt, -1);
}
