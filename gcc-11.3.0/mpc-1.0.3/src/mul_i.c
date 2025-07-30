/* mpc_mul_i -- Multiply a complex number by plus or minus i.

Copyright (C) 2005, 2009, 2010, 2011 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include "mpc-impl.h"

int
mpc_mul_i (mpc_ptr a, mpc_srcptr b, int sign, mpc_rnd_t rnd)
/* if sign is >= 0, multiply by i, otherwise by -i */
{
  int   inex_re, inex_im;
  mpfr_t tmp;

  /* Treat the most probable case of compatible precisions first */
  if (     MPC_PREC_RE (b) == MPC_PREC_IM (a)
        && MPC_PREC_IM (b) == MPC_PREC_RE (a))
  {
     if (a == b)
        mpfr_swap (mpc_realref (a), mpc_imagref (a));
     else
     {
        mpfr_set (mpc_realref (a), mpc_imagref (b), GMP_RNDN);
        mpfr_set (mpc_imagref (a), mpc_realref (b), GMP_RNDN);
     }
     if (sign >= 0)
        MPFR_CHANGE_SIGN (mpc_realref (a));
     else
        MPFR_CHANGE_SIGN (mpc_imagref (a));
     inex_re = 0;
     inex_im = 0;
  }
  else
  {
     if (a == b)
     {
        mpfr_init2 (tmp, MPC_PREC_RE (a));
        if (sign >= 0)
        {
           inex_re = mpfr_neg (tmp, mpc_imagref (b), MPC_RND_RE (rnd));
           inex_im = mpfr_set (mpc_imagref (a), mpc_realref (b), MPC_RND_IM (rnd));
        }
        else
        {
           inex_re = mpfr_set (tmp, mpc_imagref (b), MPC_RND_RE (rnd));
           inex_im = mpfr_neg (mpc_imagref (a), mpc_realref (b), MPC_RND_IM (rnd));
        }
        mpfr_clear (mpc_realref (a));
        mpc_realref (a)[0] = tmp [0];
     }
     else
        if (sign >= 0)
        {
           inex_re = mpfr_neg (mpc_realref (a), mpc_imagref (b), MPC_RND_RE (rnd));
           inex_im = mpfr_set (mpc_imagref (a), mpc_realref (b), MPC_RND_IM (rnd));
        }
        else
        {
           inex_re = mpfr_set (mpc_realref (a), mpc_imagref (b), MPC_RND_RE (rnd));
           inex_im = mpfr_neg (mpc_imagref (a), mpc_realref (b), MPC_RND_IM (rnd));
        }
  }

  return MPC_INEX(inex_re, inex_im);
}
