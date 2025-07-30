/* tpow -- test file for mpc_pow.

Copyright (C) 2009, 2011 INRIA

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

#include "mpc-tests.h"

static void
reuse_bug (void)
{
   /* bug found by the automatic builds on
      http://hydra.nixos.org/build/1469029/log/raw */
   mpc_t x, y, z;
   mp_prec_t prec = 2;

   for (prec = 2; prec <= 20; prec ++)
     {
       mpc_init2 (x, prec);
       mpc_init2 (y, prec);
       mpc_init2 (z, prec);
   
       mpfr_set_ui (mpc_realref (x), 0ul, GMP_RNDN);
       mpfr_set_ui_2exp (mpc_imagref (x), 3ul, -2, GMP_RNDN);
       mpc_set_ui (y, 8ul, MPC_RNDNN);

       mpc_pow (z, x, y, MPC_RNDNN);
       mpc_pow (y, x, y, MPC_RNDNN);
       if (mpfr_signbit (mpc_imagref (y)) != mpfr_signbit (mpc_imagref (z)))
         {
           printf ("Error: regression, reuse_bug reproduced\n");
           exit (1);
         }

       mpc_clear (x);
       mpc_clear (y);
       mpc_clear (z);
     }
}


int
main (void)
{
  DECL_FUNC (C_CC, f, mpc_pow);

  test_start ();

  reuse_bug ();

  data_check (f, "pow.dat");
  tgeneric (f, 2, 1024, 7, 10);

  test_end ();

  return 0;
}
