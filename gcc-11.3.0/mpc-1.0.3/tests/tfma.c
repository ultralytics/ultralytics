/* tfma -- test file for mpc_fma.

Copyright (C) 2011, 2012 INRIA

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
cmpfma (mpc_srcptr a, mpc_srcptr b, mpc_srcptr c, mpc_rnd_t rnd)
   /* computes a*b+c with the naive and fast functions using the rounding
      mode rnd and compares the results and return values.                                          
      In our current test suite, all input precisions are the same, and we
      use this precision also for the result.
   */
{
   mpc_t z, t;
   int   inex_z, inex_t;

   mpc_init2 (z, MPC_MAX_PREC (a));
   mpc_init2 (t, MPC_MAX_PREC (a));

   inex_z = mpc_fma_naive (z, a, b, c, rnd);
   inex_t = mpc_fma (t, a, b, c, rnd);

   if (mpc_cmp (z, t) != 0 || inex_z != inex_t) {
      fprintf (stderr, "fma_naive and fma differ for rnd=(%s,%s)\n",
               mpfr_print_rnd_mode(MPC_RND_RE(rnd)),
               mpfr_print_rnd_mode(MPC_RND_IM(rnd)));
      MPC_OUT (a);
      MPC_OUT (b);
      MPC_OUT (c);
      MPC_OUT (z);
      MPC_OUT (t);
      if (inex_z != inex_t) {
         fprintf (stderr, "inex_re (z): %s\n", MPC_INEX_STR (inex_z));
         fprintf (stderr, "inex_re (t): %s\n", MPC_INEX_STR (inex_t));
      }
      exit (1);
   }

  mpc_clear (z);
  mpc_clear (t);
}


static void
check_random (void)
{
   mpfr_prec_t prec;
   int rnd_re, rnd_im;
   mpc_t a, b, c;

   mpc_init2 (a, 1000);
   mpc_init2 (b, 1000);
   mpc_init2 (c, 1000);

   for (prec = 2; prec < 1000; prec = (mpfr_prec_t) (prec * 1.1 + 1)) {
      mpc_set_prec (a, prec);
      mpc_set_prec (b, prec);
      mpc_set_prec (c, prec);

      test_default_random (a, -1024, 1024, 128, 0);
      test_default_random (b, -1024, 1024, 128, 0);
      test_default_random (c, -1024, 1024, 128, 0);

      for (rnd_re = 0; rnd_re < 4; rnd_re ++)
         for (rnd_im = 0; rnd_im < 4; rnd_im ++)
            cmpfma (a, b, c, MPC_RND (rnd_re, rnd_im));
   }

   mpc_clear (a);
   mpc_clear (b);
   mpc_clear (c);
}


int
main (void)
{
  DECL_FUNC (CCCC, f, mpc_fma);

  test_start ();

  check_random ();

  data_check (f, "fma.dat");
  tgeneric (f, 2, 1024, 1, 256);

  test_end ();

  return 0;
}
