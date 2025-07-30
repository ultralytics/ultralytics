/* tpow_fr -- test file for mpc_pow_fr.

Copyright (C) 2009, 2011, 2012 INRIA

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
test_reuse (void)
{
  mpc_t z;
  mpfr_t y;
  int inex;

  mpfr_init2 (y, 2);
  mpc_init2 (z, 2);
  mpc_set_si_si (z, 0, -1, MPC_RNDNN);
  mpfr_neg (mpc_realref (z), mpc_realref (z), GMP_RNDN);
  mpc_div_2ui (z, z, 4, MPC_RNDNN);
  mpfr_set_ui (y, 512, GMP_RNDN);
  inex = mpc_pow_fr (z, z, y, MPC_RNDNN);
  if (MPC_INEX_RE(inex) != 0 || MPC_INEX_IM(inex) != 0 ||
      mpfr_cmp_ui_2exp (mpc_realref(z), 1, -2048) != 0 ||
      mpfr_cmp_ui (mpc_imagref(z), 0) != 0 || mpfr_signbit (mpc_imagref(z)) == 0)
    {
      printf ("Error in test_reuse, wrong ternary value or output\n");
      printf ("inex=(%d %d)\n", MPC_INEX_RE(inex), MPC_INEX_IM(inex));
      printf ("z="); mpc_out_str (stdout, 2, 0, z, MPC_RNDNN); printf ("\n");
      exit (1);
    }
  mpfr_clear (y);
  mpc_clear (z);
}

int
main (void)
{
  DECL_FUNC (CCF, f, mpc_pow_fr);
  test_start ();

  test_reuse ();
  data_check (f, "pow_fr.dat");
  tgeneric (f, 2, 1024, 7, 10);

  test_end ();

  return 0;
}
