/* tatan -- test file for mpc_atan.

Copyright (C) 2009, 2012 INRIA

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

#if 0
/* tests intermediate underflow; WONTFIX */
static int
test_underflow (void)
{
  mpc_t z;
  mpfr_exp_t emin = mpfr_get_emin ();

  mpfr_set_emin (-10);
  mpc_init2 (z, 21);
  mpfr_set_si (mpc_realref(z), -1, GMP_RNDZ);
  mpfr_set_ui_2exp (mpc_imagref(z), 1, 20, GMP_RNDZ);
  mpfr_add_ui (mpc_imagref(z), mpc_imagref(z), 1, GMP_RNDZ);
  mpfr_div_2exp (mpc_imagref(z), mpc_imagref(z), 20, GMP_RNDZ);
  mpc_atan (z, z, MPC_RNDNN);
  if (mpfr_cmp_si_2exp (mpc_realref(z), -1066635, 20) != 0 ||
      mpfr_cmp_si_2exp (mpc_imagref(z), 1687619, 22))
    {
      printf ("Error in test_coverage\n");
      printf ("expected (-1066635/2^20 1687619/2^22)\n");
      printf ("got      ");
      mpc_out_str (stdout, 10, 20, z, MPC_RNDNN);
      printf ("\n");
      exit (1);
    }
  mpc_clear (z);
  mpfr_set_emin (emin);
}
#endif


int
main (void)
{
  DECL_FUNC (CC, f, mpc_atan);

  test_start ();

  data_check (f, "atan.dat");
  tgeneric (f, 2, 512, 5, 128);

  test_end ();

  return 0;
}

