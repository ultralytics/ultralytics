/* tui_div -- test file for mpc_ui_div.

Copyright (C) 2008, 2011 INRIA

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
special (void)
{
  mpc_t a, b;

  mpc_init2 (a, 10);
  mpc_init2 (b, 10);

  mpc_set_ui_ui (a, 2, 4, MPC_RNDNN);
  mpc_ui_div (b, 10, a, MPC_RNDNN);
  if (mpc_cmp_si_si (b, 1, -2) != 0)
    {
      printf ("10/(2,4) failed\n");
      printf ("expected (1,-2)\n");
      printf ("got      ");
      mpc_out_str (stdout, 10, 0, b, MPC_RNDNN);
      printf ("\n");
      exit (1);
    }

  /* 0/(-1-0*I) should give (-0, +0) */
  mpfr_set_str (mpc_realref(a), "-1", 10, GMP_RNDN);
  mpfr_set_str (mpc_imagref(a), "-0", 10, GMP_RNDN);
  mpc_ui_div (b, 0, a, MPC_RNDNN);
  if ((mpc_cmp_si_si (b, 0, 0) != 0) || (MPFR_SIGN (mpc_realref(b)) > 0)
      || (MPFR_SIGN (mpc_imagref(b)) < 0))
    {
      printf ("0/(-1,-0) failed\n");
      printf ("expected (-0,+0)\n");
      printf ("got      ");
      mpc_out_str (stdout, 10, 0, b, MPC_RNDNN);
      printf ("\n");
      exit (1);
    }

  mpc_set_ui_ui (a, 1, 0, MPC_RNDNN);
  mpc_ui_div (b, 1, a, MPC_RNDNN);
  if (mpc_cmp_si_si (b, 1, 0) != 0)
    {
      printf ("1/(1,0) failed\n");
      printf ("expected (1,0)\n");
      printf ("got      ");
      mpc_out_str (stdout, 10, 0, b, MPC_RNDNN);
      printf ("\n");
      exit (1);
    }

  /* problem reported by Timo Hartmann with mpc-0.7, 21 Oct 2009 */
  mpc_set_ui_ui (a, 4, 0, MPC_RNDNN);
  mpc_ui_div (b, 1, a, MPC_RNDNN);
  if (mpfr_cmp_ui_2exp (mpc_realref(b), 1, -2) != 0 ||
      mpfr_cmp_ui (mpc_imagref(b), 0) != 0 || mpfr_signbit (mpc_imagref(b)) != 0)
    {
      printf ("1/(4,0) failed\n");
      printf ("expected (1/4,0)\n");
      printf ("got      ");
      mpc_out_str (stdout, 10, 0, b, MPC_RNDNN);
      printf ("\n");
      exit (1);
    }

  mpc_clear (a);
  mpc_clear (b);
}

int
main (void)
{
  DECL_FUNC (CUC, f, mpc_ui_div);

  test_start ();

  special ();

  tgeneric (f, 2, 1024, 7, 4096);

  test_end ();

  return 0;
}
