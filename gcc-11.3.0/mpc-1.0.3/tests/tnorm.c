/* tnorm -- test file for mpc_norm.

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
test_underflow (void)
{
  mpfr_exp_t emin = mpfr_get_emin ();
  mpc_t z;
  mpfr_t f;
  int inex;
  
  mpfr_set_emin (-1); /* smallest positive number is 0.5*2^emin = 0.25 */
  mpc_init2 (z, 10);
  mpfr_set_ui_2exp (mpc_realref (z), 1023, -11, GMP_RNDN); /* exact */
  mpfr_set_ui_2exp (mpc_imagref (z), 1023, -11, GMP_RNDN); /* exact */
  mpfr_init2 (f, 10);

  inex = mpc_norm (f, z, GMP_RNDZ); /* should give 511/1024 */
  if (inex >= 0)
    {
      printf ("Error in underflow case (1)\n");
      printf ("expected inex < 0, got %d\n", inex);
      exit (1);
    }
  if (mpfr_cmp_ui_2exp (f, 511, -10) != 0)
    {
      printf ("Error in underflow case (1)\n");
      printf ("got      ");
      mpfr_dump (f);
      printf ("expected ");
      mpfr_set_ui_2exp (f, 511, -10, GMP_RNDZ);
      mpfr_dump (f);
      exit (1);
    }

  inex = mpc_norm (f, z, GMP_RNDN); /* should give 511/1024 */
  if (inex >= 0)
    {
      printf ("Error in underflow case (2)\n");
      printf ("expected inex < 0, got %d\n", inex);
      exit (1);
    }
  if (mpfr_cmp_ui_2exp (f, 511, -10) != 0)
    {
      printf ("Error in underflow case (2)\n");
      printf ("got      ");
      mpfr_dump (f);
      printf ("expected ");
      mpfr_set_ui_2exp (f, 511, -10, GMP_RNDZ);
      mpfr_dump (f);
      exit (1);
    }

  inex = mpc_norm (f, z, GMP_RNDU); /* should give 1023/2048 */
  if (inex <= 0)
    {
      printf ("Error in underflow case (3)\n");
      printf ("expected inex > 0, got %d\n", inex);
      exit (1);
    }
  if (mpfr_cmp_ui_2exp (f, 1023, -11) != 0)
    {
      printf ("Error in underflow case (3)\n");
      printf ("got      ");
      mpfr_dump (f);
      printf ("expected ");
      mpfr_set_ui_2exp (f, 1023, -11, GMP_RNDZ);
      mpfr_dump (f);
      exit (1);
    }

  mpc_clear (z);
  mpfr_clear (f);
  mpfr_set_emin (emin);
}

int
main (void)
{
  DECL_FUNC (FC, f, mpc_norm);

  test_start ();

  data_check (f, "norm.dat");
  tgeneric (f, 2, 1024, 1, 4096);
  test_underflow ();

  test_end ();

  return 0;
}
