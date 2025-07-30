/* tmul_i -- test file for mpc_mul_i.

Copyright (C)  2008, 2009, 2010, 2011 INRIA

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
check_different_precisions(void)
{
  /* check reuse when real and imaginary part have different precisions. */
  mpc_t z, expected, got;
  int res;

  mpc_init2(z, 128);
  mpc_init2(expected, 128);
  mpc_init2(got, 128);

  /* change precision of one part */
  mpfr_set_prec (mpc_imagref (z), 32);
  mpfr_set_prec (mpc_imagref (expected), 32);
  mpfr_set_prec (mpc_imagref (got), 32);

  mpfr_set_str (mpc_realref (z), "0x100000000fp-32", 16, GMP_RNDN);
  mpfr_set_str (mpc_imagref (z), "-1", 2, GMP_RNDN);
  mpfr_set_str (mpc_realref (expected), "+1", 2, GMP_RNDN);
  mpfr_set_str (mpc_imagref (expected), "0x100000000fp-32", 16, GMP_RNDN);

  mpc_set (got, z, MPC_RNDNN);
  res = mpc_mul_i (got, got, +1, MPC_RNDNN);
  if (MPC_INEX_RE(res) != 0 || MPC_INEX_IM(res) >=0)
    {
      printf("Wrong inexact flag for mpc_mul_i(z, z, n)\n"
             "     got (re=%2d, im=%2d)\nexpected (re= 0, im=-1)\n",
             MPC_INEX_RE(res), MPC_INEX_IM(res));
      exit(1);
    }
  if (mpc_cmp(got, expected) != 0)
    {
      printf ("Error for mpc_mul_i(z, z, n) for\n");
      MPC_OUT (z);
      printf ("n=+1\n");
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }

  mpc_neg (expected, expected, MPC_RNDNN);
  mpc_set (got, z, MPC_RNDNN);
  mpc_mul_i (got, got, -1, MPC_RNDNN);
  if (mpc_cmp(got, expected) != 0)
    {
      printf ("Error for mpc_mul_i(z, z, n) for\n");
      MPC_OUT (z);
      printf ("n=-1\n");
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }

  mpc_clear (z);
  mpc_clear (expected);
  mpc_clear (got);
}

int
main (void)
{
  DECL_FUNC (CCI, f, mpc_mul_i);

  test_start ();

  check_different_precisions ();
  tgeneric (f, 2, 1024, 7, -1);

  test_end ();

  return 0;
}
