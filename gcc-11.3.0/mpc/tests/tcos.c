/* tcos -- test file for mpc_cos.

Copyright (C) 2008, 2009, 2011 INRIA

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
bug20090105 (void)
{
  /* this bug appeared on 32-bits machines */
  mpc_t op, expected, got;
  mpc_init2 (op, 324);
  mpc_init2 (expected, 324);
  mpc_init2 (got, 324);

  mpfr_set_str (mpc_realref(op), "-3.f1813b1487372434fea4414a520f65a343a16d0ec1ffb"
                "b2b880154db8d63377ce788fc4215c450300@1", 16, GMP_RNDN);
  mpfr_set_str (mpc_imagref(op), "-2.b7a0c80bcacf1ccbbac614bf53a58b672b1b503161bee"
                "59a82e46a23570b652f7ba5f01ef766d1c50", 16,GMP_RNDN);
  mpfr_set_str (mpc_realref(expected), "7.57c5b08a2b11b660d906a354289b0724b9c4b237"
                "95abe33424e8d9858e534bd5d776ddd18e34b0240", 16, GMP_RNDN);
  mpfr_set_str (mpc_imagref(expected), "-1.f41a389646d068e0263561cb3c5d1df763945ad"
                "ed9339f2a98387a3c4f97dbfd8a08b7d0af2f11b46", 16,GMP_RNDN);

  mpc_cos (got, op, MPC_RNDNN);
  if (mpc_cmp (got, expected) != 0)
    TEST_FAILED ("mpc_cos", op, got, expected, MPC_RNDNN);

  mpc_clear (got);
  mpc_clear(expected);
  mpc_clear (op);
}

int
main (void)
{
  DECL_FUNC (CC, f, mpc_cos);

  test_start ();

  data_check (f, "cos.dat");
  tgeneric (f, 2, 512, 7, 7);

  bug20090105 ();

  test_end ();

  return 0;
}
