/* test file for mpc_cosh.

Copyright (C) 2008, 2009, 2010, 2011 INRIA

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

#include <stdlib.h>
#include "mpc-tests.h"

static void
pure_real_argument (void)
{
  /* cosh(x -i*0) = cosh(x) +i*0 if x<0 */
  /* cosh(x -i*0) = cosh(x) -i*0 if x>0 */
  /* cosh(x +i*0) = cosh(x) -i*0 if x<0 */
  /* cosh(x -i*0) = cosh(x) +i*0 if x>0 */
  mpc_t u;
  mpc_t z;
  mpc_t cosh_z;

  mpc_init2 (z, 2);
  mpc_init2 (u, 100);
  mpc_init2 (cosh_z, 100);

  /* cosh(1 +i*0) = cosh(1) +i*0 */
  mpc_set_ui_ui (z, 1, 0, MPC_RNDNN);
  mpfr_cosh (mpc_realref (u), mpc_realref (z), GMP_RNDN);
  mpfr_set_ui (mpc_imagref (u), 0, GMP_RNDN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  /* cosh(1 -i*0) = cosh(1) -i*0 */
  mpc_conj (z, z, MPC_RNDNN);
  mpc_conj (u, u, MPC_RNDNN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || !mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  /* cosh(-1 +i*0) = cosh(1) -i*0 */
  mpc_neg (z, z, MPC_RNDNN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || !mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  /* cosh(-1 -i*0) = cosh(1) +i*0 */
  mpc_conj (z, z, MPC_RNDNN);
  mpc_conj (u, u, MPC_RNDNN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  mpc_clear (cosh_z);
  mpc_clear (z);
  mpc_clear (u);
}

static void
pure_imaginary_argument (void)
{
  /* cosh(+0 +i*y) = cos y +i*0*sin y */
  /* cosh(-0 +i*y) = cos y -i*0*sin y */
  mpc_t u;
  mpc_t z;
  mpc_t cosh_z;

  mpc_init2 (z, 2);
  mpc_init2 (u, 100);
  mpc_init2 (cosh_z, 100);

  /* cosh(+0 +i) = cos(1) + i*0 */
  mpc_set_ui_ui (z, 0, 1, MPC_RNDNN);
  mpfr_cos (mpc_realref (u), mpc_imagref (z), GMP_RNDN);
  mpfr_set_ui (mpc_imagref (u), 0, GMP_RNDN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  /* cosh(+0 -i) = cos(1) - i*0 */
  mpc_conj (z, z, MPC_RNDNN);
  mpc_conj (u, u, MPC_RNDNN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || !mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  /* cosh(-0 +i) = cos(1) - i*0 */
  mpc_neg (z, z, MPC_RNDNN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || !mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  /* cosh(-0 -i) = cos(1) + i*0 */
  mpc_conj (z, z, MPC_RNDNN);
  mpc_conj (u, u, MPC_RNDNN);
  mpc_cosh (cosh_z, z, MPC_RNDNN);
  if (mpc_cmp (cosh_z, u) != 0 || mpfr_signbit (mpc_imagref (cosh_z)))
    TEST_FAILED ("mpc_cosh", z, cosh_z, u, MPC_RNDNN);

  mpc_clear (cosh_z);
  mpc_clear (z);
  mpc_clear (u);
}

int
main (void)
{
  DECL_FUNC(CC, f,mpc_cosh);

  test_start ();

  data_check (f, "cosh.dat");
  tgeneric (f, 2, 512, 7, 7);

  pure_real_argument ();
  pure_imaginary_argument ();

  test_end ();

  return 0;
}
