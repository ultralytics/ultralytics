/* ttan -- test file for mpc_tan.

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

#include <stdlib.h>
#include "mpc-tests.h"

static void
pure_real_argument (void)
{
  /* tan(x -i*0) = tan(x) -i*0 */
  /* tan(x +i*0) = tan(x) +i*0 */
  mpfr_t x;
  mpfr_t tan_x;
  mpc_t z;
  mpc_t tan_z;

  mpfr_init2 (x, 79);
  mpfr_init2 (tan_x, 113);
  mpc_init2 (z, 79);
  mpc_init2 (tan_z, 113);

  /* tan(1 +i*0) = tan(1) +i*0 */
  mpc_set_ui_ui (z, 1, 0, MPC_RNDNN);
  mpfr_set_ui (x, 1, GMP_RNDN);
  mpfr_tan (tan_x, x, GMP_RNDN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_realref (tan_z), tan_x) != 0
      || !mpfr_zero_p (mpc_imagref (tan_z)) || mpfr_signbit (mpc_imagref (tan_z)))
    {
      printf ("mpc_tan(1 + i * 0) failed\n");
      exit (1);
    }

  /* tan(1 -i*0) = tan(1) -i*0 */
  mpc_conj (z, z, MPC_RNDNN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_realref (tan_z), tan_x) != 0
      || !mpfr_zero_p (mpc_imagref (tan_z)) || !mpfr_signbit (mpc_imagref (tan_z)))
    {
      printf ("mpc_tan(1 - i * 0) failed\n");
      exit (1);
    }

  /* tan(Pi/2 +i*0) = +Inf +i*0 */
  mpfr_const_pi (x, GMP_RNDN);
  mpfr_div_2ui (x, x, 1, GMP_RNDN);
  mpfr_set (mpc_realref (z), x, GMP_RNDN);
  mpfr_set_ui (mpc_imagref (z), 0, GMP_RNDN);
  mpfr_tan (tan_x, x, GMP_RNDN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_realref (tan_z), tan_x) != 0
      || !mpfr_zero_p (mpc_imagref (tan_z)) || mpfr_signbit (mpc_imagref (tan_z)))
    {
      printf ("mpc_tan(Pi/2 + i * 0) failed\n");
      exit (1);
    }

  /* tan(Pi/2 -i*0) = +Inf -i*0 */
  mpc_conj (z, z, MPC_RNDNN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_realref (tan_z), tan_x) != 0
      || !mpfr_zero_p (mpc_imagref (tan_z)) || !mpfr_signbit (mpc_imagref (tan_z)))
    {
      printf ("mpc_tan(Pi/2 - i * 0) failed\n");
      exit (1);
    }

  /* tan(-Pi/2 +i*0) = -Inf +i*0 */
  mpfr_neg (x, x, GMP_RNDN);
  mpc_neg (z, z, MPC_RNDNN);
  mpfr_tan (tan_x, x, GMP_RNDN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_realref (tan_z), tan_x) != 0
      || !mpfr_zero_p (mpc_imagref (tan_z)) || mpfr_signbit (mpc_imagref (tan_z)))
    {
      printf ("mpc_tan(-Pi/2 + i * 0) failed\n");
      exit (1);
    }

  /* tan(-Pi/2 -i*0) = -Inf -i*0 */
  mpc_conj (z, z, MPC_RNDNN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_realref (tan_z), tan_x) != 0
      || !mpfr_zero_p (mpc_imagref (tan_z)) || !mpfr_signbit (mpc_imagref (tan_z)))
    {
      printf ("mpc_tan(-Pi/2 - i * 0) failed\n");
      exit (1);
    }

  mpc_clear (tan_z);
  mpc_clear (z);
  mpfr_clear (tan_x);
  mpfr_clear (x);
}

static void
pure_imaginary_argument (void)
{
  /* tan(-0 +i*y) = -0 +i*tanh(y) */
  /* tan(+0 +i*y) = +0 +i*tanh(y) */
  mpfr_t y;
  mpfr_t tanh_y;
  mpc_t z;
  mpc_t tan_z;
  mpfr_prec_t prec = (mpfr_prec_t) 111;

  mpfr_init2 (y, 2);
  mpfr_init2 (tanh_y, prec);
  mpc_init2 (z, 2);
  mpc_init2 (tan_z, prec);

  /* tan(0 +i) = +0 +i*tanh(1) */
  mpc_set_ui_ui (z, 0, 1, MPC_RNDNN);
  mpfr_set_ui (y, 1, GMP_RNDN);
  mpfr_tanh (tanh_y, y, GMP_RNDN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_imagref (tan_z), tanh_y) != 0
      || !mpfr_zero_p (mpc_realref (tan_z)) || mpfr_signbit (mpc_realref (tan_z)))
    {
      mpc_t c99;

      mpc_init2 (c99, prec);
      mpfr_set_ui (mpc_realref (c99), 0, GMP_RNDN);
      mpfr_set (mpc_imagref (c99), tanh_y, GMP_RNDN);

      TEST_FAILED ("mpc_tan", z, tan_z, c99, MPC_RNDNN);
    }

  /* tan(0 -i) = +0 +i*tanh(-1) */
  mpc_conj (z, z, MPC_RNDNN);
  mpfr_neg (tanh_y, tanh_y, GMP_RNDN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_imagref (tan_z), tanh_y) != 0
      || !mpfr_zero_p (mpc_realref (tan_z)) || mpfr_signbit (mpc_realref (tan_z)))
    {
      mpc_t c99;

      mpc_init2 (c99, prec);
      mpfr_set_ui (mpc_realref (c99), 0, GMP_RNDN);
      mpfr_set (mpc_imagref (c99), tanh_y, GMP_RNDN);

      TEST_FAILED ("mpc_tan", z, tan_z, c99, MPC_RNDNN);
    }

  /* tan(-0 +i) = -0 +i*tanh(1) */
  mpc_neg (z, z, MPC_RNDNN);
  mpfr_neg (tanh_y, tanh_y, GMP_RNDN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_imagref (tan_z), tanh_y) != 0
      || !mpfr_zero_p (mpc_realref (tan_z)) || !mpfr_signbit (mpc_realref (tan_z)))
    {
      mpc_t c99;

      mpc_init2 (c99, prec);
      mpfr_set_ui (mpc_realref (c99), 0, GMP_RNDN);
      mpfr_set (mpc_imagref (c99), tanh_y, GMP_RNDN);

      TEST_FAILED ("mpc_tan", z, tan_z, c99, MPC_RNDNN);
    }

  /* tan(-0 -i) = -0 +i*tanh(-1) */
  mpc_conj (z, z, MPC_RNDNN);
  mpfr_neg (tanh_y, tanh_y, GMP_RNDN);
  mpc_tan (tan_z, z, MPC_RNDNN);
  if (mpfr_cmp (mpc_imagref (tan_z), tanh_y) != 0
      || !mpfr_zero_p (mpc_realref (tan_z)) || !mpfr_signbit (mpc_realref (tan_z)))
    {
      mpc_t c99;

      mpc_init2 (c99, prec);
      mpfr_set_ui (mpc_realref (c99), 0, GMP_RNDN);
      mpfr_set (mpc_imagref (c99), tanh_y, GMP_RNDN);

      TEST_FAILED ("mpc_tan", z, tan_z, c99, MPC_RNDNN);
    }

  mpc_clear (tan_z);
  mpc_clear (z);
  mpfr_clear (tanh_y);
  mpfr_clear (y);
}

int
main (void)
{
  DECL_FUNC (CC, f, mpc_tan);

  test_start ();

  data_check (f, "tan.dat");
  tgeneric (f, 2, 512, 7, 4);

  pure_real_argument ();
  pure_imaginary_argument ();

  test_end ();

  return 0;
}
