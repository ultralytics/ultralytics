/* Test LIMB_HIGHBIT_TO_MASK.

Copyright 2004 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


/* There's very little to these tests, but it's nice to have them since if
   something has gone wrong with the arithmetic right shift business in
   LIMB_HIGHBIT_TO_MASK then the only symptom is likely to be failures in
   udiv_qrnnd_preinv, which would not be easy to diagnose.  */

int
main (void)
{
  ASSERT_ALWAYS (LIMB_HIGHBIT_TO_MASK (0) == 0);
  ASSERT_ALWAYS (LIMB_HIGHBIT_TO_MASK (GMP_LIMB_HIGHBIT) == MP_LIMB_T_MAX);
  ASSERT_ALWAYS (LIMB_HIGHBIT_TO_MASK (MP_LIMB_T_MAX) == MP_LIMB_T_MAX);
  ASSERT_ALWAYS (LIMB_HIGHBIT_TO_MASK (GMP_LIMB_HIGHBIT >> 1) == 0);
  ASSERT_ALWAYS (LIMB_HIGHBIT_TO_MASK (MP_LIMB_T_MAX >> 1) == 0);

  exit (0);
}
