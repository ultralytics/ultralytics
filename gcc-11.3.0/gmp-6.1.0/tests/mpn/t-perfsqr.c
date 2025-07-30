/* Test mpn_perfect_square_p data.

Copyright 2002 Free Software Foundation, Inc.

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

#include "mpn/perfsqr.h"


#define PERFSQR_MOD_MASK   ((CNST_LIMB(1) << PERFSQR_MOD_BITS) - 1)

void
check_mod_2 (mp_limb_t d, mp_limb_t inv, mp_limb_t got_hi, mp_limb_t got_lo)
{
  int        want[2*GMP_LIMB_BITS], got;
  unsigned   r, idx;
  mp_limb_t  q;

  ASSERT_ALWAYS (d <= numberof (want));
  ASSERT_ALWAYS (((inv * d) & PERFSQR_MOD_MASK) == 1);
  ASSERT_ALWAYS (MP_LIMB_T_MAX / d >= PERFSQR_MOD_MASK);

  /* the squares mod d */
  for (r = 0; r < d; r++)
    want[r] = 0;
  for (r = 0; r < d; r++)
    want[(r*r)%d] = 1;

  /* for each remainder mod d, expect the table data to correctly identify
     it as a residue or non-residue */
  for (r = 0; r < d; r++)
    {
      /* as per PERFSQR_MOD_IDX */
      q = ((r) * (inv)) & PERFSQR_MOD_MASK;
      idx = (q * (d)) >> PERFSQR_MOD_BITS;

      if (idx >= GMP_LIMB_BITS)
        got = (got_hi >> (idx - GMP_LIMB_BITS)) & 1;
      else
        got = (got_lo >> idx) & 1;

      if (got != want[r])
        {
          printf ("Wrong generated data\n");
          printf ("  d=%u\n", (unsigned) d);
          printf ("  r=%u\n", r);
          printf ("  idx=%u\n", idx);
          printf ("  got  %d\n", got);
          printf ("  want %d\n", want[r]);
          abort ();
        }
    }
}

/* Check the generated data in perfsqr.h. */
void
check_mod (void)
{
#define PERFSQR_MOD_34(r, up, usize)       { r = 0; } /* so r isn't unused */
#define PERFSQR_MOD_PP(r, up, usize)       { r = 0; }
#define PERFSQR_MOD_1(r, d, inv, mask)     check_mod_2 (d, inv, CNST_LIMB(0), mask)
#define PERFSQR_MOD_2(r, d, inv, mhi, mlo) check_mod_2 (d, inv, mhi, mlo)

  PERFSQR_MOD_TEST (dummy, dummy);
}

/* Check PERFSQR_PP, if in use. */
void
check_pp (void)
{
#ifdef PERFSQR_PP
  ASSERT_ALWAYS_LIMB (PERFSQR_PP);
  ASSERT_ALWAYS_LIMB (PERFSQR_PP_NORM);
  ASSERT_ALWAYS_LIMB (PERFSQR_PP_INVERTED);

  /* preinv stuff only for nails==0 */
  if (GMP_NAIL_BITS == 0)
    {
      ASSERT_ALWAYS (PERFSQR_PP_NORM
                     == PERFSQR_PP << refmpn_count_leading_zeros (PERFSQR_PP));
      ASSERT_ALWAYS (PERFSQR_PP_INVERTED
                     == refmpn_invert_limb (PERFSQR_PP_NORM));
    }
#endif
}

int
main (void)
{
  tests_start ();

  check_mod ();
  check_pp ();

  tests_end ();
  exit (0);
}
