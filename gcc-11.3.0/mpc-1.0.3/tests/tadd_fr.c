/* tadd_fr -- test file for mpc_add_fr.

Copyright (C) 2008, 2010, 2012 INRIA

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
check_ternary_value (mpfr_prec_t prec_max, mpfr_prec_t step)
{
  mpfr_prec_t prec;
  mpc_t z;
  mpfr_t f;

  mpc_init2 (z, 2);
  mpfr_init (f);

  for (prec = 2; prec < prec_max; prec += step)
    {
      mpc_set_prec (z, prec);
      mpfr_set_prec (f, prec);

      mpc_set_ui (z, 1, MPC_RNDNN);
      mpfr_set_ui (f, 1, GMP_RNDN);
      if (mpc_add_fr (z, z, f, MPC_RNDNZ))
        {
          printf ("Error in mpc_add_fr: 1+1 should be exact\n");
          exit (1);
        }

      mpc_set_ui (z, 1, MPC_RNDNN);
      mpc_mul_2ui (z, z, (unsigned long int) prec, MPC_RNDNN);
      if (mpc_add_fr (z, z, f, MPC_RNDNN) == 0)
        {
          fprintf (stderr, "Error in mpc_add_fr: 2^prec+1 cannot be exact\n");
          exit (1);
        }
    }
  mpc_clear (z);
  mpfr_clear (f);
}

int
main (void)
{
  DECL_FUNC (CCF, f, mpc_add_fr);
  test_start ();

  check_ternary_value (1024, 1);

  data_check (f, "add_fr.dat");
  tgeneric (f, 2, 1024, 7, 10);

  test_end ();
  return 0;
}
