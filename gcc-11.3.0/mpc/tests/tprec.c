/* tprec -- Test file for mpc_set_prec, mpc_get_prec and mpc_get_prec2.

Copyright (C) 2009, 2011 INRIA

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

int
main (void)
{
  mpc_t z;
  mpfr_prec_t prec, pr, pi;

  mpc_init2 (z, 1000);

  for (prec = 2; prec <= 1000; prec++)
    {
      /* check set_prec/get_prec */
      mpfr_set_prec (mpc_realref (z), prec);
      mpfr_set_prec (mpc_imagref (z), prec + 1);
      if (mpc_get_prec (z) != 0)
        {
          printf ("Error in mpc_get_prec for prec (re) = %lu, "
                  "prec (im) = %lu\n", (unsigned long int) prec,
                  (unsigned long int) prec + 1ul);

          exit (1);
        }

      mpc_get_prec2 (&pr, &pi, z);
      if (pr != prec || pi != prec + 1)
        {
          printf ("Error in mpc_get_prec2 for prec (re) = %lu, "
                  "prec (im) = %lu\n", (unsigned long int) prec,
                  (unsigned long int) prec + 1ul);

          exit (1);
        }

      mpc_set_prec (z, prec);
      if (mpc_get_prec (z) != prec)
        {
          printf ("Error in mpc_get_prec for prec = %lu\n",
                  (unsigned long int) prec);

          exit (1);
        }
    }

  mpc_clear (z);

  return 0;
}
