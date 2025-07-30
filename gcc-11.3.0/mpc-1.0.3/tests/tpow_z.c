/* tpow_z -- test file for mpc_pow_z.

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

#include <limits.h> /* for CHAR_BIT */
#include "mpc-tests.h"

int
main (void)
{
   mpc_t z;
   mpz_t t;

   test_start ();

   mpc_init2 (z, 5);
   mpz_init_set_ui (t, 1ul);
   mpc_set_ui_ui (z, 17ul, 42ul, MPC_RNDNN);
   mpc_pow_z (z, z, t, MPC_RNDNN);
   if (mpc_cmp_si_si (z, 17l, 42l) != 0) {
         printf ("Error for mpc_pow_z (1)\n");
         exit (1);
   }
   mpz_set_si (t, -1l);
   mpc_set_ui_ui (z, 1ul, 1ul, MPC_RNDNN);
   mpc_pow_z (z, z, t, MPC_RNDNN);
   mpc_mul_ui (z, z, 2ul, MPC_RNDNN);
   if (mpc_cmp_si_si (z, 1l, -1l) != 0) {
         printf ("Error for mpc_pow_z (-1)\n");
         exit (1);
   }
   mpz_set_ui (t, 1ul);
   mpz_mul_2exp (t, t, sizeof (long) * CHAR_BIT);
   mpc_set_ui_ui (z, 0ul, 1ul, MPC_RNDNN);
   mpc_pow_z (z, z, t, MPC_RNDNN);
   if (mpc_cmp_si_si (z, 1l, 0l) != 0) {
         printf ("Error for mpc_pow_z (4*large)\n");
         exit (1);
   }
   mpc_clear (z);
   mpz_clear (t);

   test_end ();

   return 0;
}
