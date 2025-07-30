/* tswap -- Test file for mpc_swap.

Copyright (C) 2011 INRIA

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

int
main (void)
{
   mpc_t x, y, x2, y2;

   mpc_init2 (x, 50);
   mpc_init2 (x2, 50);
   mpc_init2 (y, 100);
   mpc_init2 (y2, 100);

   mpc_set_ui_ui (x,  1ul, 2ul, MPC_RNDNN);
   mpc_set_ui_ui (x2, 1ul, 2ul, MPC_RNDNN);
   mpc_set_ui_ui (y,  3ul, 4ul, MPC_RNDNN);
   mpc_set_ui_ui (y2, 3ul, 4ul, MPC_RNDNN);

   mpc_swap (x, y);

   if (   mpfr_get_prec (mpc_realref (x)) != mpfr_get_prec (mpc_realref (y2))
       || mpfr_get_prec (mpc_imagref (x)) != mpfr_get_prec (mpc_imagref (y2))
       || mpfr_get_prec (mpc_realref (y)) != mpfr_get_prec (mpc_realref (x2))
       || mpfr_get_prec (mpc_imagref (y)) != mpfr_get_prec (mpc_imagref (x2))
       || mpc_cmp (x, y2) != 0
       || mpc_cmp (y, x2) != 0)
      exit (1);

   mpc_clear (x);
   mpc_clear (x2);
   mpc_clear (y);
   mpc_clear (y2);

   return 0;
}
