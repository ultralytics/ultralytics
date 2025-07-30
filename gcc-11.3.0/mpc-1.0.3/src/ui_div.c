/* mpc_ui_div -- Divide an unsigned long int by a complex number.

Copyright (C) 2002, 2009 INRIA

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

#include <limits.h>
#include "mpc-impl.h"

int
mpc_ui_div (mpc_ptr a, unsigned long int b, mpc_srcptr c, mpc_rnd_t rnd)
{
  int inex;
  mpc_t bb;

  mpc_init2 (bb, sizeof(unsigned long int) * CHAR_BIT);
  mpc_set_ui (bb, b, rnd); /* exact */
  inex = mpc_div (a, bb, c, rnd);
  mpc_clear (bb);

  return inex;
}
