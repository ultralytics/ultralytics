/* tsqrt -- test file for mpc_sqrt.

Copyright (C) 2008 INRIA

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
  DECL_FUNC (CC, f, mpc_sqrt);

  test_start ();

  data_check (f, "sqrt.dat");
  tgeneric (f, 2, 1024, 7, 256);

  test_end ();

  return 0;
}
