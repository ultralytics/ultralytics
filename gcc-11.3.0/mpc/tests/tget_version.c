/* tget_version -- Test file for mpc_get_version

Copyright (C) 2002, 2003, 2004, 2005, 2007, 2008, 2009, 2010, 2011 INRIA

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

#include <string.h>
#include <stdlib.h>
#include "mpc-tests.h"

int
main (void)
{
#ifdef __MPIR_VERSION
  printf ("MPIR: include %d.%d.%d, lib %s\n",
          __MPIR_VERSION, __MPIR_VERSION_MINOR, __MPIR_VERSION_PATCHLEVEL,
          mpir_version);
#else
  printf ("GMP: include %d.%d.%d, lib %s\n",
          __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR, __GNU_MP_VERSION_PATCHLEVEL,
          gmp_version);
#endif
  printf ("MPFR: include %s, lib %s\n",
          MPFR_VERSION_STRING,
          mpfr_get_version ());
  printf ("MPC: include %s, lib %s\n", MPC_VERSION_STRING,
          mpc_get_version ());

  if (strcmp (mpc_get_version (), MPC_VERSION_STRING) != 0)
    {
      printf ("Error: header and library do not match\n"
              "mpc_get_version: \"%s\"\nMPC_VERSION_STRING: \"%s\"\n",
              mpc_get_version(), MPC_VERSION_STRING);
      exit (1);
    }

#ifdef MPC_CC
  printf ("C compiler: %s\n", MPC_CC);
#endif
#ifdef MPC_GCC
  printf ("GCC: %s\n", MPC_GCC);
#endif
#ifdef MPC_GCC_VERSION
  printf ("GCC version: %s\n", MPC_GCC_VERSION);
#endif

  return 0;
}
