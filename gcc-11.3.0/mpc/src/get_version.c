/* mpc_get_version -- MPC version

Copyright (C) 2008, 2009, 2010, 2011, 2012, 2014, 2015 INRIA

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

#include "mpc-impl.h"

#if MPFR_VERSION_MAJOR < 3
/* The following are functions defined for compatibility with mpfr < 3;
   logically, they should be defined in a separate file, but then gcc
   complains about an empty translation unit with mpfr >= 3.            */

void
mpfr_set_zero (mpfr_ptr z, int s)
{
   mpfr_set_ui (z, 0ul, GMP_RNDN);
   if (s < 0)
      mpfr_neg (z, z, GMP_RNDN);
}

int
mpfr_regular_p (mpfr_srcptr z)
{
   return (mpfr_number_p (z) && !mpfr_zero_p (z));
}
#endif /* mpfr < 3 */


const char *
mpc_get_version (void)
{
  return "1.0.3";
}

