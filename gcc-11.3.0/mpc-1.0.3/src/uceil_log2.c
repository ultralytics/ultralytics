/* mpc_ceil_log2 - returns ceil(log(d)/log(2))

Copyright (C) 2004, 2009, 2010 INRIA

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

/* returns ceil(log(d)/log(2)) if d > 0 */
/* Don't use count_leading_zeros since it is in longlong.h */
mpfr_prec_t
mpc_ceil_log2 (mpfr_prec_t d)
{
  mpfr_prec_t exp;

  for (exp = 0; d > 1; d = (d + 1) / 2)
    exp++;
  return exp;
}
