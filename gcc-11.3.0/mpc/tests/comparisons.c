/* comparisons.c -- Comparison functions.

Copyright (C) 2008, 2009, 2011 INRIA

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

/* comparisons, see description in mpc-tests.h */
int
same_mpfr_value (mpfr_ptr got, mpfr_ptr ref, int known_sign)
{
   /* The sign of zeroes and infinities is checked only when
      known_sign is true.                                    */
   if (mpfr_nan_p (got))
      return mpfr_nan_p (ref);
   if (mpfr_inf_p (got))
      return mpfr_inf_p (ref) &&
            (!known_sign || mpfr_signbit (got) == mpfr_signbit (ref));
   if (mpfr_zero_p (got))
      return mpfr_zero_p (ref) &&
            (!known_sign || mpfr_signbit (got) == mpfr_signbit (ref));
   return mpfr_cmp (got, ref) == 0;
}

int
same_mpc_value (mpc_ptr got, mpc_ptr ref, known_signs_t known_signs)
{
   return    same_mpfr_value (mpc_realref (got), mpc_realref (ref), known_signs.re)
          && same_mpfr_value (mpc_imagref (got), mpc_imagref (ref), known_signs.im);
}
