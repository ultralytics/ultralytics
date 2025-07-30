/* gmp_randinit (state, algorithm, ...) -- Initialize a random state.

Copyright 1999-2002 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include <stdarg.h>

#include "gmp.h"
#include "gmp-impl.h"

void
gmp_randinit (gmp_randstate_t rstate, gmp_randalg_t alg, ...)
{
  va_list ap;
  va_start (ap, alg);

  switch (alg) {
  case GMP_RAND_ALG_LC:
    if (! gmp_randinit_lc_2exp_size (rstate, va_arg (ap, unsigned long)))
      gmp_errno |= GMP_ERROR_INVALID_ARGUMENT;
    break;
  default:
    gmp_errno |= GMP_ERROR_UNSUPPORTED_ARGUMENT;
    break;
  }
  va_end (ap);
}
