/* Test file for multiple mpfr.h inclusion and va_list related functions

Copyright 2011-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#if HAVE_STDARG

#if _MPFR_EXP_FORMAT == 4
/* If mpfr_exp_t is defined as intmax_t, intmax_t must be defined before
   the inclusion of mpfr.h (this test doesn't use mpfr-impl.h). */
# include <stdint.h>
#endif

/* Assume that this is in fact a header inclusion for some library
   that uses MPFR, i.e. this inclusion is hidden in another one.
   MPFR currently (rev 6704) fails to handle this case. */
#include <mpfr.h>

#include <stdarg.h>
#define MPFR_USE_VA_LIST /* necessary due to GMP bug concerning inclusions */
#include <mpfr.h>

#include <stdio.h>
#define MPFR_USE_FILE /* necessary due to GMP bug concerning inclusions */
#include <mpfr.h>

#include "mpfr-test.h"

static void
test (FILE *fout, const char *fmt, ...)
{
  int (*fct) (FILE*, __gmp_const char*, va_list);

  fct = mpfr_vfprintf;
  if (0)
    {
      va_list ap;
      va_start (ap, fmt);
      fct (fout, fmt, ap);
      va_end (ap);
    }
}

int
main (void)
{
  tests_start_mpfr ();
  test (stdout, "%d\n", 0);
  tests_end_mpfr ();
  return 0;
}

#else  /* HAVE_STDARG */

/* The test is disabled. */

int
main (void)
{
  return 77;
}

#endif  /* HAVE_STDARG */
