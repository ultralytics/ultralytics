/* __gmp_asprintf_memory etc -- formatted output to allocated space.

Copyright 2001, 2002 Free Software Foundation, Inc.

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


/* These routines are in a separate file so that the mpz_t, mpq_t and mpf_t
   operator<< routines can avoid dragging vsnprintf into the link (via
   __gmp_asprintf_format).  */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"


int
__gmp_asprintf_memory (struct gmp_asprintf_t *d, const char *str, size_t len)
{
  GMP_ASPRINTF_T_NEED (d, len);
  memcpy (d->buf + d->size, str, len);
  d->size += len;
  return len;
}

int
__gmp_asprintf_reps (struct gmp_asprintf_t *d, int c, int reps)
{
  GMP_ASPRINTF_T_NEED (d, reps);
  memset (d->buf + d->size, c, reps);
  d->size += reps;
  return reps;
}

int
__gmp_asprintf_final (struct gmp_asprintf_t *d)
{
  char  *buf = d->buf;
  ASSERT (d->alloc >= d->size + 1);
  buf[d->size] = '\0';
  __GMP_REALLOCATE_FUNC_MAYBE_TYPE (buf, d->alloc, d->size+1, char);
  *d->result = buf;
  return 0;
}
