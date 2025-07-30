/* gmp_vasprintf -- formatted output to an allocated space.

Copyright 2001 Free Software Foundation, Inc.

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"

#if ! HAVE_VSNPRINTF
#define vsnprintf  __gmp_replacement_vsnprintf
#endif


/* vasprintf isn't used since we prefer all GMP allocs to go through
   __gmp_allocate_func, and in particular we don't want the -1 return from
   vasprintf for out-of-memory, instead __gmp_allocate_func should handle
   that.  Using vsnprintf unfortunately means we might have to re-run it if
   our current space is insufficient.

   The initial guess for the needed space is an arbitrary 256 bytes.  If
   that (and any extra GMP_ASPRINTF_T_NEED might give) isn't enough then an
   ISO C99 standard vsnprintf will tell us what we really need.

   GLIBC 2.0.x vsnprintf returns either -1 or space-1 to indicate overflow,
   without giving any indication how much is really needed.  In this case
   keep trying with double the space each time.

   A return of space-1 is success on a C99 vsnprintf, but we're not
   bothering to identify which style vsnprintf we've got, so just take the
   pessimistic option and assume it's glibc 2.0.x.

   Notice the use of ret+2 for the new space in the C99 case.  This ensures
   the next vsnprintf return value will be space-2, which is unambiguously
   successful.  But actually GMP_ASPRINTF_T_NEED() will realloc to even
   bigger than that ret+2.

   vsnprintf might trash it's given ap, so copy it in case we need to use it
   more than once.  See comments with gmp_snprintf_format.  */

static int
gmp_asprintf_format (struct gmp_asprintf_t *d, const char *fmt,
                     va_list orig_ap)
{
  int      ret;
  va_list  ap;
  size_t   space = 256;

  for (;;)
    {
      GMP_ASPRINTF_T_NEED (d, space);
      space = d->alloc - d->size;
      va_copy (ap, orig_ap);
      ret = vsnprintf (d->buf + d->size, space, fmt, ap);
      if (ret == -1)
        {
          ASSERT (strlen (d->buf + d->size) == space-1);
          ret = space-1;
        }

      /* done if output fits in our space */
      if (ret < space-1)
        break;

      if (ret == space-1)
        space *= 2;     /* possible glibc 2.0.x, so double */
      else
        space = ret+2;  /* C99, so now know space required */
    }

  d->size += ret;
  return ret;
}

const struct doprnt_funs_t  __gmp_asprintf_funs = {
  (doprnt_format_t) gmp_asprintf_format,
  (doprnt_memory_t) __gmp_asprintf_memory,
  (doprnt_reps_t)   __gmp_asprintf_reps,
  (doprnt_final_t)  __gmp_asprintf_final
};

int
gmp_vasprintf (char **result, const char *fmt, va_list ap)
{
  struct gmp_asprintf_t  d;
  GMP_ASPRINTF_T_INIT (d, result);
  return __gmp_doprnt (&__gmp_asprintf_funs, &d, fmt, ap);
}
