/* __gmp_snprintf_funs -- support for gmp_snprintf and gmp_vsnprintf.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

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

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"


#if ! HAVE_VSNPRINTF
#define vsnprintf  __gmp_replacement_vsnprintf
#endif


/* glibc 2.0.x vsnprintf returns either -1 or size-1 for an overflow, with
   no indication how big the output would have been.  It's necessary to
   re-run to determine that size.

   "size-1" would mean success from a C99 vsnprintf, and the re-run is
   unnecessary in this case, but we don't bother to try to detect what sort
   of vsnprintf we've got.  size-1 should occur rarely in normal
   circumstances.

   vsnprintf might trash it's given ap (it does for instance in glibc 2.1.3
   on powerpc), so copy it in case we need to use it to probe for the size
   output that would have been produced.  Note there's no need to preserve
   it for our callers, just for ourselves.  */

static int
gmp_snprintf_format (struct gmp_snprintf_t *d, const char *fmt,
                     va_list orig_ap)
{
  int      ret, step, alloc, avail;
  va_list  ap;
  char     *p;

  ASSERT (d->size >= 0);

  avail = d->size;
  if (avail > 1)
    {
      va_copy (ap, orig_ap);
      ret = vsnprintf (d->buf, avail, fmt, ap);
      if (ret == -1)
        {
          ASSERT (strlen (d->buf) == avail-1);
          ret = avail-1;
        }

      step = MIN (ret, avail-1);
      d->size -= step;
      d->buf += step;

      if (ret != avail-1)
        return ret;

      /* probably glibc 2.0.x truncated output, probe for actual size */
      alloc = MAX (128, ret);
    }
  else
    {
      /* no space to write anything, just probe for size */
      alloc = 128;
    }

  do
    {
      alloc *= 2;
      p = __GMP_ALLOCATE_FUNC_TYPE (alloc, char);
      va_copy (ap, orig_ap);
      ret = vsnprintf (p, alloc, fmt, ap);
      (*__gmp_free_func) (p, alloc);
    }
  while (ret == alloc-1 || ret == -1);

  return ret;
}

static int
gmp_snprintf_memory (struct gmp_snprintf_t *d, const char *str, size_t len)
{
  size_t n;

  ASSERT (d->size >= 0);

  if (d->size > 1)
    {
      n = MIN (d->size-1, len);
      memcpy (d->buf, str, n);
      d->buf += n;
      d->size -= n;
    }
  return len;
}

static int
gmp_snprintf_reps (struct gmp_snprintf_t *d, int c, int reps)
{
  size_t n;

  ASSERT (reps >= 0);
  ASSERT (d->size >= 0);

  if (d->size > 1)
    {
      n = MIN (d->size-1, reps);
      memset (d->buf, c, n);
      d->buf += n;
      d->size -= n;
    }
  return reps;
}

static int
gmp_snprintf_final (struct gmp_snprintf_t *d)
{
  if (d->size >= 1)
    d->buf[0] = '\0';
  return 0;
}

const struct doprnt_funs_t  __gmp_snprintf_funs = {
  (doprnt_format_t) gmp_snprintf_format,
  (doprnt_memory_t) gmp_snprintf_memory,
  (doprnt_reps_t)   gmp_snprintf_reps,
  (doprnt_final_t)  gmp_snprintf_final
};
