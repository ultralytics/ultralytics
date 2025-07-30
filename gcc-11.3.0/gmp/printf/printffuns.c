/* __gmp_fprintf_funs -- support for formatted output to FILEs.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

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
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"

/* SunOS 4 stdio.h doesn't provide a prototype for this */
#if ! HAVE_DECL_VFPRINTF
int vfprintf (FILE *, const char *, va_list);
#endif


static int
gmp_fprintf_memory (FILE *fp, const char *str, size_t len)
{
  return fwrite (str, 1, len, fp);
}

/* glibc putc is a function, at least when it's in multi-threaded mode or
   some such, so fwrite chunks instead of making many calls. */
static int
gmp_fprintf_reps (FILE *fp, int c, int reps)
{
  char  buf[256];
  int   i, piece, ret;
  ASSERT (reps >= 0);

  memset (buf, c, MIN (reps, sizeof (buf)));
  for (i = reps; i > 0; i -= sizeof (buf))
    {
      piece = MIN (i, sizeof (buf));
      ret = fwrite (buf, 1, piece, fp);
      if (ret == -1)
        return ret;
      ASSERT (ret == piece);
    }

  return reps;
}

const struct doprnt_funs_t  __gmp_fprintf_funs = {
  (doprnt_format_t) vfprintf,
  (doprnt_memory_t) gmp_fprintf_memory,
  (doprnt_reps_t)   gmp_fprintf_reps,
};
