/* __gmp_sprintf_funs -- support for gmp_sprintf and gmp_vsprintf.

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
#include <stdlib.h>
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"


/* The data parameter "bufp" points to a "char *buf" which is the next
   character to be written, having started as the destination from the
   application.  This is then increased each time output is produced.  */


/* If vsprintf returns -1 then pass it upwards.  It doesn't matter that
   "*bufp" is ruined in this case, since gmp_doprint will bail out
   immediately anyway.  */
static int
gmp_sprintf_format (char **bufp, const char *fmt, va_list ap)
{
  char  *buf = *bufp;
  int   ret;
  vsprintf (buf, fmt, ap);
  ret = strlen (buf);
  *bufp = buf + ret;
  return ret;
}

static int
gmp_sprintf_memory (char **bufp, const char *str, size_t len)
{
  char  *buf = *bufp;
  *bufp = buf + len;
  memcpy (buf, str, len);
  return len;
}

static int
gmp_sprintf_reps (char **bufp, int c, int reps)
{
  char  *buf = *bufp;
  ASSERT (reps >= 0);
  *bufp = buf + reps;
  memset (buf, c, reps);
  return reps;
}

static int
gmp_sprintf_final (char **bufp)
{
  char  *buf = *bufp;
  *buf = '\0';
  return 0;
}

const struct doprnt_funs_t  __gmp_sprintf_funs = {
  (doprnt_format_t) gmp_sprintf_format,
  (doprnt_memory_t) gmp_sprintf_memory,
  (doprnt_reps_t)   gmp_sprintf_reps,
  (doprnt_final_t)  gmp_sprintf_final
};
