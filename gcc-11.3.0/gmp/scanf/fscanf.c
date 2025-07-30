/* gmp_fscanf -- formatted input from a FILE.

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

#include "gmp.h"
#include "gmp-impl.h"


int
gmp_fscanf (FILE *fp, const char *fmt, ...)
{
  va_list  ap;
  int      ret;
  va_start (ap, fmt);

  ret = __gmp_doscan (&__gmp_fscanf_funs, fp, fmt, ap);
  va_end (ap);
  return ret;
}
