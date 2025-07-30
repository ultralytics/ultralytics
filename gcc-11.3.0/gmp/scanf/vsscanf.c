/* gmp_vsscanf -- formatted input from a string.

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

#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"


int
gmp_vsscanf (const char *s, const char *fmt, va_list ap)
{
#if SSCANF_WRITABLE_INPUT
  /* We only actually need this if there's standard C types in fmt, and if
     "s" is not already writable, but it's too much trouble to check that,
     and in any case this writable sscanf input business is only for a few
     old systems. */
  size_t size;
  char   *alloc;
  int    ret;
  size = strlen (s) + 1;
  alloc = (char *) (*__gmp_allocate_func) (size);
  memcpy (alloc, s, size);
  s = alloc;
  ret = __gmp_doscan (&__gmp_sscanf_funs, (void *) &s, fmt, ap);
  (*__gmp_free_func) (alloc, size);
  return ret;

#else
  return __gmp_doscan (&__gmp_sscanf_funs, (void *) &s, fmt, ap);
#endif
}
