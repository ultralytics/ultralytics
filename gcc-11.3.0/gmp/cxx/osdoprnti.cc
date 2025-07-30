/* __gmp_doprnt_integer_ios -- integer formatted output to an ostream.

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

#include <iostream>
#include <stdarg.h>   /* for va_list and hence doprnt_funs_t */
#include <string.h>   /* for strlen */

#include "gmp.h"
#include "gmp-impl.h"

using namespace std;


/* The gmp_asprintf support routines never give an error, so
   __gmp_doprnt_integer shouldn't fail and it's return can just be checked
   with an ASSERT.  */

ostream&
__gmp_doprnt_integer_ostream (ostream &o, struct doprnt_params_t *p,
                              char *s)
{
  struct gmp_asprintf_t   d;
  char  *result;
  int   ret;

  /* don't show leading zeros the way printf does */
  p->prec = -1;

  GMP_ASPRINTF_T_INIT (d, &result);
  ret = __gmp_doprnt_integer (&__gmp_asprintf_funs_noformat, &d, p, s);
  ASSERT (ret != -1);
  __gmp_asprintf_final (&d);
  (*__gmp_free_func) (s, strlen(s)+1);

  gmp_allocated_string  t (result);
  return o.write (t.str, t.len);
}
