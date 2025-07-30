/* operator<< -- mpf formatted output to an ostream.

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

#include <clocale>
#include <iostream>
#include <stdarg.h>    /* for va_list and hence doprnt_funs_t */
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"

using namespace std;


/* The gmp_asprintf support routines never give an error, so
   __gmp_doprnt_mpf shouldn't fail and it's return can just be checked with
   an ASSERT.  */

ostream&
operator<< (ostream &o, mpf_srcptr f)
{
  struct doprnt_params_t  param;
  struct gmp_asprintf_t   d;
  char  *result;
  int   ret;

  __gmp_doprnt_params_from_ios (&param, o);

#if HAVE_STD__LOCALE
  char  point[2];
  point[0] = use_facet< numpunct<char> >(o.getloc()).decimal_point();
  point[1] = '\0';
#else
  const char *point = GMP_DECIMAL_POINT;
#endif

  GMP_ASPRINTF_T_INIT (d, &result);
  ret = __gmp_doprnt_mpf (&__gmp_asprintf_funs_noformat, &d, &param, point, f);
  ASSERT (ret != -1);
  __gmp_asprintf_final (&d);

  gmp_allocated_string  t (result);
  return o.write (t.str, t.len);
}
