/* Manipulable localeconv and nl_langinfo.

Copyright 2001, 2002, 2014 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include "config.h"

#if HAVE_NL_TYPES_H
#include <nl_types.h>  /* for nl_item */
#endif

#if HAVE_LANGINFO_H
#include <langinfo.h>  /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h>    /* for lconv */
#endif


/* Replace the libc localeconv and nl_langinfo with ones we can manipulate.

   This is done in a C file since if it was in a C++ file then we'd have to
   match the "throw" or lack thereof declared for localeconv in <locale.h>.
   g++ 3.2 gives an error about mismatched throws under "-pedantic", other
   C++ compilers may very possibly do so too.  */

extern char point_string[];

#if HAVE_LOCALECONV
struct lconv *
localeconv (void)
#if defined __cplusplus && defined __GLIBC__
  throw()
#endif
{
  static struct lconv  l;
  l.decimal_point = point_string;
  return &l;
}
#endif

#if HAVE_NL_LANGINFO
char *
nl_langinfo (nl_item n)
#if defined __cplusplus && defined __GLIBC__
  throw()
#endif
{
  return point_string;
}
#endif
