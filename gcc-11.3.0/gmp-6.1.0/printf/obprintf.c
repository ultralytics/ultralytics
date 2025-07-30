/* gmp_obstack_printf -- formatted output to an obstack.

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

#include "config.h"

#if HAVE_OBSTACK_VPRINTF

#include <stdarg.h>
#include <obstack.h>
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"


int
gmp_obstack_printf (struct obstack *ob, const char *fmt, ...)
{
  va_list  ap;
  int      ret;

  va_start (ap, fmt);

  ASSERT (! MEM_OVERLAP_P (obstack_base(ob), obstack_object_size(ob),
                           fmt, strlen(fmt)+1));

  ret = __gmp_doprnt (&__gmp_obstack_printf_funs, ob, fmt, ap);
  va_end (ap);
  return ret;
}

#endif /* HAVE_OBSTACK_VPRINTF */
