/* mpz expression evaluation, simple part */

/*
Copyright 2000, 2001 Free Software Foundation, Inc.

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

#include <stdio.h>
#include "gmp.h"
#include "expr-impl.h"


int
mpexpr_va_to_var (void *var[], va_list ap)
{
  int   i = 0;
  void  *v;

  for (;;)
    {
      v = va_arg (ap, void *);
      if (v == NULL)
	break;
      if (i >= MPEXPR_VARIABLES)
	return MPEXPR_RESULT_BAD_VARIABLE;
      var[i++] = v;
    }

  while (i < MPEXPR_VARIABLES)
    var[i++] = NULL;

  return MPEXPR_RESULT_OK;
}
