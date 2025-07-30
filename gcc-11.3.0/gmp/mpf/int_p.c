/* mpf_integer_p -- test whether an mpf is an integer */

/*
Copyright 2001, 2002, 2014-2015 Free Software Foundation, Inc.

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

#include "gmp.h"
#include "gmp-impl.h"


int
mpf_integer_p (mpf_srcptr f) __GMP_NOTHROW
{
  mp_srcptr fp;
  mp_exp_t exp;
  mp_size_t size;

  size = SIZ (f);
  exp = EXP (f);
  if (exp <= 0)
    return (size == 0);  /* zero is an integer,
			    others have only fraction limbs */
  size = ABS (size);

  /* Ignore zeroes at the low end of F.  */
  for (fp = PTR (f); *fp == 0; ++fp)
    --size;

  /* no fraction limbs */
  return size <= exp;
}
