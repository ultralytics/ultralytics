/* Test gmp_randinit_set.

Copyright 2003 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

/* expect after a gmp_randinit_set that the new and old generators will
   produce the same sequence of numbers */
void
check_one (const char *name, gmp_randstate_ptr src)
{
  gmp_randstate_t dst;
  mpz_t  sz, dz;
  int    i;

  gmp_randinit_set (dst, src);
  mpz_init (sz);
  mpz_init (dz);

  for (i = 0; i < 20; i++)
    {
      mpz_urandomb (sz, src, 123);
      mpz_urandomb (dz, dst, 123);

      if (mpz_cmp (sz, dz) != 0)
        {
          printf     ("gmp_randinit_set didn't duplicate randstate\n");
          printf     ("  algorithm: %s\n", name);
          gmp_printf ("  from src:  %#Zx\n", sz);
          gmp_printf ("  from dst:  %#Zx\n", dz);
          abort ();
        }
    }

  mpz_clear (sz);
  mpz_clear (dz);
  gmp_randclear (dst);
}

int
main (int argc, char *argv[])
{
  tests_start ();

  call_rand_algs (check_one);

  tests_end ();
  exit (0);
}
