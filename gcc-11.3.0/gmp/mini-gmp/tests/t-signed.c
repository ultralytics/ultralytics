/* Exercise some mpz_..._si functions.

Copyright 2013 Free Software Foundation, Inc.

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

#include "testutils.h"

int
check_si (mpz_t sz, mpz_t oz, long si, long oi, int c)
{
  mpz_t t;
  int fail;

  if (mpz_cmp_si (sz, oi) != c)
    {
      printf ("mpz_cmp_si (sz, %ld) != %i.\n", oi, c);
      printf (" sz="); mpz_out_str (stdout, 10, sz); printf ("\n");
      abort ();
    }

  if ((si < oi ? -1 : si > oi) != c)
    return 1;

  mpz_init_set_si (t, si);

  if ((fail = mpz_cmp_si (sz, si)) != 0)
    printf ("mpz_cmp_si (sz, %ld) != 0.\n", si);
  if (mpz_cmp_si (oz, si) != -c)
    printf ("mpz_cmp_si (oz, %ld) != %i.\n", si, -c), fail = 1;
  if (! mpz_fits_slong_p (sz))
    printf ("mpz_fits_slong_p (sz) != 1.\n"), fail = 1;
  if (mpz_get_si (sz) != si)
    printf ("mpz_get_si (sz) != %ld.\n", si), fail = 1;
  if (mpz_cmp (t, sz) != 0)
    {
      printf ("mpz_init_set_si (%ld) failed.\n", si);
      printf (" got="); mpz_out_str (stdout, 10, t); printf ("\n");
      fail = 1;
    }

  mpz_clear (t);

  if (fail)
    {
      printf (" sz="); mpz_out_str (stdout, 10, sz); printf ("\n");
      printf (" oz="); mpz_out_str (stdout, 10, oz); printf ("\n");
      printf (" si=%ld\n", si);
      abort ();
    }

  return 0;
}

void
try_op_si (int c)
{
  long  si, oi;
  mpz_t sz, oz;

  si = c;
  mpz_init_set_si (sz, si);

  oi = si;
  mpz_init_set (oz, sz);

  do {
    si *= 2; /* c * 2^k */
    mpz_mul_2exp (sz, sz, 1);

    if (check_si (sz, oz, si, oi, c))
      {
	mpz_set (oz, sz);
	break;
      }

    oi = si + c; /* c * (2^k + 1) */
    if (c == -1)
      mpz_sub_ui (oz, sz, 1);
    else
      mpz_add_ui (oz, sz, 1);

    if (check_si (oz, sz, oi, si, c))
      break;

    oi = (si - c) * 2 + c; /* c * (2^K - 1) */
    mpz_mul_si (oz, sz, 2*c);
    if (c == -1)
      mpz_ui_sub (oz, 1, oz); /* oz = sz * 2 + 1 */
    else
      mpz_sub_ui (oz, oz, 1); /* oz = sz * 2 - 1 */
  } while (check_si (oz, sz, oi, si, c) == 0);

  mpz_clear (sz);

  if (mpz_fits_slong_p (oz))
    {
      printf ("Should not fit a signed long any more.\n");
      printf (" oz="); mpz_out_str (stdout, 10, oz); printf ("\n");
      abort ();
    }

  if (mpz_cmp_si (oz, -c) != c)
      {
	printf ("mpz_cmp_si (oz, %i) != %i.\n", c, c);
	printf (" oz="); mpz_out_str (stdout, 10, oz); printf ("\n");
	abort ();
      }

  mpz_mul_2exp (oz, oz, 1);
  if (mpz_cmp_si (oz, -c) != c)
      {
	printf ("mpz_cmp_si (oz, %i) != %i.\n", c, c);
	printf (" oz="); mpz_out_str (stdout, 10, oz); printf ("\n");
	abort ();
      }

  mpz_clear (oz);
}

void
testmain (int argc, char *argv[])
{
  try_op_si (-1);
  try_op_si (1);
}
