/* Test mpz_fits_*_p */

/*
Copyright 2001 Free Software Foundation, Inc.

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


/* Nothing sophisticated here, just exercise mpz_fits_*_p on a small amount
   of data. */

#define EXPECT_S(fun,name,answer)                                       \
  got = fun (z);                                                        \
  if (got != answer)                                                    \
    {                                                                   \
      printf ("%s (%s) got %d want %d\n", name, expr, got, answer);     \
      printf (" z size %d\n", SIZ(z));                                  \
      printf (" z dec "); mpz_out_str (stdout, 10, z); printf ("\n");   \
      printf (" z hex "); mpz_out_str (stdout, 16, z); printf ("\n");   \
      error = 1;                                                        \
    }

#define EXPECT(fun,answer)  EXPECT_S(fun,#fun,answer)

int
main (void)
{
  mpz_t       z;
  int         got;
  const char  *expr;
  int         error = 0;

  tests_start ();
  mpz_init (z);

  mpz_set_ui (z, 0L);
  expr = "0";
  EXPECT (mpz_fits_ulong_p, 1);
  EXPECT (mpz_fits_uint_p, 1);
  EXPECT (mpz_fits_ushort_p, 1);
  EXPECT (mpz_fits_slong_p, 1);
  EXPECT (mpz_fits_sint_p, 1);
  EXPECT (mpz_fits_sshort_p, 1);

  mpz_set_ui (z, 1L);
  expr = "1";
  EXPECT (mpz_fits_ulong_p, 1);
  EXPECT (mpz_fits_uint_p, 1);
  EXPECT (mpz_fits_ushort_p, 1);
  EXPECT (mpz_fits_slong_p, 1);
  EXPECT (mpz_fits_sint_p, 1);
  EXPECT (mpz_fits_sshort_p, 1);

  mpz_set_si (z, -1L);
  expr = "-1";
  EXPECT (mpz_fits_ulong_p, 0);
  EXPECT (mpz_fits_uint_p, 0);
  EXPECT (mpz_fits_ushort_p, 0);
  EXPECT (mpz_fits_slong_p, 1);
  EXPECT (mpz_fits_sint_p, 1);
  EXPECT (mpz_fits_sshort_p, 1);

  mpz_set_ui (z, 1L);
  mpz_mul_2exp (z, z, 5L*GMP_LIMB_BITS);
  expr = "2^(5*BPML)";
  EXPECT (mpz_fits_ulong_p, 0);
  EXPECT (mpz_fits_uint_p, 0);
  EXPECT (mpz_fits_ushort_p, 0);
  EXPECT (mpz_fits_slong_p, 0);
  EXPECT (mpz_fits_sint_p, 0);
  EXPECT (mpz_fits_sshort_p, 0);


  mpz_set_ui (z, (unsigned long) USHRT_MAX);
  expr = "USHRT_MAX";
  EXPECT (mpz_fits_ulong_p, 1);
  EXPECT (mpz_fits_uint_p, 1);
  EXPECT (mpz_fits_ushort_p, 1);

  mpz_set_ui (z, (unsigned long) USHRT_MAX);
  mpz_add_ui (z, z, 1L);
  expr = "USHRT_MAX + 1";
  EXPECT (mpz_fits_ushort_p, 0);


  mpz_set_ui (z, (unsigned long) UINT_MAX);
  expr = "UINT_MAX";
  EXPECT (mpz_fits_ulong_p, 1);
  EXPECT (mpz_fits_uint_p, 1);

  mpz_set_ui (z, (unsigned long) UINT_MAX);
  mpz_add_ui (z, z, 1L);
  expr = "UINT_MAX + 1";
  EXPECT (mpz_fits_uint_p, 0);


  mpz_set_ui (z, ULONG_MAX);
  expr = "ULONG_MAX";
  EXPECT (mpz_fits_ulong_p, 1);

  mpz_set_ui (z, ULONG_MAX);
  mpz_add_ui (z, z, 1L);
  expr = "ULONG_MAX + 1";
  EXPECT (mpz_fits_ulong_p, 0);


  mpz_set_si (z, (long) SHRT_MAX);
  expr = "SHRT_MAX";
  EXPECT (mpz_fits_slong_p, 1);
  EXPECT (mpz_fits_sint_p, 1);
  EXPECT (mpz_fits_sshort_p, 1);

  mpz_set_si (z, (long) SHRT_MAX);
  mpz_add_ui (z, z, 1L);
  expr = "SHRT_MAX + 1";
  EXPECT (mpz_fits_sshort_p, 0);


  mpz_set_si (z, (long) INT_MAX);
  expr = "INT_MAX";
  EXPECT (mpz_fits_slong_p, 1);
  EXPECT (mpz_fits_sint_p, 1);

  mpz_set_si (z, (long) INT_MAX);
  mpz_add_ui (z, z, 1L);
  expr = "INT_MAX + 1";
  EXPECT (mpz_fits_sint_p, 0);


  mpz_set_si (z, LONG_MAX);
  expr = "LONG_MAX";
  EXPECT (mpz_fits_slong_p, 1);

  mpz_set_si (z, LONG_MAX);
  mpz_add_ui (z, z, 1L);
  expr = "LONG_MAX + 1";
  EXPECT (mpz_fits_slong_p, 0);


  mpz_set_si (z, (long) SHRT_MIN);
  expr = "SHRT_MIN";
  EXPECT (mpz_fits_slong_p, 1);
  EXPECT (mpz_fits_sint_p, 1);
  EXPECT (mpz_fits_sshort_p, 1);

  mpz_set_si (z, (long) SHRT_MIN);
  mpz_sub_ui (z, z, 1L);
  expr = "SHRT_MIN + 1";
  EXPECT (mpz_fits_sshort_p, 0);


  mpz_set_si (z, (long) INT_MIN);
  expr = "INT_MIN";
  EXPECT (mpz_fits_slong_p, 1);
  EXPECT (mpz_fits_sint_p, 1);

  mpz_set_si (z, (long) INT_MIN);
  mpz_sub_ui (z, z, 1L);
  expr = "INT_MIN + 1";
  EXPECT (mpz_fits_sint_p, 0);


  mpz_set_si (z, LONG_MIN);
  expr = "LONG_MIN";
  EXPECT (mpz_fits_slong_p, 1);

  mpz_set_si (z, LONG_MIN);
  mpz_sub_ui (z, z, 1L);
  expr = "LONG_MIN + 1";
  EXPECT (mpz_fits_slong_p, 0);


  if (error)
    abort ();

  mpz_clear (z);
  tests_end ();
  exit (0);
}
