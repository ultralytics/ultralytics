/* Test mpf_fits_*_p

Copyright 2001, 2002, 2013 Free Software Foundation, Inc.

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


/* Nothing sophisticated here, just exercise mpf_fits_*_p on a small amount
   of data. */

#define EXPECT_S(fun,name,answer)                                        \
  got = fun (f);                                                         \
  if (got != answer)                                                     \
    {                                                                    \
      printf ("%s (%s) got %d want %d\n", name, expr, got, answer);      \
      printf (" f size %d exp %ld\n", SIZ(f), EXP(f));                   \
      printf (" f dec "); mpf_out_str (stdout, 10, 0, f); printf ("\n"); \
      printf (" f hex "); mpf_out_str (stdout, 16, 0, f); printf ("\n"); \
      error = 1;                                                         \
    }

#define EXPECT(fun,answer)  EXPECT_S(fun,#fun,answer)

int
main (void)
{
  mpf_t       f, f0p5;
  int         got;
  const char  *expr;
  int         error = 0;

  tests_start ();
  mpf_init2 (f, 200L);
  mpf_init2 (f0p5, 200L);

  /* 0.5 */
  mpf_set_ui (f0p5, 1L);
  mpf_div_2exp (f0p5, f0p5, 1L);

  mpf_set_ui (f, 0L);
  expr = "0";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);
  EXPECT (mpf_fits_ushort_p, 1);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_ui (f, 1L);
  expr = "1";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);
  EXPECT (mpf_fits_ushort_p, 1);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_si (f, -1L);
  expr = "-1";
  EXPECT (mpf_fits_ulong_p, 0);
  EXPECT (mpf_fits_uint_p, 0);
  EXPECT (mpf_fits_ushort_p, 0);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);


  mpf_set_ui (f, (unsigned long) USHRT_MAX);
  expr = "USHRT_MAX";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);
  EXPECT (mpf_fits_ushort_p, 1);

  mpf_set_ui (f, (unsigned long) USHRT_MAX);
  mpf_add (f, f, f0p5);
  expr = "USHRT_MAX + 0.5";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);
  EXPECT (mpf_fits_ushort_p, 1);

  mpf_set_ui (f, (unsigned long) USHRT_MAX);
  mpf_add_ui (f, f, 1L);
  expr = "USHRT_MAX + 1";
  EXPECT (mpf_fits_ushort_p, 0);


  mpf_set_ui (f, (unsigned long) UINT_MAX);
  expr = "UINT_MAX";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);

  mpf_set_ui (f, (unsigned long) UINT_MAX);
  mpf_add (f, f, f0p5);
  expr = "UINT_MAX + 0.5";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);

  mpf_set_ui (f, (unsigned long) UINT_MAX);
  mpf_add_ui (f, f, 1L);
  expr = "UINT_MAX + 1";
  EXPECT (mpf_fits_uint_p, 0);


  mpf_set_ui (f, ULONG_MAX);
  expr = "ULONG_MAX";
  EXPECT (mpf_fits_ulong_p, 1);

  mpf_set_ui (f, ULONG_MAX);
  mpf_add (f, f, f0p5);
  expr = "ULONG_MAX + 0.5";
  EXPECT (mpf_fits_ulong_p, 1);

  mpf_set_ui (f, ULONG_MAX);
  mpf_add_ui (f, f, 1L);
  expr = "ULONG_MAX + 1";
  EXPECT (mpf_fits_ulong_p, 0);


  mpf_set_si (f, (long) SHRT_MAX);
  expr = "SHRT_MAX";
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_si (f, (long) SHRT_MAX);
  expr = "SHRT_MAX + 0.5";
  mpf_add (f, f, f0p5);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_si (f, (long) SHRT_MAX);
  mpf_add_ui (f, f, 1L);
  expr = "SHRT_MAX + 1";
  EXPECT (mpf_fits_sshort_p, 0);


  mpf_set_si (f, (long) INT_MAX);
  expr = "INT_MAX";
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);

  mpf_set_si (f, (long) INT_MAX);
  mpf_add (f, f, f0p5);
  expr = "INT_MAX + 0.5";
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);

  mpf_set_si (f, (long) INT_MAX);
  mpf_add_ui (f, f, 1L);
  expr = "INT_MAX + 1";
  EXPECT (mpf_fits_sint_p, 0);


  mpf_set_si (f, LONG_MAX);
  expr = "LONG_MAX";
  EXPECT (mpf_fits_slong_p, 1);

  mpf_set_si (f, LONG_MAX);
  mpf_add (f, f, f0p5);
  expr = "LONG_MAX + 0.5";
  EXPECT (mpf_fits_slong_p, 1);

  mpf_set_si (f, LONG_MAX);
  mpf_add_ui (f, f, 1L);
  expr = "LONG_MAX + 1";
  EXPECT (mpf_fits_slong_p, 0);


  mpf_set_si (f, (long) SHRT_MIN);
  expr = "SHRT_MIN";
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_si (f, (long) SHRT_MIN);
  mpf_sub (f, f, f0p5);
  expr = "SHRT_MIN - 0.5";
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_si (f, (long) SHRT_MIN);
  mpf_sub_ui (f, f, 1L);
  expr = "SHRT_MIN - 1";
  EXPECT (mpf_fits_sshort_p, 0);


  mpf_set_si (f, (long) INT_MIN);
  expr = "INT_MIN";
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);

  mpf_set_si (f, (long) INT_MIN);
  mpf_sub (f, f, f0p5);
  expr = "INT_MIN - 0.5";
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);

  mpf_set_si (f, (long) INT_MIN);
  mpf_sub_ui (f, f, 1L);
  expr = "INT_MIN - 1";
  EXPECT (mpf_fits_sint_p, 0);


  mpf_set_si (f, LONG_MIN);
  expr = "LONG_MIN";
  EXPECT (mpf_fits_slong_p, 1);

  mpf_set_si (f, LONG_MIN);
  mpf_sub (f, f, f0p5);
  expr = "LONG_MIN - 0.5";
  EXPECT (mpf_fits_slong_p, 1);

  mpf_set_si (f, LONG_MIN);
  mpf_sub_ui (f, f, 1L);
  expr = "LONG_MIN - 1";
  EXPECT (mpf_fits_slong_p, 0);


  mpf_set_str_or_abort (f, "0.5", 10);
  expr = "0.5";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);
  EXPECT (mpf_fits_ushort_p, 1);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_str_or_abort (f, "-0.5", 10);
  expr = "-0.5";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);
  EXPECT (mpf_fits_ushort_p, 1);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_str_or_abort (f, "-1.5", 10);
  expr = "-1.5";
  EXPECT (mpf_fits_ulong_p, 0);
  EXPECT (mpf_fits_uint_p, 0);
  EXPECT (mpf_fits_ushort_p, 0);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);


  mpf_set_str_or_abort (f, "1.000000000000000000000000000000000001", 16);
  expr = "1.000000000000000000000000000000000001 base 16";
  EXPECT (mpf_fits_ulong_p, 1);
  EXPECT (mpf_fits_uint_p, 1);
  EXPECT (mpf_fits_ushort_p, 1);
  EXPECT (mpf_fits_slong_p, 1);
  EXPECT (mpf_fits_sint_p, 1);
  EXPECT (mpf_fits_sshort_p, 1);

  mpf_set_str_or_abort (f, "1@1000", 16);
  expr = "1@1000 base 16";
  EXPECT (mpf_fits_ulong_p, 0);
  EXPECT (mpf_fits_uint_p, 0);
  EXPECT (mpf_fits_ushort_p, 0);
  EXPECT (mpf_fits_slong_p, 0);
  EXPECT (mpf_fits_sint_p, 0);
  EXPECT (mpf_fits_sshort_p, 0);


  mpf_set_ui (f, 1L);
  mpf_mul_2exp (f, f, BITS_PER_ULONG + 1);
  mpf_sub_ui (f, f, 1L);
  expr = "2^(BITS_PER_ULONG+1) - 1";
  EXPECT (mpf_fits_ulong_p, 0);
  EXPECT (mpf_fits_uint_p, 0);
  EXPECT (mpf_fits_ushort_p, 0);
  EXPECT (mpf_fits_slong_p, 0);
  EXPECT (mpf_fits_sint_p, 0);
  EXPECT (mpf_fits_sshort_p, 0);

  mpf_set_ui (f, 1L);
  mpf_mul_2exp (f, f, BITS_PER_ULONG + 1);
  mpf_ui_sub (f, 1L, f);
  expr = "- (2^(BITS_PER_ULONG+1) - 1)";
  EXPECT (mpf_fits_ulong_p, 0);
  EXPECT (mpf_fits_uint_p, 0);
  EXPECT (mpf_fits_ushort_p, 0);
  EXPECT (mpf_fits_slong_p, 0);
  EXPECT (mpf_fits_sint_p, 0);
  EXPECT (mpf_fits_sshort_p, 0);

  mpf_set_ui (f, 1L);
  mpf_mul_2exp (f, f, BITS_PER_ULONG + 5);
  mpf_sub_ui (f, f, 1L);
  expr = "2^(BITS_PER_ULONG+5) - 1";
  EXPECT (mpf_fits_ulong_p, 0);
  EXPECT (mpf_fits_uint_p, 0);
  EXPECT (mpf_fits_ushort_p, 0);
  EXPECT (mpf_fits_slong_p, 0);
  EXPECT (mpf_fits_sint_p, 0);
  EXPECT (mpf_fits_sshort_p, 0);


  if (error)
    abort ();

  mpf_clear (f);
  mpf_clear (f0p5);
  tests_end ();
  exit (0);
}
