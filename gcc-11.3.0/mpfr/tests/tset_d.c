/* Test file for mpfr_set_d and mpfr_get_d.

Copyright 1999-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "mpfr-test.h"

int
main (int argc, char *argv[])
{
  mpfr_t x, y, z;
  unsigned long k, n;
  volatile double d;
  double dd;

  tests_start_mpfr ();
  mpfr_test_init ();

#ifndef MPFR_DOUBLE_SPEC
  printf ("Warning! The MPFR_DOUBLE_SPEC macro is not defined. This means\n"
          "that you do not have a conforming C implementation and problems\n"
          "may occur with conversions between MPFR numbers and standard\n"
          "floating-point types. Please contact the MPFR team.\n");
#elif MPFR_DOUBLE_SPEC == 0
  /*
  printf ("The type 'double' of your C implementation does not seem to\n"
          "correspond to the IEEE-754 double precision. Though code has\n"
          "been written to support such implementations, tests have been\n"
          "done only on IEEE-754 double-precision implementations and\n"
          "conversions between MPFR numbers and standard floating-point\n"
          "types may be inaccurate. You may wish to contact the MPFR team\n"
          "for further testing.\n");
  */
  printf ("The type 'double' of your C implementation does not seem to\n"
          "correspond to the IEEE-754 double precision. Such particular\n"
          "implementations are not supported yet, and conversions between\n"
          "MPFR numbers and standard floating-point types may be very\n"
          "inaccurate.\n");
  printf ("FLT_RADIX    = %ld\n", (long) FLT_RADIX);
  printf ("DBL_MANT_DIG = %ld\n", (long) DBL_MANT_DIG);
  printf ("DBL_MIN_EXP  = %ld\n", (long) DBL_MIN_EXP);
  printf ("DBL_MAX_EXP  = %ld\n", (long) DBL_MAX_EXP);
#endif

  mpfr_init (x);

#if !defined(MPFR_ERRDIVZERO)
  mpfr_set_nan (x);
  d = mpfr_get_d (x, MPFR_RNDN);
  if (! DOUBLE_ISNAN (d))
    {
      printf ("ERROR for NAN (1)\n");
#ifdef MPFR_NANISNAN
      printf ("The reason is that NAN == NAN. Please look at the configure "
              "output\nand Section \"In case of problem\" of the INSTALL "
              "file.\n");
#endif
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_d (x, d, MPFR_RNDN);
  if (! mpfr_nan_p (x))
    {
      printf ("ERROR for NAN (2)\n");
#ifdef MPFR_NANISNAN
      printf ("The reason is that NAN == NAN. Please look at the configure "
              "output\nand Section \"In case of problem\" of the INSTALL "
              "file.\n");
#endif
      exit (1);
    }
#endif  /* MPFR_ERRDIVZERO */

#ifdef HAVE_SIGNEDZ
  d = 0.0;
  mpfr_set_d (x, d, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS(x));
  d = -d;
  mpfr_set_d (x, d, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) != 0 || MPFR_IS_POS(x))
    {
      printf ("Error in mpfr_set_d on -0\n");
      exit (1);
    }
#endif  /* HAVE_SIGNEDZ */

#if !defined(MPFR_ERRDIVZERO)
  mpfr_set_inf (x, 1);
  d = mpfr_get_d (x, MPFR_RNDN);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_d (x, d, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);

  mpfr_set_inf (x, -1);
  d = mpfr_get_d (x, MPFR_RNDN);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_d (x, d, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) < 0);
#endif  /* MPFR_ERRDIVZERO */

  mpfr_set_prec (x, 2);

  /* checks that subnormals are not flushed to zero */
  d = DBL_MIN; /* 2^(-1022) */
  for (n=0; n<52; n++, d /= 2.0)
    if (d != 0.0) /* should be 2^(-1022-n) */
      {
        mpfr_set_d (x, d, MPFR_RNDN);
        if (mpfr_cmp_ui_2exp (x, 1, -1022-n))
          {
            printf ("Wrong result for d=2^(%ld), ", -1022-n);
            printf ("got ");
            mpfr_out_str (stdout, 10, 10, x, MPFR_RNDN);
            printf ("\n");
            mpfr_print_binary (x);
            puts ("");
            exit (1);
          }
      }

   /* checks that rounds to nearest sets the last
     bit to zero in case of equal distance */
   mpfr_set_d (x, 5.0, MPFR_RNDN);
   if (mpfr_cmp_ui (x, 4))
     {
       printf ("Error in tset_d: expected 4.0, got ");
       mpfr_print_binary (x); putchar('\n');
       exit (1);
     }
   mpfr_set_d (x, -5.0, MPFR_RNDN);
   if (mpfr_cmp_si (x, -4))
     {
       printf ("Error in tset_d: expected -4.0, got ");
       mpfr_print_binary (x); putchar('\n');
       exit (1);
     }

   mpfr_set_d (x, 9.84891017624509146344e-01, MPFR_RNDU);
   if (mpfr_cmp_ui (x, 1))
     {
       printf ("Error in tset_d: expected 1.0, got ");
       mpfr_print_binary (x); putchar('\n');
       exit (1);
     }

  mpfr_init2 (z, 32);
  mpfr_set_d (z, 1.0, (mpfr_rnd_t) 0);
  if (mpfr_cmp_ui (z, 1))
    {
      mpfr_print_binary (z); puts ("");
      printf ("Error: 1.0 != 1.0\n");
      exit (1);
    }
  mpfr_set_prec (x, 53);
  mpfr_init2 (y, 53);
  mpfr_set_d (x, d=-1.08007920352320089721e+150, (mpfr_rnd_t) 0);
  if (mpfr_get_d1 (x) != d)
    {
      mpfr_print_binary (x); puts ("");
      printf ("Error: get_d o set_d <> identity for d = %1.20e %1.20e\n",
              d, mpfr_get_d1 (x));
      exit (1);
    }

  mpfr_set_d (x, 8.06294740693074521573e-310, (mpfr_rnd_t) 0);
  d = -6.72658901114033715233e-165;
  mpfr_set_d (x, d, (mpfr_rnd_t) 0);
  if (d != mpfr_get_d1 (x))
    {
      mpfr_print_binary (x);
      puts ("");
      printf ("Error: get_d o set_d <> identity for d = %1.20e %1.20e\n",
              d, mpfr_get_d1 (x));
      exit (1);
    }

  n = (argc==1) ? 500000 : atoi(argv[1]);
  for (k = 1; k <= n; k++)
    {
      do
        {
          d = DBL_RAND ();
        }
#ifdef HAVE_DENORMS
      while (0);
#else
      while (ABS(d) < DBL_MIN);
#endif
      mpfr_set_d (x, d, (mpfr_rnd_t) 0);
      dd = mpfr_get_d1 (x);
      if (d != dd && !(Isnan(d) && Isnan(dd)))
        {
          printf ("Mismatch on : %1.18g != %1.18g\n", d, mpfr_get_d1 (x));
          mpfr_print_binary (x);
          puts ("");
          exit (1);
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  tests_end_mpfr ();
  return 0;
}
