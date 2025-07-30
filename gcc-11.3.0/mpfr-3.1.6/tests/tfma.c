/* Test file for mpfr_fma.

Copyright 2001-2017 Free Software Foundation, Inc.
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

#include "mpfr-test.h"

/* When a * b is exact, the FMA is equivalent to the separate operations. */
static void
test_exact (void)
{
  const char *val[] =
    { "@NaN@", "-@Inf@", "-2", "-1", "-0", "0", "1", "2", "@Inf@" };
  int sv = sizeof (val) / sizeof (*val);
  int i, j, k;
  int rnd;
  mpfr_t a, b, c, r1, r2;

  mpfr_inits2 (8, a, b, c, r1, r2, (mpfr_ptr) 0);

  for (i = 0; i < sv; i++)
    for (j = 0; j < sv; j++)
      for (k = 0; k < sv; k++)
        RND_LOOP (rnd)
          {
            if (mpfr_set_str (a, val[i], 10, MPFR_RNDN) ||
                mpfr_set_str (b, val[j], 10, MPFR_RNDN) ||
                mpfr_set_str (c, val[k], 10, MPFR_RNDN) ||
                mpfr_mul (r1, a, b, (mpfr_rnd_t) rnd) ||
                mpfr_add (r1, r1, c, (mpfr_rnd_t) rnd))
              {
                printf ("test_exact internal error for (%d,%d,%d,%d)\n",
                        i, j, k, rnd);
                exit (1);
              }
            if (mpfr_fma (r2, a, b, c, (mpfr_rnd_t) rnd))
              {
                printf ("test_exact(%d,%d,%d,%d): mpfr_fma should be exact\n",
                        i, j, k, rnd);
                exit (1);
              }
            if (MPFR_IS_NAN (r1))
              {
                if (MPFR_IS_NAN (r2))
                  continue;
                printf ("test_exact(%d,%d,%d,%d): mpfr_fma should be NaN\n",
                        i, j, k, rnd);
                exit (1);
              }
            if (! mpfr_equal_p (r1, r2) || MPFR_SIGN (r1) != MPFR_SIGN (r2))
              {
                printf ("test_exact(%d,%d,%d,%d):\nexpected ", i, j, k, rnd);
                mpfr_out_str (stdout, 10, 0, r1, MPFR_RNDN);
                printf ("\n     got ");
                mpfr_out_str (stdout, 10, 0, r2, MPFR_RNDN);
                printf ("\n");
                exit (1);
              }
          }

  mpfr_clears (a, b, c, r1, r2, (mpfr_ptr) 0);
}

static void
test_overflow1 (void)
{
  mpfr_t x, y, z, r;
  int inex;

  mpfr_inits2 (8, x, y, z, r, (mpfr_ptr) 0);
  MPFR_SET_POS (x);
  mpfr_setmax (x, mpfr_get_emax ());  /* x = 2^emax - ulp */
  mpfr_set_ui (y, 2, MPFR_RNDN);       /* y = 2 */
  mpfr_neg (z, x, MPFR_RNDN);          /* z = -x = -(2^emax - ulp) */
  mpfr_clear_flags ();
  /* The intermediate multiplication x * y overflows, but x * y + z = x
     is representable. */
  inex = mpfr_fma (r, x, y, z, MPFR_RNDN);
  if (inex || ! mpfr_equal_p (r, x))
    {
      printf ("Error in test_overflow1\nexpected ");
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
      printf (" with inex = 0\n     got ");
      mpfr_out_str (stdout, 2, 0, r, MPFR_RNDN);
      printf (" with inex = %d\n", inex);
      exit (1);
    }
  if (mpfr_overflow_p ())
    {
      printf ("Error in test_overflow1: overflow flag set\n");
      exit (1);
    }
  mpfr_clears (x, y, z, r, (mpfr_ptr) 0);
}

static void
test_overflow2 (void)
{
  mpfr_t x, y, z, r;
  int i, inex, rnd, err = 0;

  mpfr_inits2 (8, x, y, z, r, (mpfr_ptr) 0);

  MPFR_SET_POS (x);
  mpfr_setmin (x, mpfr_get_emax ());  /* x = 0.1@emax */
  mpfr_set_si (y, -2, MPFR_RNDN);      /* y = -2 */
  /* The intermediate multiplication x * y will overflow. */

  for (i = -9; i <= 9; i++)
    RND_LOOP (rnd)
      {
        int inf, overflow;

        inf = rnd == MPFR_RNDN || rnd == MPFR_RNDD || rnd == MPFR_RNDA;
        overflow = inf || i <= 0;

        inex = mpfr_set_si_2exp (z, i, mpfr_get_emin (), MPFR_RNDN);
        MPFR_ASSERTN (inex == 0);

        mpfr_clear_flags ();
        /* One has: x * y = -1@emax exactly (but not representable). */
        inex = mpfr_fma (r, x, y, z, (mpfr_rnd_t) rnd);
        if (overflow ^ (mpfr_overflow_p () != 0))
          {
            printf ("Error in test_overflow2 (i = %d, %s): wrong overflow"
                    " flag (should be %d)\n", i,
                    mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), overflow);
            err = 1;
          }
        if (mpfr_nanflag_p ())
          {
            printf ("Error in test_overflow2 (i = %d, %s): NaN flag should"
                    " not be set\n", i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            err = 1;
          }
        if (mpfr_nan_p (r))
          {
            printf ("Error in test_overflow2 (i = %d, %s): got NaN\n",
                    i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            err = 1;
          }
        else if (MPFR_SIGN (r) >= 0)
          {
            printf ("Error in test_overflow2 (i = %d, %s): wrong sign "
                    "(+ instead of -)\n", i,
                    mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            err = 1;
          }
        else if (inf && ! mpfr_inf_p (r))
          {
            printf ("Error in test_overflow2 (i = %d, %s): expected -Inf,"
                    " got\n", i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            mpfr_dump (r);
            err = 1;
          }
        else if (!inf && (mpfr_inf_p (r) ||
                          (mpfr_nextbelow (r), ! mpfr_inf_p (r))))
          {
            printf ("Error in test_overflow2 (i = %d, %s): expected -MAX,"
                    " got\n", i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            mpfr_dump (r);
            err = 1;
          }
        if (inf ? inex >= 0 : inex <= 0)
          {
            printf ("Error in test_overflow2 (i = %d, %s): wrong inexact"
                    " flag (got %d)\n", i,
                    mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), inex);
            err = 1;
          }

      }

  if (err)
    exit (1);
  mpfr_clears (x, y, z, r, (mpfr_ptr) 0);
}

static void
test_underflow1 (void)
{
  mpfr_t x, y, z, r;
  int inex, signy, signz, rnd, err = 0;

  mpfr_inits2 (8, x, y, z, r, (mpfr_ptr) 0);

  MPFR_SET_POS (x);
  mpfr_setmin (x, mpfr_get_emin ());  /* x = 0.1@emin */

  for (signy = -1; signy <= 1; signy += 2)
    {
      mpfr_set_si_2exp (y, signy, -1, MPFR_RNDN);  /* |y| = 1/2 */
      for (signz = -3; signz <= 3; signz += 2)
        {
          RND_LOOP (rnd)
            {
              mpfr_set_si (z, signz, MPFR_RNDN);
              if (ABS (signz) != 1)
                mpfr_setmax (z, mpfr_get_emax ());
              /* |z| = 1 or 2^emax - ulp */
              mpfr_clear_flags ();
              inex = mpfr_fma (r, x, y, z, (mpfr_rnd_t) rnd);
#define ERRTU1 "Error in test_underflow1 (signy = %d, signz = %d, %s)\n  "
              if (mpfr_nanflag_p ())
                {
                  printf (ERRTU1 "NaN flag is set\n", signy, signz,
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  err = 1;
                }
              if (signy < 0 && MPFR_IS_LIKE_RNDD(rnd, signz))
                mpfr_nextbelow (z);
              if (signy > 0 && MPFR_IS_LIKE_RNDU(rnd, signz))
                mpfr_nextabove (z);
              if ((mpfr_overflow_p () != 0) ^ (mpfr_inf_p (z) != 0))
                {
                  printf (ERRTU1 "wrong overflow flag\n", signy, signz,
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  err = 1;
                }
              if (mpfr_underflow_p ())
                {
                  printf (ERRTU1 "underflow flag is set\n", signy, signz,
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  err = 1;
                }
              if (! mpfr_equal_p (r, z))
                {
                  printf (ERRTU1 "got ", signy, signz,
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  mpfr_print_binary (r);
                  printf (" instead of ");
                  mpfr_print_binary (z);
                  printf ("\n");
                  err = 1;
                }
              if (inex >= 0 && (rnd == MPFR_RNDD ||
                                (rnd == MPFR_RNDZ && signz > 0) ||
                                (rnd == MPFR_RNDN && signy > 0)))
                {
                  printf (ERRTU1 "ternary value = %d instead of < 0\n",
                          signy, signz, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd),
                          inex);
                  err = 1;
                }
              if (inex <= 0 && (rnd == MPFR_RNDU ||
                                (rnd == MPFR_RNDZ && signz < 0) ||
                                (rnd == MPFR_RNDN && signy < 0)))
                {
                  printf (ERRTU1 "ternary value = %d instead of > 0\n",
                          signy, signz, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd),
                          inex);
                  err = 1;
                }
            }
        }
    }

  if (err)
    exit (1);
  mpfr_clears (x, y, z, r, (mpfr_ptr) 0);
}

static void
test_underflow2 (void)
{
  mpfr_t x, y, z, r;
  int b, i, inex, same, err = 0;

  mpfr_inits2 (32, x, y, z, r, (mpfr_ptr) 0);

  mpfr_set_si_2exp (z, 1, mpfr_get_emin (), MPFR_RNDN);   /* z = 2^emin */
  mpfr_set_si_2exp (x, 1, mpfr_get_emin (), MPFR_RNDN);   /* x = 2^emin */

  for (b = 0; b <= 1; b++)
    {
      for (i = 15; i <= 17; i++)
        {
          mpfr_set_si_2exp (y, i, -4 - MPFR_PREC (z), MPFR_RNDN);
          /*  z = 1.000...00b
           * xy =            01111
           *   or            10000
           *   or            10001
           */
          mpfr_clear_flags ();
          inex = mpfr_fma (r, x, y, z, MPFR_RNDN);
#define ERRTU2 "Error in test_underflow2 (b = %d, i = %d)\n  "
          if (__gmpfr_flags != MPFR_FLAGS_INEXACT)
            {
              printf (ERRTU2 "flags = %u instead of %u\n", b, i,
                      __gmpfr_flags, (unsigned int) MPFR_FLAGS_INEXACT);
              err = 1;
            }
          same = i == 15 || (i == 16 && b == 0);
          if (same ? (inex >= 0) : (inex <= 0))
            {
              printf (ERRTU2 "incorrect ternary value (%d instead of %c 0)\n",
                      b, i, inex, same ? '<' : '>');
              err = 1;
            }
          mpfr_set (y, z, MPFR_RNDN);
          if (!same)
            mpfr_nextabove (y);
          if (! mpfr_equal_p (r, y))
            {
              printf (ERRTU2 "expected ", b, i);
              mpfr_dump (y);
              printf ("  got      ");
              mpfr_dump (r);
              err = 1;
            }
        }
      mpfr_nextabove (z);
    }

  if (err)
    exit (1);
  mpfr_clears (x, y, z, r, (mpfr_ptr) 0);
}

static void
test_underflow3 (int n)
{
  mpfr_t x, y, z, t1, t2;
  int sign, k, s, rnd, inex1, inex2;
  mpfr_exp_t e;
  unsigned int flags1, flags2;

  mpfr_inits2 (4, x, z, t1, t2, (mpfr_ptr) 0);
  mpfr_init2 (y, 6);

  e = mpfr_get_emin () - 1;

  for (sign = 1; sign >= -1; sign -= 2)
    for (k = 1; k <= 7; k++)
      for (s = -1; s <= 1; s++)
        {
          mpfr_set_si_2exp (x, sign, e, MPFR_RNDN);
          mpfr_set_si_2exp (y, 8*k+s, -6, MPFR_RNDN);
          mpfr_neg (z, x, MPFR_RNDN);
          /* x = sign * 2^(emin-1)
             y = (8 * k + s) * 2^(-6) = k / 8 + s * 2^(-6)
             z = -x = -sign * 2^(emin-1)
             FMA(x,y,z) = sign * ((k-8) * 2^(emin-4) + s * 2^(emin-7)) exactly.
             Note: The purpose of the s * 2^(emin-7) term is to yield
             double rounding when scaling for k = 4, s != 0, MPFR_RNDN. */

          RND_LOOP (rnd)
            {
              mpfr_clear_flags ();
              inex1 = mpfr_set_si_2exp (t1, sign * (8*k+s-64), e-6,
                                        (mpfr_rnd_t) rnd);
              flags1 = __gmpfr_flags;

              mpfr_clear_flags ();
              inex2 = mpfr_fma (t2, x, y, z, (mpfr_rnd_t) rnd);
              flags2 = __gmpfr_flags;

              if (! (mpfr_equal_p (t1, t2) &&
                     SAME_SIGN (inex1, inex2) &&
                     flags1 == flags2))
                {
                  printf ("Error in test_underflow3, n = %d, sign = %d,"
                          " k = %d, s = %d, %s\n", n, sign, k, s,
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  printf ("Expected ");
                  mpfr_dump (t1);
                  printf ("  with inex ~ %d, flags =", inex1);
                  flags_out (flags1);
                  printf ("Got      ");
                  mpfr_dump (t2);
                  printf ("  with inex ~ %d, flags =", inex2);
                  flags_out (flags2);
                  exit (1);
                }
            }
        }

  mpfr_clears (x, y, z, t1, t2, (mpfr_ptr) 0);
}

static void
bug20101018 (void)
{
  mpfr_t x, y, z, t, u;
  int i;

  mpfr_init2 (x, 64);
  mpfr_init2 (y, 64);
  mpfr_init2 (z, 64);
  mpfr_init2 (t, 64);
  mpfr_init2 (u, 64);

  mpfr_set_str (x, "0xf.fffffffffffffffp-14766", 16, MPFR_RNDN);
  mpfr_set_str (y, "-0xf.fffffffffffffffp+317", 16, MPFR_RNDN);
  mpfr_set_str (z, "0x8.3ffffffffffe3ffp-14443", 16, MPFR_RNDN);
  mpfr_set_str (t, "0x8.7ffffffffffc7ffp-14444", 16, MPFR_RNDN);
  i = mpfr_fma (u, x, y, z, MPFR_RNDN);
  if (! mpfr_equal_p (u, t))
    {
      printf ("Wrong result in bug20101018 (a)\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 16, 0, t, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 16, 0, u, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  if (i <= 0)
    {
      printf ("Wrong ternary value in bug20101018 (a)\n");
      printf ("Expected > 0\n");
      printf ("Got      %d\n", i);
      exit (1);
    }

  mpfr_set_str (x, "-0xf.fffffffffffffffp-11420", 16, MPFR_RNDN);
  mpfr_set_str (y, "0xf.fffffffffffffffp+9863", 16, MPFR_RNDN);
  mpfr_set_str (z, "0x8.fffff80ffffffffp-1551", 16, MPFR_RNDN);
  mpfr_set_str (t, "0x9.fffff01ffffffffp-1552", 16, MPFR_RNDN);
  i = mpfr_fma (u, x, y, z, MPFR_RNDN);
  if (! mpfr_equal_p (u, t))
    {
      printf ("Wrong result in bug20101018 (b)\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 16, 0, t, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 16, 0, u, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  if (i <= 0)
    {
      printf ("Wrong ternary value in bug20101018 (b)\n");
      printf ("Expected > 0\n");
      printf ("Got      %d\n", i);
      exit (1);
    }

  mpfr_set_str (x, "0xf.fffffffffffffffp-2125", 16, MPFR_RNDN);
  mpfr_set_str (y, "-0xf.fffffffffffffffp-6000", 16, MPFR_RNDN);
  mpfr_set_str (z, "0x8p-8119", 16, MPFR_RNDN);
  mpfr_set_str (t, "0x8.000000000000001p-8120", 16, MPFR_RNDN);
  i = mpfr_fma (u, x, y, z, MPFR_RNDN);
  if (! mpfr_equal_p (u, t))
    {
      printf ("Wrong result in bug20101018 (c)\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 16, 0, t, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 16, 0, u, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  if (i <= 0)
    {
      printf ("Wrong ternary value in bug20101018 (c)\n");
      printf ("Expected > 0\n");
      printf ("Got      %d\n", i);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);
  mpfr_clear (u);
}

int
main (int argc, char *argv[])
{
  mpfr_t x, y, z, s;
  mpfr_exp_t emin, emax;

  tests_start_mpfr ();

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  bug20101018 ();

  mpfr_init (x);
  mpfr_init (s);
  mpfr_init (y);
  mpfr_init (z);

  /* check special cases */
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 2);
  mpfr_set_prec (z, 2);
  mpfr_set_prec (s, 2);
  mpfr_set_str (x, "-0.75", 10, MPFR_RNDN);
  mpfr_set_str (y, "0.5", 10, MPFR_RNDN);
  mpfr_set_str (z, "0.375", 10, MPFR_RNDN);
  mpfr_fma (s, x, y, z, MPFR_RNDU); /* result is 0 */
  if (mpfr_cmp_ui(s, 0))
    {
      printf("Error: -0.75 * 0.5 + 0.375 should be equal to 0 for prec=2\n");
      exit(1);
    }

  mpfr_set_prec (x, 27);
  mpfr_set_prec (y, 27);
  mpfr_set_prec (z, 27);
  mpfr_set_prec (s, 27);
  mpfr_set_str_binary (x, "1.11111111111111111111111111e-1");
  mpfr_set (y, x, MPFR_RNDN);
  mpfr_set_str_binary (z, "-1.00011110100011001011001001e-1");
  if (mpfr_fma (s, x, y, z, MPFR_RNDN) >= 0)
    {
      printf ("Wrong inexact flag for x=y=1-2^(-27)\n");
      exit (1);
    }

  mpfr_set_nan (x);
  mpfr_urandomb (y, RANDS);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_nan_p (s))
    {
      printf ("evaluation of function in x=NAN does not return NAN");
      exit (1);
    }

  mpfr_set_nan (y);
  mpfr_urandomb (x, RANDS);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_nan_p(s))
    {
      printf ("evaluation of function in y=NAN does not return NAN");
      exit (1);
    }

  mpfr_set_nan (z);
  mpfr_urandomb (y, RANDS);
  mpfr_urandomb (x, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_nan_p (s))
    {
      printf ("evaluation of function in z=NAN does not return NAN");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, 1);
  mpfr_set_inf (z, 1);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_inf_p (s) || mpfr_sgn (s) < 0)
    {
      printf ("Error for (+inf) * (+inf) + (+inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_set_inf (y, -1);
  mpfr_set_inf (z, 1);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_inf_p (s) || mpfr_sgn (s) < 0)
    {
      printf ("Error for (-inf) * (-inf) + (+inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, -1);
  mpfr_set_inf (z, -1);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_inf_p (s) || mpfr_sgn (s) > 0)
    {
      printf ("Error for (+inf) * (-inf) + (-inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_set_inf (y, 1);
  mpfr_set_inf (z, -1);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_inf_p (s) || mpfr_sgn (s) > 0)
    {
      printf ("Error for (-inf) * (+inf) + (-inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_nan_p (s))
    {
      printf ("evaluation of function in x=INF y=0  does not return NAN");
      exit (1);
    }

  mpfr_set_inf (y, 1);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_nan_p (s))
    {
      printf ("evaluation of function in x=0 y=INF does not return NAN");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_urandomb (y, RANDS); /* always positive */
  mpfr_set_inf (z, -1);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_nan_p (s))
    {
      printf ("evaluation of function in x=INF y>0 z=-INF does not return NAN");
      exit (1);
    }

  mpfr_set_inf (y, 1);
  mpfr_urandomb (x, RANDS);
  mpfr_set_inf (z, -1);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_nan_p (s))
    {
      printf ("evaluation of function in x>0 y=INF z=-INF does not return NAN");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_urandomb (y, RANDS);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_inf_p (s) || mpfr_sgn (s) < 0)
    {
      printf ("evaluation of function in x=INF does not return INF");
      exit (1);
    }

  mpfr_set_inf (y, 1);
  mpfr_urandomb (x, RANDS);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_inf_p (s) || mpfr_sgn (s) < 0)
    {
      printf ("evaluation of function in y=INF does not return INF");
      exit (1);
    }

  mpfr_set_inf (z, 1);
  mpfr_urandomb (x, RANDS);
  mpfr_urandomb (y, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (!mpfr_inf_p (s) || mpfr_sgn (s) < 0)
    {
      printf ("evaluation of function in z=INF does not return INF");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_urandomb (y, RANDS);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (! mpfr_equal_p (s, z))
    {
      printf ("evaluation of function in x=0 does not return z\n");
      exit (1);
    }

  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_urandomb (x, RANDS);
  mpfr_urandomb (z, RANDS);
  mpfr_fma (s, x, y, z, MPFR_RNDN);
  if (! mpfr_equal_p (s, z))
    {
      printf ("evaluation of function in y=0 does not return z\n");
      exit (1);
    }

  {
    mpfr_prec_t prec;
    mpfr_t t, slong;
    mpfr_rnd_t rnd;
    int inexact, compare;
    unsigned int n;

    mpfr_prec_t p0 = 2, p1 = 200;
    unsigned int N = 200;

    mpfr_init (t);
    mpfr_init (slong);

    /* generic test */
    for (prec = p0; prec <= p1; prec++)
      {
        mpfr_set_prec (x, prec);
        mpfr_set_prec (y, prec);
        mpfr_set_prec (z, prec);
        mpfr_set_prec (s, prec);
        mpfr_set_prec (t, prec);

        for (n = 0; n < N; n++)
          {
            mpfr_urandomb (x, RANDS);
            mpfr_urandomb (y, RANDS);
            mpfr_urandomb (z, RANDS);

            if (randlimb () % 2)
              mpfr_neg (x, x, MPFR_RNDN);
            if (randlimb () % 2)
              mpfr_neg (y, y, MPFR_RNDN);
            if (randlimb () % 2)
              mpfr_neg (z, z, MPFR_RNDN);

            rnd = RND_RAND ();
            mpfr_set_prec (slong, 2 * prec);
            if (mpfr_mul (slong, x, y, rnd))
              {
                printf ("x*y should be exact\n");
                exit (1);
              }
            compare = mpfr_add (t, slong, z, rnd);
            inexact = mpfr_fma (s, x, y, z, rnd);
            if (! mpfr_equal_p (s, t))
              {
                printf ("results differ for x=");
                mpfr_out_str (stdout, 2, prec, x, MPFR_RNDN);
                printf ("  y=");
                mpfr_out_str (stdout, 2, prec, y, MPFR_RNDN);
                printf ("  z=");
                mpfr_out_str (stdout, 2, prec, z, MPFR_RNDN);
                printf (" prec=%u rnd_mode=%s\n", (unsigned int) prec,
                        mpfr_print_rnd_mode (rnd));
                printf ("got      ");
                mpfr_out_str (stdout, 2, prec, s, MPFR_RNDN);
                puts ("");
                printf ("expected ");
                mpfr_out_str (stdout, 2, prec, t, MPFR_RNDN);
                puts ("");
                printf ("approx  ");
                mpfr_print_binary (slong);
                puts ("");
                exit (1);
              }
            if (! SAME_SIGN (inexact, compare))
              {
                printf ("Wrong inexact flag for rnd=%s: expected %d, got %d\n",
                        mpfr_print_rnd_mode (rnd), compare, inexact);
                printf (" x="); mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
                printf (" y="); mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
                printf (" z="); mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
                printf (" s="); mpfr_out_str (stdout, 2, 0, s, MPFR_RNDN);
                printf ("\n");
                exit (1);
              }
          }
      }
    mpfr_clear (t);
    mpfr_clear (slong);

  }
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (s);

  test_exact ();

  test_overflow1 ();
  test_overflow2 ();
  test_underflow1 ();
  test_underflow2 ();
  test_underflow3 (1);

  set_emin (MPFR_EMIN_MIN);
  set_emax (MPFR_EMAX_MAX);
  test_overflow1 ();
  test_overflow2 ();
  test_underflow1 ();
  test_underflow2 ();
  test_underflow3 (2);
  set_emin (emin);
  set_emax (emax);

  tests_end_mpfr ();
  return 0;
}
