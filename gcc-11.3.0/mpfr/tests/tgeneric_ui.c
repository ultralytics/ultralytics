/* Generic test file for functions with one mpfr_t argument and an integer.

Copyright 2005-2017 Free Software Foundation, Inc.
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

/* define INTEGER_TYPE to what we want */
#ifndef INTEGER_TYPE
# define INTEGER_TYPE mp_limb_t
#endif
#ifndef RAND_FUNCTION
# define RAND_FUNCTION(x) mpfr_urandomb ((x), RANDS)
#endif
#ifndef INT_RAND_FUNCTION
# define INT_RAND_FUNCTION() (INTEGER_TYPE) randlimb ()
#endif

static void
test_generic_ui (mpfr_prec_t p0, mpfr_prec_t p1, unsigned int N)
{
  mpfr_prec_t prec, yprec;
  mpfr_t x, y, z, t;
  INTEGER_TYPE u;
  mpfr_rnd_t rnd;
  int inexact, compare, compare2;
  unsigned int n;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);
  mpfr_init (t);

  /* generic test */
  for (prec = p0; prec <= p1; prec++)
    {
      mpfr_set_prec (x, prec);
      mpfr_set_prec (z, prec);
      mpfr_set_prec (t, prec);
      yprec = prec + 10;

      for (n = 0; n <= N; n++)
        {
          if (n > 1 || prec < p1)
            RAND_FUNCTION (x);
          else
            {
              /* Special cases tested in precision p1 if n <= 1. */
              mpfr_set_si (x, n == 0 ? 1 : -1, MPFR_RNDN);
              mpfr_set_exp (x, mpfr_get_emin ());
            }
          if (n < 2 || n > 3 || prec < p1)
            u = INT_RAND_FUNCTION ();
          else
            {
              /* Special cases tested in precision p1 if n = 2 or 3. */
              if ((INTEGER_TYPE) -1 < 0)  /* signed, type long assumed */
                u = n == 2 ? LONG_MIN : LONG_MAX;
              else  /* unsigned */
                u = n == 2 ? 0 : -1;
            }
          rnd = RND_RAND ();
          mpfr_set_prec (y, yprec);
          compare = TEST_FUNCTION (y, x, u, rnd);
          if (mpfr_can_round (y, yprec, rnd, rnd, prec))
            {
              mpfr_set (t, y, rnd);
              inexact = TEST_FUNCTION (z, x, u, rnd);
              if (mpfr_cmp (t, z))
                {
                  printf ("results differ for x=");
                  mpfr_out_str (stdout, 2, prec, x, MPFR_RNDN);
                  printf ("\nu=%lu", (unsigned long) u);
                  printf (" prec=%lu rnd_mode=%s\n",
                          (unsigned long ) prec, mpfr_print_rnd_mode (rnd));
#ifdef TEST_FUNCTION_NAME
                  printf ("Function: %s\n", TEST_FUNCTION_NAME);
#endif
                  printf ("got      ");
                  mpfr_out_str (stdout, 2, prec, z, MPFR_RNDN);
                  puts ("");
                  printf ("expected ");
                  mpfr_out_str (stdout, 2, prec, t, MPFR_RNDN);
                  puts ("");
                  printf ("approx  ");
                  mpfr_print_binary (y);
                  puts ("");
                  exit (1);
                }
              compare2 = mpfr_cmp (t, y);
              /* if rounding to nearest, cannot know the sign of t - f(x)
                 because of composed rounding: y = o(f(x)) and t = o(y) */
              if (compare * compare2 >= 0)
                compare = compare + compare2;
              else
                compare = inexact; /* cannot determine sign(t-f(x)) */
              if (! SAME_SIGN (inexact, compare))
                {
                  printf ("Wrong inexact flag for rnd=%s: expected %d, got %d"
                          "\n", mpfr_print_rnd_mode (rnd), compare, inexact);
                  printf ("x = "); mpfr_dump (x);
                  printf ("u = %lu\n", (unsigned long) u);
                  printf ("y = "); mpfr_dump (y);
                  printf ("t = "); mpfr_dump (t);
                  exit (1);
                }
            }
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);
}

#undef RAND_FUNCTION
#undef INTEGER_TYPE
#undef TEST_FUNCTION
#undef TEST_FUNCTION_NAME
#undef test_generic_ui
#undef INT_RAND_FUNCTION
