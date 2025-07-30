/* tprintf.c -- test file for mpfr_printf and mpfr_vprintf

Copyright 2008-2017 Free Software Foundation, Inc.
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

#if HAVE_STDARG
#include <stdarg.h>

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <errno.h>

#include "mpfr-intmax.h"
#include "mpfr-test.h"
#define STDOUT_FILENO 1

#if MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0)

#define QUOTE(X) NAME(X)
#define NAME(X) #X

/* unlike other tests, we print out errors to stderr because stdout might be
   redirected */
#define check_length(num_test, var, value, var_spec)                    \
  if ((var) != (value))                                                 \
    {                                                                   \
      fprintf (stderr, "Error in test #%d: mpfr_printf printed %"       \
               QUOTE(var_spec)" characters instead of %d\n",            \
               (num_test), (var), (value));                             \
      exit (1);                                                         \
    }

#define check_length_with_cmp(num_test, var, value, cmp, var_spec)      \
  if (cmp != 0)                                                         \
    {                                                                   \
      mpfr_fprintf (stderr, "Error in test #%d, mpfr_printf printed %"  \
                    QUOTE(var_spec)" characters instead of %d\n",       \
                    (num_test), (var), (value));                        \
      exit (1);                                                         \
    }

/* limit for random precision in random() */
const int prec_max_printf = 5000;
/* boolean: is stdout redirected to a file ? */
int stdout_redirect;

static void
check (const char *fmt, mpfr_t x)
{
  if (mpfr_printf (fmt, x) == -1)
    {
      fprintf (stderr, "Error in mpfr_printf(\"%s\", ...)\n", fmt);

      exit (1);
    }
  putchar ('\n');
}

static void
check_vprintf (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  if (mpfr_vprintf (fmt, ap) == -1)
    {
      fprintf (stderr, "Error in mpfr_vprintf(\"%s\", ...)\n", fmt);

      va_end (ap);
      exit (1);
    }
  putchar ('\n');
  va_end (ap);
}

static void
check_vprintf_failure (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  if (mpfr_vprintf (fmt, ap) != -1)
   {
      putchar ('\n');
      fprintf (stderr, "Error in mpfr_vprintf(\"%s\", ...)\n", fmt);

      va_end (ap);
      exit (1);
    }
  putchar ('\n');
  va_end (ap);
}

static void
check_vprintf_overflow (const char *fmt, ...)
{
  va_list ap;
  int r, e;

  va_start (ap, fmt);
  errno = 0;
  r = mpfr_vprintf (fmt, ap);
  e = errno;
  va_end (ap);

  if (r != -1
#ifdef EOVERFLOW
      || e != EOVERFLOW
#endif
      )
    {
      putchar ('\n');
      fprintf (stderr, "Error in mpfr_vprintf(\"%s\", ...)\n"
               "Got r = %d, errno = %d\n", fmt, r, e);
      exit (1);
    }

  putchar ('\n');
}

static void
check_invalid_format (void)
{
  int i = 0;

  /* format in disorder */
  check_vprintf_failure ("blah %l2.1d blah", i);
  check_vprintf_failure ("blah %2.1#d blah", i);

  /* incomplete format */
  check_vprintf_failure ("%", i);
  check_vprintf_failure ("% (missing conversion specifier)", i);
  check_vprintf_failure ("missing conversion specifier %h", i);
  check_vprintf_failure ("this should fail %.l because of missing conversion specifier "
                         "(or doubling %%)", i);
  check_vprintf_failure ("%L", i);
  check_vprintf_failure ("%hh. ", i);
  check_vprintf_failure ("blah %j.");
  check_vprintf_failure ("%ll blah");
  check_vprintf_failure ("blah%t blah");
  check_vprintf_failure ("%z ");
  check_vprintf_failure ("%F (missing conversion specifier)");
  check_vprintf_failure ("%Q (missing conversion specifier)");
  check_vprintf_failure ("%M (missing conversion specifier)");
  check_vprintf_failure ("%N (missing conversion specifier)");
  check_vprintf_failure ("%Z (missing conversion specifier)");
  check_vprintf_failure ("%R (missing conversion specifier)");
  check_vprintf_failure ("%R");
  check_vprintf_failure ("%P (missing conversion specifier)");

  /* conversion specifier with wrong length specifier */
  check_vprintf_failure ("%ha", i);
  check_vprintf_failure ("%hhe", i);
  check_vprintf_failure ("%jf", i);
  check_vprintf_failure ("%lg", i);
  check_vprintf_failure ("%tA", i);
  check_vprintf_failure ("%zE", i);
  check_vprintf_failure ("%Ld", i);
  check_vprintf_failure ("%Qf", i);
  check_vprintf_failure ("%MG", i);
  check_vprintf_failure ("%Na", i);
  check_vprintf_failure ("%ZE", i);
  check_vprintf_failure ("%PG", i);
  check_vprintf_failure ("%Fu", i);
  check_vprintf_failure ("%Rx", i);
}

static void
check_long_string (void)
{
  /* this test is VERY expensive both in time (~1 min on core2 @ 2.40GHz) and
     in memory (~2.5 GB) */
  mpfr_t x;

  mpfr_init2 (x, INT_MAX);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_nextabove (x);

  check_vprintf_overflow ("%Rb", x);
  check_vprintf_overflow ("%RA %RA %Ra %Ra", x, x, x, x);

  mpfr_clear (x);
}

static void
check_special (void)
{
  mpfr_t x;

  mpfr_init (x);

  mpfr_set_inf (x, 1);
  check ("%Ra", x);
  check ("%Rb", x);
  check ("%Re", x);
  check ("%Rf", x);
  check ("%Rg", x);
  check_vprintf ("%Ra", x);
  check_vprintf ("%Rb", x);
  check_vprintf ("%Re", x);
  check_vprintf ("%Rf", x);
  check_vprintf ("%Rg", x);

  mpfr_set_inf (x, -1);
  check ("%Ra", x);
  check ("%Rb", x);
  check ("%Re", x);
  check ("%Rf", x);
  check ("%Rg", x);
  check_vprintf ("%Ra", x);
  check_vprintf ("%Rb", x);
  check_vprintf ("%Re", x);
  check_vprintf ("%Rf", x);
  check_vprintf ("%Rg", x);

  mpfr_set_nan (x);
  check ("%Ra", x);
  check ("%Rb", x);
  check ("%Re", x);
  check ("%Rf", x);
  check ("%Rg", x);
  check_vprintf ("%Ra", x);
  check_vprintf ("%Rb", x);
  check_vprintf ("%Re", x);
  check_vprintf ("%Rf", x);
  check_vprintf ("%Rg", x);

  mpfr_clear (x);
}

static void
check_mixed (void)
{
  int ch = 'a';
#ifndef NPRINTF_HH
  signed char sch = -1;
  unsigned char uch = 1;
#endif
  short sh = -1;
  unsigned short ush = 1;
  int i = -1;
  int j = 1;
  unsigned int ui = 1;
  long lo = -1;
  unsigned long ulo = 1;
  float f = -1.25;
  double d = -1.25;
#if !defined(NPRINTF_T) || !defined(NPRINTF_L)
  long double ld = -1.25;
#endif

#ifndef NPRINTF_T
  ptrdiff_t p = 1, saved_p;
#endif
  size_t sz = 1;

  mpz_t mpz;
  mpq_t mpq;
  mpf_t mpf;
  mpfr_rnd_t rnd = MPFR_RNDN;

  mpfr_t mpfr;
  mpfr_prec_t prec;

  mpz_init (mpz);
  mpz_set_ui (mpz, ulo);
  mpq_init (mpq);
  mpq_set_si (mpq, lo, ulo);
  mpf_init (mpf);
  mpf_set_q (mpf, mpq);
  mpfr_init (mpfr);
  mpfr_set_f (mpfr, mpf, MPFR_RNDN);
  prec = mpfr_get_prec (mpfr);

  check_vprintf ("a. %Ra, b. %u, c. %lx%n", mpfr, ui, ulo, &j);
  check_length (1, j, 22, d);
  check_vprintf ("a. %c, b. %Rb, c. %u, d. %li%ln", i, mpfr, i, lo, &ulo);
  check_length (2, ulo, 36, lu);
  check_vprintf ("a. %hi, b. %*f, c. %Re%hn", ush, 3, f, mpfr, &ush);
  check_length (3, ush, 29, hu);
  check_vprintf ("a. %hi, b. %f, c. %#.2Rf%n", sh, d, mpfr, &i);
  check_length (4, i, 29, d);
  check_vprintf ("a. %R*A, b. %Fe, c. %i%zn", rnd, mpfr, mpf, sz, &sz);
  check_length (5, (unsigned long) sz, 34, lu); /* no format specifier '%zu' in C89 */
  check_vprintf ("a. %Pu, b. %c, c. %RUG, d. %Zi%Zn", prec, ch, mpfr, mpz, &mpz);
  check_length_with_cmp (6, mpz, 24, mpz_cmp_ui (mpz, 24), Zi);
  check_vprintf ("%% a. %#.0RNg, b. %Qx%Rn c. %p",
                 mpfr, mpq, &mpfr, (void *) &i);
  check_length_with_cmp (7, mpfr, 15, mpfr_cmp_ui (mpfr, 15), Rg);

#ifndef NPRINTF_T
  saved_p = p;
  check_vprintf ("%% a. %RNg, b. %Qx, c. %td%tn", mpfr, mpq, p, &p);
  if (p != 20)
    mpfr_fprintf (stderr, "Error in test 8, got '%% a. %RNg, b. %Qx, c. %td'\n", mpfr, mpq, saved_p);
  check_length (8, (long) p, 20, ld); /* no format specifier '%td' in C89 */
#endif

#ifndef NPRINTF_L
  check_vprintf ("a. %RA, b. %Lf, c. %QX%zn", mpfr, ld, mpq, &sz);
  check_length (9, (unsigned long) sz, 30, lu); /* no format specifier '%zu' in C89 */
#endif

#ifndef NPRINTF_HH
  check_vprintf ("a. %hhi, b. %Ra, c. %hhu%hhn", sch, mpfr, uch, &uch);
  check_length (10, (unsigned int) uch, 22, u); /* no format specifier '%hhu' in C89 */
#endif

#if defined(HAVE_LONG_LONG) && !defined(NPRINTF_LL)
  {
    long long llo = -1;
    unsigned long long ullo = 1;

    check_vprintf ("a. %Re, b. %llx%Qn", mpfr, ullo, &mpq);
    check_length_with_cmp (11, mpq, 16, mpq_cmp_ui (mpq, 16, 1), Qu);
    check_vprintf ("a. %lli, b. %Rf%lln", llo, mpfr, &ullo);
    check_length (12, ullo, 19, llu);
  }
#endif

#if defined(_MPFR_H_HAVE_INTMAX_T) && !defined(NPRINTF_J)
  {
    intmax_t im = -1;
    uintmax_t uim = 1;

    check_vprintf ("a. %*RA, b. %ji%Fn", 10, mpfr, im, &mpf);
    check_length_with_cmp (31, mpf, 20, mpf_cmp_ui (mpf, 20), Fg);
    check_vprintf ("a. %.*Re, b. %jx%jn", 10, mpfr, uim, &im);
    check_length (32, (long) im, 25, li); /* no format specifier "%ji" in C89 */
  }
#endif

  mpfr_clear (mpfr);
  mpf_clear (mpf);
  mpq_clear (mpq);
  mpz_clear (mpz);
}

static void
check_random (int nb_tests)
{
  int i;
  mpfr_t x;
  mpfr_rnd_t rnd;
  char flag[] =
    {
      '-',
      '+',
      ' ',
      '#',
      '0', /* no ambiguity: first zeros are flag zero*/
      '\''
    };
  char specifier[] =
    {
      'a',
      'b',
      'e',
      'f',
      'g'
    };
  mpfr_exp_t old_emin, old_emax;

  old_emin = mpfr_get_emin ();
  old_emax = mpfr_get_emax ();

  mpfr_init (x);

  for (i = 0; i < nb_tests; ++i)
    {
      int ret;
      int j, jmax;
      int spec, prec;
#define FMT_SIZE 13
      char fmt[FMT_SIZE]; /* at most something like "%-+ #0'.*R*f" */
      char *ptr = fmt;

      tests_default_random (x, 256, MPFR_EMIN_MIN, MPFR_EMAX_MAX, 0);
      rnd = (mpfr_rnd_t) RND_RAND ();

      spec = (int) (randlimb () % 5);
      jmax = (spec == 3 || spec == 4) ? 6 : 5; /* ' flag only with %f or %g */
      /* advantage small precision */
      prec = (randlimb () % 2) ? 10 : prec_max_printf;
      prec = (int) (randlimb () % prec);
      if (spec == 3
          && (mpfr_get_exp (x) > prec_max_printf
              || mpfr_get_exp (x) < -prec_max_printf))
        /*  change style 'f' to style 'e' when number x is very large or very
            small*/
        --spec;

      *ptr++ = '%';
      for (j = 0; j < jmax; j++)
        {
          if (randlimb () % 3 == 0)
            *ptr++ = flag[j];
        }
      *ptr++ = '.';
      *ptr++ = '*';
      *ptr++ = 'R';
      *ptr++ = '*';
      *ptr++ = specifier[spec];
      *ptr = '\0';
      MPFR_ASSERTD (ptr - fmt < FMT_SIZE);

      mpfr_printf ("mpfr_printf(\"%s\", %d, %s, %Re)\n", fmt, prec,
                   mpfr_print_rnd_mode (rnd), x);
      ret = mpfr_printf (fmt, prec, rnd, x);
      if (ret == -1)
        {
          if (spec == 3
              && (MPFR_GET_EXP (x) > INT_MAX || MPFR_GET_EXP (x) < -INT_MAX))
            /* normal failure: x is too large to be output with full precision */
            {
              mpfr_printf ("too large !");
            }
          else
            {
              printf ("Error in mpfr_printf(\"%s\", %d, %s, ...)",
                      fmt, prec, mpfr_print_rnd_mode (rnd));

              if (stdout_redirect)
                {
                  if ((fflush (stdout) == EOF) || (fclose (stdout) == -1))
                    {
                      perror ("check_random");
                      exit (1);
                    }
                }
              exit (1);
            }
        }
      putchar ('\n');
    }

  mpfr_set_emin (old_emin);
  mpfr_set_emax (old_emax);

  mpfr_clear (x);
}

int
main (int argc, char *argv[])
{
  int N;

  tests_start_mpfr ();

  /* with no argument: prints to /dev/null,
     tprintf N: prints N tests to stdout */
  if (argc == 1)
    {
      N = 1000;
      stdout_redirect = 1;
      if (freopen ("/dev/null", "w", stdout) == NULL)
        {
          /* We failed to open this device, try with a dummy file */
          if (freopen ("tprintf_out.txt", "w", stdout) == NULL)
            {
              /* Output the error message to stderr since it is not
                 a message about a wrong result in MPFR. Anyway the
                 stdandard output may have changed. */
              fprintf (stderr, "Can't open /dev/null or a temporary file\n");
              exit (1);
            }
        }
    }
  else
    {
      stdout_redirect = 0;
      N = atoi (argv[1]);
    }

  check_invalid_format ();
  check_special ();
  check_mixed ();

  /* expensive tests */
  if (getenv ("MPFR_CHECK_LARGEMEM") != NULL)
    check_long_string();

  check_random (N);

  if (stdout_redirect)
    {
      if ((fflush (stdout) == EOF) || (fclose (stdout) == -1))
        perror ("main");
    }
  tests_end_mpfr ();
  return 0;
}

#else  /* MPFR_VERSION */

int
main (void)
{
  printf ("Warning! Test disabled for this MPFR version.\n");
  return 0;
}

#endif  /* MPFR_VERSION */

#else  /* HAVE_STDARG */

int
main (void)
{
  /* We have nothing to test. */
  return 77;
}

#endif  /* HAVE_STDARG */
