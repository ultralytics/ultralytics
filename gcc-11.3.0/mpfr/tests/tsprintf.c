/* tsprintf.c -- test file for mpfr_sprintf, mpfr_vsprintf, mpfr_snprintf,
   and mpfr_vsnprintf

Copyright 2007-2017 Free Software Foundation, Inc.
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

#ifdef HAVE_STDARG
#include <stdarg.h>

#include <stdlib.h>
#include <float.h>

#ifdef HAVE_LOCALE_H
#include <locale.h>
#endif

#include "mpfr-test.h"

#if MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0)

const int prec_max_printf = 5000; /* limit for random precision in
                                     random_double() */
#define BUF_SIZE 65536

const char pinf_str[] = "inf";
const char pinf_uc_str[] = "INF";
const char minf_str[] = "-inf";
const char minf_uc_str[] = "-INF";
const char nan_str[] = "nan";
const char nan_uc_str[] = "NAN";

/* 1. compare expected string with the string BUFFER returned by
   mpfr_sprintf(buffer, fmt, x)
   2. then test mpfr_snprintf (buffer, p, fmt, x) with a random p. */
static int
check_sprintf (const char *expected, const char *fmt, mpfr_srcptr x)
{
  int n0, n1, p;
  char buffer[BUF_SIZE];

  /* test mpfr_sprintf */
  n0 = mpfr_sprintf (buffer, fmt, x);
  if (strcmp (buffer, expected) != 0)
    {
      printf ("Error in mpfr_sprintf (s, \"%s\", x);\n", fmt);
      printf ("expected: \"%s\"\ngot:      \"%s\"\n", expected, buffer);

      exit (1);
    }

  /* test mpfr_snprintf */
  p = (int) (randlimb () % n0);
  if (p == 0 && (randlimb () & 1) == 0)
    {
      n1 = mpfr_snprintf (NULL, 0, fmt, x);
    }
  else
    {
      buffer[p] = 17;
      n1 = mpfr_snprintf (buffer, p, fmt, x);
      if (buffer[p] != 17)
        {
          printf ("Buffer overflow in mpfr_snprintf for p = %d!\n", p);
          exit (1);
        }
    }
  if (n0 != n1)
    {
      printf ("Error in mpfr_snprintf (s, %d, \"%s\", x) return value\n",
              p, fmt);
      printf ("expected: %d\ngot:      %d\n", n0, n1);
      exit (1);
    }
  if ((p > 1 && strncmp (expected, buffer, p-1) != 0)
      || (p == 1 && buffer[0] != '\0'))
    {
      char part_expected[BUF_SIZE];
      strncpy (part_expected, expected, p);
      part_expected[p-1] = '\0';
      printf ("Error in mpfr_vsnprintf (s, %d, \"%s\", ...);\n", p, fmt);
      printf ("expected: \"%s\"\ngot:      \"%s\"\n", part_expected, buffer);
      exit (1);
    }
  return n0;
}

/* 1. compare expected string with the string BUFFER returned by
   mpfr_vsprintf(buffer, fmt, ...)
   2. then, test mpfr_vsnprintf. */
static int
check_vsprintf (const char *expected, const char *fmt, ...)
{
  int n0, n1, p;
  char buffer[BUF_SIZE];
  va_list ap0, ap1;
  va_start (ap0, fmt);
  va_start (ap1, fmt);

  n0 = mpfr_vsprintf (buffer, fmt, ap0);
  if (strcmp (buffer, expected) != 0)
    {
      printf ("Error in mpfr_vsprintf (s, \"%s\", ...);\n", fmt);
      printf ("expected: \"%s\"\ngot:      \"%s\"\n", expected, buffer);

      va_end (ap0);
      va_end (ap1);
      exit (1);
    }
  va_end (ap0);

  /* test mpfr_snprintf */
  p = (int) (randlimb () % n0);
  if (p == 0 && (randlimb () & 1) == 0)
    {
      n1 = mpfr_vsnprintf (NULL, 0, fmt, ap1);
    }
  else
    {
      buffer[p] = 17;
      n1 = mpfr_vsnprintf (buffer, p, fmt, ap1);
      if (buffer[p] != 17)
        {
          printf ("Buffer overflow in mpfr_vsnprintf for p = %d!\n", p);
          exit (1);
        }
    }
  if (n0 != n1)
    {
      printf ("Error in mpfr_vsnprintf (s, %d, \"%s\", ...) return value\n",
              p, fmt);
      printf ("expected: %d\ngot:      %d\n", n0, n1);

      va_end (ap1);
      exit (1);
    }
  if ((p > 1 && strncmp (expected, buffer, p-1) != 0)
      || (p == 1 && buffer[0] != '\0'))
    {
      char part_expected[BUF_SIZE];
      strncpy (part_expected, expected, p);
      part_expected[p-1] = '\0';
      printf ("Error in mpfr_vsnprintf (s, %d, \"%s\", ...);\n", p, fmt);
      printf ("expected: \"%s\"\ngot:      \"%s\"\n", part_expected, buffer);

      va_end (ap1);
      exit (1);
    }

  va_end (ap1);
  return n0;
}

static void
native_types (void)
{
  int c = 'a';
  int i = -1;
  unsigned int ui = 1;
  double d = -1.25;
  char s[] = "test";

  char buf[255];

  sprintf (buf, "%c", c);
  check_vsprintf (buf, "%c", c);

  sprintf (buf, "%d", i);
  check_vsprintf (buf, "%d", i);

  sprintf (buf, "%e", d);
  check_vsprintf (buf, "%e", d);

  sprintf (buf, "%f", d);
  check_vsprintf (buf, "%f", d);

  sprintf (buf, "%i", i);
  check_vsprintf (buf, "%i", i);

  sprintf (buf, "%g", d);
  check_vsprintf (buf, "%g", d);

  sprintf (buf, "%o", i);
  check_vsprintf (buf, "%o", i);

  sprintf (buf, "%s", s);
  check_vsprintf (buf, "%s", s);

  sprintf (buf, "--%s++", "");
  check_vsprintf (buf, "--%s++", "");

  sprintf (buf, "%u", ui);
  check_vsprintf (buf, "%u", ui);

  sprintf (buf, "%x", ui);
  check_vsprintf (buf, "%x", ui);
}

static int
decimal (void)
{
  mpfr_prec_t p = 128;
  mpfr_t x;
  mpfr_t z;
  mpfr_init (z);
  mpfr_init2 (x, p);

  /* specifier 'P' for precision */
  check_vsprintf ("128", "%Pu", p);
  check_vsprintf ("00128", "%.5Pu", p);

  /* special numbers */
  mpfr_set_inf (x, 1);
  check_sprintf (pinf_str, "%Re", x);
  check_sprintf (pinf_str, "%RUe", x);
  check_sprintf (pinf_uc_str, "%RE", x);
  check_sprintf (pinf_uc_str, "%RDE", x);
  check_sprintf (pinf_str, "%Rf", x);
  check_sprintf (pinf_str, "%RYf", x);
  check_sprintf (pinf_uc_str, "%RF", x);
  check_sprintf (pinf_uc_str, "%RZF", x);
  check_sprintf (pinf_str, "%Rg", x);
  check_sprintf (pinf_str, "%RNg", x);
  check_sprintf (pinf_uc_str, "%RG", x);
  check_sprintf (pinf_uc_str, "%RUG", x);
  check_sprintf ("       inf", "%010Re", x);
  check_sprintf ("       inf", "%010RDe", x);

  mpfr_set_inf (x, -1);
  check_sprintf (minf_str, "%Re", x);
  check_sprintf (minf_str, "%RYe", x);
  check_sprintf (minf_uc_str, "%RE", x);
  check_sprintf (minf_uc_str, "%RZE", x);
  check_sprintf (minf_str, "%Rf", x);
  check_sprintf (minf_str, "%RNf", x);
  check_sprintf (minf_uc_str, "%RF", x);
  check_sprintf (minf_uc_str, "%RUF", x);
  check_sprintf (minf_str, "%Rg", x);
  check_sprintf (minf_str, "%RDg", x);
  check_sprintf (minf_uc_str, "%RG", x);
  check_sprintf (minf_uc_str, "%RYG", x);
  check_sprintf ("      -inf", "%010Re", x);
  check_sprintf ("      -inf", "%010RZe", x);

  mpfr_set_nan (x);
  check_sprintf (nan_str, "%Re", x);
  check_sprintf (nan_str, "%RNe", x);
  check_sprintf (nan_uc_str, "%RE", x);
  check_sprintf (nan_uc_str, "%RUE", x);
  check_sprintf (nan_str, "%Rf", x);
  check_sprintf (nan_str, "%RDf", x);
  check_sprintf (nan_uc_str, "%RF", x);
  check_sprintf (nan_uc_str, "%RYF", x);
  check_sprintf (nan_str, "%Rg", x);
  check_sprintf (nan_str, "%RZg", x);
  check_sprintf (nan_uc_str, "%RG", x);
  check_sprintf (nan_uc_str, "%RNG", x);
  check_sprintf ("       nan", "%010Re", x);

  /* positive numbers */
  mpfr_set_str (x, "18993474.61279296875", 10, MPFR_RNDN);
  mpfr_set_ui (z, 0, MPFR_RNDD);

  /* simplest case right justified */
  check_sprintf ("      1.899347461279296875e+07", "%30Re", x);
  check_sprintf ("                         2e+07", "%30.0Re", x);
  check_sprintf ("               18993474.612793", "%30Rf", x);
  check_sprintf ("              18993474.6127930", "%30.7Rf", x);
  check_sprintf ("                   1.89935e+07", "%30Rg", x);
  check_sprintf ("                         2e+07", "%30.0Rg", x);
  check_sprintf ("          18993474.61279296875", "%30.19Rg", x);
  check_sprintf ("                         0e+00", "%30.0Re", z);
  check_sprintf ("                             0", "%30.0Rf", z);
  check_sprintf ("                        0.0000", "%30.4Rf", z);
  check_sprintf ("                             0", "%30.0Rg", z);
  check_sprintf ("                             0", "%30.4Rg", z);
  /* sign or space, pad with leading zeros */
  check_sprintf (" 000001.899347461279296875E+07", "% 030RE", x);
  check_sprintf (" 0000000000000000001.89935E+07", "% 030RG", x);
  check_sprintf (" 0000000000000000000000002E+07", "% 030.0RE", x);
  check_sprintf (" 0000000000000000000000000E+00", "% 030.0RE", z);
  check_sprintf (" 00000000000000000000000000000", "% 030.0RF", z);
  /* sign + or -, left justified */
  check_sprintf ("+1.899347461279296875e+07     ", "%+-30Re", x);
  check_sprintf ("+2e+07                        ", "%+-30.0Re", x);
  check_sprintf ("+0e+00                        ", "%+-30.0Re", z);
  check_sprintf ("+0                            ", "%+-30.0Rf", z);
  /* decimal point, left justified, precision and rounding parameter */
  check_vsprintf ("1.9E+07   ", "%#-10.*R*E", 1, MPFR_RNDN, x);
  check_vsprintf ("2.E+07    ", "%#*.*R*E", -10, 0, MPFR_RNDN, x);
  check_vsprintf ("2.E+07    ", "%#-10.*R*G", 0, MPFR_RNDN, x);
  check_vsprintf ("0.E+00    ", "%#-10.*R*E", 0, MPFR_RNDN, z);
  check_vsprintf ("0.        ", "%#-10.*R*F", 0, MPFR_RNDN, z);
  check_vsprintf ("0.        ", "%#-10.*R*G", 0, MPFR_RNDN, z);
  /* sign or space */
  check_sprintf (" 1.899e+07", "% .3RNe", x);
  check_sprintf (" 2e+07",     "% .0RNe", x);
  /* sign + or -, decimal point, pad with leading zeros */
  check_sprintf ("+0001.8E+07", "%0+#11.1RZE", x);
  check_sprintf ("+00001.E+07", "%0+#11.0RZE", x);
  check_sprintf ("+0000.0E+00", "%0+#11.1RZE", z);
  check_sprintf ("+00000000.0", "%0+#11.1RZF", z);
  /* pad with leading zero */
  check_sprintf ("0000001.899347461279296875e+07", "%030RDe", x);
  check_sprintf ("00000000000000000000000001e+07", "%030.0RDe", x);
  /* sign or space, decimal point, left justified */
  check_sprintf (" 1.8E+07   ", "%- #11.1RDE", x);
  check_sprintf (" 1.E+07    ", "%- #11.0RDE", x);

  /* negative numbers */
  mpfr_mul_si (x, x, -1, MPFR_RNDD);
  mpfr_mul_si (z, z, -1, MPFR_RNDD);

  /* sign + or - */
  check_sprintf ("  -1.8e+07", "%+10.1RUe", x);
  check_sprintf ("    -1e+07", "%+10.0RUe", x);
  check_sprintf ("    -0e+00", "%+10.0RUe", z);
  check_sprintf ("        -0", "%+10.0RUf", z);


  /* neighborhood of 1 */
  mpfr_set_str (x, "0.99993896484375", 10, MPFR_RNDN);
  check_sprintf ("9.9993896484375E-01 ", "%-20RE", x);
  check_sprintf ("9.9993896484375E-01 ", "%-20.RE", x);
  check_sprintf ("1E+00               ", "%-20.0RE", x);
  check_sprintf ("1.0E+00             ", "%-20.1RE", x);
  check_sprintf ("1.00E+00            ", "%-20.2RE", x);
  check_sprintf ("9.999E-01           ", "%-20.3RE", x);
  check_sprintf ("9.9994E-01          ", "%-20.4RE", x);
  check_sprintf ("0.999939            ", "%-20RF", x);
  check_sprintf ("0.999939            ", "%-20.RF", x);
  check_sprintf ("1                   ", "%-20.0RF", x);
  check_sprintf ("1.0                 ", "%-20.1RF", x);
  check_sprintf ("1.00                ", "%-20.2RF", x);
  check_sprintf ("1.000               ", "%-20.3RF", x);
  check_sprintf ("0.9999              ", "%-20.4RF", x);
  check_sprintf ("0.999939            ", "%-#20RF", x);
  check_sprintf ("0.999939            ", "%-#20.RF", x);
  check_sprintf ("1.                  ", "%-#20.0RF", x);
  check_sprintf ("1.0                 ", "%-#20.1RF", x);
  check_sprintf ("1.00                ", "%-#20.2RF", x);
  check_sprintf ("1.000               ", "%-#20.3RF", x);
  check_sprintf ("0.9999              ", "%-#20.4RF", x);
  check_sprintf ("1                   ", "%-20.0RG", x);
  check_sprintf ("1                   ", "%-20.1RG", x);
  check_sprintf ("1                   ", "%-20.2RG", x);
  check_sprintf ("1                   ", "%-20.3RG", x);
  check_sprintf ("0.9999              ", "%-20.4RG", x);
  check_sprintf ("0.999939            ", "%-#20RG", x);
  check_sprintf ("0.999939            ", "%-#20.RG", x);
  check_sprintf ("1.                  ", "%-#20.0RG", x);
  check_sprintf ("1.                  ", "%-#20.1RG", x);
  check_sprintf ("1.0                 ", "%-#20.2RG", x);
  check_sprintf ("1.00                ", "%-#20.3RG", x);
  check_sprintf ("0.9999              ", "%-#20.4RG", x);

  /* multiple of 10 */
  mpfr_set_str (x, "1e17", 10, MPFR_RNDN);
  check_sprintf ("1e+17", "%Re", x);
  check_sprintf ("1.000e+17", "%.3Re", x);
  check_sprintf ("100000000000000000", "%.0Rf", x);
  check_sprintf ("100000000000000000.0", "%.1Rf", x);
  check_sprintf ("100000000000000000.000000", "%'Rf", x);
  check_sprintf ("100000000000000000.0", "%'.1Rf", x);

  mpfr_ui_div (x, 1, x, MPFR_RNDN); /* x=1e-17 */
  check_sprintf ("1e-17", "%Re", x);
  check_sprintf ("0.000000", "%Rf", x);
  check_sprintf ("1e-17", "%Rg", x);
  check_sprintf ("0.0", "%.1RDf", x);
  check_sprintf ("0.0", "%.1RZf", x);
  check_sprintf ("0.1", "%.1RUf", x);
  check_sprintf ("0.1", "%.1RYf", x);
  check_sprintf ("0", "%.0RDf", x);
  check_sprintf ("0", "%.0RZf", x);
  check_sprintf ("1", "%.0RUf", x);
  check_sprintf ("1", "%.0RYf", x);

  /* multiple of 10 with 'g' style */
  mpfr_set_str (x, "10", 10, MPFR_RNDN);
  check_sprintf ("10", "%Rg", x);
  check_sprintf ("1e+01", "%.0Rg", x);
  check_sprintf ("1e+01", "%.1Rg", x);
  check_sprintf ("10", "%.2Rg", x);

  mpfr_ui_div (x, 1, x, MPFR_RNDN);
  check_sprintf ("0.1", "%Rg", x);
  check_sprintf ("0.1", "%.0Rg", x);
  check_sprintf ("0.1", "%.1Rg", x);

  mpfr_set_str (x, "1000", 10, MPFR_RNDN);
  check_sprintf ("1000", "%Rg", x);
  check_sprintf ("1e+03", "%.0Rg", x);
  check_sprintf ("1e+03", "%.3Rg", x);
  check_sprintf ("1000", "%.4Rg", x);

  mpfr_ui_div (x, 1, x, MPFR_RNDN);
  check_sprintf ("0.001", "%Rg", x);
  check_sprintf ("0.001", "%.0Rg", x);
  check_sprintf ("0.001", "%.1Rg", x);

  mpfr_set_str (x, "100000", 10, MPFR_RNDN);
  check_sprintf ("100000", "%Rg", x);
  check_sprintf ("1e+05", "%.0Rg", x);
  check_sprintf ("1e+05", "%.5Rg", x);
  check_sprintf ("100000", "%.6Rg", x);

  mpfr_ui_div (x, 1, x, MPFR_RNDN);
  check_sprintf ("1e-05", "%Rg", x);
  check_sprintf ("1e-05", "%.0Rg", x);
  check_sprintf ("1e-05", "%.1Rg", x);

  /* check rounding mode */
  mpfr_set_str (x, "0.0076", 10, MPFR_RNDN);
  check_sprintf ("0.007", "%.3RDF", x);
  check_sprintf ("0.007", "%.3RZF", x);
  check_sprintf ("0.008", "%.3RF", x);
  check_sprintf ("0.008", "%.3RUF", x);
  check_sprintf ("0.008", "%.3RYF", x);
  check_vsprintf ("0.008", "%.3R*F", MPFR_RNDA, x);

  /* check limit between %f-style and %g-style */
  mpfr_set_str (x, "0.0000999", 10, MPFR_RNDN);
  check_sprintf ("0.0001",   "%.0Rg", x);
  check_sprintf ("9e-05",    "%.0RDg", x);
  check_sprintf ("0.0001",   "%.1Rg", x);
  check_sprintf ("0.0001",   "%.2Rg", x);
  check_sprintf ("9.99e-05", "%.3Rg", x);

  /* trailing zeros */
  mpfr_set_si_2exp (x, -1, -15, MPFR_RNDN); /* x=-2^-15 */
  check_sprintf ("-3.0517578125e-05", "%.30Rg", x);
  check_sprintf ("-3.051757812500000000000000000000e-05", "%.30Re", x);
  check_sprintf ("-3.05175781250000000000000000000e-05", "%#.30Rg", x);
  check_sprintf ("-0.000030517578125000000000000000", "%.30Rf", x);

  /* bug 20081023 */
  check_sprintf ("-3.0517578125e-05", "%.30Rg", x);
  mpfr_set_str (x, "1.9999", 10, MPFR_RNDN);
  check_sprintf ("1.999900  ", "%-#10.7RG", x);
  check_sprintf ("1.9999    ", "%-10.7RG", x);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  check_sprintf ("1.", "%#.1Rg", x);
  check_sprintf ("1.   ", "%-#5.1Rg", x);
  check_sprintf ("  1.0", "%#5.2Rg", x);
  check_sprintf ("1.00000000000000000000000000000", "%#.30Rg", x);
  check_sprintf ("1", "%.30Rg", x);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  check_sprintf ("0.", "%#.1Rg", x);
  check_sprintf ("0.   ", "%-#5.1Rg", x);
  check_sprintf ("  0.0", "%#5.2Rg", x);
  check_sprintf ("0.00000000000000000000000000000", "%#.30Rg", x);
  check_sprintf ("0", "%.30Rg", x);

  /* following tests with precision 53 bits */
  mpfr_set_prec (x, 53);

  /* Exponent zero has a plus sign */
  mpfr_set_str (x, "-9.95645044213728791504536275169812142849e-01", 10,
                MPFR_RNDN);
  check_sprintf ("-1.0e+00", "%- #0.1Re", x);

  /* Decimal point and no figure after it with '#' flag and 'G' style */
  mpfr_set_str (x, "-9.90597761233942053494e-01", 10, MPFR_RNDN);
  check_sprintf ("-1.", "%- #0.1RG", x);

  /* precision zero */
  mpfr_set_d (x, 9.5, MPFR_RNDN);
  check_sprintf ("9",    "%.0RDf", x);
  check_sprintf ("10",    "%.0RUf", x);

  mpfr_set_d (x, 19.5, MPFR_RNDN);
  check_sprintf ("19",    "%.0RDf", x);
  check_sprintf ("20",    "%.0RUf", x);

  mpfr_set_d (x, 99.5, MPFR_RNDN);
  check_sprintf ("99",    "%.0RDf", x);
  check_sprintf ("100",   "%.0RUf", x);

  mpfr_set_d (x, -9.5, MPFR_RNDN);
  check_sprintf ("-10",    "%.0RDf", x);
  check_sprintf ("-10",    "%.0RYf", x);
  check_sprintf ("-10",    "%.0Rf", x);
  check_sprintf ("-1e+01", "%.0Re", x);
  check_sprintf ("-1e+01", "%.0Rg", x);
  mpfr_set_ui_2exp (x, 1, -1, MPFR_RNDN);
  check_sprintf ("0",      "%.0Rf", x);
  check_sprintf ("5e-01",  "%.0Re", x);
  check_sprintf ("0.5",    "%.0Rg", x);
  mpfr_set_ui_2exp (x, 3, -1, MPFR_RNDN);
  check_sprintf ("2",      "%.0Rf", x);
  mpfr_set_ui_2exp (x, 5, -1, MPFR_RNDN);
  check_sprintf ("2",      "%.0Rf", x);
  mpfr_set_ui (x, 0x1f, MPFR_RNDN);
  check_sprintf ("0x1p+5", "%.0Ra", x);
  mpfr_set_ui (x, 3, MPFR_RNDN);
  check_sprintf ("1p+2",   "%.0Rb", x);

  /* round to next ten power with %f but not with %g */
  mpfr_set_str (x, "-6.64464380544039223686e-02", 10, MPFR_RNDN);
  check_sprintf ("-0.1",  "%.1Rf", x);
  check_sprintf ("-0.0",  "%.1RZf", x);
  check_sprintf ("-0.07", "%.1Rg", x);
  check_sprintf ("-0.06", "%.1RZg", x);

  /* round to next ten power and do not remove trailing zeros */
  mpfr_set_str (x, "9.98429393291486722006e-02", 10, MPFR_RNDN);
  check_sprintf ("0.1",   "%#.1Rg", x);
  check_sprintf ("0.10",  "%#.2Rg", x);
  check_sprintf ("0.099", "%#.2RZg", x);

  /* Halfway cases */
  mpfr_set_str (x, "1.5", 10, MPFR_RNDN);
  check_sprintf ("2e+00", "%.0Re", x);
  mpfr_set_str (x, "2.5", 10, MPFR_RNDN);
  check_sprintf ("2e+00", "%.0Re", x);
  mpfr_set_str (x, "9.5", 10, MPFR_RNDN);
  check_sprintf ("1e+01", "%.0Re", x);
  mpfr_set_str (x, "1.25", 10, MPFR_RNDN);
  check_sprintf ("1.2e+00", "%.1Re", x);
  mpfr_set_str (x, "1.75", 10, MPFR_RNDN);
  check_sprintf ("1.8e+00", "%.1Re", x);
  mpfr_set_str (x, "-0.5", 10, MPFR_RNDN);
  check_sprintf ("-0", "%.0Rf", x);
  mpfr_set_str (x, "1.25", 10, MPFR_RNDN);
  check_sprintf ("1.2", "%.1Rf", x);
  mpfr_set_str (x, "1.75", 10, MPFR_RNDN);
  check_sprintf ("1.8", "%.1Rf", x);
  mpfr_set_str (x, "1.5", 10, MPFR_RNDN);
  check_sprintf ("2", "%.1Rg", x);
  mpfr_set_str (x, "2.5", 10, MPFR_RNDN);
  check_sprintf ("2", "%.1Rg", x);
  mpfr_set_str (x, "9.25", 10, MPFR_RNDN);
  check_sprintf ("9.2", "%.2Rg", x);
  mpfr_set_str (x, "9.75", 10, MPFR_RNDN);
  check_sprintf ("9.8", "%.2Rg", x);

  /* assertion failure in r6320 */
  mpfr_set_str (x, "-9.996", 10, MPFR_RNDN);
  check_sprintf ("-10.0", "%.1Rf", x);

  mpfr_clears (x, z, (mpfr_ptr) 0);
  return 0;
}

static int
hexadecimal (void)
{
  mpfr_t x, z;
  mpfr_inits2 (64, x, z, (mpfr_ptr) 0);

  /* special */
  mpfr_set_inf (x, 1);
  check_sprintf (pinf_str, "%Ra", x);
  check_sprintf (pinf_str, "%RUa", x);
  check_sprintf (pinf_str, "%RDa", x);
  check_sprintf (pinf_uc_str, "%RA", x);
  check_sprintf (pinf_uc_str, "%RYA", x);
  check_sprintf (pinf_uc_str, "%RZA", x);
  check_sprintf (pinf_uc_str, "%RNA", x);

  mpfr_set_inf (x, -1);
  check_sprintf (minf_str, "%Ra", x);
  check_sprintf (minf_str, "%RYa", x);
  check_sprintf (minf_str, "%RZa", x);
  check_sprintf (minf_str, "%RNa", x);
  check_sprintf (minf_uc_str, "%RA", x);
  check_sprintf (minf_uc_str, "%RUA", x);
  check_sprintf (minf_uc_str, "%RDA", x);

  mpfr_set_nan (x);
  check_sprintf (nan_str, "%Ra", x);
  check_sprintf (nan_uc_str, "%RA", x);

  /* regular numbers */
  mpfr_set_str (x, "FEDCBA9.87654321", 16, MPFR_RNDN);
  mpfr_set_ui (z, 0, MPFR_RNDZ);

  /* simplest case right justified */
  check_sprintf ("   0xf.edcba987654321p+24", "%25Ra", x);
  check_sprintf ("   0xf.edcba987654321p+24", "%25RUa", x);
  check_sprintf ("   0xf.edcba987654321p+24", "%25RDa", x);
  check_sprintf ("   0xf.edcba987654321p+24", "%25RYa", x);
  check_sprintf ("   0xf.edcba987654321p+24", "%25RZa", x);
  check_sprintf ("   0xf.edcba987654321p+24", "%25RNa", x);
  check_sprintf ("                  0x1p+28", "%25.0Ra", x);
  check_sprintf ("                   0x0p+0", "%25.0Ra", z);
  /* sign or space, pad with leading zeros */
  check_sprintf (" 0X00F.EDCBA987654321P+24", "% 025RA", x);
  check_sprintf (" 0X000000000000000001P+28", "% 025.0RA", x);
  check_sprintf (" 0X0000000000000000000P+0", "% 025.0RA", z);
  /* sign + or -, left justified */
  check_sprintf ("+0xf.edcba987654321p+24  ", "%+-25Ra", x);
  check_sprintf ("+0x1p+28                 ", "%+-25.0Ra", x);
  check_sprintf ("+0x0p+0                  ", "%+-25.0Ra", z);
  /* decimal point, left justified, precision and rounding parameter */
  check_vsprintf ("0XF.FP+24 ", "%#-10.*R*A", 1, MPFR_RNDN, x);
  check_vsprintf ("0X1.P+28  ", "%#-10.*R*A", 0, MPFR_RNDN, x);
  check_vsprintf ("0X0.P+0   ", "%#-10.*R*A", 0, MPFR_RNDN, z);
  /* sign or space */
  check_sprintf (" 0xf.eddp+24", "% .3RNa", x);
  check_sprintf (" 0x1p+28",     "% .0RNa", x);
  /* sign + or -, decimal point, pad with leading zeros */
  check_sprintf ("+0X0F.EP+24", "%0+#11.1RZA", x);
  check_sprintf ("+0X00F.P+24", "%0+#11.0RZA", x);
  check_sprintf ("+0X000.0P+0", "%0+#11.1RZA", z);
  /* pad with leading zero */
  check_sprintf ("0x0000f.edcba987654321p+24", "%026RDa", x);
  check_sprintf ("0x0000000000000000000fp+24", "%026.0RDa", x);
  /* sign or space, decimal point, left justified */
  check_sprintf (" 0XF.EP+24 " , "%- #11.1RDA", x);
  check_sprintf (" 0XF.P+24  " , "%- #11.0RDA", x);

  mpfr_mul_si (x, x, -1, MPFR_RNDD);
  mpfr_mul_si (z, z, -1, MPFR_RNDD);

  /* sign + or - */
  check_sprintf ("-0xf.ep+24", "%+10.1RUa", x);
  check_sprintf ("  -0xfp+24", "%+10.0RUa", x);
  check_sprintf ("   -0x0p+0", "%+10.0RUa", z);

  /* rounding bit is zero */
  mpfr_set_str (x, "0xF.7", 16, MPFR_RNDN);
  check_sprintf ("0XFP+0", "%.0RNA", x);
  /* tie case in round to nearest mode */
  mpfr_set_str (x, "0x0.8800000000000000p+3", 16, MPFR_RNDN);
  check_sprintf ("0x9.p-1", "%#.0RNa", x);
  mpfr_set_str (x, "-0x0.9800000000000000p+3", 16, MPFR_RNDN);
  check_sprintf ("-0xap-1", "%.0RNa", x);
  /* trailing zeros in fractional part */
  check_sprintf ("-0X4.C0000000000000000000P+0", "%.20RNA", x);
  /* rounding bit is one and the first non zero bit is far away */
  mpfr_set_prec (x, 1024);
  mpfr_set_ui_2exp (x, 29, -1, MPFR_RNDN);
  mpfr_nextabove (x);
  check_sprintf ("0XFP+0", "%.0RNA", x);

  /* with more than one limb */
  mpfr_set_prec (x, 300);
  mpfr_set_str (x, "0xf.ffffffffffffffffffffffffffffffffffffffffffffffffffff"
                "fffffffffffffffff", 16, MPFR_RNDN);
  check_sprintf ("0x1p+4 [300]", "%.0RNa [300]", x);
  check_sprintf ("0xfp+0 [300]", "%.0RZa [300]", x);
  check_sprintf ("0x1p+4 [300]", "%.0RYa [300]", x);
  check_sprintf ("0xfp+0 [300]", "%.0RDa [300]", x);
  check_sprintf ("0x1p+4 [300]", "%.0RUa [300]", x);
  check_sprintf ("0x1.0000000000000000000000000000000000000000p+4",
                 "%.40RNa", x);
  check_sprintf ("0xf.ffffffffffffffffffffffffffffffffffffffffp+0",
                 "%.40RZa", x);
  check_sprintf ("0x1.0000000000000000000000000000000000000000p+4",
                 "%.40RYa", x);
  check_sprintf ("0xf.ffffffffffffffffffffffffffffffffffffffffp+0",
                 "%.40RDa", x);
  check_sprintf ("0x1.0000000000000000000000000000000000000000p+4",
                 "%.40RUa", x);

  mpfr_set_str (x, "0xf.7fffffffffffffffffffffffffffffffffffffffffffffffffff"
                "ffffffffffffffffff", 16, MPFR_RNDN);
  check_sprintf ("0XFP+0", "%.0RNA", x);
  check_sprintf ("0XFP+0", "%.0RZA", x);
  check_sprintf ("0X1P+4", "%.0RYA", x);
  check_sprintf ("0XFP+0", "%.0RDA", x);
  check_sprintf ("0X1P+4", "%.0RUA", x);
  check_sprintf ("0XF.8P+0", "%.1RNA", x);
  check_sprintf ("0XF.7P+0", "%.1RZA", x);
  check_sprintf ("0XF.8P+0", "%.1RYA", x);
  check_sprintf ("0XF.7P+0", "%.1RDA", x);
  check_sprintf ("0XF.8P+0", "%.1RUA", x);

  /* do not round up to the next power of the base */
  mpfr_set_str (x, "0xf.fffffffffffffffffffffffffffffffffffffeffffffffffffff"
                "ffffffffffffffffff", 16, MPFR_RNDN);
  check_sprintf ("0xf.ffffffffffffffffffffffffffffffffffffff00p+0",
                 "%.40RNa", x);
  check_sprintf ("0xf.fffffffffffffffffffffffffffffffffffffeffp+0",
                 "%.40RZa", x);
  check_sprintf ("0xf.ffffffffffffffffffffffffffffffffffffff00p+0",
                 "%.40RYa", x);
  check_sprintf ("0xf.fffffffffffffffffffffffffffffffffffffeffp+0",
                 "%.40RDa", x);
  check_sprintf ("0xf.ffffffffffffffffffffffffffffffffffffff00p+0",
                 "%.40RUa", x);

  mpfr_clears (x, z, (mpfr_ptr) 0);
  return 0;
}

static int
binary (void)
{
  mpfr_t x;
  mpfr_t z;
  mpfr_inits2 (64, x, z, (mpfr_ptr) 0);

  /* special */
  mpfr_set_inf (x, 1);
  check_sprintf (pinf_str, "%Rb", x);

  mpfr_set_inf (x, -1);
  check_sprintf (minf_str, "%Rb", x);

  mpfr_set_nan (x);
  check_sprintf (nan_str, "%Rb", x);

  /* regular numbers */
  mpfr_set_str (x, "1110010101.1001101", 2, MPFR_RNDN);
  mpfr_set_ui (z, 0, MPFR_RNDN);

  /* simplest case: right justified */
  check_sprintf ("    1.1100101011001101p+9", "%25Rb", x);
  check_sprintf ("                     0p+0", "%25Rb", z);
  /* sign or space, pad with leading zeros */
  check_sprintf (" 0001.1100101011001101p+9", "% 025Rb", x);
  check_sprintf (" 000000000000000000000p+0", "% 025Rb", z);
  /* sign + or -, left justified */
  check_sprintf ("+1.1100101011001101p+9   ", "%+-25Rb", x);
  check_sprintf ("+0p+0                    ", "%+-25Rb", z);
  /* sign or space */
  check_sprintf (" 1.110p+9",  "% .3RNb", x);
  check_sprintf (" 1.1101p+9", "% .4RNb", x);
  check_sprintf (" 0.0000p+0", "% .4RNb", z);
  /* sign + or -, decimal point, pad with leading zeros */
  check_sprintf ("+00001.1p+9", "%0+#11.1RZb", x);
  check_sprintf ("+0001.0p+10", "%0+#11.1RNb", x);
  check_sprintf ("+000000.p+0", "%0+#11.0RNb", z);
  /* pad with leading zero */
  check_sprintf ("00001.1100101011001101p+9", "%025RDb", x);
  /* sign or space, decimal point (unused), left justified */
  check_sprintf (" 1.1p+9    ", "%- #11.1RDb", x);
  check_sprintf (" 1.p+9     ", "%- #11.0RDb", x);
  check_sprintf (" 1.p+10    ", "%- #11.0RUb", x);
  check_sprintf (" 1.p+9     ", "%- #11.0RZb", x);
  check_sprintf (" 1.p+10    ", "%- #11.0RYb", x);
  check_sprintf (" 1.p+10    ", "%- #11.0RNb", x);

  mpfr_mul_si (x, x, -1, MPFR_RNDD);
  mpfr_mul_si (z, z, -1, MPFR_RNDD);

  /* sign + or - */
  check_sprintf ("   -1.1p+9", "%+10.1RUb", x);
  check_sprintf ("   -0.0p+0", "%+10.1RUb", z);

  /* precision 0 */
  check_sprintf ("-1p+10", "%.0RNb", x);
  check_sprintf ("-1p+10", "%.0RDb", x);
  check_sprintf ("-1p+9",  "%.0RUb", x);
  check_sprintf ("-1p+9",  "%.0RZb", x);
  check_sprintf ("-1p+10", "%.0RYb", x);
  /* round to next base power */
  check_sprintf ("-1.0p+10", "%.1RNb", x);
  check_sprintf ("-1.0p+10", "%.1RDb", x);
  check_sprintf ("-1.0p+10", "%.1RYb", x);
  /* do not round to next base power */
  check_sprintf ("-1.1p+9", "%.1RUb", x);
  check_sprintf ("-1.1p+9", "%.1RZb", x);
  /* rounding bit is zero */
  check_sprintf ("-1.11p+9", "%.2RNb", x);
  /* tie case in round to nearest mode */
  check_sprintf ("-1.1100101011001101p+9", "%.16RNb", x);
  /* trailing zeros in fractional part */
  check_sprintf ("-1.110010101100110100000000000000p+9", "%.30RNb", x);

  mpfr_clears (x, z, (mpfr_ptr) 0);
  return 0;
}

static int
mixed (void)
{
  int n1;
  int n2;
  int i = 121;
#ifndef NPRINTF_L
  long double d = 1. / 31.;
#endif
  mpf_t mpf;
  mpq_t mpq;
  mpz_t mpz;
  mpfr_t x;
  mpfr_rnd_t rnd;

  mpf_init (mpf);
  mpf_set_ui (mpf, 40);
  mpf_div_ui (mpf, mpf, 31); /* mpf = 40.0 / 31.0 */
  mpq_init (mpq);
  mpq_set_ui (mpq, 123456, 4567890);
  mpz_init (mpz);
  mpz_fib_ui (mpz, 64);
  mpfr_init (x);
  mpfr_set_str (x, "-12345678.875", 10, MPFR_RNDN);
  rnd = MPFR_RNDD;

  check_vsprintf ("121%", "%i%%", i);
  check_vsprintf ("121% -1.2345678875E+07", "%i%% %RNE", i, x);
  check_vsprintf ("121, -12345679", "%i, %.0Rf", i, x);
  check_vsprintf ("10610209857723, -1.2345678875e+07", "%Zi, %R*e", mpz, rnd,
                  x);
  check_vsprintf ("-12345678.9, 121", "%.1Rf, %i", x, i);
  check_vsprintf ("-12345678, 1e240/45b352", "%.0R*f, %Qx", MPFR_RNDZ, x, mpq);
  n1 = check_vsprintf ("121, -12345678.875000000000, 1.290323", "%i, %.*Rf, %Ff%n",
                       i, 12, x, mpf, &n2);
  if (n1 != n2)
    {
      printf ("error in number of characters written by mpfr_vsprintf\n");
      printf ("expected: %d\n", n2);
      printf ("     got: %d\n", n1);
      exit (1);
    }

#ifndef NPRINTF_L
  check_vsprintf ("00000010610209857723, -1.2345678875e+07, 0.032258",
                  "%.*Zi, %R*e, %Lf", 20, mpz, rnd, x, d);
#endif

  mpf_clear (mpf);
  mpq_clear (mpq);
  mpz_clear (mpz);
  mpfr_clear (x);
  return 0;
}

#if MPFR_LCONV_DPTS

/* Check with locale "da_DK". On most platforms, decimal point is ','
   and thousands separator is '.'; the test is not performed if this
   is not the case or if the locale doesn't exist. */
static int
locale_da_DK (void)
{
  mpfr_prec_t p = 128;
  mpfr_t x;

  if (setlocale (LC_ALL, "da_DK") == 0 ||
      localeconv()->decimal_point[0] != ',' ||
      localeconv()->thousands_sep[0] != '.')
    return 0;

  mpfr_init2 (x, p);

  /* positive numbers */
  mpfr_set_str (x, "18993474.61279296875", 10, MPFR_RNDN);

  /* simplest case right justified with thousands separator */
  check_sprintf ("      1,899347461279296875e+07", "%'30Re", x);
  check_sprintf ("                   1,89935e+07", "%'30Rg", x);
  check_sprintf ("        18.993.474,61279296875", "%'30.19Rg", x);
  check_sprintf ("             18.993.474,612793", "%'30Rf", x);

  /* sign or space, pad, thousands separator with leading zeros */
  check_sprintf (" 000001,899347461279296875E+07", "%' 030RE", x);
  check_sprintf (" 0000000000000000001,89935E+07", "%' 030RG", x);
  check_sprintf (" 000000018.993.474,61279296875", "%' 030.19RG", x);
  check_sprintf (" 00000000000018.993.474,612793", "%' 030RF", x);

  mpfr_set_ui (x, 50, MPFR_RNDN);
  mpfr_exp10 (x, x, MPFR_RNDN);
  check_sprintf ("100000000000000000000000000000000000000000000000000", "%.0Rf",
                 x);
  check_sprintf
    ("100.000.000.000.000.000.000.000.000.000.000.000.000.000.000.000.000,",
     "%'#.0Rf", x);
  check_sprintf
    ("100.000.000.000.000.000.000.000.000.000.000.000.000.000.000.000.000,0000",
     "%'.4Rf", x);

  mpfr_clear (x);
  return 0;
}

#endif  /* MPFR_LCONV_DPTS */

/* check concordance between mpfr_asprintf result with a regular mpfr float
   and with a regular double float */
static int
random_double (void)
{
  mpfr_t x; /* random regular mpfr float */
  double y; /* regular double float (equal to x) */

  char flag[] =
    {
      '-',
      '+',
      ' ',
      '#',
      '0', /* no ambiguity: first zeros are flag zero */
      '\'' /* SUS extension */
    };
  /* no 'a': mpfr and glibc do not have the same semantic */
  char specifier[] =
    {
      'e',
      'f',
      'g',
      'E',
      'f', /* SUSv2 doesn't accept %F, but %F and %f are the same for
              regular numbers */
      'G',
    };
  int spec; /* random index in specifier[] */
  int prec; /* random value for precision field */

  /* in the format string for mpfr_t variable, the maximum length is
     reached by something like "%-+ #0'.*Rf", that is 12 characters. */
#define FMT_MPFR_SIZE 12
  char fmt_mpfr[FMT_MPFR_SIZE];
  char *ptr_mpfr;

  /* in the format string for double variable, the maximum length is
     reached by something like "%-+ #0'.*f", that is 11 characters. */
#define FMT_SIZE 11
  char fmt[FMT_SIZE];
  char *ptr;

  int xi;
  char *xs;
  int yi;
  char *ys;

  int i, j, jmax;

  mpfr_init2 (x, MPFR_LDBL_MANT_DIG);

  for (i = 0; i < 1000; ++i)
    {
      /* 1. random double */
      do
        {
          y = DBL_RAND ();
        }
#ifdef HAVE_DENORMS
      while (0);
#else
      while (ABS(y) < DBL_MIN);
#endif

      if (randlimb () % 2 == 0)
        y = -y;

      mpfr_set_d (x, y, MPFR_RNDN);
      if (y != mpfr_get_d (x, MPFR_RNDN))
        /* conversion error: skip this one */
        continue;

      /* 2. build random format strings fmt_mpfr and fmt */
      ptr_mpfr = fmt_mpfr;
      ptr = fmt;
      *ptr_mpfr++ = *ptr++ = '%';
      /* random specifier 'e', 'f', 'g', 'E', 'F', or 'G' */
      spec = (int) (randlimb() % 6);
      /* random flags, but no ' flag with %e or with non-glibc */
#if __MPFR_GLIBC(1,0)
      jmax = (spec == 0 || spec == 3) ? 5 : 6;
#else
      jmax = 5;
#endif
      for (j = 0; j < jmax; j++)
        {
          if (randlimb() % 3 == 0)
            *ptr_mpfr++ = *ptr++ = flag[j];
        }
      *ptr_mpfr++ = *ptr++ = '.';
      *ptr_mpfr++ = *ptr++ = '*';
      *ptr_mpfr++ = 'R';
      *ptr_mpfr++ = *ptr++ = specifier[spec];
      *ptr_mpfr = *ptr = '\0';
      MPFR_ASSERTN (ptr - fmt < FMT_SIZE);
      MPFR_ASSERTN (ptr_mpfr - fmt_mpfr < FMT_MPFR_SIZE);

      /* advantage small precision */
      if (randlimb() % 2 == 0)
        prec = (int) (randlimb() % 10);
      else
        prec = (int) (randlimb() % prec_max_printf);

      /* 3. calls and checks */
      /* the double float case is handled by the libc asprintf through
         gmp_asprintf */
      xi = mpfr_asprintf (&xs, fmt_mpfr, prec, x);
      yi = mpfr_asprintf (&ys, fmt, prec, y);

      /* test if XS and YS differ, beware that ISO C99 doesn't specify
         the sign of a zero exponent (the C99 rationale says: "The sign
         of a zero exponent in %e format is unspecified.  The committee
         knows of different implementations and choose not to require
         implementations to document their behaviour in this case
         (by making this be implementation defined behaviour).  Most
         implementations use a "+" sign, e.g., 1.2e+00; but there is at
         least one implementation that uses the sign of the unlimited
         precision result, e.g., the 0.987 would be 9.87e-01, so could
         end up as 1e-00 after rounding to one digit of precision."),
         while mpfr always uses '+' */
      if (xi != yi
          || ((strcmp (xs, ys) != 0)
              && (spec == 1 || spec == 4
                  || ((strstr (xs, "e+00") == NULL
                       || strstr (ys, "e-00") == NULL)
                      && (strstr (xs, "E+00") == NULL
                          || strstr (ys, "E-00") == NULL)))))
        {
          mpfr_printf ("Error in mpfr_asprintf(\"%s\", %d, %Re)\n",
                       fmt_mpfr, prec, x);
          printf ("expected: %s\n", ys);
          printf ("     got: %s\n", xs);
          printf ("xi=%d yi=%d spec=%d\n", xi, yi, spec);

          exit (1);
        }

      mpfr_free_str (xs);
      mpfr_free_str (ys);
    }

  mpfr_clear (x);
  return 0;
}

static void
bug20080610 (void)
{
  /* bug on icc found on June 10, 2008 */
  /* this is not a bug but a different implementation choice: ISO C99 doesn't
     specify the sign of a zero exponent (see note in random_double above). */
  mpfr_t x;
  double y;
  int xi;
  char *xs;
  int yi;
  char *ys;

  mpfr_init2 (x, MPFR_LDBL_MANT_DIG);

  y = -9.95645044213728791504536275169812142849e-01;
  mpfr_set_d (x, y, MPFR_RNDN);

  xi = mpfr_asprintf (&xs, "%- #0.*Re", 1, x);
  yi = mpfr_asprintf (&ys, "%- #0.*e", 1, y);

  if (xi != yi || strcmp (xs, ys) != 0)
    {
      printf ("Error in bug20080610\n");
      printf ("expected: %s\n", ys);
      printf ("     got: %s\n", xs);
      printf ("xi=%d yi=%d\n", xi, yi);

      exit (1);
    }

  mpfr_free_str (xs);
  mpfr_free_str (ys);
  mpfr_clear (x);
}

static void
bug20081214 (void)
{
 /* problem with glibc 2.3.6, December 14, 2008:
    the system asprintf outputs "-1.0" instead of "-1.". */
  mpfr_t x;
  double y;
  int xi;
  char *xs;
  int yi;
  char *ys;

  mpfr_init2 (x, MPFR_LDBL_MANT_DIG);

  y = -9.90597761233942053494e-01;
  mpfr_set_d (x, y, MPFR_RNDN);

  xi = mpfr_asprintf (&xs, "%- #0.*RG", 1, x);
  yi = mpfr_asprintf (&ys, "%- #0.*G", 1, y);

  if (xi != yi || strcmp (xs, ys) != 0)
    {
      mpfr_printf ("Error in bug20081214\n"
                   "mpfr_asprintf(\"%- #0.*Re\", 1, %Re)\n", x);
      printf ("expected: %s\n", ys);
      printf ("     got: %s\n", xs);
      printf ("xi=%d yi=%d\n", xi, yi);

      exit (1);
    }

  mpfr_free_str (xs);
  mpfr_free_str (ys);
  mpfr_clear (x);
}

static void
bug20111102 (void)
{
  mpfr_t t;
  char s[100];

  mpfr_init2 (t, 84);
  mpfr_set_str (t, "999.99999999999999999999", 10, MPFR_RNDN);
  mpfr_sprintf (s, "%.20RNg", t);
  if (strcmp (s, "1000") != 0)
    {
      printf ("Error in bug20111102, expected 1000, got %s\n", s);
      exit (1);
    }
  mpfr_clear (t);
}

/* In particular, the following test makes sure that the rounding
 * for %Ra and %Rb is not done on the MPFR number itself (as it
 * would overflow). Note: it has been reported on comp.std.c that
 * some C libraries behave differently on %a, but this is a bug.
 */
static void
check_emax_aux (mpfr_exp_t e)
{
  mpfr_t x;
  char *s1, s2[256];
  int i;
  mpfr_exp_t emax;

  MPFR_ASSERTN (e <= LONG_MAX);
  emax = mpfr_get_emax ();
  set_emax (e);

  mpfr_init2 (x, 16);

  mpfr_set_inf (x, 1);
  mpfr_nextbelow (x);

  i = mpfr_asprintf (&s1, "%Ra %.2Ra", x, x);
  MPFR_ASSERTN (i > 0);

  mpfr_snprintf (s2, 256, "0x7.fff8p+%ld 0x8.00p+%ld", e-3, e-3);

  if (strcmp (s1, s2) != 0)
    {
      printf ("Error in check_emax_aux for emax = ");
      if (e > LONG_MAX)
        printf ("(>LONG_MAX)\n");
      else
        printf ("%ld\n", (long) e);
      printf ("Expected %s\n", s2);
      printf ("Got      %s\n", s1);
      exit (1);
    }

  mpfr_free_str (s1);

  i = mpfr_asprintf (&s1, "%Rb %.2Rb", x, x);
  MPFR_ASSERTN (i > 0);

  mpfr_snprintf (s2, 256, "1.111111111111111p+%ld 1.00p+%ld", e-1, e);

  if (strcmp (s1, s2) != 0)
    {
      printf ("Error in check_emax_aux for emax = ");
      if (e > LONG_MAX)
        printf ("(>LONG_MAX)\n");
      else
        printf ("%ld\n", (long) e);
      printf ("Expected %s\n", s2);
      printf ("Got      %s\n", s1);
      exit (1);
    }

  mpfr_free_str (s1);

  mpfr_clear (x);
  set_emax (emax);
}

static void
check_emax (void)
{
  check_emax_aux (15);
  check_emax_aux (MPFR_EMAX_MAX);
}

static void
check_emin_aux (mpfr_exp_t e)
{
  mpfr_t x;
  char *s1, s2[256];
  int i;
  mpfr_exp_t emin;
  mpz_t ee;

  MPFR_ASSERTN (e >= LONG_MIN);
  emin = mpfr_get_emin ();
  set_emin (e);

  mpfr_init2 (x, 16);
  mpz_init (ee);

  mpfr_setmin (x, e);
  mpz_set_si (ee, e);
  mpz_sub_ui (ee, ee, 1);

  i = mpfr_asprintf (&s1, "%Ra", x);
  MPFR_ASSERTN (i > 0);

  gmp_snprintf (s2, 256, "0x1p%Zd", ee);

  if (strcmp (s1, s2) != 0)
    {
      printf ("Error in check_emin_aux for emin = %ld\n", (long) e);
      printf ("Expected %s\n", s2);
      printf ("Got      %s\n", s1);
      exit (1);
    }

  mpfr_free_str (s1);

  i = mpfr_asprintf (&s1, "%Rb", x);
  MPFR_ASSERTN (i > 0);

  gmp_snprintf (s2, 256, "1p%Zd", ee);

  if (strcmp (s1, s2) != 0)
    {
      printf ("Error in check_emin_aux for emin = %ld\n", (long) e);
      printf ("Expected %s\n", s2);
      printf ("Got      %s\n", s1);
      exit (1);
    }

  mpfr_free_str (s1);

  mpfr_clear (x);
  mpz_clear (ee);
  set_emin (emin);
}

static void
check_emin (void)
{
  check_emin_aux (-15);
  check_emin_aux (mpfr_get_emin ());
  check_emin_aux (MPFR_EMIN_MIN);
}

static void
test20161214 (void)
{
  mpfr_t x;
  char buf[32];
  const char s[] = "0x0.fffffffffffff8p+1024";
  int r;

  mpfr_init2 (x, 64);
  mpfr_set_str (x, s, 16, MPFR_RNDN);
  r = mpfr_snprintf (buf, 32, "%.*RDf", -2, x);
  MPFR_ASSERTN(r == 316);
  r = mpfr_snprintf (buf, 32, "%.*RDf", INT_MIN + 1, x);
  MPFR_ASSERTN(r == 316);
  r = mpfr_snprintf (buf, 32, "%.*RDf", INT_MIN, x);
  MPFR_ASSERTN(r == 316);
  mpfr_clear (x);
}

int
main (int argc, char **argv)
{

  tests_start_mpfr ();

#if defined(HAVE_LOCALE_H) && defined(HAVE_SETLOCALE)
  /* currently, we just check with 'C' locale */
  setlocale (LC_ALL, "C");
#endif

  bug20111102 ();
  native_types ();
  hexadecimal ();
  binary ();
  decimal ();
  mixed ();
  check_emax ();
  check_emin ();
  test20161214 ();

#if defined(HAVE_LOCALE_H) && defined(HAVE_SETLOCALE)
#if MPFR_LCONV_DPTS
  locale_da_DK ();
  /* Avoid a warning by doing the setlocale outside of this #if */
#endif
  setlocale (LC_ALL, "C");
#endif

  if (getenv ("MPFR_CHECK_LIBC_PRINTF"))
    {
      /* check against libc */
      random_double ();
      bug20081214 ();
      bug20080610 ();
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
