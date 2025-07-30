/* Test mpz_cmp_d and mpz_cmpabs_d.

Copyright 2001-2003, 2005 Free Software Foundation, Inc.

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
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


/* FIXME: Not sure if the tests here are exhaustive.  Ought to try to get
   each possible exit from mpz_cmp_d (and mpz_cmpabs_d) exercised.  */


#define SGN(n)  ((n) > 0 ? 1 : (n) < 0 ? -1 : 0)


void
check_one (const char *name, mpz_srcptr x, double y, int cmp, int cmpabs)
{
  int   got;

  got = mpz_cmp_d (x, y);
  if (SGN(got) != cmp)
    {
      int i;
      printf    ("mpz_cmp_d wrong (from %s)\n", name);
      printf    ("  got  %d\n", got);
      printf    ("  want %d\n", cmp);
    fail:
      mpz_trace ("  x", x);
      printf    ("  y %g\n", y);
      mp_trace_base=-16;
      mpz_trace ("  x", x);
      printf    ("  y %g\n", y);
      printf    ("  y");
      for (i = 0; i < sizeof(y); i++)
        printf (" %02X", (unsigned) ((unsigned char *) &y)[i]);
      printf ("\n");
      abort ();
    }

  got = mpz_cmpabs_d (x, y);
  if (SGN(got) != cmpabs)
    {
      printf    ("mpz_cmpabs_d wrong\n");
      printf    ("  got  %d\n", got);
      printf    ("  want %d\n", cmpabs);
      goto fail;
    }
}


void
check_data (void)
{
  static const struct {
    const char  *x;
    double      y;
    int         cmp, cmpabs;

  } data[] = {

    {  "0",  0.0,  0,  0 },

    {  "1",  0.0,  1,  1 },
    { "-1",  0.0, -1,  1 },

    {  "1",  0.5,  1,  1 },
    { "-1", -0.5, -1,  1 },

    {  "0",  1.0, -1, -1 },
    {  "0", -1.0,  1, -1 },

    {  "0x1000000000000000000000000000000000000000000000000", 1.0,  1, 1 },
    { "-0x1000000000000000000000000000000000000000000000000", 1.0, -1, 1 },

    {  "0",  1e100, -1, -1 },
    {  "0", -1e100,  1, -1 },

    {  "2",  1.5,   1,  1 },
    {  "2", -1.5,   1,  1 },
    { "-2",  1.5,  -1,  1 },
    { "-2", -1.5,  -1,  1 },
  };

  mpz_t  x;
  int    i;

  mpz_init (x);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (x, data[i].x, 0);
      check_one ("check_data", x, data[i].y, data[i].cmp, data[i].cmpabs);
    }

  mpz_clear (x);
}


/* Equality of integers with up to 53 bits */
void
check_onebits (void)
{
  mpz_t   x, x2;
  double  y;
  int     i;

  mpz_init_set_ui (x, 0L);
  mpz_init (x2);

  for (i = 0; i < 512; i++)
    {
      mpz_mul_2exp (x, x, 1);
      mpz_add_ui (x, x, 1L);

      y = mpz_get_d (x);
      mpz_set_d (x2, y);

      /* stop if any truncation is occurring */
      if (mpz_cmp (x, x2) != 0)
        break;

      check_one ("check_onebits", x, y, 0, 0);
      check_one ("check_onebits", x, -y, 1, 0);
      mpz_neg (x, x);
      check_one ("check_onebits", x, y, -1, 0);
      check_one ("check_onebits", x, -y, 0, 0);
      mpz_neg (x, x);
    }

  mpz_clear (x);
  mpz_clear (x2);
}


/* With the mpz differing by 1, in a limb position possibly below the double */
void
check_low_z_one (void)
{
  mpz_t          x;
  double         y;
  unsigned long  i;

  mpz_init (x);

  /* FIXME: It'd be better to base this on the float format. */
#if defined (__vax) || defined (__vax__)
#define LIM 127			/* vax fp numbers have limited range */
#else
#define LIM 512
#endif

  for (i = 1; i < LIM; i++)
    {
      mpz_set_ui (x, 1L);
      mpz_mul_2exp (x, x, i);
      y = mpz_get_d (x);

      check_one ("check_low_z_one", x, y,   0, 0);
      check_one ("check_low_z_one", x, -y,  1, 0);
      mpz_neg (x, x);
      check_one ("check_low_z_one", x, y,  -1, 0);
      check_one ("check_low_z_one", x, -y,  0, 0);
      mpz_neg (x, x);

      mpz_sub_ui (x, x, 1);

      check_one ("check_low_z_one", x, y,  -1, -1);
      check_one ("check_low_z_one", x, -y,  1, -1);
      mpz_neg (x, x);
      check_one ("check_low_z_one", x, y,  -1, -1);
      check_one ("check_low_z_one", x, -y,  1, -1);
      mpz_neg (x, x);

      mpz_add_ui (x, x, 2);

      check_one ("check_low_z_one", x, y,   1, 1);
      check_one ("check_low_z_one", x, -y,  1, 1);
      mpz_neg (x, x);
      check_one ("check_low_z_one", x, y,  -1, 1);
      check_one ("check_low_z_one", x, -y, -1, 1);
      mpz_neg (x, x);
    }

  mpz_clear (x);
}

/* Comparing 1 and 1+2^-n.  "y" is volatile to make gcc store and fetch it,
   which forces it to a 64-bit double, whereas on x86 it would otherwise
   remain on the float stack as an 80-bit long double.  */
void
check_one_2exp (void)
{
  double           e;
  mpz_t            x;
  volatile double  y;
  int              i;

  mpz_init (x);

  e = 1.0;
  for (i = 0; i < 128; i++)
    {
      e /= 2.0;
      y = 1.0 + e;
      if (y == 1.0)
        break;

      mpz_set_ui (x, 1L);
      check_one ("check_one_2exp", x,  y, -1, -1);
      check_one ("check_one_2exp", x, -y,  1, -1);

      mpz_set_si (x, -1L);
      check_one ("check_one_2exp", x,  y, -1, -1);
      check_one ("check_one_2exp", x, -y,  1, -1);
    }

  mpz_clear (x);
}

void
check_infinity (void)
{
  mpz_t   x;
  double  y = tests_infinity_d ();
  if (y == 0.0)
    return;

  mpz_init (x);

  /* 0 cmp inf */
  mpz_set_ui (x, 0L);
  check_one ("check_infinity", x,  y, -1, -1);
  check_one ("check_infinity", x, -y,  1, -1);

  /* 123 cmp inf */
  mpz_set_ui (x, 123L);
  check_one ("check_infinity", x,  y, -1, -1);
  check_one ("check_infinity", x, -y,  1, -1);

  /* -123 cmp inf */
  mpz_set_si (x, -123L);
  check_one ("check_infinity", x,  y, -1, -1);
  check_one ("check_infinity", x, -y,  1, -1);

  /* 2^5000 cmp inf */
  mpz_set_ui (x, 1L);
  mpz_mul_2exp (x, x, 5000L);
  check_one ("check_infinity", x,  y, -1, -1);
  check_one ("check_infinity", x, -y,  1, -1);

  /* -2^5000 cmp inf */
  mpz_neg (x, x);
  check_one ("check_infinity", x,  y, -1, -1);
  check_one ("check_infinity", x, -y,  1, -1);

  mpz_clear (x);
}

int
main (int argc, char *argv[])
{
  tests_start ();

  check_data ();
  check_onebits ();
  check_low_z_one ();
  check_one_2exp ();
  check_infinity ();

  tests_end ();
  exit (0);
}
