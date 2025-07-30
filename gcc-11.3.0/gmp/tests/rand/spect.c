/* spect.c -- the spectral test */

/*
Copyright 1999 Free Software Foundation, Inc.

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

/* T is upper dimension.  Z_A is the LC multiplier, which is
   relatively prime to Z_M, the LC modulus.  The result is put in
   rop[] with v[t] in rop[t-2]. */

/* BUGS: Due to lazy allocation scheme, maximum T is hard coded to MAXT. */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "gmp.h"

#include "gmpstat.h"

int g_debug = 0;

int
main (int argc, char *argv[])
{
  const char usage[] = "usage: spect [-d] a m n\n";
  int c;
  unsigned int n;
  mpz_t a, m;
  mpf_t res[GMP_SPECT_MAXT], res_min[GMP_SPECT_MAXT], f_tmp;
  int f;


  mpz_init (a);
  mpz_init (m);
  for (f = 0; f < GMP_SPECT_MAXT; f++)
    {
      mpf_init (res[f]);
      mpf_init (res_min[f]);
    }
  mpf_init (f_tmp);
  mpf_set_ui (res_min[0], 32768); /* 2^15 */
  mpf_set_ui (res_min[1], 1024); /* 2^10 */
  mpf_set_ui (res_min[2], 256); /* 2^8 */
  mpf_set_ui (res_min[3], 64); /* 2^6 */
  mpf_set_ui (res_min[4], 32); /* 2^5 */

  while ((c = getopt (argc, argv, "dh")) != -1)
    switch (c)
      {
      case 'd':			/* debug */
	g_debug++;
	break;
      case 'h':
      default:
	fputs (usage, stderr);
	exit (1);
      }
  argc -= optind;
  argv += optind;

  if (argc < 3)
    {
      fputs (usage, stderr);
      exit (1);
    }

  mpz_set_str (a, argv[0], 0);
  mpz_set_str (m, argv[1], 0);
  n = (unsigned int) atoi (argv[2]);
  if (n + 1 > GMP_SPECT_MAXT)
    n = GMP_SPECT_MAXT + 1;

  spectral_test (res, n, a, m);

  for (f = 0; f < n - 1; f++)
    {
      /* print v */
      printf ("%d: v = ", f + 2);
      mpf_out_str (stdout, 10, 4, res[f]);

#ifdef PRINT_RAISED_BY_TWO_AS_WELL
      printf (" (^2 = ");
      mpf_mul (f_tmp, res[f], res[f]);
      mpf_out_str (stdout, 10, 4, f_tmp);
      printf (")");
#endif /* PRINT_RAISED_BY_TWO_AS_WELL */

      /* print merit */
      printf (" m = ");
      merit (f_tmp, f + 2, res[f], m);
      mpf_out_str (stdout, 10, 4, f_tmp);

      if (mpf_cmp (res[f], res_min[f]) < 0)
	printf ("\t*** v too low ***");
      if (mpf_get_d (f_tmp) < .1)
	printf ("\t*** merit too low ***");

      puts ("");
    }

  mpz_clear (a);
  mpz_clear (m);
  for (f = 0; f < GMP_SPECT_MAXT; f++)
    {
      mpf_clear (res[f]);
      mpf_clear (res_min[f]);
    }
  mpf_clear (f_tmp);

  return 0;
}


void
debug_foo()
{
  if (0)
    {
      mpz_dump (0);
      mpf_dump (0);
    }
}
