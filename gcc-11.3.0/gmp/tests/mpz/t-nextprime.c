/* Test mpz_nextprime.

Copyright 2009 Free Software Foundation, Inc.

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

void
refmpz_nextprime (mpz_ptr p, mpz_srcptr t)
{
  mpz_add_ui (p, t, 1L);
  while (! mpz_probab_prime_p (p, 10))
    mpz_add_ui (p, p, 1L);
}

void
run (const char *start, int reps, const char *end, short diffs[])
{
  mpz_t x, y;
  int i;

  mpz_init_set_str (x, start, 0);
  mpz_init (y);

  for (i = 0; i < reps; i++)
    {
      mpz_nextprime (y, x);
      mpz_sub (x, y, x);
      if (diffs != NULL && diffs[i] != mpz_get_ui (x))
	{
	  gmp_printf ("diff list discrepancy\n");
	  abort ();
	}
      mpz_set (x, y);
    }

  mpz_set_str (y, end, 0);

  if (mpz_cmp (x, y) != 0)
    {
      gmp_printf ("got  %Zx\n", x);
      gmp_printf ("want %Zx\n", y);
      abort ();
    }

  mpz_clear (y);
  mpz_clear (x);
}

extern short diff1[];
extern short diff3[];
extern short diff4[];
extern short diff5[];

int
main (int argc, char **argv)
{
  int i;
  int reps = 20;
  gmp_randstate_ptr rands;
  mpz_t bs, x, nxtp, ref_nxtp;
  unsigned long size_range;

  tests_start();
  rands = RANDS;

  run ("2", 1000, "0x1ef7", diff1);

  run ("3", 1000 - 1, "0x1ef7", NULL);

  run ("0x8a43866f5776ccd5b02186e90d28946aeb0ed914", 50,
       "0x8a43866f5776ccd5b02186e90d28946aeb0eeec5", diff3);

  run ("0x10000000000000000000000000000000000000", 50,
       "0x100000000000000000000000000000000010ab", diff4);

  run ("0x1c2c26be55317530311facb648ea06b359b969715db83292ab8cf898d8b1b", 50,
       "0x1c2c26be55317530311facb648ea06b359b969715db83292ab8cf898da957", diff5);

  mpz_init (bs);
  mpz_init (x);
  mpz_init (nxtp);
  mpz_init (ref_nxtp);

  if (argc == 2)
     reps = atoi (argv[1]);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 8 + 2; /* 0..1024 bit operands */

      mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (x, rands, mpz_get_ui (bs));

/*      gmp_printf ("%ld: %Zd\n", mpz_sizeinbase (x, 2), x); */

      mpz_nextprime (nxtp, x);
      refmpz_nextprime (ref_nxtp, x);
      if (mpz_cmp (nxtp, ref_nxtp) != 0)
	abort ();
    }

  mpz_clear (bs);
  mpz_clear (x);
  mpz_clear (nxtp);
  mpz_clear (ref_nxtp);

  tests_end ();
  return 0;
}

short diff1[] =
{
  1,2,2,4,2,4,2,4,6,2,6,4,2,4,6,6,
  2,6,4,2,6,4,6,8,4,2,4,2,4,14,4,6,
  2,10,2,6,6,4,6,6,2,10,2,4,2,12,12,4,
  2,4,6,2,10,6,6,6,2,6,4,2,10,14,4,2,
  4,14,6,10,2,4,6,8,6,6,4,6,8,4,8,10,
  2,10,2,6,4,6,8,4,2,4,12,8,4,8,4,6,
  12,2,18,6,10,6,6,2,6,10,6,6,2,6,6,4,
  2,12,10,2,4,6,6,2,12,4,6,8,10,8,10,8,
  6,6,4,8,6,4,8,4,14,10,12,2,10,2,4,2,
  10,14,4,2,4,14,4,2,4,20,4,8,10,8,4,6,
  6,14,4,6,6,8,6,12,4,6,2,10,2,6,10,2,
  10,2,6,18,4,2,4,6,6,8,6,6,22,2,10,8,
  10,6,6,8,12,4,6,6,2,6,12,10,18,2,4,6,
  2,6,4,2,4,12,2,6,34,6,6,8,18,10,14,4,
  2,4,6,8,4,2,6,12,10,2,4,2,4,6,12,12,
  8,12,6,4,6,8,4,8,4,14,4,6,2,4,6,2,
  6,10,20,6,4,2,24,4,2,10,12,2,10,8,6,6,
  6,18,6,4,2,12,10,12,8,16,14,6,4,2,4,2,
  10,12,6,6,18,2,16,2,22,6,8,6,4,2,4,8,
  6,10,2,10,14,10,6,12,2,4,2,10,12,2,16,2,
  6,4,2,10,8,18,24,4,6,8,16,2,4,8,16,2,
  4,8,6,6,4,12,2,22,6,2,6,4,6,14,6,4,
  2,6,4,6,12,6,6,14,4,6,12,8,6,4,26,18,
  10,8,4,6,2,6,22,12,2,16,8,4,12,14,10,2,
  4,8,6,6,4,2,4,6,8,4,2,6,10,2,10,8,
  4,14,10,12,2,6,4,2,16,14,4,6,8,6,4,18,
  8,10,6,6,8,10,12,14,4,6,6,2,28,2,10,8,
  4,14,4,8,12,6,12,4,6,20,10,2,16,26,4,2,
  12,6,4,12,6,8,4,8,22,2,4,2,12,28,2,6,
  6,6,4,6,2,12,4,12,2,10,2,16,2,16,6,20,
  16,8,4,2,4,2,22,8,12,6,10,2,4,6,2,6,
  10,2,12,10,2,10,14,6,4,6,8,6,6,16,12,2,
  4,14,6,4,8,10,8,6,6,22,6,2,10,14,4,6,
  18,2,10,14,4,2,10,14,4,8,18,4,6,2,4,6,
  2,12,4,20,22,12,2,4,6,6,2,6,22,2,6,16,
  6,12,2,6,12,16,2,4,6,14,4,2,18,24,10,6,
  2,10,2,10,2,10,6,2,10,2,10,6,8,30,10,2,
  10,8,6,10,18,6,12,12,2,18,6,4,6,6,18,2,
  10,14,6,4,2,4,24,2,12,6,16,8,6,6,18,16,
  2,4,6,2,6,6,10,6,12,12,18,2,6,4,18,8,
  24,4,2,4,6,2,12,4,14,30,10,6,12,14,6,10,
  12,2,4,6,8,6,10,2,4,14,6,6,4,6,2,10,
  2,16,12,8,18,4,6,12,2,6,6,6,28,6,14,4,
  8,10,8,12,18,4,2,4,24,12,6,2,16,6,6,14,
  10,14,4,30,6,6,6,8,6,4,2,12,6,4,2,6,
  22,6,2,4,18,2,4,12,2,6,4,26,6,6,4,8,
  10,32,16,2,6,4,2,4,2,10,14,6,4,8,10,6,
  20,4,2,6,30,4,8,10,6,6,8,6,12,4,6,2,
  6,4,6,2,10,2,16,6,20,4,12,14,28,6,20,4,
  18,8,6,4,6,14,6,6,10,2,10,12,8,10,2,10,
  8,12,10,24,2,4,8,6,4,8,18,10,6,6,2,6,
  10,12,2,10,6,6,6,8,6,10,6,2,6,6,6,10,
  8,24,6,22,2,18,4,8,10,30,8,18,4,2,10,6,
  2,6,4,18,8,12,18,16,6,2,12,6,10,2,10,2,
  6,10,14,4,24,2,16,2,10,2,10,20,4,2,4,8,
  16,6,6,2,12,16,8,4,6,30,2,10,2,6,4,6,
  6,8,6,4,12,6,8,12,4,14,12,10,24,6,12,6,
  2,22,8,18,10,6,14,4,2,6,10,8,6,4,6,30,
  14,10,2,12,10,2,16,2,18,24,18,6,16,18,6,2,
  18,4,6,2,10,8,10,6,6,8,4,6,2,10,2,12,
  4,6,6,2,12,4,14,18,4,6,20,4,8,6,4,8,
  4,14,6,4,14,12,4,2,30,4,24,6,6,12,12,14,
  6,4,2,4,18,6,12,8
};

short diff3[] =
{
  33,32,136,116,24,22,104,114,76,278,238,162,36,44,388,134,
  130,26,312,42,138,28,24,80,138,108,270,12,330,130,98,102,
  162,34,36,170,90,34,14,6,24,66,154,218,70,132,188,88,
  80,82
};

short diff4[] =
{
  91,92,64,6,104,24,46,258,68,18,54,100,68,154,26,4,
  38,142,168,42,18,26,286,104,136,116,40,2,28,110,52,78,
  104,24,54,96,4,626,196,24,56,36,52,102,48,156,26,18,
  42,40
};

short diff5[] =
{
  268,120,320,184,396,2,94,108,20,318,274,14,64,122,220,108,
  18,174,6,24,348,32,64,116,268,162,20,156,28,110,52,428,
  196,14,262,30,194,120,300,66,268,12,428,370,212,198,192,130,
  30,80
};
