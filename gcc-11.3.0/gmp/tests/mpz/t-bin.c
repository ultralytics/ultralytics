/* Exercise mpz_bin_ui and mpz_bin_uiui.

Copyright 2000, 2001, 2010, 2012 Free Software Foundation, Inc.

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

/* Default number of generated tests. */
#define COUNT 700

void
try_mpz_bin_ui (mpz_srcptr want, mpz_srcptr n, unsigned long k)
{
  mpz_t  got;

  mpz_init (got);
  mpz_bin_ui (got, n, k);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (got, want) != 0)
    {
      printf ("mpz_bin_ui wrong\n");
      printf ("  n="); mpz_out_str (stdout, 10, n); printf ("\n");
      printf ("  k=%lu\n", k);
      printf ("  got="); mpz_out_str (stdout, 10, got); printf ("\n");
      printf ("  want="); mpz_out_str (stdout, 10, want); printf ("\n");
      abort();
    }
  mpz_clear (got);
}


void
try_mpz_bin_uiui (mpz_srcptr want, unsigned long n, unsigned long k)
{
  mpz_t  got;

  mpz_init (got);
  mpz_bin_uiui (got, n, k);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (got, want) != 0)
    {
      printf ("mpz_bin_uiui wrong\n");
      printf ("  n=%lu\n", n);
      printf ("  k=%lu\n", k);
      printf ("  got="); mpz_out_str (stdout, 10, got); printf ("\n");
      printf ("  want="); mpz_out_str (stdout, 10, want); printf ("\n");
      abort();
    }
  mpz_clear (got);
}


void
samples (void)
{
  static const struct {
    const char     *n;
    unsigned long  k;
    const char     *want;
  } data[] = {

    {   "0", 123456, "0" },
    {   "1", 543210, "0" },
    {   "2", 123321, "0" },
    {   "3", 234567, "0" },
    {   "10", 23456, "0" },

    /* negatives, using bin(-n,k)=bin(n+k-1,k) */
    {   "-1",  0,  "1"  },
    {   "-1",  1, "-1"  },
    {   "-1",  2,  "1"  },
    {   "-1",  3, "-1"  },
    {   "-1",  4,  "1"  },

    {   "-2",  0,  "1"  },
    {   "-2",  1, "-2"  },
    {   "-2",  2,  "3"  },
    {   "-2",  3, "-4"  },
    {   "-2",  4,  "5"  },
    {   "-2",  5, "-6"  },
    {   "-2",  6,  "7"  },

    {   "-3",  0,   "1"  },
    {   "-3",  1,  "-3"  },
    {   "-3",  2,   "6"  },
    {   "-3",  3, "-10"  },
    {   "-3",  4,  "15"  },
    {   "-3",  5, "-21"  },
    {   "-3",  6,  "28"  },

    /* A few random values */
    {   "41", 20,  "269128937220" },
    {   "62", 37,  "147405545359541742" },
    {   "50", 18,  "18053528883775" },
    {  "149", 21,  "19332950844468483467894649" },
  };

  mpz_t  n, want;
  int    i;

  mpz_init (n);
  mpz_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (n, data[i].n, 0);
      mpz_set_str_or_abort (want, data[i].want, 0);

      try_mpz_bin_ui (want, n, data[i].k);

      if (mpz_fits_ulong_p (n))
	try_mpz_bin_uiui (want, mpz_get_ui (n), data[i].k);
    }

  mpz_clear (n);
  mpz_clear (want);
}


/* Test some bin(2k,k) cases.  This produces some biggish numbers to
   exercise the limb accumulating code.  */
void
twos (int count)
{
  mpz_t          n, want;
  unsigned long  k;

  mpz_init (n);
  mpz_init (want);

  mpz_set_ui (want, (unsigned long) 2);
  for (k = 1; k < count; k++)
    {
      mpz_set_ui (n, 2*k);
      try_mpz_bin_ui (want, n, k);

      try_mpz_bin_uiui (want, 2*k, k);

      mpz_mul_ui (want, want, 2*(2*k+1));
      mpz_fdiv_q_ui (want, want, k+1);
    }

  mpz_clear (n);
  mpz_clear (want);
}

/* Test some random bin(n,k) cases.  This produces some biggish
   numbers to exercise the limb accumulating code.  */
void
randomwalk (int count)
{
  mpz_t          n_z, want;
  unsigned long  n, k, i, r;
  int            tests;
  gmp_randstate_ptr rands;

  rands = RANDS;
  mpz_init (n_z);
  mpz_init (want);

  k = 3;
  n = 12;
  mpz_set_ui (want, (unsigned long) 220); /* binomial(12,3) = 220 */

  for (tests = 1; tests < count; tests++)
    {
      r = gmp_urandomm_ui (rands, 62) + 1;
      for (i = r & 7; i > 0; i--)
	{
	  n++; k++;
	  mpz_mul_ui (want, want, n);
	  mpz_fdiv_q_ui (want, want, k);
	}
      for (i = r >> 3; i > 0; i--)
	{
	  n++;
	  mpz_mul_ui (want, want, n);
	  mpz_fdiv_q_ui (want, want, n - k);
	}

      mpz_set_ui (n_z, n);
      try_mpz_bin_ui (want, n_z, k);

      try_mpz_bin_uiui (want, n, k);
    }

  mpz_clear (n_z);
  mpz_clear (want);
}


/* Test all bin(n,k) cases, with 0 <= k <= n + 1 <= count.  */
void
smallexaustive (unsigned int count)
{
  mpz_t          n_z, want;
  unsigned long  n, k;

  mpz_init (n_z);
  mpz_init (want);

  for (n = 0; n < count; n++)
    {
      mpz_set_ui (want, (unsigned long) 1);
      mpz_set_ui (n_z, n);
      for (k = 0; k <= n; k++)
	{
	  try_mpz_bin_ui (want, n_z, k);
	  try_mpz_bin_uiui (want, n, k);
	  mpz_mul_ui (want, want, n - k);
	  mpz_fdiv_q_ui (want, want, k + 1);
	}
      try_mpz_bin_ui (want, n_z, k);
      try_mpz_bin_uiui (want, n, k);
    }

  mpz_clear (n_z);
  mpz_clear (want);
}

int
main (int argc, char **argv)
{
  int count;

  if (argc > 1)
    {
      char *end;
      count = strtol (argv[1], &end, 0);
      if (*end || count <= 0)
	{
	  fprintf (stderr, "Invalid test count: %s.\n", argv[1]);
	  return 1;
	}
    }
  else
    count = COUNT;

  tests_start ();

  samples ();
  smallexaustive (count >> 4);
  twos (count >> 1);
  randomwalk (count - (count >> 1));

  tests_end ();
  exit (0);
}
