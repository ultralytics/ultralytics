/* Test mpz_inp_raw and mpz_out_raw.

Copyright 2001 Free Software Foundation, Inc.

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

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#define FILENAME  "t-io_raw.tmp"


/* In the fopen, "b" selects binary mode on DOS systems, meaning no
   conversion of '\n' to and from CRLF.  It's believed systems without such
   nonsense will simply ignore the "b", but in case that's not so a plain
   "w+" is attempted if "w+b" fails.  */

FILE *
fopen_wplusb_or_die (const char *filename)
{
  FILE  *fp;
  fp = fopen (filename, "w+b");
  if (fp == NULL)
    fp = fopen (filename, "w+");

  if (fp == NULL)
    {
      printf ("Cannot create file %s\n", filename);
      abort ();
    }
  return fp;
}

/* use 0x80 to check nothing bad happens with sign extension etc */
#define BYTEVAL(i)  (((i) + 1) | 0x80)

void
check_in (void)
{
  int        i, j, zeros, neg, error = 0;
  mpz_t      want, got;
  size_t     want_ret, got_ret;
  mp_size_t  size;
  FILE       *fp;

  mpz_init (want);
  mpz_init (got);

  for (i = 0; i < 32; i++)
    {
      for (zeros = 0; zeros < 8; zeros++)
	{
	  for (neg = 0; neg <= 1; neg++)
	    {
	      want_ret = i + zeros + 4;

	      /* need this to get the twos complement right */
	      ASSERT_ALWAYS (sizeof (size) >= 4);

	      size = i + zeros;
	      if (neg)
		size = -size;

	      fp = fopen_wplusb_or_die (FILENAME);
	      for (j = 3; j >= 0; j--)
		ASSERT_ALWAYS (putc ((size >> (j*8)) & 0xFF, fp) != EOF);
	      for (j = 0; j < zeros; j++)
		ASSERT_ALWAYS (putc ('\0', fp) != EOF);
	      for (j = 0; j < i; j++)
		ASSERT_ALWAYS (putc (BYTEVAL (j), fp) != EOF);
	      /* and some trailing garbage */
	      ASSERT_ALWAYS (putc ('x', fp) != EOF);
	      ASSERT_ALWAYS (putc ('y', fp) != EOF);
	      ASSERT_ALWAYS (putc ('z', fp) != EOF);
	      ASSERT_ALWAYS (fflush (fp) == 0);
	      rewind (fp);

	      got_ret = mpz_inp_raw (got, fp);
	      ASSERT_ALWAYS (! ferror(fp));
	      ASSERT_ALWAYS (fclose (fp) == 0);

	      MPZ_CHECK_FORMAT (got);

	      if (got_ret != want_ret)
		{
		  printf ("check_in: return value wrong\n");
		  error = 1;
		}
	      if (mpz_cmp (got, want) != 0)
		{
		  printf ("check_in: result wrong\n");
		  error = 1;
		}
	      if (error)
		{
		  printf    ("  i=%d zeros=%d neg=%d\n", i, zeros, neg);
		  printf    ("  got_ret  %lu\n", (unsigned long) got_ret);
		  printf    ("  want_ret %lu\n", (unsigned long) want_ret);
		  mpz_trace ("  got      ", got);
		  mpz_trace ("  want     ", want);
		  abort ();
		}

	      mpz_neg (want, want);
	    }
	}
      mpz_mul_2exp (want, want, 8);
      mpz_add_ui (want, want, (unsigned long) BYTEVAL (i));
    }

  mpz_clear (want);
  mpz_clear (got);
}


void
check_out (void)
{
  int        i, j, neg, error = 0;
  mpz_t      z;
  char       want[256], got[256], *p;
  size_t     want_len, got_ret, got_read;
  mp_size_t  size;
  FILE       *fp;

  mpz_init (z);

  for (i = 0; i < 32; i++)
    {
      for (neg = 0; neg <= 1; neg++)
	{
	  want_len = i + 4;

	  /* need this to get the twos complement right */
	  ASSERT_ALWAYS (sizeof (size) >= 4);

	  size = i;
	  if (neg)
	    size = -size;

	  p = want;
	  for (j = 3; j >= 0; j--)
	    *p++ = size >> (j*8);
	  for (j = 0; j < i; j++)
	    *p++ = BYTEVAL (j);
	  ASSERT_ALWAYS (p <= want + sizeof (want));

	  fp = fopen_wplusb_or_die (FILENAME);
	  got_ret = mpz_out_raw (fp, z);
	  ASSERT_ALWAYS (fflush (fp) == 0);
	  rewind (fp);
	  got_read = fread (got, 1, sizeof(got), fp);
	  ASSERT_ALWAYS (! ferror(fp));
	  ASSERT_ALWAYS (fclose (fp) == 0);

	  if (got_ret != want_len)
	    {
	      printf ("check_out: wrong return value\n");
	      error = 1;
	    }
	  if (got_read != want_len)
	    {
	      printf ("check_out: wrong number of bytes read back\n");
	      error = 1;
	    }
	  if (memcmp (want, got, want_len) != 0)
	    {
	      printf ("check_out: wrong data\n");
	      error = 1;
	    }
	  if (error)
	    {
	      printf    ("  i=%d neg=%d\n", i, neg);
	      mpz_trace ("  z", z);
	      printf    ("  got_ret  %lu\n", (unsigned long) got_ret);
	      printf    ("  got_read %lu\n", (unsigned long) got_read);
	      printf    ("  want_len %lu\n", (unsigned long) want_len);
	      printf    ("  want");
	      for (j = 0; j < want_len; j++)
		printf (" %02X", (unsigned) (unsigned char) want[j]);
	      printf    ("\n");
	      printf    ("  got ");
	      for (j = 0; j < want_len; j++)
		printf (" %02X", (unsigned) (unsigned char) got[j]);
	      printf    ("\n");
	      abort ();
	    }

	  mpz_neg (z, z);
	}
      mpz_mul_2exp (z, z, 8);
      mpz_add_ui (z, z, (unsigned long) BYTEVAL (i));
    }

  mpz_clear (z);
}


void
check_rand (void)
{
  gmp_randstate_ptr  rands = RANDS;
  int        i, error = 0;
  mpz_t      got, want;
  size_t     inp_ret, out_ret;
  FILE       *fp;

  mpz_init (want);
  mpz_init (got);

  for (i = 0; i < 500; i++)
    {
      mpz_erandomb (want, rands, 10*GMP_LIMB_BITS);
      mpz_negrandom (want, rands);

      fp = fopen_wplusb_or_die (FILENAME);
      out_ret = mpz_out_raw (fp, want);
      ASSERT_ALWAYS (fflush (fp) == 0);
      rewind (fp);
      inp_ret = mpz_inp_raw (got, fp);
      ASSERT_ALWAYS (fclose (fp) == 0);

      MPZ_CHECK_FORMAT (got);

      if (inp_ret != out_ret)
	{
	  printf ("check_rand: different inp/out return values\n");
	  error = 1;
	}
      if (mpz_cmp (got, want) != 0)
	{
	  printf ("check_rand: wrong result\n");
	  error = 1;
	}
      if (error)
	{
	  printf    ("  out_ret %lu\n", (unsigned long) out_ret);
	  printf    ("  inp_ret %lu\n", (unsigned long) inp_ret);
	  mpz_trace ("  want", want);
	  mpz_trace ("  got ", got);
	  abort ();
	}
    }

  mpz_clear (got);
  mpz_clear (want);
}


int
main (void)
{
  tests_start ();
  mp_trace_base = -16;

  check_in ();
  check_out ();
  check_rand ();

  unlink (FILENAME);
  tests_end ();

  exit (0);
}
