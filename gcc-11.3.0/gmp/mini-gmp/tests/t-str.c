/*

Copyright 2012, 2013 Free Software Foundation, Inc.

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

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "testutils.h"

#define MAXBITS 400
#define COUNT 2000

#define GMP_LIMB_BITS (sizeof(mp_limb_t) * CHAR_BIT)
#define MAXLIMBS ((MAXBITS + GMP_LIMB_BITS - 1) / GMP_LIMB_BITS)

static void
test_small (void)
{
  struct {
    const char *input;
    const char *decimal;
  } data[] = {
    { "183407", "183407" },
    { " 763959", "763959" },
    { "9 81999", "981999" },
    { "10\t7398", "107398" },
    { "-9585 44", "-00958544" },
    { "-0", "0000" },
    { " -000  ", "0" },
    { "0704436", "231710" },
    { " 02503517", "689999" },
    { "0 1312143", "365667" },
    { "-03 274062", "-882738" },
    { "012\t242", "005282" },
    { "0b11010111110010001111", "883855" },
    { " 0b11001010010100001", "103585" },
    { "-0b101010110011101111", "-175343" },
    { "0b 1111111011011100110", "521958" },
    { "0b1 1111110111001000011", "1044035" },
    { " 0x53dfc", "343548" },
    { "0xfA019", "1024025" },
    { "0x 642d1", "410321" },
    { "0x5 8067", "360551" },
    { "-0xd6Be6", "-879590" },
    { "\t0B1110000100000000011", "460803" },
    { "0B\t1111110010010100101", "517285" },
    { "0B1\t010111101101110100", "359284" },
    { "-0B101\t1001101111111001", "-367609" },
    { "0B10001001010111110000", "562672" },
    { "0Xe4B7e", "936830" },
    { "0X1E4bf", "124095" },
    { "-0Xfdb90", "-1039248" },
    { "0X7fc47", "523335" },
    { "0X8167c", "530044" },
    /* Some invalid inputs */
    { "0ab", NULL },
    { "10x0", NULL },
    { "0xxab", NULL },
    { "ab", NULL },
    { "0%#", NULL },
    { "$foo", NULL },
    { NULL, NULL }
  };
  unsigned i;
  mpz_t a, b;
  mpz_init (b);

  for (i = 0; data[i].input; i++)
    {
      int res = mpz_init_set_str (a, data[i].input, 0);
      if (data[i].decimal)
	{
	  if (res != 0)
	    {
	      fprintf (stderr, "mpz_set_str returned -1, input: %s\n",
		       data[i].input);
	      abort ();
	    }
	  if (mpz_set_str (b, data[i].decimal, 10) != 0)
	    {
	      fprintf (stderr, "mpz_set_str returned -1, decimal input: %s\n",
		       data[i].input);
	      abort ();
	    }
	  if (mpz_cmp (a, b) != 0)
	    {
	      fprintf (stderr, "mpz_set_str failed for input: %s\n",
		       data[i].input);

	      dump ("got", a);
	      dump ("ref", b);
	      abort ();
	    }
	}
      else if (res != -1)
	{
	  fprintf (stderr, "mpz_set_str returned %d, invalid input: %s\n",
		   res, data[i].input);
	  abort ();
	}
      mpz_clear (a);
    }

  mpz_clear (b);
}

void
testmain (int argc, char **argv)
{
  unsigned i;
  char *ap;
  char *bp;
  char *rp;
  size_t bn, rn, arn;

  mpz_t a, b;

  FILE *tmp;

  test_small ();

  mpz_init (a);
  mpz_init (b);

  tmp = tmpfile ();
  if (!tmp)
    fprintf (stderr,
	     "Failed to create temporary file. Skipping mpz_out_str tests.\n");

  for (i = 0; i < COUNT; i++)
    {
      int base;
      for (base = 0; base <= 36; base += 1 + (base == 0))
	{
	  hex_random_str_op (MAXBITS, i&1 ? base: -base, &ap, &rp);
	  if (mpz_set_str (a, ap, 16) != 0)
	    {
	      fprintf (stderr, "mpz_set_str failed on input %s\n", ap);
	      abort ();
	    }

	  rn = strlen (rp);
	  arn = rn - (rp[0] == '-');

	  bn = mpz_sizeinbase (a, base ? base : 10);
	  if (bn < arn || bn > (arn + 1))
	    {
	      fprintf (stderr, "mpz_sizeinbase failed:\n");
	      dump ("a", a);
	      fprintf (stderr, "r = %s\n", rp);
	      fprintf (stderr, "  base %d, correct size %u, got %u\n",
		       base, (unsigned) arn, (unsigned)bn);
	      abort ();
	    }
	  bp = mpz_get_str (NULL, i&1 ? base: -base, a);
	  if (strcmp (bp, rp))
	    {
	      fprintf (stderr, "mpz_get_str failed:\n");
	      dump ("a", a);
	      fprintf (stderr, "b = %s\n", bp);
	      fprintf (stderr, "  base = %d\n", base);
	      fprintf (stderr, "r = %s\n", rp);
	      abort ();
	    }

	  /* Just a few tests with file i/o. */
	  if (tmp && i < 20)
	    {
	      size_t tn;
	      rewind (tmp);
	      tn = mpz_out_str (tmp, i&1 ? base: -base, a);
	      if (tn != rn)
		{
		  fprintf (stderr, "mpz_out_str, bad return value:\n");
		  dump ("a", a);
		  fprintf (stderr, "r = %s\n", rp);
		  fprintf (stderr, "  base %d, correct size %u, got %u\n",
			   base, (unsigned) rn, (unsigned)tn);
		  abort ();
		}
	      rewind (tmp);
	      memset (bp, 0, rn);
	      tn = fread (bp, 1, rn, tmp);
	      if (tn != rn)
		{
		  fprintf (stderr,
			   "fread failed, expected %lu bytes, got only %lu.\n",
			   (unsigned long) rn, (unsigned long) tn);
		  abort ();
		}

	      if (memcmp (bp, rp, rn) != 0)
		{
		  fprintf (stderr, "mpz_out_str failed:\n");
		  dump ("a", a);
		  fprintf (stderr, "b = %s\n", bp);
		  fprintf (stderr, "  base = %d\n", base);
		  fprintf (stderr, "r = %s\n", rp);
		  abort ();
		}
	    }

	  mpz_set_str (b, rp, base);

	  if (mpz_cmp (a, b))
	    {
	      fprintf (stderr, "mpz_set_str failed:\n");
	      fprintf (stderr, "r = %s\n", rp);
	      fprintf (stderr, "  base = %d\n", base);
	      fprintf (stderr, "r = %s\n", ap);
	      fprintf (stderr, "  base = 16\n");
	      dump ("b", b);
	      dump ("r", a);
	      abort ();
	    }

	  /* Test mpn interface */
	  if (base && mpz_sgn (a))
	    {
	      size_t i;
	      const char *absr;
	      mp_limb_t t[MAXLIMBS];
	      mp_size_t tn = mpz_size (a);

	      assert (tn <= MAXLIMBS);
	      mpn_copyi (t, a->_mp_d, tn);

	      bn = mpn_get_str ((unsigned char *) bp, base, t, tn);
	      if (bn != arn)
		{
		  fprintf (stderr, "mpn_get_str failed:\n");
		  fprintf (stderr, "returned length: %lu (bad)\n", (unsigned long) bn);
		  fprintf (stderr, "expected: %lu\n", (unsigned long) arn);
		  fprintf (stderr, "  base = %d\n", base);
		  fprintf (stderr, "r = %s\n", ap);
		  fprintf (stderr, "  base = 16\n");
		  dump ("b", b);
		  dump ("r", a);
		  abort ();
		}
	      absr = rp + (rp[0] == '-');

	      for (i = 0; i < bn; i++)
		{
		  unsigned char digit = absr[i];
		  unsigned value;
		  if (digit >= '0' && digit <= '9')
		    value = digit - '0';
		  else if (digit >= 'a' && digit <= 'z')
		    value = digit - 'a' + 10;
		  else if (digit >= 'A' && digit <= 'Z')
		    value = digit - 'A' + 10;
		  else
		    {
		      fprintf (stderr, "Internal error in test.\n");
		      abort();
		    }
		  if (bp[i] != value)
		    {
		      fprintf (stderr, "mpn_get_str failed:\n");
		      fprintf (stderr, "digit %lu: %d (bad)\n", (unsigned long) i, bp[i]);
		      fprintf (stderr, "expected: %d\n", value);
		      fprintf (stderr, "  base = %d\n", base);
		      fprintf (stderr, "r = %s\n", ap);
		      fprintf (stderr, "  base = 16\n");
		      dump ("b", b);
		      dump ("r", a);
		      abort ();
		    }
		}
	      tn = mpn_set_str (t, (unsigned char *) bp, bn, base);
	      if (tn != mpz_size (a) || mpn_cmp (t, a->_mp_d, tn))
		{
		  fprintf (stderr, "mpn_set_str failed:\n");
		  fprintf (stderr, "r = %s\n", rp);
		  fprintf (stderr, "  base = %d\n", base);
		  fprintf (stderr, "r = %s\n", ap);
		  fprintf (stderr, "  base = 16\n");
		  dump ("r", a);
		  abort ();
		}
	    }
	  free (ap);
	  testfree (bp);
	}
    }
  mpz_clear (a);
  mpz_clear (b);
}
