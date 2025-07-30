/* Test mpz_inp_str.

Copyright 2001, 2002 Free Software Foundation, Inc.

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
#include <unistd.h>		/* for unlink */
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


#define FILENAME  "t-inp_str.tmp"


void
check_data (void)
{
  static const struct {
    const char  *inp;
    int         base;
    const char  *want;
    int         want_nread;

  } data[] = {

    { "0",   10, "0", 1 },

    { "abc", 10, "0", 0 },
    { "0xf", 10, "0", 1 },
    { "ghi", 16, "0", 0 },
    { "100", 90, "0", 0 },

    {  "ff", 16,  "255", 2 },
    { "-ff", 16, "-255", 3 },
    {  "FF", 16,  "255", 2 },
    { "-FF", 16, "-255", 3 },

    {  "z", 36, "35", 1 },
    {  "Z", 36, "35", 1 },
    { "1B", 59, "70", 2 },
    {  "a", 60, "36", 1 },
    {  "A", 61, "10", 1 },

    {  "0x0",    0,   "0", 3 },
    {  "0X10",   0,  "16", 4 },
    { "-0X0",    0,   "0", 4 },
    { "-0x10",   0, "-16", 5 },

    {  "0b0",    0,  "0", 3 },
    {  "0B10",   0,  "2", 4 },
    { "-0B0",    0,  "0", 4 },
    { "-0b10",   0, "-2", 5 },

    {  "00",   0,  "0", 2 },
    {  "010",  0,  "8", 3 },
    { "-00",   0,  "0", 3 },
    { "-010",  0, "-8", 4 },

    {  "0x",     0,   "0", 2 },
    {  "0",      0,   "0", 1 },
    { " 030",   10,  "30", 4 },
  };

  mpz_t  got, want;
  long   ftell_nread;
  int    i, pre, post, j, got_nread, want_nread;
  FILE   *fp;

  mpz_init (got);
  mpz_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      for (pre = 0; pre <= 3; pre++)
	{
	  for (post = 0; post <= 2; post++)
	    {
	      mpz_set_str_or_abort (want, data[i].want, 0);
	      MPZ_CHECK_FORMAT (want);

	      /* create the file new each time to ensure its length is what
		 we want */
	      fp = fopen (FILENAME, "w+");
	      ASSERT_ALWAYS (fp != NULL);
	      for (j = 0; j < pre; j++)
		putc (' ', fp);
	      fputs (data[i].inp, fp);
	      for (j = 0; j < post; j++)
		putc (' ', fp);
	      fflush (fp);
	      ASSERT_ALWAYS (! ferror(fp));

	      rewind (fp);
	      got_nread = mpz_inp_str (got, fp, data[i].base);

	      if (got_nread != 0)
		{
		  ftell_nread = ftell (fp);
		  if (got_nread != ftell_nread)
		    {
		      printf ("mpz_inp_str nread wrong\n");
		      printf ("  inp          \"%s\"\n", data[i].inp);
		      printf ("  base         %d\n", data[i].base);
		      printf ("  pre          %d\n", pre);
		      printf ("  post         %d\n", post);
		      printf ("  got_nread    %d\n", got_nread);
		      printf ("  ftell_nread  %ld\n", ftell_nread);
		      abort ();
		    }
		}

	      /* if data[i].inp is a whole string to read and there's no post
		 whitespace then expect to have EOF */
	      if (post == 0 && data[i].want_nread == strlen(data[i].inp))
		{
		  int  c = getc(fp);
		  if (c != EOF)
		    {
		      printf ("mpz_inp_str didn't read to EOF\n");
		      printf ("  inp   \"%s\"\n", data[i].inp);
		      printf ("  base  %d\n", data[i].base);
		      printf ("  pre   %d\n", pre);
		      printf ("  post  %d\n", post);
		      printf ("  c     '%c' %#x\n", c, c);
		      abort ();
		    }
		}

	      /* only expect "pre" included in the count when non-zero */
	      want_nread = data[i].want_nread;
	      if (want_nread != 0)
		want_nread += pre;

	      if (got_nread != want_nread)
		{
		  printf ("mpz_inp_str nread wrong\n");
		  printf ("  inp         \"%s\"\n", data[i].inp);
		  printf ("  base        %d\n", data[i].base);
		  printf ("  pre         %d\n", pre);
		  printf ("  post        %d\n", post);
		  printf ("  got_nread   %d\n", got_nread);
		  printf ("  want_nread  %d\n", want_nread);
		  abort ();
		}

	      MPZ_CHECK_FORMAT (got);

	      if (mpz_cmp (got, want) != 0)
		{
		  printf ("mpz_inp_str wrong result\n");
		  printf ("  inp   \"%s\"\n", data[i].inp);
		  printf ("  base  %d\n", data[i].base);
		  mpz_trace ("  got ",  got);
		  mpz_trace ("  want", want);
		  abort ();
		}

	      ASSERT_ALWAYS (fclose (fp) == 0);
	    }
	}
    }

  mpz_clear (got);
  mpz_clear (want);
}

int
main (void)
{
  tests_start ();

  check_data ();

  unlink (FILENAME);
  tests_end ();
  exit (0);
}
