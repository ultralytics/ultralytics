/* Test mpq_inp_str.

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
#include <unistd.h>   /* for unlink */
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
    { "0/1", 10, "0", 3 },

    { "0/",   10, "0", 0 },
    { "/123", 10, "0", 0 },
    { "blah", 10, "0", 0 },
    { "123/blah", 10, "0", 0 },
    { "5 /8", 10, "5", 1 },
    { "5/ 8", 10, "0", 0 },

    {  "ff", 16,  "255", 2 },
    { "-ff", 16, "-255", 3 },
    {  "FF", 16,  "255", 2 },
    { "-FF", 16, "-255", 3 },

    { "z", 36, "35", 1 },
    { "Z", 36, "35", 1 },

    {  "0x0",    0,   "0", 3 },
    {  "0x10",   0,  "16", 4 },
    { "-0x0",    0,   "0", 4 },
    { "-0x10",   0, "-16", 5 },
    { "-0x10/5", 0, "-16/5", 7 },

    {  "00",   0,  "0", 2 },
    {  "010",  0,  "8", 3 },
    { "-00",   0,  "0", 3 },
    { "-010",  0, "-8", 4 },
  };

  mpq_t  got, want;
  long   ftell_nread;
  int    i, post, j, got_nread;
  FILE   *fp;

  mpq_init (got);
  mpq_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      for (post = 0; post <= 2; post++)
	{
	  mpq_set_str_or_abort (want, data[i].want, 0);
	  MPQ_CHECK_FORMAT (want);

	  fp = fopen (FILENAME, "w+");
	  ASSERT_ALWAYS (fp != NULL);
	  fputs (data[i].inp, fp);
	  for (j = 0; j < post; j++)
	    putc (' ', fp);
	  fflush (fp);
	  ASSERT_ALWAYS (! ferror(fp));

	  rewind (fp);
	  got_nread = mpq_inp_str (got, fp, data[i].base);

	  if (got_nread != 0)
	    {
	      ftell_nread = ftell (fp);
	      if (got_nread != ftell_nread)
		{
		  printf ("mpq_inp_str nread wrong\n");
		  printf ("  inp          \"%s\"\n", data[i].inp);
		  printf ("  base         %d\n", data[i].base);
		  printf ("  got_nread    %d\n", got_nread);
		  printf ("  ftell_nread  %ld\n", ftell_nread);
		  abort ();
		}
	    }

	  if (post == 0 && data[i].want_nread == strlen(data[i].inp))
	    {
	      int  c = getc(fp);
	      if (c != EOF)
		{
		  printf ("mpq_inp_str didn't read to EOF\n");
		  printf ("  inp         \"%s\"\n", data[i].inp);
		  printf ("  base        %d\n", data[i].base);
		  printf ("  c '%c' %#x\n", c, c);
		  abort ();
		}
	    }

	  if (got_nread != data[i].want_nread)
	    {
	      printf ("mpq_inp_str nread wrong\n");
	      printf ("  inp         \"%s\"\n", data[i].inp);
	      printf ("  base        %d\n", data[i].base);
	      printf ("  got_nread   %d\n", got_nread);
	      printf ("  want_nread  %d\n", data[i].want_nread);
	      abort ();
	    }

	  MPQ_CHECK_FORMAT (got);

	  if (! mpq_equal (got, want))
	    {
	      printf ("mpq_inp_str wrong result\n");
	      printf ("  inp   \"%s\"\n", data[i].inp);
	      printf ("  base  %d\n", data[i].base);
	      mpq_trace ("  got ",  got);
	      mpq_trace ("  want", want);
	      abort ();
	    }

	  ASSERT_ALWAYS (fclose (fp) == 0);
	}
    }

  mpq_clear (got);
  mpq_clear (want);
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
