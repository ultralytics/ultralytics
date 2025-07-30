/* Test mpf_inp_str.

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
    { "ghi", 16, "0", 0 },

    { "125",    10, "125",  3 },
    { "125e1",  10, "1250", 5 },
    { "12e+2",  10, "1200", 5 },
    { "125e-1", 10, "12.5", 6 },

    {  "ff", 16,  "255", 2 },
    { "-ff", 16, "-255", 3 },
    {  "FF", 16,  "255", 2 },
    { "-FF", 16, "-255", 3 },

    { "100",     16, "256",  3 },
    { "100@1",   16, "4096", 5 },
    { "100@10",  16, "4722366482869645213696", 6 },
    { "100@10", -16, "281474976710656",        6 },
    { "100@-1",  16, "16",   6 },
    { "10000000000000000@-10",  16, "1", 21 },
    { "10000000000@-10",       -16, "1", 15 },

    { "z", 36, "35", 1 },
    { "Z", 36, "35", 1 },
    { "z@1", 36, "1260", 3 },
    { "Z@1", 36, "1260", 3 },

    {  "0",      0,   "0", 1 },
  };

  mpf_t  got, want;
  long   ftell_nread;
  int    i, pre, post, j, got_nread, want_nread;
  FILE   *fp;

  mpf_init (got);
  mpf_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      for (pre = 0; pre <= 3; pre++)
        {
          for (post = 0; post <= 2; post++)
            {
              mpf_set_str_or_abort (want, data[i].want, 10);
              MPF_CHECK_FORMAT (want);

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
              got_nread = mpf_inp_str (got, fp, data[i].base);

              if (got_nread != 0)
                {
                  ftell_nread = ftell (fp);
                  if (got_nread != ftell_nread)
                    {
                      printf ("mpf_inp_str nread wrong\n");
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
                      printf ("mpf_inp_str didn't read to EOF\n");
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
                  printf ("mpf_inp_str nread wrong\n");
                  printf ("  inp         \"%s\"\n", data[i].inp);
                  printf ("  base        %d\n", data[i].base);
                  printf ("  pre         %d\n", pre);
                  printf ("  post        %d\n", post);
                  printf ("  got_nread   %d\n", got_nread);
                  printf ("  want_nread  %d\n", want_nread);
                  abort ();
                }

              MPF_CHECK_FORMAT (got);

              if (mpf_cmp (got, want) != 0)
                {
                  printf ("mpf_inp_str wrong result\n");
                  printf ("  inp   \"%s\"\n", data[i].inp);
                  printf ("  base  %d\n", data[i].base);
                  mpf_trace ("  got ",  got);
                  mpf_trace ("  want", want);
                  abort ();
                }

              ASSERT_ALWAYS (fclose (fp) == 0);
            }
        }
    }

  mpf_clear (got);
  mpf_clear (want);
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
