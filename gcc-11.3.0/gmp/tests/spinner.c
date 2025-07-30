/* A stupid little spinning wheel designed to make it look like useful work
   is being done.

Copyright 1999-2001 Free Software Foundation, Inc.

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

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#if HAVE_UNISTD_H
#include <unistd.h>     /* for isatty */
#endif

#include "gmp.h"
#include "gmp-impl.h"

#include "tests.h"


/* "alarm" is not available on mingw32, and the SIGALRM constant is not
   defined.  Don't bother with a spinner in this case.  */
#if ! HAVE_ALARM || ! defined (SIGALRM)
#define alarm(n)          abort()
#define signal(sig,func)  SIG_ERR
#endif


/* An application can update this to get a count printed with the spinner.
   If left at 0, no count is printed. */

unsigned long  spinner_count = 0;


int  spinner_wanted = -1;  /* -1 uninitialized, 1 wanted, 0 not */
int  spinner_tick = 1;     /* 1 ready to print, 0 not */


/*ARGSUSED*/
RETSIGTYPE
spinner_signal (int signum)
{
  spinner_tick = 1;

  if (signal (SIGALRM, spinner_signal) == SIG_ERR)
    {
      printf ("spinner_signal(): Oops, cannot reinstall SIGALRM\n");
      abort ();
    }
  alarm (1);
}


/* Initialize the spinner.

   This is done the first time spinner() is called, so an application
   doesn't need to call this directly.

   The spinner is only wanted if the output is a tty.  */

#define SPINNER_WANTED_INIT() \
  if (spinner_wanted < 0) spinner_init ()

void
spinner_init (void)
{
  spinner_wanted = isatty (fileno (stdout));
  if (spinner_wanted == -1)
    abort ();

  if (!spinner_wanted)
    return;

  if (signal (SIGALRM, spinner_signal) == SIG_ERR)
    {
      printf ("(no spinner)\r");
      spinner_tick = 0;
      return;
    }
  alarm (1);

  /* unbuffered output so the spinner will show up */
  setbuf (stdout, NULL);
}


void
spinner (void)
{
  static const char  data[] = { '|', '/', '-', '\\' };
  static int         pos = 0;

  char  buf[128];

  SPINNER_WANTED_INIT ();

  if (spinner_tick)
    {
      buf[0] = data[pos];
      pos = (pos + 1) % numberof (data);
      spinner_tick = 0;

      if (spinner_count != 0)
	{
	  sprintf (buf+1, " %lu\r", spinner_count);
	}
      else
	{
	  buf[1] = '\r';
	  buf[2] = '\0';
	}
      fputs (buf, stdout);
    }
}
