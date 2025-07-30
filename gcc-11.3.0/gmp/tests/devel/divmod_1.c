/*
Copyright 1996, 1998, 2000, 2001, 2007 Free Software Foundation, Inc.

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

#if defined (USG) || defined (__SVR4) || defined (_UNICOS) || defined (__hpux)
#include <time.h>

int
cputime ()
{
  if (CLOCKS_PER_SEC < 100000)
    return clock () * 1000 / CLOCKS_PER_SEC;
  return clock () / (CLOCKS_PER_SEC / 1000);
}
#else
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

int
cputime ()
{
  struct rusage rus;

  getrusage (0, &rus);
  return rus.ru_utime.tv_sec * 1000 + rus.ru_utime.tv_usec / 1000;
}
#endif

static void mpn_print (mp_ptr, mp_size_t);

#define M * 1000000

#ifndef CLOCK
#error "Don't know CLOCK of your machine"
#endif

#ifndef OPS
#define OPS 20000000
#endif
#ifndef SIZE
#define SIZE 1000
#endif
#ifndef TIMES
#define TIMES OPS/SIZE
#endif

#ifndef FSIZE
#define FSIZE SIZE
#endif

int
main ()
{
  mp_limb_t np[SIZE];
  mp_limb_t dx[SIZE + FSIZE + 2];
  mp_limb_t dy[SIZE + FSIZE + 2];
  mp_limb_t dlimb;
  mp_size_t nn, fn;
  mp_limb_t retx, rety;
  int test;
#if TIMES != 1
  int i;
  long t0, t;
  double cyc;
#endif

  for (test = 0; ; test++)
    {
#if TIMES == 1 && ! defined (PRINT)
      if (test % (SIZE > 100000 ? 1 : 100000 / SIZE) == 0)
	{
	  printf ("\r%u", test);
	  fflush (stdout);
	}
#endif

#ifdef RANDOM
      nn = random () % (SIZE + 1);
      fn = random () % (FSIZE + 1);
#else
      nn = SIZE;
      fn = FSIZE;
#endif

      dx[0] = 0x87654321;
      dx[nn + fn + 1] = 0x12345678;
      dy[0] = 0x87654321;
      dy[nn + fn + 1] = 0x12345678;
      mpn_random2 (np, nn);

#ifdef FIXED_DLIMB
      dlimb = FIXED_DLIMB;
#else
      do
	{
	  mpn_random2 (&dlimb, 1);
#ifdef FORCE_NORM
	  dlimb |= GMP_NUMB_HIGHBIT;
#endif
#ifdef FORCE_UNNORM
	  dlimb &= GMP_NUMB_MAX >> 1;
#endif
	}
      while (dlimb == 0);
#endif

#if defined (PRINT) || defined (XPRINT)
      printf ("N=");
      mpn_print (np, nn);
      printf ("D=");
      mpn_print (&dlimb, 1);
      printf ("nn=%ld\n", (long) nn);
#endif

#if TIMES != 1
      t0 = cputime();
      for (i = 0; i < TIMES; i++)
	mpn_divrem_1 (dx + 1, 0L, np, nn, dlimb);
      t = cputime() - t0;
      cyc = ((double) t * CLOCK) / (TIMES * nn * 1000.0);
      printf ("mpn_divrem_1 int:    %5ldms (%.3f cycles/limb) [%.2f Gb/s]\n",
	      t, cyc,
	      CLOCK/cyc*GMP_LIMB_BITS*GMP_LIMB_BITS/1e9);
      t0 = cputime();
      for (i = 0; i < TIMES; i++)
	mpn_divrem_1 (dx + 1, fn, np, 0, dlimb);
      t = cputime() - t0;
      cyc = ((double) t * CLOCK) / (TIMES * fn * 1000.0);
      printf ("mpn_divrem_1 frac:   %5ldms (%.3f cycles/limb) [%.2f Gb/s]\n",
	      t, cyc,
	      CLOCK/cyc*GMP_LIMB_BITS*GMP_LIMB_BITS/1e9);
#endif

      retx = refmpn_divrem_1 (dx + 1, fn, np, nn, dlimb);
      rety = mpn_divrem_1 (dy + 1, fn, np, nn, dlimb);

#ifndef NOCHECK
      if (retx != rety || mpn_cmp (dx, dy, fn + nn + 2) != 0)
	{
	  printf ("ERROR in test %d, nn=%ld, fn=%ld\n", test, nn, fn);
	  mpn_print (np, nn);
	  mpn_print (&dlimb, 1);
	  printf ("rq: ");
	  mpn_print (dx + 1, nn + fn);
	  printf ("rr: %*lX\n", (int) (2 * sizeof(mp_limb_t)), retx);
	  printf (" q: ");
	  mpn_print (dy + 1, nn + fn);
	  printf (" r: %*lX\n", (int) (2 * sizeof(mp_limb_t)), rety);
	  if (dy[0] != 0x87654321)
	    printf ("clobbered at low end %*lX\n", (int) (2 * sizeof(mp_limb_t)), dy[0]);
	  if (dy[nn + fn + 1] != 0x12345678)
	    printf ("clobbered at high end\n");
	  abort ();
	}
#endif
    }
}

static void
mpn_print (mp_ptr p, mp_size_t size)
{
  mp_size_t i;

  for (i = size - 1; i >= 0; i--)
    {
#ifdef _LONG_LONG_LIMB
      printf ("%0*lX%0*lX", (int) (sizeof(mp_limb_t)),
	      (unsigned long) (p[i] >> (GMP_LIMB_BITS/2)),
              (int) (sizeof(mp_limb_t)), (unsigned long) (p[i]));
#else
      printf ("%0*lX", (int) (2 * sizeof(mp_limb_t)), p[i]);
#endif
#ifdef SPACE
      if (i != 0)
	printf (" ");
#endif
    }
  puts ("");
}
