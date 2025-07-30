/*
Copyright 1996-1998, 2000, 2001, 2007, 2009 Free Software Foundation, Inc.

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

#define M * 1000000

#ifndef CLOCK
#error "Don't know CLOCK of your machine"
#endif

#ifndef OPS
#define OPS 20000000
#endif
#ifndef SIZE
#define SIZE 100
#endif
#ifndef TIMES
#define TIMES OPS/(SIZE+1)
#endif

int
main ()
{
  mp_limb_t nptr[2 * SIZE];
  mp_limb_t dptr[2 * SIZE];
  mp_limb_t qptr[2 * SIZE];
  mp_limb_t pptr[2 * SIZE + 1];
  mp_limb_t rptr[2 * SIZE];
  mp_size_t nsize, dsize, qsize, rsize, psize;
  int test;
  mp_limb_t qlimb;

  for (test = 0; ; test++)
    {
      printf ("%d\n", test);
#ifdef RANDOM
      nsize = random () % (2 * SIZE) + 1;
      dsize = random () % nsize + 1;
#else
      nsize = 2 * SIZE;
      dsize = SIZE;
#endif

      mpn_random2 (nptr, nsize);
      mpn_random2 (dptr, dsize);
      dptr[dsize - 1] |= (mp_limb_t) 1 << (GMP_LIMB_BITS - 1);

      MPN_COPY (rptr, nptr, nsize);
      qlimb = mpn_divrem (qptr, (mp_size_t) 0, rptr, nsize, dptr, dsize);
      rsize = dsize;
      qsize = nsize - dsize;
      qptr[qsize] = qlimb;
      qsize += qlimb;
      if (qsize == 0 || qsize > 2 * SIZE)
	{
	  continue;		/* bogus */
	}
      else
	{
	  mp_limb_t cy;
	  if (qsize > dsize)
	    mpn_mul (pptr, qptr, qsize, dptr, dsize);
	  else
	    mpn_mul (pptr, dptr, dsize, qptr, qsize);
	  psize = qsize + dsize;
	  psize -= pptr[psize - 1] == 0;
	  cy = mpn_add (pptr, pptr, psize, rptr, rsize);
	  pptr[psize] = cy;
	  psize += cy;
	}

      if (nsize != psize || mpn_cmp (nptr, pptr, nsize) != 0)
	abort ();
    }
}
