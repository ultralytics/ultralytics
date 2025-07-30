/* gmp_nextprime -- generate small primes reasonably efficiently for internal
   GMP needs.

   Contributed to the GNU project by Torbjorn Granlund.  Miscellaneous
   improvements by Martin Boij.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

/*
  Optimisation ideas:

  1. Unroll the sieving loops.  Should reach 1 write/cycle.  That would be a 2x
     improvement.

  2. Separate sieving with primes p < SIEVESIZE and p >= SIEVESIZE.  The latter
     will need at most one write, and thus not need any inner loop.

  3. For primes p >= SIEVESIZE, i.e., typically the majority of primes, we
     perform more than one division per sieving write.  That might dominate the
     entire run time for the nextprime function.  A incrementally initialised
     remainder table of Pi(65536) = 6542 16-bit entries could replace that
     division.
*/

#include "gmp.h"
#include "gmp-impl.h"
#include <string.h>		/* for memset */


unsigned long int
gmp_nextprime (gmp_primesieve_t *ps)
{
  unsigned long p, d, pi;
  unsigned char *sp;
  static unsigned char addtab[] =
    { 2,4,2,4,6,2,6,4,2,4,6,6,2,6,4,2,6,4,6,8,4,2,4,2,4,8,6,4,6,2,4,6,2,6,6,4,
      2,4,6,2,6,4,2,4,2,10,2,10 };
  unsigned char *addp = addtab;
  unsigned long ai;

  /* Look for already sieved primes.  A sentinel at the end of the sieving
     area allows us to use a very simple loop here.  */
  d = ps->d;
  sp = ps->s + d;
  while (*sp != 0)
    sp++;
  if (sp != ps->s + SIEVESIZE)
    {
      d = sp - ps->s;
      ps->d = d + 1;
      return ps->s0 + 2 * d;
    }

  /* Handle the number 2 separately.  */
  if (ps->s0 < 3)
    {
      ps->s0 = 3 - 2 * SIEVESIZE; /* Tricky */
      return 2;
    }

  /* Exhausted computed primes.  Resieve, then call ourselves recursively.  */

#if 0
  for (sp = ps->s; sp < ps->s + SIEVESIZE; sp++)
    *sp = 0;
#else
  memset (ps->s, 0, SIEVESIZE);
#endif

  ps->s0 += 2 * SIEVESIZE;

  /* Update sqrt_s0 as needed.  */
  while ((ps->sqrt_s0 + 1) * (ps->sqrt_s0 + 1) <= ps->s0 + 2 * SIEVESIZE - 1)
    ps->sqrt_s0++;

  pi = ((ps->s0 + 3) / 2) % 3;
  if (pi > 0)
    pi = 3 - pi;
  if (ps->s0 + 2 * pi <= 3)
    pi += 3;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 3;
    }

  pi = ((ps->s0 + 5) / 2) % 5;
  if (pi > 0)
    pi = 5 - pi;
  if (ps->s0 + 2 * pi <= 5)
    pi += 5;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 5;
    }

  pi = ((ps->s0 + 7) / 2) % 7;
  if (pi > 0)
    pi = 7 - pi;
  if (ps->s0 + 2 * pi <= 7)
    pi += 7;
  sp = ps->s + pi;
  while (sp < ps->s + SIEVESIZE)
    {
      *sp = 1, sp += 7;
    }

  p = 11;
  ai = 0;
  while (p <= ps->sqrt_s0)
    {
      pi = ((ps->s0 + p) / 2) % p;
      if (pi > 0)
	pi = p - pi;
      if (ps->s0 + 2 * pi <= p)
	  pi += p;
      sp = ps->s + pi;
      while (sp < ps->s + SIEVESIZE)
	{
	  *sp = 1, sp += p;
	}
      p += addp[ai];
      ai = (ai + 1) % 48;
    }
  ps->d = 0;
  return gmp_nextprime (ps);
}

void
gmp_init_primesieve (gmp_primesieve_t *ps)
{
  ps->s0 = 0;
  ps->sqrt_s0 = 0;
  ps->d = SIEVESIZE;
  ps->s[SIEVESIZE] = 0;		/* sentinel */
}
