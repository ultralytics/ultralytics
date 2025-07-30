/* List and count primes.
   Written by tege while on holiday in Rodupp, August 2001.
   Between 10 and 500 times faster than previous program.

Copyright 2001, 2002, 2006, 2012 Free Software Foundation, Inc.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see https://www.gnu.org/licenses/.  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* IDEAS:
 * Do not fill primes[] with real primes when the range [fr,to] is small,
   when fr,to are relatively large.  Fill primes[] with odd numbers instead.
   [Probably a bad idea, since the primes[] array would become very large.]
 * Separate small primes and large primes when sieving.  Either the Montgomery
   way (i.e., having a large array a multiple of L1 cache size), or just
   separate loops for primes <= S and primes > S.  The latter primes do not
   require an inner loop, since they will touch the sieving array at most once.
 * Pre-fill sieving array with an appropriately aligned ...00100100... pattern,
   then omit 3 from primes array.  (May require similar special handling of 3
   as we now have for 2.)
 * A large SIEVE_LIMIT currently implies very large memory usage, mainly due
   to the sieving array in make_primelist, but also because of the primes[]
   array.  We might want to stage the program, using sieve_region/find_primes
   to build primes[].  Make report() a function pointer, as part of achieving
   this.
 * Store primes[] as two arrays, one array with primes represented as delta
   values using just 8 bits (if gaps are too big, store bogus primes!)
   and one array with "rem" values.  The latter needs 32-bit values.
 * A new entry point, mpz_probab_prime_likely_p, would be useful.
 * Improve command line syntax and versatility.  "primes -f FROM -t TO",
   allow either to be omitted for open interval.  (But disallow
   "primes -c -f FROM" since that would be infinity.)  Allow printing a
   limited *number* of primes using syntax like "primes -f FROM -n NUMBER".
 * When looking for maxgaps, we should not perform any primality testing until
   we find possible record gaps.  Should speed up the searches tremendously.
 */

#include "gmp.h"

struct primes
{
  unsigned int prime;
  int rem;
};

struct primes *primes;
unsigned long n_primes;

void find_primes (unsigned char *, mpz_t, unsigned long, mpz_t);
void sieve_region (unsigned char *, mpz_t, unsigned long);
void make_primelist (unsigned long);

int flag_print = 1;
int flag_count = 0;
int flag_maxgap = 0;
unsigned long maxgap = 0;
unsigned long total_primes = 0;

void
report (mpz_t prime)
{
  total_primes += 1;
  if (flag_print)
    {
      mpz_out_str (stdout, 10, prime);
      printf ("\n");
    }
  if (flag_maxgap)
    {
      static unsigned long prev_prime_low = 0;
      unsigned long gap;
      if (prev_prime_low != 0)
	{
	  gap = mpz_get_ui (prime) - prev_prime_low;
	  if (maxgap < gap)
	    maxgap = gap;
	}
      prev_prime_low = mpz_get_ui (prime);
    }
}

int
main (int argc, char *argv[])
{
  char *progname = argv[0];
  mpz_t fr, to;
  mpz_t fr2, to2;
  unsigned long sieve_lim;
  unsigned long est_n_primes;
  unsigned char *s;
  mpz_t tmp;
  mpz_t siev_sqr_lim;

  while (argc != 1)
    {
      if (strcmp (argv[1], "-c") == 0)
	{
	  flag_count = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-p") == 0)
	{
	  flag_print = 2;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-g") == 0)
	{
	  flag_maxgap = 1;
	  argv++;
	  argc--;
	}
      else
	break;
    }

  if (flag_count || flag_maxgap)
    flag_print--;		/* clear unless an explicit -p  */

  mpz_init (fr);
  mpz_init (to);
  mpz_init (fr2);
  mpz_init (to2);

  if (argc == 3)
    {
      mpz_set_str (fr, argv[1], 0);
      if (argv[2][0] == '+')
	{
	  mpz_set_str (to, argv[2] + 1, 0);
	  mpz_add (to, to, fr);
	}
      else
	mpz_set_str (to, argv[2], 0);
    }
  else if (argc == 2)
    {
      mpz_set_ui (fr, 0);
      mpz_set_str (to, argv[1], 0);
    }
  else
    {
      fprintf (stderr, "usage: %s [-c] [-p] [-g] [from [+]]to\n", progname);
      exit (1);
    }

  mpz_set (fr2, fr);
  if (mpz_cmp_ui (fr2, 3) < 0)
    {
      mpz_set_ui (fr2, 2);
      report (fr2);
      mpz_set_ui (fr2, 3);
    }
  mpz_setbit (fr2, 0);				/* make odd */
  mpz_sub_ui (to2, to, 1);
  mpz_setbit (to2, 0);				/* make odd */

  mpz_init (tmp);
  mpz_init (siev_sqr_lim);

  mpz_sqrt (tmp, to2);
#define SIEVE_LIMIT 10000000
  if (mpz_cmp_ui (tmp, SIEVE_LIMIT) < 0)
    {
      sieve_lim = mpz_get_ui (tmp);
    }
  else
    {
      sieve_lim = SIEVE_LIMIT;
      mpz_sub (tmp, to2, fr2);
      if (mpz_cmp_ui (tmp, sieve_lim) < 0)
	sieve_lim = mpz_get_ui (tmp);	/* limit sieving for small ranges */
    }
  mpz_set_ui (siev_sqr_lim, sieve_lim + 1);
  mpz_mul_ui (siev_sqr_lim, siev_sqr_lim, sieve_lim + 1);

  est_n_primes = (size_t) (sieve_lim / log((double) sieve_lim) * 1.13) + 10;
  primes = malloc (est_n_primes * sizeof primes[0]);
  make_primelist (sieve_lim);
  assert (est_n_primes >= n_primes);

#if DEBUG
  printf ("sieve_lim = %lu\n", sieve_lim);
  printf ("n_primes = %lu (3..%u)\n",
	  n_primes, primes[n_primes - 1].prime);
#endif

#define S (1 << 15)		/* FIXME: Figure out L1 cache size */
  s = malloc (S/2);
  while (mpz_cmp (fr2, to2) <= 0)
    {
      unsigned long rsize;
      rsize = S;
      mpz_add_ui (tmp, fr2, rsize);
      if (mpz_cmp (tmp, to2) > 0)
	{
	  mpz_sub (tmp, to2, fr2);
	  rsize = mpz_get_ui (tmp) + 2;
	}
#if DEBUG
      printf ("Sieving region ["); mpz_out_str (stdout, 10, fr2);
      printf (","); mpz_add_ui (tmp, fr2, rsize - 2);
      mpz_out_str (stdout, 10, tmp); printf ("]\n");
#endif
      sieve_region (s, fr2, rsize);
      find_primes (s, fr2, rsize / 2, siev_sqr_lim);

      mpz_add_ui (fr2, fr2, S);
    }
  free (s);

  if (flag_count)
    printf ("Pi(interval) = %lu\n", total_primes);

  if (flag_maxgap)
    printf ("max gap: %lu\n", maxgap);

  return 0;
}

/* Find primes in region [fr,fr+rsize).  Requires that fr is odd and that
   rsize is even.  The sieving array s should be aligned for "long int" and
   have rsize/2 entries, rounded up to the nearest multiple of "long int".  */
void
sieve_region (unsigned char *s, mpz_t fr, unsigned long rsize)
{
  unsigned long ssize = rsize / 2;
  unsigned long start, start2, prime;
  unsigned long i;
  mpz_t tmp;

  mpz_init (tmp);

#if 0
  /* initialize sieving array */
  for (ii = 0; ii < (ssize + sizeof (long) - 1) / sizeof (long); ii++)
    ((long *) s) [ii] = ~0L;
#else
  {
    long k;
    long *se = (long *) (s + ((ssize + sizeof (long) - 1) & -sizeof (long)));
    for (k = -((ssize + sizeof (long) - 1) / sizeof (long)); k < 0; k++)
      se[k] = ~0L;
  }
#endif

  for (i = 0; i < n_primes; i++)
    {
      prime = primes[i].prime;

      if (primes[i].rem >= 0)
	{
	  start2 = primes[i].rem;
	}
      else
	{
	  mpz_set_ui (tmp, prime);
	  mpz_mul_ui (tmp, tmp, prime);
	  if (mpz_cmp (fr, tmp) <= 0)
	    {
	      mpz_sub (tmp, tmp, fr);
	      if (mpz_cmp_ui (tmp, 2 * ssize) > 0)
		break;		/* avoid overflow at next line, also speedup */
	      start = mpz_get_ui (tmp);
	    }
	  else
	    {
	      start = (prime - mpz_tdiv_ui (fr, prime)) % prime;
	      if (start % 2 != 0)
		start += prime;		/* adjust if even divisible */
	    }
	  start2 = start / 2;
	}

#if 0
      for (ii = start2; ii < ssize; ii += prime)
	s[ii] = 0;
      primes[i].rem = ii - ssize;
#else
      {
	long k;
	unsigned char *se = s + ssize; /* point just beyond sieving range */
	for (k = start2 - ssize; k < 0; k += prime)
	  se[k] = 0;
	primes[i].rem = k;
      }
#endif
    }
  mpz_clear (tmp);
}

/* Find primes in region [fr,fr+rsize), using the previously sieved s[].  */
void
find_primes (unsigned char *s, mpz_t  fr, unsigned long ssize,
	     mpz_t siev_sqr_lim)
{
  unsigned long j, ij;
  mpz_t tmp;

  mpz_init (tmp);
  for (j = 0; j < (ssize + sizeof (long) - 1) / sizeof (long); j++)
    {
      if (((long *) s) [j] != 0)
	{
	  for (ij = 0; ij < sizeof (long); ij++)
	    {
	      if (s[j * sizeof (long) + ij] != 0)
		{
		  if (j * sizeof (long) + ij >= ssize)
		    goto out;
		  mpz_add_ui (tmp, fr, (j * sizeof (long) + ij) * 2);
		  if (mpz_cmp (tmp, siev_sqr_lim) < 0 ||
		      mpz_probab_prime_p (tmp, 10))
		    report (tmp);
		}
	    }
	}
    }
 out:
  mpz_clear (tmp);
}

/* Generate a list of primes and store in the global array primes[].  */
void
make_primelist (unsigned long maxprime)
{
#if 1
  unsigned char *s;
  unsigned long ssize = maxprime / 2;
  unsigned long i, ii, j;

  s = malloc (ssize);
  memset (s, ~0, ssize);
  for (i = 3; ; i += 2)
    {
      unsigned long isqr = i * i;
      if (isqr >= maxprime)
	break;
      if (s[i * i / 2 - 1] == 0)
	continue;				/* only sieve with primes */
      for (ii = i * i / 2 - 1; ii < ssize; ii += i)
	s[ii] = 0;
    }
  n_primes = 0;
  for (j = 0; j < ssize; j++)
    {
      if (s[j] != 0)
	{
	  primes[n_primes].prime = j * 2 + 3;
	  primes[n_primes].rem = -1;
	  n_primes++;
	}
    }
  /* FIXME: This should not be needed if fencepost errors were fixed... */
  if (primes[n_primes - 1].prime > maxprime)
    n_primes--;
  free (s);
#else
  unsigned long i;
  n_primes = 0;
  for (i = 3; i <= maxprime; i += 2)
    {
      if (i < 7 || (i % 3 != 0 && i % 5 != 0 && i % 7 != 0))
	{
	  primes[n_primes].prime = i;
	  primes[n_primes].rem = -1;
	  n_primes++;
	}
    }
#endif
}
