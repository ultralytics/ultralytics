/* Factoring with Pollard's rho method.

Copyright 1995, 1997-2003, 2005, 2009, 2012, 2015 Free Software
Foundation, Inc.

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
#include <inttypes.h>

#include "gmp.h"

static unsigned char primes_diff[] = {
#define P(a,b,c) a,
#include "primes.h"
#undef P
};
#define PRIMES_PTAB_ENTRIES (sizeof(primes_diff) / sizeof(primes_diff[0]))

int flag_verbose = 0;

/* Prove primality or run probabilistic tests.  */
int flag_prove_primality = 1;

/* Number of Miller-Rabin tests to run when not proving primality. */
#define MR_REPS 25

struct factors
{
  mpz_t         *p;
  unsigned long *e;
  long nfactors;
};

void factor (mpz_t, struct factors *);

void
factor_init (struct factors *factors)
{
  factors->p = malloc (1);
  factors->e = malloc (1);
  factors->nfactors = 0;
}

void
factor_clear (struct factors *factors)
{
  int i;

  for (i = 0; i < factors->nfactors; i++)
    mpz_clear (factors->p[i]);

  free (factors->p);
  free (factors->e);
}

void
factor_insert (struct factors *factors, mpz_t prime)
{
  long    nfactors  = factors->nfactors;
  mpz_t         *p  = factors->p;
  unsigned long *e  = factors->e;
  long i, j;

  /* Locate position for insert new or increment e.  */
  for (i = nfactors - 1; i >= 0; i--)
    {
      if (mpz_cmp (p[i], prime) <= 0)
	break;
    }

  if (i < 0 || mpz_cmp (p[i], prime) != 0)
    {
      p = realloc (p, (nfactors + 1) * sizeof p[0]);
      e = realloc (e, (nfactors + 1) * sizeof e[0]);

      mpz_init (p[nfactors]);
      for (j = nfactors - 1; j > i; j--)
	{
	  mpz_set (p[j + 1], p[j]);
	  e[j + 1] = e[j];
	}
      mpz_set (p[i + 1], prime);
      e[i + 1] = 1;

      factors->p = p;
      factors->e = e;
      factors->nfactors = nfactors + 1;
    }
  else
    {
      e[i] += 1;
    }
}

void
factor_insert_ui (struct factors *factors, unsigned long prime)
{
  mpz_t pz;

  mpz_init_set_ui (pz, prime);
  factor_insert (factors, pz);
  mpz_clear (pz);
}


void
factor_using_division (mpz_t t, struct factors *factors)
{
  mpz_t q;
  unsigned long int p;
  int i;

  if (flag_verbose > 0)
    {
      printf ("[trial division] ");
    }

  mpz_init (q);

  p = mpz_scan1 (t, 0);
  mpz_fdiv_q_2exp (t, t, p);
  while (p)
    {
      factor_insert_ui (factors, 2);
      --p;
    }

  p = 3;
  for (i = 1; i <= PRIMES_PTAB_ENTRIES;)
    {
      if (! mpz_divisible_ui_p (t, p))
	{
	  p += primes_diff[i++];
	  if (mpz_cmp_ui (t, p * p) < 0)
	    break;
	}
      else
	{
	  mpz_tdiv_q_ui (t, t, p);
	  factor_insert_ui (factors, p);
	}
    }

  mpz_clear (q);
}

static int
mp_millerrabin (mpz_srcptr n, mpz_srcptr nm1, mpz_ptr x, mpz_ptr y,
		mpz_srcptr q, unsigned long int k)
{
  unsigned long int i;

  mpz_powm (y, x, q, n);

  if (mpz_cmp_ui (y, 1) == 0 || mpz_cmp (y, nm1) == 0)
    return 1;

  for (i = 1; i < k; i++)
    {
      mpz_powm_ui (y, y, 2, n);
      if (mpz_cmp (y, nm1) == 0)
	return 1;
      if (mpz_cmp_ui (y, 1) == 0)
	return 0;
    }
  return 0;
}

int
mp_prime_p (mpz_t n)
{
  int k, r, is_prime;
  mpz_t q, a, nm1, tmp;
  struct factors factors;

  if (mpz_cmp_ui (n, 1) <= 0)
    return 0;

  /* We have already casted out small primes. */
  if (mpz_cmp_ui (n, (long) FIRST_OMITTED_PRIME * FIRST_OMITTED_PRIME) < 0)
    return 1;

  mpz_inits (q, a, nm1, tmp, NULL);

  /* Precomputation for Miller-Rabin.  */
  mpz_sub_ui (nm1, n, 1);

  /* Find q and k, where q is odd and n = 1 + 2**k * q.  */
  k = mpz_scan1 (nm1, 0);
  mpz_tdiv_q_2exp (q, nm1, k);

  mpz_set_ui (a, 2);

  /* Perform a Miller-Rabin test, finds most composites quickly.  */
  if (!mp_millerrabin (n, nm1, a, tmp, q, k))
    {
      is_prime = 0;
      goto ret2;
    }

  if (flag_prove_primality)
    {
      /* Factor n-1 for Lucas.  */
      mpz_set (tmp, nm1);
      factor (tmp, &factors);
    }

  /* Loop until Lucas proves our number prime, or Miller-Rabin proves our
     number composite.  */
  for (r = 0; r < PRIMES_PTAB_ENTRIES; r++)
    {
      int i;

      if (flag_prove_primality)
	{
	  is_prime = 1;
	  for (i = 0; i < factors.nfactors && is_prime; i++)
	    {
	      mpz_divexact (tmp, nm1, factors.p[i]);
	      mpz_powm (tmp, a, tmp, n);
	      is_prime = mpz_cmp_ui (tmp, 1) != 0;
	    }
	}
      else
	{
	  /* After enough Miller-Rabin runs, be content. */
	  is_prime = (r == MR_REPS - 1);
	}

      if (is_prime)
	goto ret1;

      mpz_add_ui (a, a, primes_diff[r]);	/* Establish new base.  */

      if (!mp_millerrabin (n, nm1, a, tmp, q, k))
	{
	  is_prime = 0;
	  goto ret1;
	}
    }

  fprintf (stderr, "Lucas prime test failure.  This should not happen\n");
  abort ();

 ret1:
  if (flag_prove_primality)
    factor_clear (&factors);
 ret2:
  mpz_clears (q, a, nm1, tmp, NULL);

  return is_prime;
}

void
factor_using_pollard_rho (mpz_t n, unsigned long a, struct factors *factors)
{
  mpz_t x, z, y, P;
  mpz_t t, t2;
  unsigned long long k, l, i;

  if (flag_verbose > 0)
    {
      printf ("[pollard-rho (%lu)] ", a);
    }

  mpz_inits (t, t2, NULL);
  mpz_init_set_si (y, 2);
  mpz_init_set_si (x, 2);
  mpz_init_set_si (z, 2);
  mpz_init_set_ui (P, 1);
  k = 1;
  l = 1;

  while (mpz_cmp_ui (n, 1) != 0)
    {
      for (;;)
	{
	  do
	    {
	      mpz_mul (t, x, x);
	      mpz_mod (x, t, n);
	      mpz_add_ui (x, x, a);

	      mpz_sub (t, z, x);
	      mpz_mul (t2, P, t);
	      mpz_mod (P, t2, n);

	      if (k % 32 == 1)
		{
		  mpz_gcd (t, P, n);
		  if (mpz_cmp_ui (t, 1) != 0)
		    goto factor_found;
		  mpz_set (y, x);
		}
	    }
	  while (--k != 0);

	  mpz_set (z, x);
	  k = l;
	  l = 2 * l;
	  for (i = 0; i < k; i++)
	    {
	      mpz_mul (t, x, x);
	      mpz_mod (x, t, n);
	      mpz_add_ui (x, x, a);
	    }
	  mpz_set (y, x);
	}

    factor_found:
      do
	{
	  mpz_mul (t, y, y);
	  mpz_mod (y, t, n);
	  mpz_add_ui (y, y, a);

	  mpz_sub (t, z, y);
	  mpz_gcd (t, t, n);
	}
      while (mpz_cmp_ui (t, 1) == 0);

      mpz_divexact (n, n, t);	/* divide by t, before t is overwritten */

      if (!mp_prime_p (t))
	{
	  if (flag_verbose > 0)
	    {
	      printf ("[composite factor--restarting pollard-rho] ");
	    }
	  factor_using_pollard_rho (t, a + 1, factors);
	}
      else
	{
	  factor_insert (factors, t);
	}

      if (mp_prime_p (n))
	{
	  factor_insert (factors, n);
	  break;
	}

      mpz_mod (x, x, n);
      mpz_mod (z, z, n);
      mpz_mod (y, y, n);
    }

  mpz_clears (P, t2, t, z, x, y, NULL);
}

void
factor (mpz_t t, struct factors *factors)
{
  factor_init (factors);

  if (mpz_sgn (t) != 0)
    {
      factor_using_division (t, factors);

      if (mpz_cmp_ui (t, 1) != 0)
	{
	  if (flag_verbose > 0)
	    {
	      printf ("[is number prime?] ");
	    }
	  if (mp_prime_p (t))
	    factor_insert (factors, t);
	  else
	    factor_using_pollard_rho (t, 1, factors);
	}
    }
}

int
main (int argc, char *argv[])
{
  mpz_t t;
  int i, j, k;
  struct factors factors;

  while (argc > 1)
    {
      if (!strcmp (argv[1], "-v"))
	flag_verbose = 1;
      else if (!strcmp (argv[1], "-w"))
	flag_prove_primality = 0;
      else
	break;

      argv++;
      argc--;
    }

  mpz_init (t);
  if (argc > 1)
    {
      for (i = 1; i < argc; i++)
	{
	  mpz_set_str (t, argv[i], 0);

	  gmp_printf ("%Zd:", t);
	  factor (t, &factors);

	  for (j = 0; j < factors.nfactors; j++)
	    for (k = 0; k < factors.e[j]; k++)
	      gmp_printf (" %Zd", factors.p[j]);

	  puts ("");
	  factor_clear (&factors);
	}
    }
  else
    {
      for (;;)
	{
	  mpz_inp_str (t, stdin, 0);
	  if (feof (stdin))
	    break;

	  gmp_printf ("%Zd:", t);
	  factor (t, &factors);

	  for (j = 0; j < factors.nfactors; j++)
	    for (k = 0; k < factors.e[j]; k++)
	      gmp_printf (" %Zd", factors.p[j]);

	  puts ("");
	  factor_clear (&factors);
	}
    }

  exit (0);
}
