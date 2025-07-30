/* stat.c -- statistical tests of random number sequences. */

/*
Copyright 1999, 2000 Free Software Foundation, Inc.

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

/* Examples:

  $ gen 1000 | stat
Test 1000 real numbers.

  $ gen 30000 | stat -2 1000
Test 1000 real numbers 30 times and then test the 30 results in a
``second level''.

  $ gen -f mpz_urandomb 1000 | stat -i 0xffffffff
Test 1000 integers 0 <= X <= 2^32-1.

  $ gen -f mpz_urandomb -z 34 1000 | stat -i 0x3ffffffff
Test 1000 integers 0 <= X <= 2^34-1.

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "gmp.h"
#include "gmpstat.h"

#if !HAVE_DECL_OPTARG
extern char *optarg;
extern int optind, opterr;
#endif

#define FVECSIZ (100000L)

int g_debug = 0;

static void
print_ks_results (mpf_t f_p, mpf_t f_p_prob,
		  mpf_t f_m, mpf_t f_m_prob,
		  FILE *fp)
{
  double p, pp, m, mp;

  p = mpf_get_d (f_p);
  m = mpf_get_d (f_m);
  pp = mpf_get_d (f_p_prob);
  mp = mpf_get_d (f_m_prob);

  fprintf (fp, "%.4f (%.0f%%)\t", p, pp * 100.0);
  fprintf (fp, "%.4f (%.0f%%)\n", m, mp * 100.0);
}

static void
print_x2_table (unsigned int v, FILE *fp)
{
  double t[7];
  int f;


  fprintf (fp, "Chi-square table for v=%u\n", v);
  fprintf (fp, "1%%\t5%%\t25%%\t50%%\t75%%\t95%%\t99%%\n");
  x2_table (t, v);
  for (f = 0; f < 7; f++)
    fprintf (fp, "%.2f\t", t[f]);
  fputs ("\n", fp);
}



/* Pks () -- Distribution function for KS results with a big n (like 1000
   or so):  F(x) = 1 - pow(e, -2*x^2) [Knuth, vol 2, p.51]. */
/* gnuplot: plot [0:1] Pks(x), Pks(x) = 1-exp(-2*x**2)  */

static void
Pks (mpf_t p, mpf_t x)
{
  double dt;			/* temp double */

  mpf_set (p, x);
  mpf_mul (p, p, p);		/* p = x^2 */
  mpf_mul_ui (p, p, 2);		/* p = 2*x^2 */
  mpf_neg (p, p);		/* p = -2*x^2 */
  /* No pow() in gmp.  Use doubles. */
  /* FIXME: Use exp()? */
  dt = pow (M_E, mpf_get_d (p));
  mpf_set_d (p, dt);
  mpf_ui_sub (p, 1, p);
}

/* f_freq() -- frequency test on real numbers 0<=f<1*/
static void
f_freq (const unsigned l1runs, const unsigned l2runs,
	mpf_t fvec[], const unsigned long n)
{
  unsigned f;
  mpf_t f_p, f_p_prob;
  mpf_t f_m, f_m_prob;
  mpf_t *l1res;			/* level 1 result array */

  mpf_init (f_p);  mpf_init (f_m);
  mpf_init (f_p_prob);  mpf_init (f_m_prob);


  /* Allocate space for 1st level results. */
  l1res = (mpf_t *) malloc (l2runs * 2 * sizeof (mpf_t));
  if (NULL == l1res)
    {
      fprintf (stderr, "stat: malloc failure\n");
      exit (1);
    }

  printf ("\nEquidistribution/Frequency test on real numbers (0<=X<1):\n");
  printf ("\tKp\t\tKm\n");

  for (f = 0; f < l2runs; f++)
    {
      /*  f_printvec (fvec, n); */
      mpf_freqt (f_p, f_m, fvec + f * n, n);

      /* what's the probability of getting these results? */
      ks_table (f_p_prob, f_p, n);
      ks_table (f_m_prob, f_m, n);

      if (l1runs == 0)
	{
	  /*printf ("%u:\t", f + 1);*/
	  print_ks_results (f_p, f_p_prob, f_m, f_m_prob, stdout);
	}
      else
	{
	  /* save result */
	  mpf_init_set (l1res[f], f_p);
	  mpf_init_set (l1res[f + l2runs], f_m);
	}
    }

  /* Now, apply the KS test on the results from the 1st level rounds
     with the distribution
     F(x) = 1 - pow(e, -2*x^2)	[Knuth, vol 2, p.51] */

  if (l1runs != 0)
    {
      /*printf ("-------------------------------------\n");*/

      /* The Kp's. */
      ks (f_p, f_m, l1res, Pks, l2runs);
      ks_table (f_p_prob, f_p, l2runs);
      ks_table (f_m_prob, f_m, l2runs);
      printf ("Kp:\t");
      print_ks_results (f_p, f_p_prob, f_m, f_m_prob, stdout);

      /* The Km's. */
      ks (f_p, f_m, l1res + l2runs, Pks, l2runs);
      ks_table (f_p_prob, f_p, l2runs);
      ks_table (f_m_prob, f_m, l2runs);
      printf ("Km:\t");
      print_ks_results (f_p, f_p_prob, f_m, f_m_prob, stdout);
    }

  mpf_clear (f_p);  mpf_clear (f_m);
  mpf_clear (f_p_prob);  mpf_clear (f_m_prob);
  free (l1res);
}

/* z_freq(l1runs, l2runs, zvec, n, max) -- frequency test on integers
   0<=z<=MAX */
static void
z_freq (const unsigned l1runs,
	const unsigned l2runs,
	mpz_t zvec[],
	const unsigned long n,
	unsigned int max)
{
  mpf_t V;			/* result */
  double d_V;			/* result as a double */

  mpf_init (V);


  printf ("\nEquidistribution/Frequency test on integers (0<=X<=%u):\n", max);
  print_x2_table (max, stdout);

  mpz_freqt (V, zvec, max, n);

  d_V = mpf_get_d (V);
  printf ("V = %.2f (n = %lu)\n", d_V, n);

  mpf_clear (V);
}

unsigned int stat_debug = 0;

int
main (argc, argv)
     int argc;
     char *argv[];
{
  const char usage[] =
    "usage: stat [-d] [-2 runs] [-i max | -r max] [file]\n" \
    "       file     filename\n" \
    "       -2 runs  perform 2-level test with RUNS runs on 1st level\n" \
    "       -d       increase debugging level\n" \
    "       -i max   input is integers 0 <= Z <= MAX\n" \
    "       -r max   input is real numbers 0 <= R < 1 and use MAX as\n" \
    "                maximum value when converting real numbers to integers\n" \
    "";

  mpf_t fvec[FVECSIZ];
  mpz_t zvec[FVECSIZ];
  unsigned long int f, n, vecentries;
  char *filen;
  FILE *fp;
  int c;
  int omitoutput = 0;
  int realinput = -1;		/* 1: input is real numbers 0<=R<1;
				   0: input is integers 0 <= Z <= MAX. */
  long l1runs = 0,		/* 1st level runs */
    l2runs = 1;			/* 2nd level runs */
  mpf_t f_temp;
  mpz_t z_imax;			/* max value when converting between
				   real number and integer. */
  mpf_t f_imax_plus1;		/* f_imax + 1 stored in an mpf_t for
				   convenience */
  mpf_t f_imax_minus1;		/* f_imax - 1 stored in an mpf_t for
				   convenience */


  mpf_init (f_temp);
  mpz_init_set_ui (z_imax, 0x7fffffff);
  mpf_init (f_imax_plus1);
  mpf_init (f_imax_minus1);

  while ((c = getopt (argc, argv, "d2:i:r:")) != -1)
    switch (c)
      {
      case '2':
	l1runs = atol (optarg);
	l2runs = -1;		/* set later on */
	break;
      case 'd':			/* increase debug level */
	stat_debug++;
	break;
      case 'i':
	if (1 == realinput)
	  {
	    fputs ("stat: options -i and -r are mutually exclusive\n", stderr);
	    exit (1);
	  }
	if (mpz_set_str (z_imax, optarg, 0))
	  {
	    fprintf (stderr, "stat: bad max value %s\n", optarg);
	    exit (1);
	  }
	realinput = 0;
	break;
      case 'r':
	if (0 == realinput)
	  {
	    fputs ("stat: options -i and -r are mutually exclusive\n", stderr);
	    exit (1);
	  }
	if (mpz_set_str (z_imax, optarg, 0))
	  {
	    fprintf (stderr, "stat: bad max value %s\n", optarg);
	    exit (1);
	  }
	realinput = 1;
	break;
      case 'o':
	omitoutput = atoi (optarg);
	break;
      case '?':
      default:
	fputs (usage, stderr);
	exit (1);
      }
  argc -= optind;
  argv += optind;

  if (argc < 1)
    fp = stdin;
  else
    filen = argv[0];

  if (fp != stdin)
    if (NULL == (fp = fopen (filen, "r")))
      {
	perror (filen);
	exit (1);
      }

  if (-1 == realinput)
    realinput = 1;		/* default is real numbers */

  /* read file and fill appropriate vec */
  if (1 == realinput)		/* real input */
    {
      for (f = 0; f < FVECSIZ ; f++)
	{
	  mpf_init (fvec[f]);
	  if (!mpf_inp_str (fvec[f], fp, 10))
	    break;
	}
    }
  else				/* integer input */
    {
      for (f = 0; f < FVECSIZ ; f++)
	{
	  mpz_init (zvec[f]);
	  if (!mpz_inp_str (zvec[f], fp, 10))
	    break;
	}
    }
  vecentries = n = f;		/* number of entries read */
  fclose (fp);

  if (FVECSIZ == f)
    fprintf (stderr, "stat: warning: discarding input due to lazy allocation "\
	     "of only %ld entries.  sorry.\n", FVECSIZ);

  printf ("Got %lu numbers.\n", n);

  /* convert and fill the other vec */
  /* since fvec[] contains 0<=f<1 and we want ivec[] to contain
     0<=z<=imax and we are truncating all fractions when
     converting float to int, we have to add 1 to imax.*/
  mpf_set_z (f_imax_plus1, z_imax);
  mpf_add_ui (f_imax_plus1, f_imax_plus1, 1);
  if (1 == realinput)		/* fill zvec[] */
    {
      for (f = 0; f < n; f++)
	{
	  mpf_mul (f_temp, fvec[f], f_imax_plus1);
	  mpz_init (zvec[f]);
	  mpz_set_f (zvec[f], f_temp); /* truncating fraction */
	  if (stat_debug > 1)
	    {
	      mpz_out_str (stderr, 10, zvec[f]);
	      fputs ("\n", stderr);
	    }
	}
    }
  else				/* integer input; fill fvec[] */
    {
      /*    mpf_set_z (f_imax_minus1, z_imax);
	    mpf_sub_ui (f_imax_minus1, f_imax_minus1, 1);*/
      for (f = 0; f < n; f++)
	{
	  mpf_init (fvec[f]);
	  mpf_set_z (fvec[f], zvec[f]);
	  mpf_div (fvec[f], fvec[f], f_imax_plus1);
	  if (stat_debug > 1)
	    {
	      mpf_out_str (stderr, 10, 0, fvec[f]);
	      fputs ("\n", stderr);
	    }
	}
    }

  /* 2 levels? */
  if (1 != l2runs)
    {
      l2runs = n / l1runs;
      printf ("Doing %ld second level rounds "\
	      "with %ld entries in each round", l2runs, l1runs);
      if (n % l1runs)
	printf (" (discarding %ld entr%s)", n % l1runs,
		n % l1runs == 1 ? "y" : "ies");
      puts (".");
      n = l1runs;
    }

#ifndef DONT_FFREQ
  f_freq (l1runs, l2runs, fvec, n);
#endif
#ifdef DO_ZFREQ
  z_freq (l1runs, l2runs, zvec, n, mpz_get_ui (z_imax));
#endif

  mpf_clear (f_temp); mpz_clear (z_imax);
  mpf_clear (f_imax_plus1);
  mpf_clear (f_imax_minus1);
  for (f = 0; f < vecentries; f++)
    {
      mpf_clear (fvec[f]);
      mpz_clear (zvec[f]);
    }

  return 0;
}
