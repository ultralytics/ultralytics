/* tune-gcd-p

   Tune the choice for splitting p in divide-and-conquer gcd.

Copyright 2008, 2010, 2011 Free Software Foundation, Inc.

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

#define TUNE_GCD_P 1

#include "../mpn/gcd.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "speed.h"

/* Search for minimum over a range. FIXME: Implement golden-section /
   fibonacci search*/
static int
search (double *minp, double (*f)(void *, int), void *ctx, int start, int end)
{
  int x[4];
  double y[4];

  int best_i;

  x[0] = start;
  x[3] = end;

  y[0] = f(ctx, x[0]);
  y[3] = f(ctx, x[3]);

  for (;;)
    {
      int i;
      int length = x[3] - x[0];

      x[1] = x[0] + length/3;
      x[2] = x[0] + 2*length/3;

      y[1] = f(ctx, x[1]);
      y[2] = f(ctx, x[2]);

#if 0
      printf("%d: %f, %d: %f, %d:, %f %d: %f\n",
	     x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3]);
#endif
      for (best_i = 0, i = 1; i < 4; i++)
	if (y[i] < y[best_i])
	  best_i = i;

      if (length <= 4)
	break;

      if (best_i >= 2)
	{
	  x[0] = x[1];
	  y[0] = y[1];
	}
      else
	{
	  x[3] = x[2];
	  y[3] = y[2];
	}
    }
  *minp = y[best_i];
  return x[best_i];
}

static int
compare_double(const void *ap, const void *bp)
{
  double a = * (const double *) ap;
  double b = * (const double *) bp;

  if (a < b)
    return -1;
  else if (a > b)
    return 1;
  else
    return 0;
}

static double
median (double *v, size_t n)
{
  qsort(v, n, sizeof(*v), compare_double);

  return v[n/2];
}

#define TIME(res, code) do {				\
  double time_measurement[5];				\
  unsigned time_i;					\
							\
  for (time_i = 0; time_i < 5; time_i++)		\
    {							\
      speed_starttime();				\
      code;						\
      time_measurement[time_i] = speed_endtime();	\
    }							\
  res = median(time_measurement, 5);			\
} while (0)

struct bench_data
{
  mp_size_t n;
  mp_ptr ap;
  mp_ptr bp;
  mp_ptr up;
  mp_ptr vp;
  mp_ptr gp;
};

static double
bench_gcd (void *ctx, int p)
{
  struct bench_data *data = (struct bench_data *) ctx;
  double t;

  p_table[data->n] = p;
  TIME(t, {
      MPN_COPY (data->up, data->ap, data->n);
      MPN_COPY (data->vp, data->bp, data->n);
      mpn_gcd (data->gp, data->up, data->n, data->vp, data->n);
    });

  return t;
}

int
main(int argc, char **argv)
{
  gmp_randstate_t rands;  struct bench_data data;
  mp_size_t n;

  TMP_DECL;

  /* Unbuffered so if output is redirected to a file it isn't lost if the
     program is killed part way through.  */
  setbuf (stdout, NULL);
  setbuf (stderr, NULL);

  gmp_randinit_default (rands);

  TMP_MARK;

  data.ap = TMP_ALLOC_LIMBS (P_TABLE_SIZE);
  data.bp = TMP_ALLOC_LIMBS (P_TABLE_SIZE);
  data.up = TMP_ALLOC_LIMBS (P_TABLE_SIZE);
  data.vp = TMP_ALLOC_LIMBS (P_TABLE_SIZE);
  data.gp = TMP_ALLOC_LIMBS (P_TABLE_SIZE);

  mpn_random (data.ap, P_TABLE_SIZE);
  mpn_random (data.bp, P_TABLE_SIZE);

  memset (p_table, 0, sizeof(p_table));

  for (n = 100; n < P_TABLE_SIZE; n++)
    {
      mp_size_t p;
      mp_size_t best_p;
      double best_time;
      double lehmer_time;

      if (data.ap[n-1] == 0)
	data.ap[n-1] = 1;

      if (data.bp[n-1] == 0)
	data.bp[n-1] = 1;

      data.n = n;

      lehmer_time = bench_gcd (&data, 0);

      best_p = search (&best_time, bench_gcd, &data, n/5, 4*n/5);
      if (best_time > lehmer_time)
	best_p = 0;

      printf("%6d %6d %5.3g", n, best_p, (double) best_p / n);
      if (best_p > 0)
	{
	  double speedup = 100 * (lehmer_time - best_time) / lehmer_time;
	  printf(" %5.3g%%", speedup);
	  if (speedup < 1.0)
	    {
	      printf(" (ignored)");
	      best_p = 0;
	    }
	}
      printf("\n");

      p_table[n] = best_p;
    }
  TMP_FREE;
  gmp_randclear(rands);
  return 0;
}
