/* Test file for mpfr_custom_*

Copyright 2005-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdlib.h>

#include "mpfr-test.h"

#define BUFFER_SIZE 250
#define PREC_TESTED 200

long Buffer[BUFFER_SIZE];
char *stack = (char *) Buffer;
long *org = (long *) Buffer;
mpfr_prec_t p = PREC_TESTED;

#define ALIGNED(s) (((s) + sizeof (long) - 1) / sizeof (long) * sizeof (long))

static void *
new_st (size_t s)
{
  void *p = (void *) stack;

  if (MPFR_UNLIKELY (s > (char *) &Buffer[BUFFER_SIZE] - stack))
    {
      printf ("[INTERNAL TEST ERROR] Stack overflow.\n");
      exit (1);
    }
  stack += ALIGNED (s);
  return p;
}

static void
reset_stack (void)
{
  stack = (char *) Buffer;
}

/*************************************************************************/

 /* Alloc a new mpfr_t on the main stack */
static mpfr_ptr
new_mpfr (mpfr_prec_t p)
{
  mpfr_ptr x = (mpfr_ptr) new_st (sizeof (mpfr_t));
  void *mantissa = new_st (mpfr_custom_get_size (p));
  mpfr_custom_init (mantissa, p);
  mpfr_custom_init_set (x, 0, 0, p, mantissa);
  return x;
}

 /* Alloc a new mpfr_t on the main stack */
static mpfr_ptr
new_nan (mpfr_prec_t p)
{
  mpfr_ptr x = (mpfr_ptr) new_st (sizeof (mpfr_t));
  void *mantissa = new_st ((mpfr_custom_get_size) (p));
  (mpfr_custom_init) (mantissa, p);
  (mpfr_custom_init_set) (x, MPFR_NAN_KIND, 0, p, mantissa);
  return x;
}

 /* Alloc a new mpfr_t on the main stack */
static mpfr_ptr
new_inf (mpfr_prec_t p)
{
  mpfr_ptr x = (mpfr_ptr) new_st (sizeof (mpfr_t));
  void *mantissa = new_st ((mpfr_custom_get_size) (p));
  (mpfr_custom_init) (mantissa, p);
  (mpfr_custom_init_set) (x, -MPFR_INF_KIND, 0, p, mantissa);
  return x;
}

 /* Garbage the stack by keeping only x and save it in old_stack */
static mpfr_ptr
return_mpfr (mpfr_ptr x, char *old_stack)
{
  void *mantissa       = mpfr_custom_get_significand (x);
  size_t size_mantissa = mpfr_custom_get_size (mpfr_get_prec (x));
  mpfr_ptr newx;

  memmove (old_stack, x, sizeof (mpfr_t));
  memmove (old_stack + ALIGNED (sizeof (mpfr_t)), mantissa, size_mantissa);
  newx = (mpfr_ptr) old_stack;
  mpfr_custom_move (newx, old_stack + ALIGNED (sizeof (mpfr_t)));
  stack = old_stack + ALIGNED (sizeof (mpfr_t)) + ALIGNED (size_mantissa);
  return newx;
}

 /* Garbage the stack by keeping only x and save it in old_stack */
static mpfr_ptr
return_mpfr_func (mpfr_ptr x, char *old_stack)
{
  void *mantissa       = (mpfr_custom_get_significand) (x);
  size_t size_mantissa = (mpfr_custom_get_size) (mpfr_get_prec (x));
  mpfr_ptr newx;

  memmove (old_stack, x, sizeof (mpfr_t));
  memmove (old_stack + ALIGNED (sizeof (mpfr_t)), mantissa, size_mantissa);
  newx = (mpfr_ptr) old_stack;
  (mpfr_custom_move) (newx, old_stack + ALIGNED (sizeof (mpfr_t)));
  stack = old_stack + ALIGNED (sizeof (mpfr_t)) + ALIGNED (size_mantissa);
  return newx;
}

/*************************************************************************/

static void
test1 (void)
{
  mpfr_ptr x, y;

  reset_stack ();
  org = (long *) stack;

  x = new_mpfr (p);
  y = new_mpfr (p);
  mpfr_set_ui (x, 42, MPFR_RNDN);
  mpfr_set_ui (y, 17, MPFR_RNDN);
  mpfr_add (y, x, y, MPFR_RNDN);
  y = return_mpfr (y, (char *) org);
  if ((long *) y != org || mpfr_cmp_ui (y, 59) != 0)
    {
      printf ("Compact (1) failed!\n");
      exit (1);
    }

  x = new_mpfr (p);
  y = new_mpfr (p);
  mpfr_set_ui (x, 4217, MPFR_RNDN);
  mpfr_set_ui (y, 1742, MPFR_RNDN);
  mpfr_add (y, x, y, MPFR_RNDN);
  y = return_mpfr_func (y, (char *) org);
  if ((long *) y != org || mpfr_cmp_ui (y, 5959) != 0)
    {
      printf ("Compact (5) failed!\n");
      exit (1);
    }

  reset_stack ();
}

static void
test_nan_inf_zero (void)
{
  mpfr_ptr val;
  int sign;
  int kind;

  reset_stack ();

  val = new_mpfr (MPFR_PREC_MIN);
  mpfr_set_nan (val);
  kind = (mpfr_custom_get_kind) (val);
  if (kind != MPFR_NAN_KIND)
    {
      printf ("mpfr_custom_get_kind error: ");
      mpfr_dump (val);
      printf (" is kind %d instead of %d\n", kind, (int) MPFR_NAN_KIND);
      exit (1);
    }

  val = new_nan (MPFR_PREC_MIN);
  if (!mpfr_nan_p(val))
    {
      printf ("Error: mpfr_custom_init_set doesn't set NAN mpfr.\n");
      exit (1);
    }

  val = new_inf (MPFR_PREC_MIN);
  if (!mpfr_inf_p(val) || mpfr_sgn(val) >= 0)
    {
      printf ("Error: mpfr_custom_init_set doesn't set -INF mpfr.\n");
      exit (1);
    }

  sign = 1;
  mpfr_set_inf (val, sign);
  kind = (mpfr_custom_get_kind) (val);
  if ((ABS (kind) != MPFR_INF_KIND) || (SIGN (kind) != SIGN (sign)))
    {
      printf ("mpfr_custom_get_kind error: ");
      mpfr_dump (val);
      printf (" is kind %d instead of %d\n", kind, (int) MPFR_INF_KIND);
      printf (" have sign %d instead of %d\n", SIGN (kind), SIGN (sign));
      exit (1);
    }

  sign = -1;
  mpfr_set_zero (val, sign);
  kind = (mpfr_custom_get_kind) (val);
  if ((ABS (kind) != MPFR_ZERO_KIND) || (SIGN (kind) != SIGN (sign)))
    {
      printf ("mpfr_custom_get_kind error: ");
      mpfr_dump (val);
      printf (" is kind %d instead of %d\n", kind, (int) MPFR_ZERO_KIND);
      printf (" have sign %d instead of %d\n", SIGN (kind), SIGN (sign));
      exit (1);
    }

  reset_stack ();
}

/*************************************************************************/

/* We build the MPFR variable each time it is needed */
/* a[0] is the kind, a[1] is the exponent, &a[2] is the mantissa */
static long *
dummy_new (void)
{
  long *r;

  r = (long *) new_st (ALIGNED (2 * sizeof (long)) +
                       ALIGNED (mpfr_custom_get_size (p)));
  (mpfr_custom_init) (&r[2], p);
  r[0] = (int) MPFR_NAN_KIND;
  r[1] = 0;
  return r;
}

static long *
dummy_set_si (long si)
{
  mpfr_t x;
  long * r = dummy_new ();
  (mpfr_custom_init_set) (x, MPFR_REGULAR_KIND, 0, p, &r[2]);
  mpfr_set_si (x, si, MPFR_RNDN);
  r[0] = mpfr_custom_get_kind (x);
  r[1] = mpfr_custom_get_exp (x);
  return r;
}

static long *
dummy_add (long *a, long *b)
{
  mpfr_t x, y, z;
  long *r = dummy_new ();
  mpfr_custom_init_set (x, 0 + MPFR_REGULAR_KIND, 0, p, &r[2]);
  (mpfr_custom_init_set) (y, a[0], a[1], p, &a[2]);
  mpfr_custom_init_set (z, 0 + b[0], b[1], p, &b[2]);
  mpfr_add (x, y, z, MPFR_RNDN);
  r[0] = (mpfr_custom_get_kind) (x);
  r[1] = (mpfr_custom_get_exp) (x);
  return r;
}

static long *
dummy_compact (long *r, long *org_stack)
{
  memmove (org_stack, r,
           ALIGNED (2*sizeof (long)) + ALIGNED ((mpfr_custom_get_size) (p)));
  return org_stack;
}

/*************************************************************************/

static void
test2 (void)
{
  mpfr_t x;
  long *a, *b, *c;

  reset_stack ();
  org = (long *) stack;

  a = dummy_set_si (42);
  b = dummy_set_si (17);
  c = dummy_add (a, b);
  c = dummy_compact (c, org);
  (mpfr_custom_init_set) (x, c[0], c[1], p, &c[2]);
  if (c != org || mpfr_cmp_ui (x, 59) != 0)
    {
      printf ("Compact (2) failed! c=%p a=%p\n", (void *) c, (void *) a);
      mpfr_dump (x);
      exit (1);
    }

  a = dummy_set_si (42);
  b = dummy_set_si (-17);
  c = dummy_add (a, b);
  c = dummy_compact (c, org);
  (mpfr_custom_init_set) (x, c[0], c[1], p, &c[2]);
  if (c != org || mpfr_cmp_ui (x, 25) != 0)
    {
      printf ("Compact (6) failed! c=%p a=%p\n", (void *) c, (void *) a);
      mpfr_dump (x);
      exit (1);
    }

  reset_stack ();
}


int
main (void)
{
  tests_start_mpfr ();
  /* We test iff long = mp_limb_t */
  if (sizeof (long) == sizeof (mp_limb_t))
    {
      test1 ();
      test2 ();
      test_nan_inf_zero ();
    }
  tests_end_mpfr ();
  return 0;
}
