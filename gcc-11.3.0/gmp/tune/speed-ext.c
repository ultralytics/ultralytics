/* An example of extending the speed program to measure routines not in GMP.

Copyright 1999, 2000, 2002, 2003, 2005 Free Software Foundation, Inc.

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


/* The extension here is three versions of an mpn arithmetic mean.  These
   aren't meant to be particularly useful, just examples.

   You can run something like the following to compare their speeds.

           ./speed-ext -s 1-20 -c mean_calls mean_open mean_open2

   On RISC chips, mean_open() might be fastest if the compiler is doing a
   good job.  On the register starved x86s, mean_calls will be fastest.


   Notes:

   SPEED_EXTRA_PROTOS and SPEED_EXTRA_ROUTINES are macros that get expanded
   by speed.c in useful places.  SPEED_EXTRA_PROTOS goes after the header
   files, and SPEED_EXTRA_ROUTINES goes in the array of available routines.

   The advantage of this #include "speed.c" scheme is that there's no
   editing of a copy of that file, and new features in new versions of it
   will be immediately available.

   In a real program the routines mean_calls() etc would probably be in
   separate C or assembler source files, and just the measuring
   speed_mean_calls() etc would be here.  Linking against other libraries
   for things to measure is perfectly possible too.

   When attempting to compare two versions of the same named routine, say
   like the generic and assembler versions of mpn_add_n(), creative use of
   cc -D or #define is suggested, so one or both can be renamed and linked
   into the same program.  It'll be much easier to compare them side by side
   than with separate programs for each.

   common.c has notes on writing speed measuring routines.

   Remember to link against tune/libspeed.la (or tune/.libs/libspeed.a if
   not using libtool) to get common.o and other objects needed by speed.c.  */


#define SPEED_EXTRA_PROTOS                                              \
  double speed_mean_calls (struct speed_params *s);			\
  double speed_mean_open  (struct speed_params *s);			\
  double speed_mean_open2 (struct speed_params *s);

#define SPEED_EXTRA_ROUTINES            \
  { "mean_calls",  speed_mean_calls  }, \
  { "mean_open",   speed_mean_open   }, \
  { "mean_open2",  speed_mean_open2  },

#include "speed.c"


/* A straightforward implementation calling mpn subroutines.

   wp,size is set to (xp,size + yp,size) / 2.  The return value is the
   remainder from the division.  The other versions are the same.  */

mp_limb_t
mean_calls (mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size)
{
  mp_limb_t  c, ret;

  ASSERT (size >= 1);

  c = mpn_add_n (wp, xp, yp, size);
  ret = mpn_rshift (wp, wp, size, 1) >> (GMP_LIMB_BITS-1);
  wp[size-1] |= (c << (GMP_LIMB_BITS-1));
  return ret;
}


/* An open-coded version, making one pass over the data.  The right shift is
   done as the added limbs are produced.  The addition code follows
   mpn/generic/add_n.c. */

mp_limb_t
mean_open (mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size)
{
  mp_limb_t  w, wprev, x, y, c, ret;
  mp_size_t  i;

  ASSERT (size >= 1);

  x = xp[0];
  y = yp[0];

  wprev = x + y;
  c = (wprev < x);
  ret = (wprev & 1);

#define RSHIFT(hi,lo)   (((lo) >> 1) | ((hi) << (GMP_LIMB_BITS-1)))

  for (i = 1; i < size; i++)
    {
      x = xp[i];
      y = yp[i];

      w = x + c;
      c = (w < x);
      w += y;
      c += (w < y);

      wp[i-1] = RSHIFT (w, wprev);
      wprev = w;
    }

  wp[i-1] = RSHIFT (c, wprev);

  return ret;
}


/* Another one-pass version, but right shifting the source limbs rather than
   the result limbs.  There's not much chance of this being better than the
   above, but it's an alternative at least. */

mp_limb_t
mean_open2 (mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size)
{
  mp_limb_t  w, x, y, xnext, ynext, c, ret;
  mp_size_t  i;

  ASSERT (size >= 1);

  x = xp[0];
  y = yp[0];

  /* ret is the low bit of x+y, c is the carry out of that low bit add */
  ret = (x ^ y) & 1;
  c   = (x & y) & 1;

  for (i = 0; i < size-1; i++)
    {
      xnext = xp[i+1];
      ynext = yp[i+1];
      x = RSHIFT (xnext, x);
      y = RSHIFT (ynext, y);

      w = x + c;
      c = (w < x);
      w += y;
      c += (w < y);
      wp[i] = w;

      x = xnext;
      y = ynext;
    }

  wp[i] = (x >> 1) + (y >> 1) + c;

  return ret;
}


/* The speed measuring routines are the same apart from which function they
   run, so a macro is used.  Actually this macro is the same as
   SPEED_ROUTINE_MPN_BINARY_N.  */

#define SPEED_ROUTINE_MEAN(mean_fun)                    \
  {                                                     \
    unsigned  i;                                        \
    mp_ptr    wp;                                       \
    double    t;                                        \
    TMP_DECL;                                  \
                                                        \
    SPEED_RESTRICT_COND (s->size >= 1);                 \
                                                        \
    TMP_MARK;                                  \
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);   \
                                                        \
    speed_operand_src (s, s->xp, s->size);              \
    speed_operand_src (s, s->yp, s->size);              \
    speed_operand_dst (s, wp, s->size);                 \
    speed_cache_fill (s);                               \
                                                        \
    speed_starttime ();                                 \
    i = s->reps;                                        \
    do                                                  \
      mean_fun (wp, s->xp, s->yp, s->size);             \
    while (--i != 0);                                   \
    t = speed_endtime ();                               \
                                                        \
    TMP_FREE;                                  \
    return t;                                           \
  }

double
speed_mean_calls (struct speed_params *s)
{
  SPEED_ROUTINE_MEAN (mean_calls);
}

double
speed_mean_open (struct speed_params *s)
{
  SPEED_ROUTINE_MEAN (mean_open);
}

double
speed_mean_open2 (struct speed_params *s)
{
  SPEED_ROUTINE_MEAN (mean_open2);
}
