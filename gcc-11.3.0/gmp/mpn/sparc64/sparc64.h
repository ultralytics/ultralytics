/* UltraSPARC 64 support macros.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2003 Free Software Foundation, Inc.

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


#define LOW32(x)   ((x) & 0xFFFFFFFF)
#define HIGH32(x)  ((x) >> 32)


/* Halfword number i in src is accessed as src[i+HALF_ENDIAN_ADJ(i)].
   Plain src[i] would be incorrect in big endian, HALF_ENDIAN_ADJ has the
   effect of swapping the two halves in this case.  */
#if HAVE_LIMB_BIG_ENDIAN
#define HALF_ENDIAN_ADJ(i)  (1 - (((i) & 1) << 1))   /* +1 even, -1 odd */
#endif
#if HAVE_LIMB_LITTLE_ENDIAN
#define HALF_ENDIAN_ADJ(i)  0                        /* no adjust */
#endif
#ifndef HALF_ENDIAN_ADJ
Error, error, unknown limb endianness;
#endif


/* umul_ppmm_lowequal sets h to the high limb of q*d, assuming the low limb
   of that product is equal to l.  dh and dl are the 32-bit halves of d.

   |-----high----||----low-----|
   +------+------+
   |             |                 ph = qh * dh
   +------+------+
          +------+------+
          |             |          pm1 = ql * dh
          +------+------+
          +------+------+
          |             |          pm2 = qh * dl
          +------+------+
                 +------+------+
                 |             |   pl = ql * dl (not calculated)
                 +------+------+

   Knowing that the low 64 bits is equal to l means that LOW(pm1) + LOW(pm2)
   + HIGH(pl) == HIGH(l).  The only thing we need from those product parts
   is whether they produce a carry into the high.

   pm_l = LOW(pm1)+LOW(pm2) is done to contribute its carry, then the only
   time there's a further carry from LOW(pm_l)+HIGH(pl) is if LOW(pm_l) >
   HIGH(l).  pl is never actually calculated.  */

#define umul_ppmm_lowequal(h, q, d, dh, dl, l)  \
  do {                                          \
    mp_limb_t  ql, qh, ph, pm1, pm2, pm_l;      \
    ASSERT (dh == HIGH32(d));                   \
    ASSERT (dl == LOW32(d));                    \
    ASSERT (q*d == l);                          \
                                                \
    ql = LOW32 (q);                             \
    qh = HIGH32 (q);                            \
                                                \
    pm1 = ql * dh;                              \
    pm2 = qh * dl;                              \
    ph  = qh * dh;                              \
                                                \
    pm_l = LOW32 (pm1) + LOW32 (pm2);           \
                                                \
    (h) = ph + HIGH32 (pm1) + HIGH32 (pm2)      \
      + HIGH32 (pm_l) + ((pm_l << 32) > l);     \
                                                \
    ASSERT_HIGH_PRODUCT (h, q, d);              \
  } while (0)


/* Set h to the high of q*d, assuming the low limb of that product is equal
   to l, and that d fits in 32-bits.

   |-----high----||----low-----|
          +------+------+
          |             |          pm = qh * dl
          +------+------+
                 +------+------+
                 |             |   pl = ql * dl (not calculated)
                 +------+------+

   Knowing that LOW(pm) + HIGH(pl) == HIGH(l) (mod 2^32) means that the only
   time there's a carry from that sum is when LOW(pm) > HIGH(l).  There's no
   need to calculate pl to determine this.  */

#define umul_ppmm_half_lowequal(h, q, d, l)     \
  do {                                          \
    mp_limb_t pm;                               \
    ASSERT (q*d == l);                          \
    ASSERT (HIGH32(d) == 0);                    \
                                                \
    pm = HIGH32(q) * d;                         \
    (h) = HIGH32(pm) + ((pm << 32) > l);        \
    ASSERT_HIGH_PRODUCT (h, q, d);              \
  } while (0)


/* check that h is the high limb of x*y */
#if WANT_ASSERT
#define ASSERT_HIGH_PRODUCT(h, x, y)    \
  do {                                  \
    mp_limb_t  want_h, dummy;           \
    umul_ppmm (want_h, dummy, x, y);    \
    ASSERT (h == want_h);               \
  } while (0)
#else
#define ASSERT_HIGH_PRODUCT(h, q, d)    \
  do { } while (0)
#endif


/* Multiply u anv v, where v < 2^32.  */
#define umul_ppmm_s(w1, w0, u, v)					\
  do {									\
    UWtype __x0, __x2;							\
    UWtype __ul, __vl, __uh;						\
    UWtype __u = (u), __v = (v);					\
									\
    __ul = __ll_lowpart (__u);						\
    __uh = __ll_highpart (__u);						\
    __vl = __ll_lowpart (__v);						\
									\
    __x0 = (UWtype) __ul * __vl;					\
    __x2 = (UWtype) __uh * __vl;					\
									\
    (w1) = (__x2 + (__x0 >> W_TYPE_SIZE/2)) >> W_TYPE_SIZE/2;		\
    (w0) = (__x2 << W_TYPE_SIZE/2) + __x0;				\
  } while (0)

/* Count the leading zeros on a limb, but assuming it fits in 32 bits.
   The count returned will be in the range 32 to 63.
   This is the 32-bit generic C count_leading_zeros from longlong.h. */
#define count_leading_zeros_32(count, x)                                      \
  do {                                                                        \
    mp_limb_t  __xr = (x);                                                    \
    unsigned   __a;                                                           \
    ASSERT ((x) != 0);                                                        \
    ASSERT ((x) <= CNST_LIMB(0xFFFFFFFF));                                    \
    __a = __xr < ((UWtype) 1 << 16) ? (__xr < ((UWtype) 1 << 8) ? 1 : 8 + 1)  \
      : (__xr < ((UWtype) 1 << 24)  ? 16 + 1 : 24 + 1);                       \
                                                                              \
    (count) = W_TYPE_SIZE + 1 - __a - __clz_tab[__xr >> __a];                 \
  } while (0)


/* Set inv to a 32-bit inverse floor((b*(b-d)-1) / d), knowing that d fits
   32 bits and is normalized (high bit set).  */
#define invert_half_limb(inv, d)                \
  do {                                          \
    mp_limb_t  _n;                              \
    ASSERT ((d) <= 0xFFFFFFFF);                 \
    ASSERT ((d) & 0x80000000);                  \
    _n = (((mp_limb_t) -(d)) << 32) - 1;        \
    (inv) = (mp_limb_t) (unsigned) (_n / (d));  \
  } while (0)


/* Divide nh:nl by d, setting q to the quotient and r to the remainder.
   q, r, nh and nl are 32-bits each, d_limb is 32-bits but in an mp_limb_t,
   dinv_limb is similarly a 32-bit inverse but in an mp_limb_t.  */

#define udiv_qrnnd_half_preinv(q, r, nh, nl, d_limb, dinv_limb)         \
  do {                                                                  \
    unsigned   _n2, _n10, _n1, _nadj, _q11n, _xh, _r, _q;               \
    mp_limb_t  _n, _x;                                                  \
    ASSERT (d_limb <= 0xFFFFFFFF);                                      \
    ASSERT (dinv_limb <= 0xFFFFFFFF);                                   \
    ASSERT (d_limb & 0x80000000);                                       \
    ASSERT (nh < d_limb);                                               \
    _n10 = (nl);                                                        \
    _n2 = (nh);                                                         \
    _n1 = (int) _n10 >> 31;                                             \
    _nadj = _n10 + (_n1 & d_limb);                                      \
    _x = dinv_limb * (_n2 - _n1) + _nadj;                               \
    _q11n = ~(_n2 + HIGH32 (_x));             /* -q1-1 */               \
    _n = ((mp_limb_t) _n2 << 32) + _n10;                                \
    _x = _n + d_limb * _q11n;                 /* n-q1*d-d */            \
    _xh = HIGH32 (_x) - d_limb;               /* high(n-q1*d-d) */      \
    ASSERT (_xh == 0 || _xh == ~0);                                     \
    _r = _x + (d_limb & _xh);                 /* addback */             \
    _q = _xh - _q11n;                         /* q1+1-addback */        \
    ASSERT (_r < d_limb);                                               \
    ASSERT (d_limb * _q + _r == _n);                                    \
    (r) = _r;                                                           \
    (q) = _q;                                                           \
  } while (0)


