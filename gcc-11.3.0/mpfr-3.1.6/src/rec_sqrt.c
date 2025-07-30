/* mpfr_rec_sqrt -- inverse square root

Copyright 2008-2017 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>

#define MPFR_NEED_LONGLONG_H /* for umul_ppmm */
#include "mpfr-impl.h"

#define LIMB_SIZE(x) ((((x)-1)>>MPFR_LOG2_GMP_NUMB_BITS) + 1)

#define MPFR_COM_N(x,y,n)                               \
  {                                                     \
    mp_size_t i;                                        \
    for (i = 0; i < n; i++)                             \
      *((x)+i) = ~*((y)+i);                             \
  }

/* Put in X a p-bit approximation of 1/sqrt(A),
   where X = {x, n}/B^n, n = ceil(p/GMP_NUMB_BITS),
   A = 2^(1+as)*{a, an}/B^an, as is 0 or 1, an = ceil(ap/GMP_NUMB_BITS),
   where B = 2^GMP_NUMB_BITS.

   We have 1 <= A < 4 and 1/2 <= X < 1.

   The error in the approximate result with respect to the true
   value 1/sqrt(A) is bounded by 1 ulp(X), i.e., 2^{-p} since 1/2 <= X < 1.

   Note: x and a are left-aligned, i.e., the most significant bit of
   a[an-1] is set, and so is the most significant bit of the output x[n-1].

   If p is not a multiple of GMP_NUMB_BITS, the extra low bits of the input
   A are taken into account to compute the approximation of 1/sqrt(A), but
   whether or not they are zero, the error between X and 1/sqrt(A) is bounded
   by 1 ulp(X) [in precision p].
   The extra low bits of the output X (if p is not a multiple of GMP_NUMB_BITS)
   are set to 0.

   Assumptions:
   (1) A should be normalized, i.e., the most significant bit of a[an-1]
       should be 1. If as=0, we have 1 <= A < 2; if as=1, we have 2 <= A < 4.
   (2) p >= 12
   (3) {a, an} and {x, n} should not overlap
   (4) GMP_NUMB_BITS >= 12 and is even

   Note: this routine is much more efficient when ap is small compared to p,
   including the case where ap <= GMP_NUMB_BITS, thus it can be used to
   implement an efficient mpfr_rec_sqrt_ui function.

   References:
   [1] Modern Computer Algebra, Richard Brent and Paul Zimmermann,
   http://www.loria.fr/~zimmerma/mca/pub226.html
*/
static void
mpfr_mpn_rec_sqrt (mpfr_limb_ptr x, mpfr_prec_t p,
                   mpfr_limb_srcptr a, mpfr_prec_t ap, int as)

{
  /* the following T1 and T2 are bipartite tables giving initial
     approximation for the inverse square root, with 13-bit input split in
     5+4+4, and 11-bit output. More precisely, if 2048 <= i < 8192,
     with i = a*2^8 + b*2^4 + c, we use for approximation of
     2048/sqrt(i/2048) the value x = T1[16*(a-8)+b] + T2[16*(a-8)+c].
     The largest error is obtained for i = 2054, where x = 2044,
     and 2048/sqrt(i/2048) = 2045.006576...
  */
  static short int T1[384] = {
2040, 2033, 2025, 2017, 2009, 2002, 1994, 1987, 1980, 1972, 1965, 1958, 1951,
1944, 1938, 1931, /* a=8 */
1925, 1918, 1912, 1905, 1899, 1892, 1886, 1880, 1874, 1867, 1861, 1855, 1849,
1844, 1838, 1832, /* a=9 */
1827, 1821, 1815, 1810, 1804, 1799, 1793, 1788, 1783, 1777, 1772, 1767, 1762,
1757, 1752, 1747, /* a=10 */
1742, 1737, 1733, 1728, 1723, 1718, 1713, 1709, 1704, 1699, 1695, 1690, 1686,
1681, 1677, 1673, /* a=11 */
1669, 1664, 1660, 1656, 1652, 1647, 1643, 1639, 1635, 1631, 1627, 1623, 1619,
1615, 1611, 1607, /* a=12 */
1603, 1600, 1596, 1592, 1588, 1585, 1581, 1577, 1574, 1570, 1566, 1563, 1559,
1556, 1552, 1549, /* a=13 */
1545, 1542, 1538, 1535, 1532, 1528, 1525, 1522, 1518, 1515, 1512, 1509, 1505,
1502, 1499, 1496, /* a=14 */
1493, 1490, 1487, 1484, 1481, 1478, 1475, 1472, 1469, 1466, 1463, 1460, 1457,
1454, 1451, 1449, /* a=15 */
1446, 1443, 1440, 1438, 1435, 1432, 1429, 1427, 1424, 1421, 1419, 1416, 1413,
1411, 1408, 1405, /* a=16 */
1403, 1400, 1398, 1395, 1393, 1390, 1388, 1385, 1383, 1380, 1378, 1375, 1373,
1371, 1368, 1366, /* a=17 */
1363, 1360, 1358, 1356, 1353, 1351, 1349, 1346, 1344, 1342, 1340, 1337, 1335,
1333, 1331, 1329, /* a=18 */
1327, 1325, 1323, 1321, 1319, 1316, 1314, 1312, 1310, 1308, 1306, 1304, 1302,
1300, 1298, 1296, /* a=19 */
1294, 1292, 1290, 1288, 1286, 1284, 1282, 1280, 1278, 1276, 1274, 1272, 1270,
1268, 1266, 1265, /* a=20 */
1263, 1261, 1259, 1257, 1255, 1253, 1251, 1250, 1248, 1246, 1244, 1242, 1241,
1239, 1237, 1235, /* a=21 */
1234, 1232, 1230, 1229, 1227, 1225, 1223, 1222, 1220, 1218, 1217, 1215, 1213,
1212, 1210, 1208, /* a=22 */
1206, 1204, 1203, 1201, 1199, 1198, 1196, 1195, 1193, 1191, 1190, 1188, 1187,
1185, 1184, 1182, /* a=23 */
1181, 1180, 1178, 1177, 1175, 1174, 1172, 1171, 1169, 1168, 1166, 1165, 1163,
1162, 1160, 1159, /* a=24 */
1157, 1156, 1154, 1153, 1151, 1150, 1149, 1147, 1146, 1144, 1143, 1142, 1140,
1139, 1137, 1136, /* a=25 */
1135, 1133, 1132, 1131, 1129, 1128, 1127, 1125, 1124, 1123, 1121, 1120, 1119,
1117, 1116, 1115, /* a=26 */
1114, 1113, 1111, 1110, 1109, 1108, 1106, 1105, 1104, 1103, 1101, 1100, 1099,
1098, 1096, 1095, /* a=27 */
1093, 1092, 1091, 1090, 1089, 1087, 1086, 1085, 1084, 1083, 1081, 1080, 1079,
1078, 1077, 1076, /* a=28 */
1075, 1073, 1072, 1071, 1070, 1069, 1068, 1067, 1065, 1064, 1063, 1062, 1061,
1060, 1059, 1058, /* a=29 */
1057, 1056, 1055, 1054, 1052, 1051, 1050, 1049, 1048, 1047, 1046, 1045, 1044,
1043, 1042, 1041, /* a=30 */
1040, 1039, 1038, 1037, 1036, 1035, 1034, 1033, 1032, 1031, 1030, 1029, 1028,
1027, 1026, 1025 /* a=31 */
};
  static unsigned char T2[384] = {
    7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, /* a=8 */
    6, 5, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 0, 0, /* a=9 */
    5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, /* a=10 */
    4, 4, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, /* a=11 */
    3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, /* a=12 */
    3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, /* a=13 */
    3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, /* a=14 */
    2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, /* a=15 */
    2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, /* a=16 */
    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, /* a=17 */
    3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, /* a=18 */
    2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, /* a=19 */
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, /* a=20 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, /* a=21 */
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* a=22 */
    2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, /* a=23 */
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* a=24 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, /* a=25 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, /* a=26 */
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* a=27 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, /* a=28 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, /* a=29 */
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* a=30 */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  /* a=31 */
};
  mp_size_t n = LIMB_SIZE(p);   /* number of limbs of X */
  mp_size_t an = LIMB_SIZE(ap); /* number of limbs of A */

  /* A should be normalized */
  MPFR_ASSERTD((a[an - 1] & MPFR_LIMB_HIGHBIT) != 0);
  /* We should have enough bits in one limb and GMP_NUMB_BITS should be even.
     Since that does not depend on MPFR, we always check this. */
  MPFR_ASSERTN((GMP_NUMB_BITS >= 12) && ((GMP_NUMB_BITS & 1) == 0));
  /* {a, an} and {x, n} should not overlap */
  MPFR_ASSERTD((a + an <= x) || (x + n <= a));
  MPFR_ASSERTD(p >= 11);

  if (MPFR_UNLIKELY(an > n)) /* we can cut the input to n limbs */
    {
      a += an - n;
      an = n;
    }

  if (p == 11) /* should happen only from recursive calls */
    {
      unsigned long i, ab, ac;
      mp_limb_t t;

      /* take the 12+as most significant bits of A */
      i = a[an - 1] >> (GMP_NUMB_BITS - (12 + as));
      /* if one wants faithful rounding for p=11, replace #if 0 by #if 1 */
      ab = i >> 4;
      ac = (ab & 0x3F0) | (i & 0x0F);
      t = (mp_limb_t) T1[ab - 0x80] + (mp_limb_t) T2[ac - 0x80];
      x[0] = t << (GMP_NUMB_BITS - p);
    }
  else /* p >= 12 */
    {
      mpfr_prec_t h, pl;
      mpfr_limb_ptr r, s, t, u;
      mp_size_t xn, rn, th, ln, tn, sn, ahn, un;
      mp_limb_t neg, cy, cu;
      MPFR_TMP_DECL(marker);

      /* compared to Algorithm 3.9 of [1], we have {a, an} = A/2 if as=0,
         and A/4 if as=1. */

      /* h = max(11, ceil((p+3)/2)) is the bitsize of the recursive call */
      h = (p < 18) ? 11 : (p >> 1) + 2;

      xn = LIMB_SIZE(h);       /* limb size of the recursive Xh */
      rn = LIMB_SIZE(2 * h);   /* a priori limb size of Xh^2 */
      ln = n - xn;             /* remaining limbs to be computed */

      /* Since |Xh - A^{-1/2}| <= 2^{-h}, then by multiplying by Xh + A^{-1/2}
         we get |Xh^2 - 1/A| <= 2^{-h+1}, thus |A*Xh^2 - 1| <= 2^{-h+3},
         thus the h-3 most significant bits of t should be zero,
         which is in fact h+1+as-3 because of the normalization of A.
         This corresponds to th=floor((h+1+as-3)/GMP_NUMB_BITS) limbs.

         More precisely we have |Xh^2 - 1/A| <= 2^{-h} * (Xh + A^{-1/2})
         <= 2^{-h} * (2 A^{-1/2} + 2^{-h}) <= 2.001 * 2^{-h} * A^{-1/2}
         since A < 4 and h >= 11, thus
         |A*Xh^2 - 1| <= 2.001 * 2^{-h} * A^{1/2} <= 1.001 * 2^{2-h}.
         This is sufficient to prove that the upper limb of {t,tn} below is
         less that 0.501 * 2^GMP_NUMB_BITS, thus cu = 0 below.
      */
      th = (h + 1 + as - 3) >> MPFR_LOG2_GMP_NUMB_BITS;
      tn = LIMB_SIZE(2 * h + 1 + as);

      /* we need h+1+as bits of a */
      ahn = LIMB_SIZE(h + 1 + as); /* number of high limbs of A
                                      needed for the recursive call*/
      if (MPFR_UNLIKELY(ahn > an))
        ahn = an;
      mpfr_mpn_rec_sqrt (x + ln, h, a + an - ahn, ahn * GMP_NUMB_BITS, as);
      /* the most h significant bits of X are set, X has ceil(h/GMP_NUMB_BITS)
         limbs, the low (-h) % GMP_NUMB_BITS bits are zero */

      /* compared to Algorithm 3.9 of [1], we have {x+ln,xn} = X_h */

      MPFR_TMP_MARK (marker);
      /* first step: square X in r, result is exact */
      un = xn + (tn - th);
      /* We use the same temporary buffer to store r and u: r needs 2*xn
         limbs where u needs xn+(tn-th) limbs. Since tn can store at least
         2h bits, and th at most h bits, then tn-th can store at least h bits,
         thus tn - th >= xn, and reserving the space for u is enough. */
      MPFR_ASSERTD(2 * xn <= un);
      u = r = MPFR_TMP_LIMBS_ALLOC (un);
      if (2 * h <= GMP_NUMB_BITS) /* xn=rn=1, and since p <= 2h-3, n=1,
                                     thus ln = 0 */
        {
          MPFR_ASSERTD(ln == 0);
          cy = x[0] >> (GMP_NUMB_BITS >> 1);
          r ++;
          r[0] = cy * cy;
        }
      else if (xn == 1) /* xn=1, rn=2 */
        umul_ppmm(r[1], r[0], x[ln], x[ln]);
      else
        {
          mpn_mul_n (r, x + ln, x + ln, xn);
          /* we have {r, 2*xn} = X_h^2 */
          if (rn < 2 * xn)
            r ++;
        }
      /* now the 2h most significant bits of {r, rn} contains X^2, r has rn
         limbs, and the low (-2h) % GMP_NUMB_BITS bits are zero */

      /* Second step: s <- A * (r^2), and truncate the low ap bits,
         i.e., at weight 2^{-2h} (s is aligned to the low significant bits)
       */
      sn = an + rn;
      s = MPFR_TMP_LIMBS_ALLOC (sn);
      if (rn == 1) /* rn=1 implies n=1, since rn*GMP_NUMB_BITS >= 2h,
                           and 2h >= p+3 */
        {
          /* necessarily p <= GMP_NUMB_BITS-3: we can ignore the two low
             bits from A */
          /* since n=1, and we ensured an <= n, we also have an=1 */
          MPFR_ASSERTD(an == 1);
          umul_ppmm (s[1], s[0], r[0], a[0]);
        }
      else
        {
          /* we have p <= n * GMP_NUMB_BITS
             2h <= rn * GMP_NUMB_BITS with p+3 <= 2h <= p+4
             thus n <= rn <= n + 1 */
          MPFR_ASSERTD(rn <= n + 1);
          /* since we ensured an <= n, we have an <= rn */
          MPFR_ASSERTD(an <= rn);
          mpn_mul (s, r, rn, a, an);
          /* s should be near B^sn/2^(1+as), thus s[sn-1] is either
             100000... or 011111... if as=0, or
             010000... or 001111... if as=1.
             We ignore the bits of s after the first 2h+1+as ones.
             We have {s, rn+an} = A*X_h^2/2 if as=0, A*X_h^2/4 if as=1. */
        }

      /* We ignore the bits of s after the first 2h+1+as ones: s has rn + an
         limbs, where rn = LIMBS(2h), an=LIMBS(a), and tn = LIMBS(2h+1+as). */
      t = s + sn - tn; /* pointer to low limb of the high part of t */
      /* the upper h-3 bits of 1-t should be zero,
         where 1 corresponds to the most significant bit of t[tn-1] if as=0,
         and to the 2nd most significant bit of t[tn-1] if as=1 */

      /* compute t <- 1 - t, which is B^tn - {t, tn+1},
         with rounding toward -Inf, i.e., rounding the input t toward +Inf.
         We could only modify the low tn - th limbs from t, but it gives only
         a small speedup, and would make the code more complex.
      */
      neg = t[tn - 1] & (MPFR_LIMB_HIGHBIT >> as);
      if (neg == 0) /* Ax^2 < 1: we have t = th + eps, where 0 <= eps < ulp(th)
                       is the part truncated above, thus 1 - t rounded to -Inf
                       is 1 - th - ulp(th) */
        {
          /* since the 1+as most significant bits of t are zero, set them
             to 1 before the one-complement */
          t[tn - 1] |= MPFR_LIMB_HIGHBIT | (MPFR_LIMB_HIGHBIT >> as);
          MPFR_COM_N (t, t, tn);
          /* we should add 1 here to get 1-th complement, and subtract 1 for
             -ulp(th), thus we do nothing */
        }
      else /* negative case: we want 1 - t rounded toward -Inf, i.e.,
              th + eps rounded toward +Inf, which is th + ulp(th):
              we discard the bit corresponding to 1,
              and we add 1 to the least significant bit of t */
        {
          t[tn - 1] ^= neg;
          mpn_add_1 (t, t, tn, 1);
        }
      tn -= th; /* we know at least th = floor((h+1+as-3)/GMP_NUMB_LIMBS) of
                   the high limbs of {t, tn} are zero */

      /* tn = rn - th, where rn * GMP_NUMB_BITS >= 2*h and
         th * GMP_NUMB_BITS <= h+1+as-3, thus tn > 0 */
      MPFR_ASSERTD(tn > 0);

      /* u <- x * t, where {t, tn} contains at least h+3 bits,
         and {x, xn} contains h bits, thus tn >= xn */
      MPFR_ASSERTD(tn >= xn);
      if (tn == 1) /* necessarily xn=1 */
        umul_ppmm (u[1], u[0], t[0], x[ln]);
      else
        mpn_mul (u, t, tn, x + ln, xn);

      /* we have {u, tn+xn} = T_l X_h/2 if as=0, T_l X_h/4 if as=1 */

      /* we have already discarded the upper th high limbs of t, thus we only
         have to consider the upper n - th limbs of u */
      un = n - th; /* un cannot be zero, since p <= n*GMP_NUMB_BITS,
                      h = ceil((p+3)/2) <= (p+4)/2,
                      th*GMP_NUMB_BITS <= h-1 <= p/2+1,
                      thus (n-th)*GMP_NUMB_BITS >= p/2-1.
                   */
      MPFR_ASSERTD(un > 0);
      u += (tn + xn) - un; /* xn + tn - un = xn + (original_tn - th) - (n - th)
                                           = xn + original_tn - n
                              = LIMBS(h) + LIMBS(2h+1+as) - LIMBS(p) > 0
                              since 2h >= p+3 */
      MPFR_ASSERTD(tn + xn > un); /* will allow to access u[-1] below */

      /* In case as=0, u contains |x*(1-Ax^2)/2|, which is exactly what we
         need to add or subtract.
         In case as=1, u contains |x*(1-Ax^2)/4|, thus we need to multiply
         u by 2. */

      if (as == 1)
        /* shift on un+1 limbs to get most significant bit of u[-1] into
           least significant bit of u[0] */
        mpn_lshift (u - 1, u - 1, un + 1, 1);

      /* now {u,un} represents U / 2 from Algorithm 3.9 */

      pl = n * GMP_NUMB_BITS - p;       /* low bits from x */
      /* We want that the low pl bits are zero after rounding to nearest,
         thus we round u to nearest at bit pl-1 of u[0] */
      if (pl > 0)
        {
          cu = mpn_add_1 (u, u, un, u[0] & (MPFR_LIMB_ONE << (pl - 1)));
          /* mask bits 0..pl-1 of u[0] */
          u[0] &= ~MPFR_LIMB_MASK(pl);
        }
      else /* round bit is in u[-1] */
        cu = mpn_add_1 (u, u, un, u[-1] >> (GMP_NUMB_BITS - 1));
      MPFR_ASSERTN(cu == 0);

      /* We already have filled {x + ln, xn = n - ln}, and we want to add or
         subtract {u, un} at position x.
         un = n - th, where th contains <= h+1+as-3<=h-1 bits
         ln = n - xn, where xn contains >= h bits
         thus un > ln.
         Warning: ln might be zero.
      */
      MPFR_ASSERTD(un > ln);
      /* we can have un = ln + 2, for example with GMP_NUMB_BITS=32 and
         p=62, as=0, then h=33, n=2, th=0, xn=2, thus un=2 and ln=0. */
      MPFR_ASSERTD(un == ln + 1 || un == ln + 2);
      /* the high un-ln limbs of u will overlap the low part of {x+ln,xn},
         we need to add or subtract the overlapping part {u + ln, un - ln} */
      /* Warning! th may be 0, in which case the mpn_add_1 and mpn_sub_1
         below (with size = th) mustn't be used. */
      if (neg == 0)
        {
          if (ln > 0)
            MPN_COPY (x, u, ln);
          cy = mpn_add (x + ln, x + ln, xn, u + ln, un - ln);
          /* cy is the carry at x + (ln + xn) = x + n */
        }
      else /* negative case */
        {
          /* subtract {u+ln, un-ln} from {x+ln,un} */
          cy = mpn_sub (x + ln, x + ln, xn, u + ln, un - ln);
          /* cy is the borrow at x + (ln + xn) = x + n */

          /* cy cannot be non-zero, since the most significant bit of Xh is 1,
             and the correction is bounded by 2^{-h+3} */
          MPFR_ASSERTD(cy == 0);
          if (ln > 0)
            {
              MPFR_COM_N (x, u, ln);
              /* we must add one for the 2-complement ... */
              cy = mpn_add_1 (x, x, n, MPFR_LIMB_ONE);
              /* ... and subtract 1 at x[ln], where n = ln + xn */
              cy -= mpn_sub_1 (x + ln, x + ln, xn, MPFR_LIMB_ONE);
            }
        }

      /* cy can be 1 when A=1, i.e., {a, n} = B^n. In that case we should
         have X = B^n, and setting X to 1-2^{-p} satisties the error bound
         of 1 ulp. */
      if (MPFR_UNLIKELY(cy != 0))
        {
          cy -= mpn_sub_1 (x, x, n, MPFR_LIMB_ONE << pl);
          MPFR_ASSERTD(cy == 0);
        }

      MPFR_TMP_FREE (marker);
    }
}

int
mpfr_rec_sqrt (mpfr_ptr r, mpfr_srcptr u, mpfr_rnd_t rnd_mode)
{
  mpfr_prec_t rp, up, wp;
  mp_size_t rn, wn;
  int s, cy, inex;
  mpfr_limb_ptr x;
  MPFR_TMP_DECL(marker);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (u), mpfr_log_prec, u, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (r), mpfr_log_prec, r, inex));

  /* special values */
  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(u)))
    {
      if (MPFR_IS_NAN(u))
        {
          MPFR_SET_NAN(r);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_ZERO(u)) /* 1/sqrt(+0) = 1/sqrt(-0) = +Inf */
        {
          /* 0+ or 0- */
          MPFR_SET_INF(r);
          MPFR_SET_POS(r);
          mpfr_set_divby0 ();
          MPFR_RET(0); /* Inf is exact */
        }
      else
        {
          MPFR_ASSERTD(MPFR_IS_INF(u));
          /* 1/sqrt(-Inf) = NAN */
          if (MPFR_IS_NEG(u))
            {
              MPFR_SET_NAN(r);
              MPFR_RET_NAN;
            }
          /* 1/sqrt(+Inf) = +0 */
          MPFR_SET_POS(r);
          MPFR_SET_ZERO(r);
          MPFR_RET(0);
        }
    }

  /* if u < 0, 1/sqrt(u) is NaN */
  if (MPFR_UNLIKELY(MPFR_IS_NEG(u)))
    {
      MPFR_SET_NAN(r);
      MPFR_RET_NAN;
    }

  MPFR_SET_POS(r);

  rp = MPFR_PREC(r); /* output precision */
  up = MPFR_PREC(u); /* input precision */
  wp = rp + 11;      /* initial working precision */

  /* Let u = U*2^e, where e = EXP(u), and 1/2 <= U < 1.
     If e is even, we compute an approximation of X of (4U)^{-1/2},
     and the result is X*2^(-(e-2)/2) [case s=1].
     If e is odd, we compute an approximation of X of (2U)^{-1/2},
     and the result is X*2^(-(e-1)/2) [case s=0]. */

  /* parity of the exponent of u */
  s = 1 - ((mpfr_uexp_t) MPFR_GET_EXP (u) & 1);

  rn = LIMB_SIZE(rp);

  /* for the first iteration, if rp + 11 fits into rn limbs, we round up
     up to a full limb to maximize the chance of rounding, while avoiding
     to allocate extra space */
  wp = rp + 11;
  if (wp < rn * GMP_NUMB_BITS)
    wp = rn * GMP_NUMB_BITS;
  for (;;)
    {
      MPFR_TMP_MARK (marker);
      wn = LIMB_SIZE(wp);
      if (r == u || wn > rn) /* out of place, i.e., we cannot write to r */
        x = MPFR_TMP_LIMBS_ALLOC (wn);
      else
        x = MPFR_MANT(r);
      mpfr_mpn_rec_sqrt (x, wp, MPFR_MANT(u), up, s);
      /* If the input was not truncated, the error is at most one ulp;
         if the input was truncated, the error is at most two ulps
         (see algorithms.tex). */
      if (MPFR_LIKELY (mpfr_round_p (x, wn, wp - (wp < up),
                                     rp + (rnd_mode == MPFR_RNDN))))
        break;

      /* We detect only now the exact case where u=2^(2e), to avoid
         slowing down the average case. This can happen only when the
         mantissa is exactly 1/2 and the exponent is odd. */
      if (s == 0 && mpfr_cmp_ui_2exp (u, 1, MPFR_EXP(u) - 1) == 0)
        {
          mpfr_prec_t pl = wn * GMP_NUMB_BITS - wp;

          /* we should have x=111...111 */
          mpn_add_1 (x, x, wn, MPFR_LIMB_ONE << pl);
          x[wn - 1] = MPFR_LIMB_HIGHBIT;
          s += 2;
          break; /* go through */
        }
      MPFR_TMP_FREE(marker);

      wp += GMP_NUMB_BITS;
    }
  cy = mpfr_round_raw (MPFR_MANT(r), x, wp, 0, rp, rnd_mode, &inex);
  MPFR_EXP(r) = - (MPFR_EXP(u) - 1 - s) / 2;
  if (MPFR_UNLIKELY(cy != 0))
    {
      MPFR_EXP(r) ++;
      MPFR_MANT(r)[rn - 1] = MPFR_LIMB_HIGHBIT;
    }
  MPFR_TMP_FREE(marker);
  return mpfr_check_range (r, inex, rnd_mode);
}
