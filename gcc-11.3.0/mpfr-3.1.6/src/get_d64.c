/* mpfr_get_decimal64 -- convert a multiple precision floating-point number
                         to a IEEE 754r decimal64 float

See http://gcc.gnu.org/ml/gcc/2006-06/msg00691.html,
http://gcc.gnu.org/onlinedocs/gcc/Decimal-Float.html,
and TR 24732 <http://www.open-std.org/jtc1/sc22/wg14/www/projects#24732>.

Copyright 2006-2017 Free Software Foundation, Inc.
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

#include <stdlib.h> /* for strtol */
#include "mpfr-impl.h"

#define ISDIGIT(c) ('0' <= c && c <= '9')

#ifdef MPFR_WANT_DECIMAL_FLOATS

#ifndef DEC64_MAX
# define DEC64_MAX 9.999999999999999E384dd
#endif

#ifdef DPD_FORMAT
static int T[1000] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32,
  33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 80, 81, 82, 83, 84, 85, 86, 87, 88,
  89, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 112, 113, 114, 115, 116,
  117, 118, 119, 120, 121, 10, 11, 42, 43, 74, 75, 106, 107, 78, 79, 26, 27,
  58, 59, 90, 91, 122, 123, 94, 95, 128, 129, 130, 131, 132, 133, 134, 135,
  136, 137, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 160, 161, 162,
  163, 164, 165, 166, 167, 168, 169, 176, 177, 178, 179, 180, 181, 182, 183,
  184, 185, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 208, 209, 210,
  211, 212, 213, 214, 215, 216, 217, 224, 225, 226, 227, 228, 229, 230, 231,
  232, 233, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 138, 139, 170,
  171, 202, 203, 234, 235, 206, 207, 154, 155, 186, 187, 218, 219, 250, 251,
  222, 223, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 272, 273, 274,
  275, 276, 277, 278, 279, 280, 281, 288, 289, 290, 291, 292, 293, 294, 295,
  296, 297, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 320, 321, 322,
  323, 324, 325, 326, 327, 328, 329, 336, 337, 338, 339, 340, 341, 342, 343,
  344, 345, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 368, 369, 370,
  371, 372, 373, 374, 375, 376, 377, 266, 267, 298, 299, 330, 331, 362, 363,
  334, 335, 282, 283, 314, 315, 346, 347, 378, 379, 350, 351, 384, 385, 386,
  387, 388, 389, 390, 391, 392, 393, 400, 401, 402, 403, 404, 405, 406, 407,
  408, 409, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 432, 433, 434,
  435, 436, 437, 438, 439, 440, 441, 448, 449, 450, 451, 452, 453, 454, 455,
  456, 457, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 480, 481, 482,
  483, 484, 485, 486, 487, 488, 489, 496, 497, 498, 499, 500, 501, 502, 503,
  504, 505, 394, 395, 426, 427, 458, 459, 490, 491, 462, 463, 410, 411, 442,
  443, 474, 475, 506, 507, 478, 479, 512, 513, 514, 515, 516, 517, 518, 519,
  520, 521, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 544, 545, 546,
  547, 548, 549, 550, 551, 552, 553, 560, 561, 562, 563, 564, 565, 566, 567,
  568, 569, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 592, 593, 594,
  595, 596, 597, 598, 599, 600, 601, 608, 609, 610, 611, 612, 613, 614, 615,
  616, 617, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 522, 523, 554,
  555, 586, 587, 618, 619, 590, 591, 538, 539, 570, 571, 602, 603, 634, 635,
  606, 607, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 656, 657, 658,
  659, 660, 661, 662, 663, 664, 665, 672, 673, 674, 675, 676, 677, 678, 679,
  680, 681, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 704, 705, 706,
  707, 708, 709, 710, 711, 712, 713, 720, 721, 722, 723, 724, 725, 726, 727,
  728, 729, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 752, 753, 754,
  755, 756, 757, 758, 759, 760, 761, 650, 651, 682, 683, 714, 715, 746, 747,
  718, 719, 666, 667, 698, 699, 730, 731, 762, 763, 734, 735, 768, 769, 770,
  771, 772, 773, 774, 775, 776, 777, 784, 785, 786, 787, 788, 789, 790, 791,
  792, 793, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 816, 817, 818,
  819, 820, 821, 822, 823, 824, 825, 832, 833, 834, 835, 836, 837, 838, 839,
  840, 841, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 864, 865, 866,
  867, 868, 869, 870, 871, 872, 873, 880, 881, 882, 883, 884, 885, 886, 887,
  888, 889, 778, 779, 810, 811, 842, 843, 874, 875, 846, 847, 794, 795, 826,
  827, 858, 859, 890, 891, 862, 863, 896, 897, 898, 899, 900, 901, 902, 903,
  904, 905, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 928, 929, 930,
  931, 932, 933, 934, 935, 936, 937, 944, 945, 946, 947, 948, 949, 950, 951,
  952, 953, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 976, 977, 978,
  979, 980, 981, 982, 983, 984, 985, 992, 993, 994, 995, 996, 997, 998, 999,
  1000, 1001, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 906,
  907, 938, 939, 970, 971, 1002, 1003, 974, 975, 922, 923, 954, 955, 986,
  987, 1018, 1019, 990, 991, 12, 13, 268, 269, 524, 525, 780, 781, 46, 47, 28,
  29, 284, 285, 540, 541, 796, 797, 62, 63, 44, 45, 300, 301, 556, 557, 812,
  813, 302, 303, 60, 61, 316, 317, 572, 573, 828, 829, 318, 319, 76, 77,
  332, 333, 588, 589, 844, 845, 558, 559, 92, 93, 348, 349, 604, 605, 860,
  861, 574, 575, 108, 109, 364, 365, 620, 621, 876, 877, 814, 815, 124, 125,
  380, 381, 636, 637, 892, 893, 830, 831, 14, 15, 270, 271, 526, 527, 782,
  783, 110, 111, 30, 31, 286, 287, 542, 543, 798, 799, 126, 127, 140, 141,
  396, 397, 652, 653, 908, 909, 174, 175, 156, 157, 412, 413, 668, 669, 924,
  925, 190, 191, 172, 173, 428, 429, 684, 685, 940, 941, 430, 431, 188, 189,
  444, 445, 700, 701, 956, 957, 446, 447, 204, 205, 460, 461, 716, 717, 972,
  973, 686, 687, 220, 221, 476, 477, 732, 733, 988, 989, 702, 703, 236, 237,
  492, 493, 748, 749, 1004, 1005, 942, 943, 252, 253, 508, 509, 764, 765,
  1020, 1021, 958, 959, 142, 143, 398, 399, 654, 655, 910, 911, 238, 239, 158,
  159, 414, 415, 670, 671, 926, 927, 254, 255};
#endif

/* construct a decimal64 NaN */
static _Decimal64
get_decimal64_nan (void)
{
  union ieee_double_extract x;
  union ieee_double_decimal64 y;

  x.s.exp = 1984; /* G[0]..G[4] = 11111: quiet NaN */
  y.d = x.d;
  return y.d64;
}

/* construct the decimal64 Inf with given sign */
static _Decimal64
get_decimal64_inf (int negative)
{
  union ieee_double_extract x;
  union ieee_double_decimal64 y;

  x.s.sig = (negative) ? 1 : 0;
  x.s.exp = 1920; /* G[0]..G[4] = 11110: Inf */
  y.d = x.d;
  return y.d64;
}

/* construct the decimal64 zero with given sign */
static _Decimal64
get_decimal64_zero (int negative)
{
  union ieee_double_decimal64 y;

  /* zero has the same representation in binary64 and decimal64 */
  y.d = negative ? DBL_NEG_ZERO : 0.0;
  return y.d64;
}

/* construct the decimal64 smallest non-zero with given sign */
static _Decimal64
get_decimal64_min (int negative)
{
  return negative ? - 1E-398dd : 1E-398dd;
}

/* construct the decimal64 largest finite number with given sign */
static _Decimal64
get_decimal64_max (int negative)
{
  return negative ? - DEC64_MAX : DEC64_MAX;
}

/* one-to-one conversion:
   s is a decimal string representing a number x = m * 10^e which must be
   exactly representable in the decimal64 format, i.e.
   (a) the mantissa m has at most 16 decimal digits
   (b1) -383 <= e <= 384 with m integer multiple of 10^(-15), |m| < 10
   (b2) or -398 <= e <= 369 with m integer, |m| < 10^16.
   Assumes s is neither NaN nor +Inf nor -Inf.
*/
static _Decimal64
string_to_Decimal64 (char *s)
{
  long int exp = 0;
  char m[17];
  long n = 0; /* mantissa length */
  char *endptr[1];
  union ieee_double_extract x;
  union ieee_double_decimal64 y;
#ifdef DPD_FORMAT
  unsigned int G, d1, d2, d3, d4, d5;
#endif

  /* read sign */
  if (*s == '-')
    {
      x.s.sig = 1;
      s ++;
    }
  else
    x.s.sig = 0;
  /* read mantissa */
  while (ISDIGIT (*s))
    m[n++] = *s++;
  exp = n;
  if (*s == '.')
    {
      s ++;
      while (ISDIGIT (*s))
        m[n++] = *s++;
    }
  /* we have exp digits before decimal point, and a total of n digits */
  exp -= n; /* we will consider an integer mantissa */
  MPFR_ASSERTN(n <= 16);
  if (*s == 'E' || *s == 'e')
    exp += strtol (s + 1, endptr, 10);
  else
    *endptr = s;
  MPFR_ASSERTN(**endptr == '\0');
  MPFR_ASSERTN(-398 <= exp && exp <= (long) (385 - n));
  while (n < 16)
    {
      m[n++] = '0';
      exp --;
    }
  /* now n=16 and -398 <= exp <= 369 */
  m[n] = '\0';

  /* compute biased exponent */
  exp += 398;

  MPFR_ASSERTN(exp >= -15);
  if (exp < 0)
    {
      int i;
      n = -exp;
      /* check the last n digits of the mantissa are zero */
      for (i = 1; i <= n; i++)
        MPFR_ASSERTN(m[16 - n] == '0');
      /* shift the first (16-n) digits to the right */
      for (i = 16 - n - 1; i >= 0; i--)
        m[i + n] = m[i];
      /* zero the first n digits */
      for (i = 0; i < n; i ++)
        m[i] = '0';
      exp = 0;
    }

  /* now convert to DPD or BID */
#ifdef DPD_FORMAT
#define CH(d) (d - '0')
  if (m[0] >= '8')
    G = (3 << 11) | ((exp & 768) << 1) | ((CH(m[0]) & 1) << 8);
  else
    G = ((exp & 768) << 3) | (CH(m[0]) << 8);
  /* now the most 5 significant bits of G are filled */
  G |= exp & 255;
  d1 = T[100 * CH(m[1]) + 10 * CH(m[2]) + CH(m[3])]; /* 10-bit encoding */
  d2 = T[100 * CH(m[4]) + 10 * CH(m[5]) + CH(m[6])]; /* 10-bit encoding */
  d3 = T[100 * CH(m[7]) + 10 * CH(m[8]) + CH(m[9])]; /* 10-bit encoding */
  d4 = T[100 * CH(m[10]) + 10 * CH(m[11]) + CH(m[12])]; /* 10-bit encoding */
  d5 = T[100 * CH(m[13]) + 10 * CH(m[14]) + CH(m[15])]; /* 10-bit encoding */
  x.s.exp = G >> 2;
  x.s.manh = ((G & 3) << 18) | (d1 << 8) | (d2 >> 2);
  x.s.manl = (d2 & 3) << 30;
  x.s.manl |= (d3 << 20) | (d4 << 10) | d5;
#else /* BID format */
  {
    mp_size_t rn;
    mp_limb_t rp[2];
    int case_i = strcmp (m, "9007199254740992") < 0;

    for (n = 0; n < 16; n++)
      m[n] -= '0';
    rn = mpn_set_str (rp, (unsigned char *) m, 16, 10);
    if (rn == 1)
      rp[1] = 0;
#if GMP_NUMB_BITS > 32
    rp[1] = rp[1] << (GMP_NUMB_BITS - 32);
    rp[1] |= rp[0] >> 32;
    rp[0] &= 4294967295UL;
#endif
    if (case_i)
      {  /* s < 2^53: case i) */
        x.s.exp = exp << 1;
        x.s.manl = rp[0];           /* 32 bits */
        x.s.manh = rp[1] & 1048575; /* 20 low bits */
        x.s.exp |= rp[1] >> 20;     /* 1 bit */
      }
    else /* s >= 2^53: case ii) */
      {
        x.s.exp = 1536 | (exp >> 1);
        x.s.manl = rp[0];
        x.s.manh = (rp[1] ^ 2097152) | ((exp & 1) << 19);
      }
  }
#endif /* DPD_FORMAT */
  y.d = x.d;
  return y.d64;
}

_Decimal64
mpfr_get_decimal64 (mpfr_srcptr src, mpfr_rnd_t rnd_mode)
{
  int negative;
  mpfr_exp_t e;

  /* the encoding of NaN, Inf, zero is the same under DPD or BID */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (src)))
    {
      if (MPFR_IS_NAN (src))
        return get_decimal64_nan ();

      negative = MPFR_IS_NEG (src);

      if (MPFR_IS_INF (src))
        return get_decimal64_inf (negative);

      MPFR_ASSERTD (MPFR_IS_ZERO(src));
      return get_decimal64_zero (negative);
    }

  e = MPFR_GET_EXP (src);
  negative = MPFR_IS_NEG (src);

  if (MPFR_UNLIKELY(rnd_mode == MPFR_RNDA))
    rnd_mode = negative ? MPFR_RNDD : MPFR_RNDU;

  /* the smallest decimal64 number is 10^(-398),
     with 2^(-1323) < 10^(-398) < 2^(-1322) */
  if (MPFR_UNLIKELY (e < -1323)) /* src <= 2^(-1324) < 1/2*10^(-398) */
    {
      if (rnd_mode == MPFR_RNDZ || rnd_mode == MPFR_RNDN
          || (rnd_mode == MPFR_RNDD && negative == 0)
          || (rnd_mode == MPFR_RNDU && negative != 0))
        return get_decimal64_zero (negative);
      else /* return the smallest non-zero number */
        return get_decimal64_min (negative);
    }
  /* the largest decimal64 number is just below 10^(385) < 2^1279 */
  else if (MPFR_UNLIKELY (e > 1279)) /* then src >= 2^1279 */
    {
      if (rnd_mode == MPFR_RNDZ
          || (rnd_mode == MPFR_RNDU && negative != 0)
          || (rnd_mode == MPFR_RNDD && negative == 0))
        return get_decimal64_max (negative);
      else
        return get_decimal64_inf (negative);
    }
  else
    {
      /* we need to store the sign (1), the mantissa (16), and the terminating
         character, thus we need at least 18 characters in s */
      char s[23];
      mpfr_get_str (s, &e, 10, 16, src, rnd_mode);
      /* the smallest normal number is 1.000...000E-383,
         which corresponds to s=[0.]1000...000 and e=-382 */
      if (e < -382)
        {
          /* the smallest subnormal number is 0.000...001E-383 = 1E-398,
             which corresponds to s=[0.]1000...000 and e=-397 */
          if (e < -397)
            {
              if (rnd_mode == MPFR_RNDN && e == -398)
                {
                  /* If 0.5E-398 < |src| < 1E-398 (smallest subnormal),
                     src should round to +/- 1E-398 in MPFR_RNDN. */
                  mpfr_get_str (s, &e, 10, 1, src, MPFR_RNDA);
                  return e == -398 && s[negative] <= '5' ?
                    get_decimal64_zero (negative) :
                    get_decimal64_min (negative);
                }
              if (rnd_mode == MPFR_RNDZ || rnd_mode == MPFR_RNDN
                  || (rnd_mode == MPFR_RNDD && negative == 0)
                  || (rnd_mode == MPFR_RNDU && negative != 0))
                return get_decimal64_zero (negative);
              else /* return the smallest non-zero number */
                return get_decimal64_min (negative);
            }
          else
            {
              mpfr_exp_t e2;
              long digits = 16 - (-382 - e);
              /* if e = -397 then 16 - (-382 - e) = 1 */
              mpfr_get_str (s, &e2, 10, digits, src, rnd_mode);
              /* Warning: we can have e2 = e + 1 here, when rounding to
                 nearest or away from zero. */
              s[negative + digits] = 'E';
              sprintf (s + negative + digits + 1, "%ld",
                       (long int)e2 - digits);
              return string_to_Decimal64 (s);
            }
        }
      /* the largest number is 9.999...999E+384,
         which corresponds to s=[0.]9999...999 and e=385 */
      else if (e > 385)
        {
          if (rnd_mode == MPFR_RNDZ
              || (rnd_mode == MPFR_RNDU && negative != 0)
              || (rnd_mode == MPFR_RNDD && negative == 0))
            return get_decimal64_max (negative);
          else
            return get_decimal64_inf (negative);
        }
      else /* -382 <= e <= 385 */
        {
          s[16 + negative] = 'E';
          sprintf (s + 17 + negative, "%ld", (long int)e - 16);
          return string_to_Decimal64 (s);
        }
    }
}

#endif /* MPFR_WANT_DECIMAL_FLOATS */
