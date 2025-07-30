/* Fat binary x86 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 2000-2003, 2011 Free Software Foundation, Inc.

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

#define GMP_LIMB_BITS 32
#define GMP_LIMB_BYTES 4


/* mpn_divexact_1 is faster than mpn_divrem_1 at all sizes.  The only time
   this might not be true currently is for actual 80386 and 80486 chips,
   where mpn/x86/dive_1.asm might be slower than mpn/x86/divrem_1.asm, but
   that's not worth worrying about.  */
#define DIVEXACT_1_THRESHOLD  0

/* Only some of the x86s have an mpn_preinv_divrem_1, but we set
   USE_PREINV_DIVREM_1 so that all callers use it, and then let the
   __gmpn_cpuvec pointer go to plain mpn_divrem_1 if there's not an actual
   preinv.  */
#define USE_PREINV_DIVREM_1   1

#define BMOD_1_TO_MOD_1_THRESHOLD           20

/* mpn_sqr_basecase is faster than mpn_mul_basecase at all sizes, no need
   for mpn_sqr to call the latter.  */
#define SQR_BASECASE_THRESHOLD 0

/* Sensible fallbacks for these, when not taken from a cpu-specific
   gmp-mparam.h.  */
#define MUL_TOOM22_THRESHOLD      20
#define MUL_TOOM33_THRESHOLD     130
#define SQR_TOOM2_THRESHOLD       30
#define SQR_TOOM3_THRESHOLD      200

/* These are values more or less in the middle of what the typical x86 chips
   come out as.  For a fat binary it's necessary to have values for these,
   since the defaults for MUL_FFT_TABLE and SQR_FFT_TABLE otherwise come out
   as non-constant array initializers.  FIXME: Perhaps these should be done
   in the cpuvec structure like other thresholds.  */
#define MUL_FFT_TABLE  { 464, 928, 1920, 3584, 10240, 40960, 0 }
#define MUL_FFT_MODF_THRESHOLD          400
#define MUL_FFT_THRESHOLD              2000

#define SQR_FFT_TABLE  { 528, 1184, 1920, 4608, 14336, 40960, 0 }
#define SQR_FFT_MODF_THRESHOLD          500
#define SQR_FFT_THRESHOLD              3000
