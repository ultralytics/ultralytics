/* VAX gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 2000-2002 Free Software Foundation, Inc.

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

/* These numbers were measured manually using the tune/speed program.
   The standard tune/tuneup takes too long.  (VAX 8800) */

#define MUL_TOOM22_THRESHOLD             14
#define MUL_TOOM33_THRESHOLD            110

#define SQR_BASECASE_THRESHOLD            6
#define SQR_TOOM2_THRESHOLD              42
#define SQR_TOOM3_THRESHOLD             250

/* #define DIV_SB_PREINV_THRESHOLD         */
/* #define DIV_DC_THRESHOLD                */
/* #define POWM_THRESHOLD                  */

/* #define GCD_ACCEL_THRESHOLD             */
/* #define JACOBI_BASE_METHOD              */

/* #define DIVREM_1_NORM_THRESHOLD         */
/* #define DIVREM_1_UNNORM_THRESHOLD       */
/* #define MOD_1_NORM_THRESHOLD            */
/* #define MOD_1_UNNORM_THRESHOLD          */
/* #define USE_PREINV_DIVREM_1             */
/* #define USE_PREINV_MOD_1                */
/* #define DIVREM_2_THRESHOLD              */
/* #define DIVEXACT_1_THRESHOLD            */
/* #define MODEXACT_1_ODD_THRESHOLD        */

/* #define GET_STR_DC_THRESHOLD            */
/* #define GET_STR_PRECOMPUTE_THRESHOLD    */
#define SET_STR_THRESHOLD              3400
