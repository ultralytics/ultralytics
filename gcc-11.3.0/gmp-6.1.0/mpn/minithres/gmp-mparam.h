/* Minimal values gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 2000, 2006, 2008-2010, 2012 Free Software
Foundation, Inc.

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

/* The values in this file are not currently minimal.
   Trimming them further would be good.  */

#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          2
#define MOD_1U_TO_MOD_1_1_THRESHOLD          2
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         3
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         4
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD      1
#define USE_PREINV_DIVREM_1                  1  /* native */
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD            3

#define MUL_TOOM22_THRESHOLD                 8
#define MUL_TOOM33_THRESHOLD                20
#define MUL_TOOM44_THRESHOLD                24
#define MUL_TOOM6H_THRESHOLD                70 /* FIXME */
#define MUL_TOOM8H_THRESHOLD                86

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      50 /* FIXME */
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      50 /* FIXME */
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      50 /* FIXME */
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      50 /* FIXME */

#define SQR_BASECASE_THRESHOLD               0
#define SQR_TOOM2_THRESHOLD                  8
#define SQR_TOOM3_THRESHOLD                 20
#define SQR_TOOM4_THRESHOLD                 24
#define SQR_TOOM6H_THRESHOLD                70 /* FIXME */
#define SQR_TOOM8H_THRESHOLD                86

#define MULMOD_BNM1_THRESHOLD            10
#define SQRMOD_BNM1_THRESHOLD            10

#define MUL_FFT_TABLE  {64, 256, 1024, 4096, 8192, 65536, 0}
#define MUL_FFT_MODF_THRESHOLD  65
#define MUL_FFT_THRESHOLD      200

#define SQR_FFT_TABLE  {64, 256, 1024, 4096, 8192, 65536, 0}
#define SQR_FFT_MODF_THRESHOLD  65
#define SQR_FFT_THRESHOLD      200

#define MULLO_BASECASE_THRESHOLD             0
#define MULLO_DC_THRESHOLD                   2
#define MULLO_MUL_N_THRESHOLD                4
#define SQRLO_BASECASE_THRESHOLD             0
#define SQRLO_DC_THRESHOLD                   2
#define SQRLO_SQR_THRESHOLD                  4


#define DC_DIV_QR_THRESHOLD                  6
#define DC_DIVAPPR_Q_THRESHOLD               6
#define DC_BDIV_QR_THRESHOLD                 4
#define DC_BDIV_Q_THRESHOLD                  4

#define INV_MULMOD_BNM1_THRESHOLD            2
#define INV_NEWTON_THRESHOLD                 6
#define INV_APPR_THRESHOLD                   4

#define BINV_NEWTON_THRESHOLD                6
#define REDC_1_TO_REDC_N_THRESHOLD           9

#define MU_DIV_QR_THRESHOLD                  8
#define MU_DIVAPPR_Q_THRESHOLD               8
#define MUPI_DIV_QR_THRESHOLD                8
#define MU_BDIV_QR_THRESHOLD                 8
#define MU_BDIV_Q_THRESHOLD                  8

#define MATRIX22_STRASSEN_THRESHOLD          2
#define HGCD_THRESHOLD                      10
#define GCD_DC_THRESHOLD                    20
#define GCDEXT_SCHOENHAGE_THRESHOLD         20
#define JACOBI_BASE_METHOD                   1

#define GET_STR_DC_THRESHOLD                 4
#define GET_STR_PRECOMPUTE_THRESHOLD        10
#define SET_STR_THRESHOLD                   64
#define SET_STR_PRECOMPUTE_THRESHOLD       100

#define FAC_ODD_THRESHOLD                    0  /* always */
#define FAC_DSC_THRESHOLD                   70
