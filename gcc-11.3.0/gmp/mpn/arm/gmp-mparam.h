/* gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2003, 2009, 2010 Free Software Foundation,
Inc.

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

/* 1193MHz ARM (gcc55.fsffrance.org) */

#define DIVREM_1_NORM_THRESHOLD              0  /* preinv always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD         56
#define MOD_1U_TO_MOD_1_1_THRESHOLD         11
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD     MP_SIZE_T_MAX
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     71
#define USE_PREINV_DIVREM_1                  1  /* preinv always */
#define DIVREM_2_THRESHOLD                   0  /* preinv always */
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD           41

#define MUL_TOOM22_THRESHOLD                36
#define MUL_TOOM33_THRESHOLD               125
#define MUL_TOOM44_THRESHOLD               193
#define MUL_TOOM6H_THRESHOLD               303
#define MUL_TOOM8H_THRESHOLD               418

#define MUL_TOOM32_TO_TOOM43_THRESHOLD     125
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     176
#define MUL_TOOM42_TO_TOOM53_THRESHOLD     114
#define MUL_TOOM42_TO_TOOM63_THRESHOLD     129

#define SQR_BASECASE_THRESHOLD              12
#define SQR_TOOM2_THRESHOLD                 78
#define SQR_TOOM3_THRESHOLD                137
#define SQR_TOOM4_THRESHOLD                212
#define SQR_TOOM6_THRESHOLD                306
#define SQR_TOOM8_THRESHOLD                422

#define MULMOD_BNM1_THRESHOLD               20
#define SQRMOD_BNM1_THRESHOLD               26

#define MUL_FFT_MODF_THRESHOLD             436  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    436, 5}, {     27, 6}, {     28, 7}, {     15, 6}, \
    {     32, 7}, {     17, 6}, {     35, 7}, {     19, 6}, \
    {     39, 7}, {     29, 8}, {     15, 7}, {     35, 8}, \
    {     19, 7}, {     41, 8}, {     23, 7}, {     49, 8}, \
    {     27, 9}, {     15, 8}, {     31, 7}, {     63, 8}, \
    {    256, 9}, {    512,10}, {   1024,11}, {   2048,12}, \
    {   4096,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 28
#define MUL_FFT_THRESHOLD                 5760

#define SQR_FFT_MODF_THRESHOLD             404  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    404, 5}, {     13, 4}, {     27, 5}, {     27, 6}, \
    {     28, 7}, {     15, 6}, {     32, 7}, {     17, 6}, \
    {     35, 7}, {     29, 8}, {     15, 7}, {     35, 8}, \
    {     19, 7}, {     41, 8}, {     23, 7}, {     47, 8}, \
    {     27, 9}, {     15, 8}, {     39, 9}, {    512,10}, \
    {   1024,11}, {   2048,12}, {   4096,13}, {   8192,14}, \
    {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 26
#define SQR_FFT_THRESHOLD                 3776

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                 137
#define MULLO_MUL_N_THRESHOLD            11479

#define DC_DIV_QR_THRESHOLD                150
#define DC_DIVAPPR_Q_THRESHOLD             494
#define DC_BDIV_QR_THRESHOLD               148
#define DC_BDIV_Q_THRESHOLD                345

#define INV_MULMOD_BNM1_THRESHOLD           70
#define INV_NEWTON_THRESHOLD               474
#define INV_APPR_THRESHOLD                 478

#define BINV_NEWTON_THRESHOLD              542
#define REDC_1_TO_REDC_N_THRESHOLD         117

#define MU_DIV_QR_THRESHOLD               2089
#define MU_DIVAPPR_Q_THRESHOLD            2172
#define MUPI_DIV_QR_THRESHOLD              225
#define MU_BDIV_QR_THRESHOLD              1528
#define MU_BDIV_Q_THRESHOLD               2089

#define MATRIX22_STRASSEN_THRESHOLD         16
#define HGCD_THRESHOLD                     197
#define GCD_DC_THRESHOLD                   902
#define GCDEXT_DC_THRESHOLD                650
#define JACOBI_BASE_METHOD                   2

#define GET_STR_DC_THRESHOLD                20
#define GET_STR_PRECOMPUTE_THRESHOLD        39
#define SET_STR_DC_THRESHOLD              1045
#define SET_STR_PRECOMPUTE_THRESHOLD      2147
