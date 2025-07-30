/* AMD K6 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 2000-2004, 2009, 2010 Free Software Foundation,
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


/* 450MHz K6-2 */

#define MOD_1_NORM_THRESHOLD                12
#define MOD_1_UNNORM_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define MOD_1N_TO_MOD_1_1_THRESHOLD         41
#define MOD_1U_TO_MOD_1_1_THRESHOLD         32
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         3
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD    128
#define USE_PREINV_DIVREM_1                  0
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                20
#define MUL_TOOM33_THRESHOLD                69
#define MUL_TOOM44_THRESHOLD               106
#define MUL_TOOM6H_THRESHOLD               157
#define MUL_TOOM8H_THRESHOLD               199

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      73
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      69
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      65
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      64

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 32
#define SQR_TOOM3_THRESHOLD                 97
#define SQR_TOOM4_THRESHOLD                143
#define SQR_TOOM6_THRESHOLD                222
#define SQR_TOOM8_THRESHOLD                272

#define MULMOD_BNM1_THRESHOLD               13
#define SQRMOD_BNM1_THRESHOLD               17

#define MUL_FFT_MODF_THRESHOLD             476  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    476, 5}, {     17, 6}, {      9, 5}, {     19, 6}, \
    {     11, 5}, {     23, 6}, {     17, 7}, {      9, 6}, \
    {     19, 7}, {     11, 6}, {     23, 7}, {     13, 6}, \
    {     27, 7}, {     15, 6}, {     31, 7}, {     17, 6}, \
    {     35, 7}, {     21, 8}, {     11, 7}, {     27, 8}, \
    {     15, 7}, {     35, 8}, {     19, 7}, {     39, 8}, \
    {     23, 7}, {     47, 8}, {     27, 9}, {     15, 8}, \
    {     31, 7}, {     63, 8}, {     39, 9}, {     23, 8}, \
    {     51,10}, {     15, 9}, {     31, 8}, {     67, 9}, \
    {     47,10}, {     31, 9}, {     79,10}, {     47, 9}, \
    {     95,11}, {     31,10}, {     63, 9}, {    135,10}, \
    {     79, 9}, {    167,10}, {     95, 9}, {    191,10}, \
    {    111,11}, {     63,10}, {    127, 9}, {    255,10}, \
    {    143, 9}, {    287,10}, {    159,11}, {     95,10}, \
    {    191, 9}, {    383,12}, {     63,11}, {    127,10}, \
    {    255, 9}, {    511,10}, {    271, 9}, {    543,10}, \
    {    287,11}, {    159,10}, {    351,11}, {    191,10}, \
    {    415, 9}, {    831,11}, {    223,12}, {    127,11}, \
    {    255,10}, {    543,11}, {    287,10}, {    575,11}, \
    {    351,10}, {    703,12}, {    191,11}, {    415,10}, \
    {    831,13}, {    127,12}, {    255,11}, {    543,10}, \
    {   1087,11}, {    575,12}, {    319,11}, {    703,12}, \
    {    383,11}, {    831,12}, {    447,11}, {    895,13}, \
    {    255,12}, {    511,11}, {   1087,12}, {    575,11}, \
    {   1151,12}, {    703,13}, {    383,12}, {    959,14}, \
    {    255,13}, {    511,12}, {   1215,13}, {   8192,14}, \
    {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 106
#define MUL_FFT_THRESHOLD                 7424

#define SQR_FFT_MODF_THRESHOLD             432  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    432, 5}, {     17, 6}, {      9, 5}, {     19, 6}, \
    {     11, 5}, {     23, 6}, {     21, 7}, {     11, 6}, \
    {     24, 7}, {     13, 6}, {     27, 7}, {     15, 6}, \
    {     31, 7}, {     21, 8}, {     11, 7}, {     29, 8}, \
    {     15, 7}, {     35, 8}, {     19, 7}, {     39, 8}, \
    {     23, 7}, {     49, 8}, {     27, 9}, {     15, 8}, \
    {     39, 9}, {     23, 7}, {     93, 8}, {     47, 7}, \
    {     95, 8}, {     51,10}, {     15, 9}, {     31, 8}, \
    {     67, 9}, {     39, 8}, {     79, 9}, {     47, 8}, \
    {     95, 9}, {     55,10}, {     31, 9}, {     71, 8}, \
    {    143, 9}, {     79,10}, {     47, 9}, {     95,11}, \
    {     31,10}, {     63, 9}, {    135,10}, {     79, 9}, \
    {    167,10}, {     95, 9}, {    191,11}, {     63,10}, \
    {    127, 9}, {    255,10}, {    143, 9}, {    287, 8}, \
    {    575,10}, {    159, 9}, {    319,11}, {     95,10}, \
    {    191,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    511,10}, {    271, 9}, {    543,10}, {    287,11}, \
    {    159,10}, {    319, 9}, {    639,10}, {    351, 9}, \
    {    703,11}, {    191,10}, {    415,11}, {    223,12}, \
    {    127,11}, {    255,10}, {    543,11}, {    287,10}, \
    {    607,11}, {    319,10}, {    639,11}, {    351,10}, \
    {    703,12}, {    191,11}, {    415,10}, {    831,13}, \
    {    127,12}, {    255,11}, {    543,10}, {   1087,11}, \
    {    607,12}, {    319,11}, {    703,12}, {    383,11}, \
    {    831,12}, {    447,13}, {    255,12}, {    511,11}, \
    {   1087,12}, {    575,11}, {   1215,12}, {    703,13}, \
    {    383,12}, {    895,14}, {    255,13}, {    511,12}, \
    {   1215,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 112
#define SQR_FFT_THRESHOLD                 7040

#define MULLO_BASECASE_THRESHOLD             3
#define MULLO_DC_THRESHOLD                  60
#define MULLO_MUL_N_THRESHOLD            13463

#define DC_DIV_QR_THRESHOLD                 78
#define DC_DIVAPPR_Q_THRESHOLD             252
#define DC_BDIV_QR_THRESHOLD                84
#define DC_BDIV_Q_THRESHOLD                171

#define INV_MULMOD_BNM1_THRESHOLD           55
#define INV_NEWTON_THRESHOLD               234
#define INV_APPR_THRESHOLD                 236

#define BINV_NEWTON_THRESHOLD              268
#define REDC_1_TO_REDC_N_THRESHOLD          67

#define MU_DIV_QR_THRESHOLD               1308
#define MU_DIVAPPR_Q_THRESHOLD            1142
#define MUPI_DIV_QR_THRESHOLD              134
#define MU_BDIV_QR_THRESHOLD              1164
#define MU_BDIV_Q_THRESHOLD               1164

#define MATRIX22_STRASSEN_THRESHOLD         15
#define HGCD_THRESHOLD                     182
#define GCD_DC_THRESHOLD                   591
#define GCDEXT_DC_THRESHOLD                472
#define JACOBI_BASE_METHOD                   2

#define GET_STR_DC_THRESHOLD                24
#define GET_STR_PRECOMPUTE_THRESHOLD        40
#define SET_STR_DC_THRESHOLD               834
#define SET_STR_PRECOMPUTE_THRESHOLD      2042
