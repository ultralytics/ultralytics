/* PowerPC-32 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2004, 2008-2011 Free Software Foundation, Inc.

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

/* 3550 MHz POWER7/T4 */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_1P_METHOD                      1
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          7
#define MOD_1U_TO_MOD_1_1_THRESHOLD          7
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        34
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     15
#define USE_PREINV_DIVREM_1                  1
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD           34

#define MUL_TOOM22_THRESHOLD                20
#define MUL_TOOM33_THRESHOLD                89
#define MUL_TOOM44_THRESHOLD               130
#define MUL_TOOM6H_THRESHOLD               286
#define MUL_TOOM8H_THRESHOLD               363

#define MUL_TOOM32_TO_TOOM43_THRESHOLD     121
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     114
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      89
#define MUL_TOOM42_TO_TOOM63_THRESHOLD     113

#define SQR_BASECASE_THRESHOLD               4
#define SQR_TOOM2_THRESHOLD                 50
#define SQR_TOOM3_THRESHOLD                 89
#define SQR_TOOM4_THRESHOLD                154
#define SQR_TOOM6_THRESHOLD                222
#define SQR_TOOM8_THRESHOLD                381

#define MULMID_TOOM42_THRESHOLD             40

#define MULMOD_BNM1_THRESHOLD               18
#define SQRMOD_BNM1_THRESHOLD               17

#define POWM_SEC_TABLE  4,35,225,780,2212

#define MUL_FFT_MODF_THRESHOLD             476  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    476, 5}, {     21, 6}, {     11, 5}, {     23, 6}, \
    {     12, 5}, {     25, 6}, {     13, 5}, {     27, 6}, \
    {     14, 5}, {     29, 6}, {     21, 7}, {     11, 6}, \
    {     25, 7}, {     13, 6}, {     29, 7}, {     15, 6}, \
    {     31, 7}, {     17, 6}, {     35, 7}, {     19, 6}, \
    {     39, 7}, {     21, 8}, {     11, 7}, {     27, 8}, \
    {     15, 7}, {     35, 8}, {     19, 7}, {     41, 8}, \
    {     27, 9}, {     15, 8}, {     39, 9}, {     23, 8}, \
    {     51,10}, {     15, 9}, {     31, 8}, {     67, 9}, \
    {     39, 8}, {     79, 9}, {     47, 8}, {     95, 9}, \
    {     55,10}, {     31, 9}, {     79,10}, {     47, 9}, \
    {     95,11}, {     31,10}, {     63, 9}, {    135,10}, \
    {     79, 9}, {    159,10}, {     95,11}, {     63,10}, \
    {    159,11}, {     95,12}, {     63,11}, {    127,10}, \
    {    255, 9}, {    511,10}, {    271, 9}, {    543, 8}, \
    {   1087,11}, {    159,10}, {    319, 9}, {    639,10}, \
    {    335, 9}, {    671, 8}, {   1343,10}, {    351,11}, \
    {    191,10}, {    415, 9}, {    831,10}, {    431,11}, \
    {    223,12}, {   4096,13}, {   8192,14}, {  16384,15}, \
    {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 77
#define MUL_FFT_THRESHOLD                 5312

#define SQR_FFT_MODF_THRESHOLD             344  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    344, 5}, {     21, 6}, {     11, 5}, {     23, 6}, \
    {     21, 7}, {     11, 6}, {     24, 7}, {     13, 6}, \
    {     27, 7}, {     15, 6}, {     31, 7}, {     21, 8}, \
    {     11, 7}, {     27, 8}, {     15, 7}, {     33, 8}, \
    {     19, 7}, {     39, 8}, {     27, 9}, {     15, 8}, \
    {     39, 9}, {     23, 8}, {     47,10}, {     15, 9}, \
    {     31, 8}, {     63, 9}, {     39, 8}, {     79, 9}, \
    {     47,10}, {     31, 9}, {     79,10}, {     47,11}, \
    {     31,10}, {     63, 9}, {    135,10}, {     79, 9}, \
    {    159,10}, {     95, 9}, {    191,11}, {     63,10}, \
    {    127, 9}, {    255, 8}, {    511, 9}, {    271,10}, \
    {    143, 9}, {    287, 8}, {    575, 9}, {    303,10}, \
    {    159,11}, {     95,10}, {    191,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    511,10}, {    271, 9}, \
    {    543, 8}, {   1087,10}, {    287, 9}, {    575,10}, \
    {    303,11}, {    159,10}, {    319, 9}, {    639,10}, \
    {    335, 9}, {    671,10}, {    351, 9}, {    703,11}, \
    {    191,10}, {    383, 9}, {    767,10}, {    415, 9}, \
    {    831,11}, {    223,10}, {    447,12}, {   4096,13}, \
    {   8192,14}, {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 79
#define SQR_FFT_THRESHOLD                 3712

#define MULLO_BASECASE_THRESHOLD             2
#define MULLO_DC_THRESHOLD                  34
#define MULLO_MUL_N_THRESHOLD            10323

#define DC_DIV_QR_THRESHOLD                 52
#define DC_DIVAPPR_Q_THRESHOLD             202
#define DC_BDIV_QR_THRESHOLD                68
#define DC_BDIV_Q_THRESHOLD                152

#define INV_MULMOD_BNM1_THRESHOLD           66
#define INV_NEWTON_THRESHOLD               226
#define INV_APPR_THRESHOLD                 189

#define BINV_NEWTON_THRESHOLD              292
#define REDC_1_TO_REDC_N_THRESHOLD          79

#define MU_DIV_QR_THRESHOLD               1442
#define MU_DIVAPPR_Q_THRESHOLD            1442
#define MUPI_DIV_QR_THRESHOLD               91
#define MU_BDIV_QR_THRESHOLD              1308
#define MU_BDIV_Q_THRESHOLD               1442

#define MATRIX22_STRASSEN_THRESHOLD         16
#define HGCD_THRESHOLD                     126
#define HGCD_APPR_THRESHOLD                139
#define HGCD_REDUCE_THRESHOLD             2681
#define GCD_DC_THRESHOLD                   573
#define GCDEXT_DC_THRESHOLD                448
#define JACOBI_BASE_METHOD                   4

#define GET_STR_DC_THRESHOLD                 9
#define GET_STR_PRECOMPUTE_THRESHOLD        20
#define SET_STR_DC_THRESHOLD               834
#define SET_STR_PRECOMPUTE_THRESHOLD      1888
