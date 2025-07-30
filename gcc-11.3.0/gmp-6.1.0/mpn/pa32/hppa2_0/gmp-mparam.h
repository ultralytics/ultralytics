/* gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2002, 2009, 2010 Free Software Foundation,
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

/* 552 MHz PA8600 (gcc61.fsffrance.org) */

#define DIVREM_1_NORM_THRESHOLD              3
#define DIVREM_1_UNNORM_THRESHOLD            3
#define MOD_1_NORM_THRESHOLD                 3
#define MOD_1_UNNORM_THRESHOLD               4
#define MOD_1N_TO_MOD_1_1_THRESHOLD         11
#define MOD_1U_TO_MOD_1_1_THRESHOLD          8
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        22
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     28
#define USE_PREINV_DIVREM_1                  1
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD           36

#define MUL_TOOM22_THRESHOLD                18
#define MUL_TOOM33_THRESHOLD                65
#define MUL_TOOM44_THRESHOLD               166
#define MUL_TOOM6H_THRESHOLD               202
#define MUL_TOOM8H_THRESHOLD               333

#define MUL_TOOM32_TO_TOOM43_THRESHOLD     105
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     138
#define MUL_TOOM42_TO_TOOM53_THRESHOLD     105
#define MUL_TOOM42_TO_TOOM63_THRESHOLD     102

#define SQR_BASECASE_THRESHOLD               7
#define SQR_TOOM2_THRESHOLD                 55
#define SQR_TOOM3_THRESHOLD                 93
#define SQR_TOOM4_THRESHOLD                250
#define SQR_TOOM6_THRESHOLD                306
#define SQR_TOOM8_THRESHOLD                527

#define MULMOD_BNM1_THRESHOLD               13
#define SQRMOD_BNM1_THRESHOLD               15

#define MUL_FFT_MODF_THRESHOLD             244  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    244, 5}, {      8, 4}, {     17, 5}, {     13, 6}, \
    {      7, 5}, {     17, 6}, {      9, 5}, {     19, 6}, \
    {     13, 7}, {      7, 6}, {     17, 7}, {      9, 6}, \
    {     19, 7}, {     11, 6}, {     24, 7}, {     13, 8}, \
    {      7, 7}, {     19, 8}, {     11, 7}, {     25, 8}, \
    {     15, 7}, {     33, 8}, {     23, 9}, {     15, 8}, \
    {     39, 9}, {     23,10}, {     15, 9}, {     31, 8}, \
    {     67, 9}, {     39, 8}, {     79, 9}, {     47,10}, \
    {     31, 9}, {     71, 8}, {    143, 9}, {     79,10}, \
    {     47,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255, 9}, {    135, 8}, {    271, 9}, {    143,10}, \
    {     79, 9}, {    159, 8}, {    319, 9}, {    175, 8}, \
    {    351,10}, {     95, 9}, {    191, 8}, {    383, 9}, \
    {    207,11}, {     63,10}, {    127, 9}, {    255, 8}, \
    {    511, 9}, {    271,10}, {    143, 9}, {    287, 8}, \
    {    575,10}, {    159, 9}, {    319,10}, {    175, 9}, \
    {    351,11}, {     95,10}, {    191, 9}, {    383,10}, \
    {    207, 9}, {    415,12}, {     63,11}, {    127,10}, \
    {    255, 9}, {    511,10}, {    271, 9}, {    543, 8}, \
    {   1087,10}, {    287, 9}, {    575,10}, {    303,11}, \
    {    159,10}, {    351, 9}, {    703, 8}, {   1407,11}, \
    {    191,10}, {    415, 9}, {    831,11}, {    223, 9}, \
    {    895,10}, {    479,12}, {    127,11}, {    255,10}, \
    {    543, 9}, {   1087,11}, {    287,10}, {    607, 9}, \
    {   1215,11}, {    351,10}, {    703, 9}, {   1407,12}, \
    {    191,11}, {    415,10}, {    831,11}, {    479,13}, \
    {   8192,14}, {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 107
#define MUL_FFT_THRESHOLD                 2112

#define SQR_FFT_MODF_THRESHOLD             240  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    240, 5}, {      8, 4}, {     17, 5}, {     19, 6}, \
    {     17, 7}, {      9, 6}, {     20, 7}, {     11, 6}, \
    {     23, 7}, {     13, 8}, {      7, 7}, {     19, 8}, \
    {     11, 7}, {     25, 8}, {     15, 7}, {     33, 8}, \
    {     19, 7}, {     39, 8}, {     23, 9}, {     15, 8}, \
    {     39, 9}, {     23,10}, {     15, 9}, {     31, 8}, \
    {     63, 9}, {     47,10}, {     31, 9}, {     63, 8}, \
    {    127, 9}, {     71, 8}, {    143, 9}, {     79,10}, \
    {     47,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255, 7}, {    511, 9}, {    135, 8}, {    271, 9}, \
    {    143,10}, {     79, 9}, {    159, 8}, {    319, 9}, \
    {    175, 8}, {    351, 7}, {    703,10}, {     95, 9}, \
    {    191, 8}, {    383, 9}, {    207,11}, {     63,10}, \
    {    127, 9}, {    255, 8}, {    511, 9}, {    271,10}, \
    {    143, 9}, {    287, 8}, {    575,10}, {    159, 9}, \
    {    319,10}, {    175, 9}, {    351, 8}, {    703,11}, \
    {     95,10}, {    191, 9}, {    383,10}, {    207, 9}, \
    {    415,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    511,10}, {    271, 9}, {    543, 8}, {   1087,10}, \
    {    287, 9}, {    575,11}, {    159,10}, {    319, 9}, \
    {    639,10}, {    351, 9}, {    703, 8}, {   1407,11}, \
    {    191,10}, {    415, 9}, {    831,11}, {    223, 8}, \
    {   1791,10}, {    479, 9}, {    959,12}, {    127,11}, \
    {    255,10}, {    543,11}, {    287,10}, {    607,11}, \
    {    319,10}, {    639,11}, {    351,10}, {    703, 9}, \
    {   1407,12}, {    191,11}, {    415,10}, {    831,11}, \
    {    479,10}, {    959,13}, {   8192,14}, {  16384,15}, \
    {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 109
#define SQR_FFT_THRESHOLD                 1600

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                 116
#define MULLO_MUL_N_THRESHOLD             3574

#define DC_DIV_QR_THRESHOLD                100
#define DC_DIVAPPR_Q_THRESHOLD             348
#define DC_BDIV_QR_THRESHOLD               109
#define DC_BDIV_Q_THRESHOLD                254

#define INV_MULMOD_BNM1_THRESHOLD           34
#define INV_NEWTON_THRESHOLD               276
#define INV_APPR_THRESHOLD                 276

#define BINV_NEWTON_THRESHOLD              278
#define REDC_1_TO_REDC_N_THRESHOLD          78

#define MU_DIV_QR_THRESHOLD                979
#define MU_DIVAPPR_Q_THRESHOLD             263
#define MUPI_DIV_QR_THRESHOLD              102
#define MU_BDIV_QR_THRESHOLD               807
#define MU_BDIV_Q_THRESHOLD               1187

#define MATRIX22_STRASSEN_THRESHOLD         11
#define HGCD_THRESHOLD                     100
#define GCD_DC_THRESHOLD                   379
#define GCDEXT_DC_THRESHOLD                249
#define JACOBI_BASE_METHOD                   2

#define GET_STR_DC_THRESHOLD                 7
#define GET_STR_PRECOMPUTE_THRESHOLD        16
#define SET_STR_DC_THRESHOLD               270
#define SET_STR_PRECOMPUTE_THRESHOLD       782
