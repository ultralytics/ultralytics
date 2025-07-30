/* Intel P6/mmx gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2005, 2009, 2010 Free Software Foundation,
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


/* NOTE: In a fat binary build SQR_TOOM2_THRESHOLD here cannot be more than the
   value in mpn/x86/p6/gmp-mparam.h.  The latter is used as a hard limit in
   mpn/x86/p6/sqr_basecase.asm.  */


/* 800 MHz P6 model 8 */

#define MOD_1_NORM_THRESHOLD                 4
#define MOD_1_UNNORM_THRESHOLD               4
#define MOD_1N_TO_MOD_1_1_THRESHOLD          9
#define MOD_1U_TO_MOD_1_1_THRESHOLD          7
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         8
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        10
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     17
#define USE_PREINV_DIVREM_1                  1  /* native */
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD           49

#define MUL_TOOM22_THRESHOLD                22
#define MUL_TOOM33_THRESHOLD                73
#define MUL_TOOM44_THRESHOLD               193
#define MUL_TOOM6H_THRESHOLD               254
#define MUL_TOOM8H_THRESHOLD               381

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      73
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     122
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      73
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      80

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 30
#define SQR_TOOM3_THRESHOLD                 81
#define SQR_TOOM4_THRESHOLD                142
#define SQR_TOOM6_THRESHOLD                258
#define SQR_TOOM8_THRESHOLD                399

#define MULMOD_BNM1_THRESHOLD               15
#define SQRMOD_BNM1_THRESHOLD               18

#define MUL_FFT_MODF_THRESHOLD             476  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    476, 5}, {     21, 6}, {     11, 5}, {     23, 6}, \
    {     21, 7}, {     11, 6}, {     25, 7}, {     13, 6}, \
    {     27, 7}, {     15, 6}, {     31, 7}, {     21, 8}, \
    {     11, 7}, {     27, 8}, {     15, 7}, {     35, 8}, \
    {     19, 7}, {     41, 8}, {     23, 7}, {     47, 8}, \
    {     27, 9}, {     15, 8}, {     31, 7}, {     63, 8}, \
    {     39, 9}, {     23, 8}, {     51,10}, {     15, 9}, \
    {     31, 8}, {     67, 9}, {     39, 8}, {     79, 9}, \
    {     47, 8}, {     95, 9}, {     55,10}, {     31, 9}, \
    {     63, 8}, {    127, 9}, {     79,10}, {     47, 9}, \
    {     95,11}, {     31,10}, {     63, 9}, {    135,10}, \
    {     79, 9}, {    167,10}, {     95, 9}, {    199,10}, \
    {    111,11}, {     63,10}, {    127, 9}, {    255, 8}, \
    {    511,10}, {    143, 9}, {    287, 8}, {    575,10}, \
    {    159,11}, {     95,10}, {    191, 9}, {    383,10}, \
    {    207,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    511,10}, {    271, 9}, {    543, 8}, {   1087,10}, \
    {    287, 9}, {    575,11}, {    159,10}, {    319, 9}, \
    {    639,10}, {    351, 9}, {    703,11}, {    191,10}, \
    {    383, 9}, {    767,10}, {    415, 9}, {    831,11}, \
    {    223,10}, {    447,12}, {    127,11}, {    255,10}, \
    {    543, 9}, {   1087,11}, {    287,10}, {    607, 9}, \
    {   1215,11}, {    319,10}, {    671,11}, {    351,10}, \
    {    703,12}, {    191,11}, {    383,10}, {    767,11}, \
    {    415,10}, {    831,11}, {    447,13}, {    127,12}, \
    {    255,11}, {    543,10}, {   1087,11}, {    607,10}, \
    {   1215,12}, {    319,11}, {    671,10}, {   1343,11}, \
    {    703,10}, {   1407,11}, {    735,12}, {    383,11}, \
    {    831,12}, {    447,11}, {    959,10}, {   1919,13}, \
    {    255,12}, {    511,11}, {   1087,12}, {    575,11}, \
    {   1215,10}, {   2431,12}, {    639,11}, {   1343,12}, \
    {    703,11}, {   1471,13}, {    383,12}, {    767,11}, \
    {   1535,12}, {    831,11}, {   1727,12}, {    959,11}, \
    {   1919,14}, {    255,13}, {    511,12}, {   1215,11}, \
    {   2431,13}, {    639,12}, {   1471,11}, {   2943,13}, \
    {    767,12}, {   1727,13}, {    895,12}, {   1919,11}, \
    {   3839,14}, {    511,13}, {   1023,12}, {   2111,13}, \
    {   1151,12}, {   2431,13}, {   1279,12}, {   2559,13}, \
    {   1407,12}, {   2943,14}, {    767,13}, {   1663,12}, \
    {   3327,13}, {   1919,12}, {   3839,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 160
#define MUL_FFT_THRESHOLD                 7040

#define SQR_FFT_MODF_THRESHOLD             376  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    376, 5}, {     21, 6}, {     11, 5}, {     23, 6}, \
    {     21, 7}, {     11, 6}, {     24, 7}, {     13, 6}, \
    {     27, 7}, {     15, 6}, {     31, 7}, {     21, 8}, \
    {     11, 7}, {     27, 8}, {     15, 7}, {     33, 8}, \
    {     19, 7}, {     39, 8}, {     23, 7}, {     47, 8}, \
    {     27, 9}, {     15, 8}, {     39, 9}, {     23, 8}, \
    {     51,10}, {     15, 9}, {     31, 8}, {     67, 9}, \
    {     39, 8}, {     79, 9}, {     47, 8}, {     95, 9}, \
    {     55,10}, {     31, 9}, {     79,10}, {     47, 9}, \
    {     95,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255, 9}, {    135,10}, {     79, 9}, {    167,10}, \
    {     95, 9}, {    191, 8}, {    383,10}, {    111,11}, \
    {     63,10}, {    127, 9}, {    255, 8}, {    511, 9}, \
    {    271,10}, {    143, 9}, {    287, 8}, {    575, 9}, \
    {    303, 8}, {    607,10}, {    159, 9}, {    319,11}, \
    {     95,10}, {    191, 9}, {    383,10}, {    207,12}, \
    {     63,11}, {    127,10}, {    255, 9}, {    511,10}, \
    {    271, 9}, {    543,10}, {    287, 9}, {    575,10}, \
    {    303,11}, {    159,10}, {    319, 9}, {    639,10}, \
    {    351, 9}, {    703,11}, {    191,10}, {    383, 9}, \
    {    767,10}, {    415, 9}, {    831,11}, {    223,10}, \
    {    479,12}, {    127,11}, {    255,10}, {    543, 9}, \
    {   1087,11}, {    287,10}, {    607, 9}, {   1215,11}, \
    {    319,10}, {    671,11}, {    351,10}, {    703,12}, \
    {    191,11}, {    383,10}, {    767,11}, {    415,10}, \
    {    831,11}, {    479,13}, {    127,12}, {    255,11}, \
    {    543,10}, {   1087,11}, {    607,10}, {   1215,12}, \
    {    319,11}, {    671,10}, {   1343,11}, {    703,10}, \
    {   1407,11}, {    735,12}, {    383,11}, {    831,12}, \
    {    447,11}, {    959,10}, {   1919,13}, {    255,12}, \
    {    511,11}, {   1087,12}, {    575,11}, {   1215,10}, \
    {   2431,12}, {    639,11}, {   1343,12}, {    703,11}, \
    {   1407,13}, {    383,12}, {    831,11}, {   1727,12}, \
    {    959,11}, {   1919,14}, {    255,13}, {    511,12}, \
    {   1215,11}, {   2431,13}, {    639,12}, {   1471,11}, \
    {   2943,13}, {    767,12}, {   1727,13}, {    895,12}, \
    {   1919,11}, {   3839,14}, {    511,13}, {   1023,12}, \
    {   2111,13}, {   1151,12}, {   2431,13}, {   1407,12}, \
    {   2943,14}, {    767,13}, {   1535,12}, {   3071,13}, \
    {   1663,12}, {   3455,13}, {   1919,12}, {   3839,15}, \
    {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 161
#define SQR_FFT_THRESHOLD                 3712

#define MULLO_BASECASE_THRESHOLD             8
#define MULLO_DC_THRESHOLD                  60
#define MULLO_MUL_N_THRESHOLD            13765

#define DC_DIV_QR_THRESHOLD                 83
#define DC_DIVAPPR_Q_THRESHOLD             246
#define DC_BDIV_QR_THRESHOLD                76
#define DC_BDIV_Q_THRESHOLD                175

#define INV_MULMOD_BNM1_THRESHOLD           42
#define INV_NEWTON_THRESHOLD               268
#define INV_APPR_THRESHOLD                 250

#define BINV_NEWTON_THRESHOLD              276
#define REDC_1_TO_REDC_N_THRESHOLD          74

#define MU_DIV_QR_THRESHOLD               1442
#define MU_DIVAPPR_Q_THRESHOLD            1442
#define MUPI_DIV_QR_THRESHOLD              132
#define MU_BDIV_QR_THRESHOLD              1142
#define MU_BDIV_Q_THRESHOLD               1334

#define MATRIX22_STRASSEN_THRESHOLD         18
#define HGCD_THRESHOLD                     121
#define GCD_DC_THRESHOLD                   478
#define GCDEXT_DC_THRESHOLD                361
#define JACOBI_BASE_METHOD                   4

#define GET_STR_DC_THRESHOLD                13
#define GET_STR_PRECOMPUTE_THRESHOLD        26
#define SET_STR_DC_THRESHOLD               272
#define SET_STR_PRECOMPUTE_THRESHOLD      1074
