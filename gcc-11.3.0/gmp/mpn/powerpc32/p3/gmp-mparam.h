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

/* 450 MHz POWER3 */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_1P_METHOD                      2
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          3
#define MOD_1U_TO_MOD_1_1_THRESHOLD          2
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        12
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        18
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD      8
#define USE_PREINV_DIVREM_1                  1
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                10
#define MUL_TOOM33_THRESHOLD                38
#define MUL_TOOM44_THRESHOLD                58
#define MUL_TOOM6H_THRESHOLD               129
#define MUL_TOOM8H_THRESHOLD               212

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      65
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      63
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      59
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      64

#define SQR_BASECASE_THRESHOLD               0  /* always */
#define SQR_TOOM2_THRESHOLD                 14
#define SQR_TOOM3_THRESHOLD                 53
#define SQR_TOOM4_THRESHOLD                 76
#define SQR_TOOM6_THRESHOLD                106
#define SQR_TOOM8_THRESHOLD                284

#define MULMOD_BNM1_THRESHOLD                9
#define SQRMOD_BNM1_THRESHOLD                9

#define MUL_FFT_MODF_THRESHOLD             220  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    220, 5}, {     13, 6}, {      7, 5}, {     15, 6}, \
    {      9, 5}, {     19, 6}, {     13, 7}, {      7, 6}, \
    {     16, 7}, {     13, 8}, {      7, 7}, {     19, 8}, \
    {     11, 7}, {     23, 9}, {      7, 8}, {     15, 7}, \
    {     33, 8}, {     23, 9}, {     15, 8}, {     35, 9}, \
    {     23,10}, {     15, 9}, {     31, 8}, {     67, 9}, \
    {     39, 8}, {     79, 9}, {     47,10}, {     31, 9}, \
    {     63, 8}, {    127, 9}, {     71, 8}, {    143, 9}, \
    {     79,10}, {     47,11}, {     31,10}, {     63, 9}, \
    {    127, 8}, {    255, 9}, {    143,10}, {     79, 9}, \
    {    159, 8}, {    319, 9}, {    175, 8}, {    351,10}, \
    {     95, 9}, {    191, 8}, {    383,10}, {    111,11}, \
    {     63,10}, {    127, 9}, {    255,10}, {    143, 9}, \
    {    287, 8}, {    575,10}, {    159, 9}, {    319,10}, \
    {    175, 9}, {    351,11}, {     95,10}, {    191, 9}, \
    {    383,10}, {    207, 9}, {    415,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    511,10}, {    287, 9}, \
    {    575,11}, {    159,10}, {    351, 9}, {    703, 8}, \
    {   1407,11}, {    191,10}, {    415,11}, {    223,10}, \
    {    447, 9}, {    895,12}, {   4096,13}, {   8192,14}, \
    {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 82
#define MUL_FFT_THRESHOLD                 2688

#define SQR_FFT_MODF_THRESHOLD             176  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    176, 5}, {     13, 6}, {      7, 5}, {     15, 6}, \
    {     13, 7}, {      7, 6}, {     16, 7}, {      9, 6}, \
    {     19, 7}, {     11, 6}, {     23, 7}, {     13, 8}, \
    {      7, 7}, {     19, 8}, {     11, 7}, {     23, 9}, \
    {      7, 8}, {     15, 7}, {     31, 8}, {     23, 9}, \
    {     15, 8}, {     39, 9}, {     23,10}, {     15, 9}, \
    {     31, 8}, {     63, 9}, {     39, 8}, {     79, 9}, \
    {     47, 8}, {     95,10}, {     31, 9}, {     63, 8}, \
    {    127, 9}, {     71, 8}, {    143, 7}, {    287, 6}, \
    {    575, 9}, {     79, 8}, {    159,10}, {     47, 9}, \
    {     95,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255, 9}, {    143, 8}, {    287, 7}, {    575,10}, \
    {     79, 9}, {    159, 8}, {    319, 9}, {    175,10}, \
    {     95, 9}, {    191, 8}, {    383,10}, {    111, 9}, \
    {    223,11}, {     63,10}, {    127, 9}, {    255,10}, \
    {    143, 9}, {    287, 8}, {    575,10}, {    159, 9}, \
    {    319,10}, {    175,11}, {     95,10}, {    191, 9}, \
    {    383,10}, {    223,12}, {     63,11}, {    127,10}, \
    {    287, 9}, {    575,11}, {    159,10}, {    351, 9}, \
    {    703, 8}, {   1407,11}, {    191,10}, {    383,11}, \
    {    223,10}, {    447, 9}, {    895,12}, {   4096,13}, \
    {   8192,14}, {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 87
#define SQR_FFT_THRESHOLD                 1728

#define MULLO_BASECASE_THRESHOLD             2
#define MULLO_DC_THRESHOLD                  33
#define MULLO_MUL_N_THRESHOLD             5240

#define DC_DIV_QR_THRESHOLD                 32
#define DC_DIVAPPR_Q_THRESHOLD             123
#define DC_BDIV_QR_THRESHOLD                34
#define DC_BDIV_Q_THRESHOLD                 84

#define INV_MULMOD_BNM1_THRESHOLD           42
#define INV_NEWTON_THRESHOLD               129
#define INV_APPR_THRESHOLD                 124

#define BINV_NEWTON_THRESHOLD              148
#define REDC_1_TO_REDC_N_THRESHOLD          38

#define MU_DIV_QR_THRESHOLD                748
#define MU_DIVAPPR_Q_THRESHOLD             748
#define MUPI_DIV_QR_THRESHOLD               59
#define MU_BDIV_QR_THRESHOLD               562
#define MU_BDIV_Q_THRESHOLD                654

#define MATRIX22_STRASSEN_THRESHOLD         11
#define HGCD_THRESHOLD                      76
#define GCD_DC_THRESHOLD                   205
#define GCDEXT_DC_THRESHOLD                174
#define JACOBI_BASE_METHOD                   1

#define GET_STR_DC_THRESHOLD                14
#define GET_STR_PRECOMPUTE_THRESHOLD        27
#define SET_STR_DC_THRESHOLD               181
#define SET_STR_PRECOMPUTE_THRESHOLD       525
