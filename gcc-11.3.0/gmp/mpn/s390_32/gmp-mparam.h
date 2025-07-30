/* S/390-32 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 2000-2011 Free Software Foundation, Inc.

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

/* 770 MHz IBM z900 running in 32-bit mode, using just traditional insns */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            5
#define MOD_1_1P_METHOD                      2
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               5
#define MOD_1N_TO_MOD_1_1_THRESHOLD      MP_SIZE_T_MAX  /* never */
#define MOD_1U_TO_MOD_1_1_THRESHOLD         15
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        30
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD  MP_SIZE_T_MAX  /* never */
#define USE_PREINV_DIVREM_1                  1
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                19
#define MUL_TOOM33_THRESHOLD               114
#define MUL_TOOM44_THRESHOLD               166
#define MUL_TOOM6H_THRESHOLD               226
#define MUL_TOOM8H_THRESHOLD               333

#define MUL_TOOM32_TO_TOOM43_THRESHOLD     106
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     122
#define MUL_TOOM42_TO_TOOM53_THRESHOLD     105
#define MUL_TOOM42_TO_TOOM63_THRESHOLD     113

#define SQR_BASECASE_THRESHOLD               7
#define SQR_TOOM2_THRESHOLD                 40
#define SQR_TOOM3_THRESHOLD                126
#define SQR_TOOM4_THRESHOLD                192
#define SQR_TOOM6_THRESHOLD                246
#define SQR_TOOM8_THRESHOLD                357

#define MULMID_TOOM42_THRESHOLD             28

#define MULMOD_BNM1_THRESHOLD               12
#define SQRMOD_BNM1_THRESHOLD               18

#define MUL_FFT_MODF_THRESHOLD             244  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    244, 5}, {     13, 6}, {      7, 5}, {     15, 6}, \
    {      8, 5}, {     17, 6}, {     13, 7}, {      7, 6}, \
    {     16, 7}, {      9, 6}, {     19, 7}, {     11, 6}, \
    {     23, 7}, {     13, 8}, {      7, 7}, {     19, 8}, \
    {     11, 7}, {     25, 9}, {      7, 8}, {     15, 7}, \
    {     33, 8}, {     19, 7}, {     39, 8}, {     23, 7}, \
    {     47, 8}, {     27, 9}, {     15, 8}, {     39, 9}, \
    {     23, 8}, {     47,10}, {     15, 9}, {     31, 8}, \
    {     63, 9}, {     39, 8}, {     79, 9}, {     47,10}, \
    {     31, 9}, {     63, 8}, {    127, 9}, {     71, 8}, \
    {    143, 9}, {     79,10}, {     47,11}, {   2048,12}, \
    {   4096,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 48
#define MUL_FFT_THRESHOLD                 2688

#define SQR_FFT_MODF_THRESHOLD             216  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    216, 5}, {      7, 4}, {     15, 5}, {     17, 6}, \
    {     13, 7}, {      7, 6}, {     17, 7}, {      9, 6}, \
    {     20, 7}, {     11, 6}, {     23, 7}, {     13, 8}, \
    {      7, 7}, {     19, 8}, {     11, 7}, {     25, 9}, \
    {      7, 8}, {     15, 7}, {     33, 8}, {     19, 7}, \
    {     39, 8}, {     23, 9}, {     15, 8}, {     39, 9}, \
    {     23, 8}, {     47,10}, {     15, 9}, {     31, 8}, \
    {     63, 9}, {     39, 8}, {     79, 9}, {     47,10}, \
    {     31, 9}, {     63, 8}, {    127, 9}, {     71, 8}, \
    {    143, 9}, {     79,10}, {     47,11}, {   2048,12}, \
    {   4096,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 44
#define SQR_FFT_THRESHOLD                 1856

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                  61
#define MULLO_MUL_N_THRESHOLD             5240

#define DC_DIV_QR_THRESHOLD                 70
#define DC_DIVAPPR_Q_THRESHOLD             234
#define DC_BDIV_QR_THRESHOLD                59
#define DC_BDIV_Q_THRESHOLD                137

#define INV_MULMOD_BNM1_THRESHOLD           36
#define INV_NEWTON_THRESHOLD               327
#define INV_APPR_THRESHOLD                 268

#define BINV_NEWTON_THRESHOLD              324
#define REDC_1_TO_REDC_N_THRESHOLD          63

#define MU_DIV_QR_THRESHOLD               1099
#define MU_DIVAPPR_Q_THRESHOLD            1360
#define MUPI_DIV_QR_THRESHOLD              138
#define MU_BDIV_QR_THRESHOLD               889
#define MU_BDIV_Q_THRESHOLD               1234

#define MATRIX22_STRASSEN_THRESHOLD         18
#define HGCD_THRESHOLD                     167
#define GCD_DC_THRESHOLD                   518
#define GCDEXT_DC_THRESHOLD                378
#define JACOBI_BASE_METHOD                   2

#define GET_STR_DC_THRESHOLD                14
#define GET_STR_PRECOMPUTE_THRESHOLD        25
#define SET_STR_DC_THRESHOLD               577
#define SET_STR_PRECOMPUTE_THRESHOLD      1217
