/* UltraSPARC T 32-bit gmp-mparam.h -- Compiler/machine parameter header file.

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

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            3
#define MOD_1_1P_METHOD                      2
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          9
#define MOD_1U_TO_MOD_1_1_THRESHOLD         10
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        21
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     22
#define USE_PREINV_DIVREM_1                  1
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD           35

#define MUL_TOOM22_THRESHOLD                14
#define MUL_TOOM33_THRESHOLD                98
#define MUL_TOOM44_THRESHOLD               166
#define MUL_TOOM6H_THRESHOLD               226
#define MUL_TOOM8H_THRESHOLD               333

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      97
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     139
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      97
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      98
#define MUL_TOOM43_TO_TOOM54_THRESHOLD     120

#define SQR_BASECASE_THRESHOLD               6
#define SQR_TOOM2_THRESHOLD                 34
#define SQR_TOOM3_THRESHOLD                110
#define SQR_TOOM4_THRESHOLD                178
#define SQR_TOOM6_THRESHOLD                240
#define SQR_TOOM8_THRESHOLD                333

#define MULMID_TOOM42_THRESHOLD             22

#define MULMOD_BNM1_THRESHOLD                9
#define SQRMOD_BNM1_THRESHOLD               13

#define MUL_FFT_MODF_THRESHOLD             280  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    280, 5}, {     15, 6}, {      8, 5}, {     17, 6}, \
    {      9, 5}, {     19, 6}, {     13, 7}, {      7, 6}, \
    {     17, 7}, {      9, 6}, {     20, 7}, {     11, 6}, \
    {     23, 7}, {     13, 8}, {      7, 7}, {     21, 8}, \
    {     11, 7}, {     25, 9}, {      7, 8}, {     15, 7}, \
    {     33, 8}, {     19, 7}, {     41, 8}, {     23, 7}, \
    {     49, 8}, {     27, 9}, {     15, 8}, {     31, 7}, \
    {     63, 8}, {     39, 9}, {     23, 8}, {     47,10}, \
    {     15, 9}, {     31, 8}, {     67, 9}, {     39, 8}, \
    {     79, 9}, {     47,10}, {     31, 9}, {     79,10}, \
    {     47,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255,10}, {     79, 9}, {    159, 8}, {    319,10}, \
    {     95, 9}, {    191, 8}, {    383,11}, {     63,10}, \
    {    127, 9}, {    255,10}, {    143, 9}, {    287,10}, \
    {    159, 9}, {    319,10}, {    175,11}, {     95,10}, \
    {    191, 9}, {    383,12}, {   4096,13}, {   8192,14}, \
    {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 66
#define MUL_FFT_THRESHOLD                 3712

#define SQR_FFT_MODF_THRESHOLD             240  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    240, 5}, {     15, 6}, {      8, 5}, {     17, 6}, \
    {     13, 7}, {      7, 6}, {     17, 7}, {      9, 6}, \
    {     20, 7}, {     11, 6}, {     23, 7}, {     13, 8}, \
    {      7, 7}, {     19, 8}, {     11, 7}, {     25, 9}, \
    {      7, 8}, {     15, 7}, {     33, 8}, {     19, 7}, \
    {     39, 8}, {     23, 7}, {     47, 8}, {     27, 9}, \
    {     15, 8}, {     39, 9}, {     23, 8}, {     47,10}, \
    {     15, 9}, {     31, 8}, {     63, 9}, {     39, 8}, \
    {     79, 9}, {     47,10}, {     31, 9}, {     63, 8}, \
    {    127, 9}, {     71, 8}, {    143, 9}, {     79,10}, \
    {     47,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255, 9}, {    143,10}, {     79, 9}, {    159, 8}, \
    {    319, 9}, {    175,10}, {     95, 9}, {    191, 8}, \
    {    383, 9}, {    207,11}, {     63,10}, {    127, 9}, \
    {    255,10}, {    143, 9}, {    287,10}, {    159, 9}, \
    {    319,10}, {    175,11}, {     95,10}, {    191, 9}, \
    {    383,10}, {    207,12}, {   4096,13}, {   8192,14}, \
    {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 70
#define SQR_FFT_THRESHOLD                 2624

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                  51
#define MULLO_MUL_N_THRESHOLD             6633

#define DC_DIV_QR_THRESHOLD                 51
#define DC_DIVAPPR_Q_THRESHOLD             202
#define DC_BDIV_QR_THRESHOLD                47
#define DC_BDIV_Q_THRESHOLD                124

#define INV_MULMOD_BNM1_THRESHOLD           26
#define INV_NEWTON_THRESHOLD               266
#define INV_APPR_THRESHOLD                 222

#define BINV_NEWTON_THRESHOLD              296
#define REDC_1_TO_REDC_N_THRESHOLD          59

#define MU_DIV_QR_THRESHOLD               1334
#define MU_DIVAPPR_Q_THRESHOLD            1499
#define MUPI_DIV_QR_THRESHOLD              116
#define MU_BDIV_QR_THRESHOLD              1057
#define MU_BDIV_Q_THRESHOLD               1334

#define POWM_SEC_TABLE  6,35,213,724,2618

#define MATRIX22_STRASSEN_THRESHOLD         15
#define HGCD_THRESHOLD                      84
#define HGCD_APPR_THRESHOLD                101
#define HGCD_REDUCE_THRESHOLD             1437
#define GCD_DC_THRESHOLD                   372
#define GCDEXT_DC_THRESHOLD                253
#define JACOBI_BASE_METHOD                   2

#define GET_STR_DC_THRESHOLD                12
#define GET_STR_PRECOMPUTE_THRESHOLD        27
#define SET_STR_DC_THRESHOLD               399
#define SET_STR_PRECOMPUTE_THRESHOLD       885

#define FAC_DSC_THRESHOLD                  179
#define FAC_ODD_THRESHOLD                   29
