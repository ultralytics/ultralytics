/* Sparc64 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2002, 2004, 2006, 2008-2010 Free Software
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

#define GMP_LIMB_BITS 64
#define GMP_LIMB_BYTES 8

/* 1000 MHz ultrasparc t1 running GNU/Linux */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_1P_METHOD                      2
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD         13
#define MOD_1U_TO_MOD_1_1_THRESHOLD      MP_SIZE_T_MAX
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     34
#define USE_PREINV_DIVREM_1                  1
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                 8
#define MUL_TOOM33_THRESHOLD                50
#define MUL_TOOM44_THRESHOLD                99
#define MUL_TOOM6H_THRESHOLD               125
#define MUL_TOOM8H_THRESHOLD               187

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      65
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      77
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      65
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      50
#define MUL_TOOM43_TO_TOOM54_THRESHOLD      34

#define SQR_BASECASE_THRESHOLD               0  /* always */
#define SQR_TOOM2_THRESHOLD                 14
#define SQR_TOOM3_THRESHOLD                 57
#define SQR_TOOM4_THRESHOLD                133
#define SQR_TOOM6_THRESHOLD                156
#define SQR_TOOM8_THRESHOLD                260

#define MULMID_TOOM42_THRESHOLD             12

#define MULMOD_BNM1_THRESHOLD                7
#define SQRMOD_BNM1_THRESHOLD                7

#define MUL_FFT_MODF_THRESHOLD             176  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    176, 5}, {      7, 6}, {      4, 5}, {      9, 6}, \
    {      5, 5}, {     11, 6}, {     11, 7}, {      6, 6}, \
    {     13, 7}, {      7, 6}, {     15, 7}, {      9, 8}, \
    {      5, 7}, {     13, 8}, {      7, 7}, {     15, 6}, \
    {     32, 7}, {     24, 8}, {     21, 9}, {     11, 8}, \
    {     23,10}, {      7, 9}, {     15, 8}, {     33, 9}, \
    {     19, 8}, {     39, 9}, {     23,10}, {     15, 9}, \
    {     43,10}, {     23,11}, {     15,10}, {     31, 9}, \
    {     63, 8}, {    127, 9}, {     67,10}, {     39, 9}, \
    {     79, 8}, {    159,10}, {     47, 9}, {     95,11}, \
    {   2048,12}, {   4096,13}, {   8192,14}, {  16384,15}, \
    {  32768,16}, {  65536,17}, { 131072,18}, { 262144,19}, \
    { 524288,20}, {1048576,21}, {2097152,22}, {4194304,23}, \
    {8388608,24} }
#define MUL_FFT_TABLE3_SIZE 53
#define MUL_FFT_THRESHOLD                 1728


#define SQR_FFT_MODF_THRESHOLD             148  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    148, 5}, {      7, 6}, {      4, 5}, {      9, 6}, \
    {      5, 5}, {     11, 6}, {     11, 7}, {      6, 6}, \
    {     13, 7}, {      7, 6}, {     15, 7}, {     13, 8}, \
    {      7, 7}, {     16, 8}, {      9, 6}, {     38, 7}, \
    {     20, 8}, {     11, 7}, {     24, 8}, {     13, 9}, \
    {      7, 7}, {     30, 8}, {     19, 9}, {     11, 8}, \
    {     25,10}, {      7, 9}, {     15, 8}, {     31, 9}, \
    {     19, 8}, {     39, 9}, {     27,10}, {     15, 9}, \
    {     39,10}, {     23, 9}, {     47, 8}, {     95, 9}, \
    {     51,11}, {     15,10}, {     31, 8}, {    127,10}, \
    {     39, 9}, {     79, 8}, {    159,10}, {     47, 9}, \
    {     95,11}, {   2048,12}, {   4096,13}, {   8192,14}, \
    {  16384,15}, {  32768,16}, {  65536,17}, { 131072,18}, \
    { 262144,19}, { 524288,20}, {1048576,21}, {2097152,22}, \
    {4194304,23}, {8388608,24} }
#define SQR_FFT_TABLE3_SIZE 58
#define SQR_FFT_THRESHOLD                 1344

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                  28
#define MULLO_MUL_N_THRESHOLD             3176

#define DC_DIV_QR_THRESHOLD                 27
#define DC_DIVAPPR_Q_THRESHOLD             106
#define DC_BDIV_QR_THRESHOLD                27
#define DC_BDIV_Q_THRESHOLD                 62

#define INV_MULMOD_BNM1_THRESHOLD           14
#define INV_NEWTON_THRESHOLD               163
#define INV_APPR_THRESHOLD                 117

#define BINV_NEWTON_THRESHOLD              166
#define REDC_1_TO_REDC_N_THRESHOLD          31

#define MU_DIV_QR_THRESHOLD                734
#define MU_DIVAPPR_Q_THRESHOLD             748
#define MUPI_DIV_QR_THRESHOLD               67
#define MU_BDIV_QR_THRESHOLD               562
#define MU_BDIV_Q_THRESHOLD                734

#define POWM_SEC_TABLE  4,29,188,643,2741

#define MATRIX22_STRASSEN_THRESHOLD         11
#define HGCD_THRESHOLD                      58
#define HGCD_APPR_THRESHOLD                 55
#define HGCD_REDUCE_THRESHOLD              637
#define GCD_DC_THRESHOLD                   186
#define GCDEXT_DC_THRESHOLD                140
#define JACOBI_BASE_METHOD                   3

#define GET_STR_DC_THRESHOLD                20
#define GET_STR_PRECOMPUTE_THRESHOLD        33
#define SET_STR_DC_THRESHOLD               268
#define SET_STR_PRECOMPUTE_THRESHOLD       960

#define FAC_DSC_THRESHOLD                  268
#define FAC_ODD_THRESHOLD                    0  /* always */
