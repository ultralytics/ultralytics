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

/* 500 MHz ultrasparc2 running GNU/Linux */

#define DIVREM_1_NORM_THRESHOLD              3
#define DIVREM_1_UNNORM_THRESHOLD            4
#define MOD_1_NORM_THRESHOLD                 3
#define MOD_1_UNNORM_THRESHOLD               3
#define MOD_1N_TO_MOD_1_1_THRESHOLD      MP_SIZE_T_MAX  /* never */
#define MOD_1U_TO_MOD_1_1_THRESHOLD         22
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        27
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD  MP_SIZE_T_MAX  /* never */
#define USE_PREINV_DIVREM_1                  1
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                30
#define MUL_TOOM33_THRESHOLD               187
#define MUL_TOOM44_THRESHOLD               278
#define MUL_TOOM6H_THRESHOLD               278
#define MUL_TOOM8H_THRESHOLD               357

#define MUL_TOOM32_TO_TOOM43_THRESHOLD     201
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     199
#define MUL_TOOM42_TO_TOOM53_THRESHOLD     154
#define MUL_TOOM42_TO_TOOM63_THRESHOLD     107

#define SQR_BASECASE_THRESHOLD              13
#define SQR_TOOM2_THRESHOLD                 69
#define SQR_TOOM3_THRESHOLD                116
#define SQR_TOOM4_THRESHOLD                336
#define SQR_TOOM6_THRESHOLD                336
#define SQR_TOOM8_THRESHOLD                454

#define MULMOD_BNM1_THRESHOLD               17
#define SQRMOD_BNM1_THRESHOLD               23

#define MUL_FFT_MODF_THRESHOLD             248  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    248, 5}, {      9, 4}, {     19, 6}, {      5, 5}, \
    {     15, 6}, {      8, 5}, {     17, 6}, {     21, 7}, \
    {     19, 8}, {     11, 7}, {     25, 8}, {     15, 7}, \
    {     31, 8}, {     27, 9}, {     15, 8}, {     33, 9}, \
    {     19, 8}, {     39, 9}, {     27,10}, {     15, 9}, \
    {     39,10}, {     23, 9}, {     47,11}, {     15,10}, \
    {     31, 9}, {     67,10}, {     39, 9}, {     79,10}, \
    {     47,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255,10}, {     71, 9}, {    143, 8}, {    287,10}, \
    {     79,11}, {     47,12}, {   4096,13}, {   8192,14}, \
    {  16384,15}, {  32768,16}, {  65536,17}, { 131072,18}, \
    { 262144,19}, { 524288,20}, {1048576,21}, {2097152,22}, \
    {4194304,23}, {8388608,24} }
#define MUL_FFT_TABLE3_SIZE 50
#define MUL_FFT_THRESHOLD                 1984

#define SQR_FFT_MODF_THRESHOLD             236  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    236, 5}, {      8, 4}, {     17, 5}, {     19, 6}, \
    {     10, 5}, {     21, 6}, {     19, 7}, {     10, 6}, \
    {     21, 7}, {     21, 8}, {     21, 9}, {     11, 8}, \
    {     23, 9}, {     19, 8}, {     43, 9}, {     23,10}, \
    {     15, 9}, {     43,10}, {     23,11}, {     15,10}, \
    {     31, 9}, {     63,10}, {     47, 8}, {    191,11}, \
    {     31,10}, {     63, 8}, {    255, 7}, {    511, 9}, \
    {    135, 8}, {    271,10}, {     71, 9}, {    143, 8}, \
    {    287, 7}, {    575,11}, {     47, 9}, {    191, 8}, \
    {    383,12}, {   4096,13}, {   8192,14}, {  16384,15}, \
    {  32768,16}, {  65536,17}, { 131072,18}, { 262144,19}, \
    { 524288,20}, {1048576,21}, {2097152,22}, {4194304,23}, \
    {8388608,24} }
#define SQR_FFT_TABLE3_SIZE 49
#define SQR_FFT_THRESHOLD                 1120

#define MULLO_BASECASE_THRESHOLD            16
#define MULLO_DC_THRESHOLD                  41
#define MULLO_MUL_N_THRESHOLD             3791

#define DC_DIV_QR_THRESHOLD                 27
#define DC_DIVAPPR_Q_THRESHOLD             100
#define DC_BDIV_QR_THRESHOLD                47
#define DC_BDIV_Q_THRESHOLD                174

#define INV_MULMOD_BNM1_THRESHOLD           58
#define INV_NEWTON_THRESHOLD                13
#define INV_APPR_THRESHOLD                   9

#define BINV_NEWTON_THRESHOLD              187
#define REDC_1_TO_REDC_2_THRESHOLD          10
#define REDC_2_TO_REDC_N_THRESHOLD         115

#define MU_DIV_QR_THRESHOLD                680
#define MU_DIVAPPR_Q_THRESHOLD             618
#define MUPI_DIV_QR_THRESHOLD                0  /* always */
#define MU_BDIV_QR_THRESHOLD               748
#define MU_BDIV_Q_THRESHOLD                889

#define MATRIX22_STRASSEN_THRESHOLD         13
#define HGCD_THRESHOLD                      53
#define GCD_DC_THRESHOLD                   283
#define GCDEXT_DC_THRESHOLD                186
#define JACOBI_BASE_METHOD                   2

#define GET_STR_DC_THRESHOLD                13
#define GET_STR_PRECOMPUTE_THRESHOLD        16
#define SET_STR_DC_THRESHOLD               390
#define SET_STR_PRECOMPUTE_THRESHOLD      1665
