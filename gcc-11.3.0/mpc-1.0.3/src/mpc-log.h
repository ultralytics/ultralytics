/* mpc-log.h -- Include file to enable function call logging; replaces mpc.h.

Copyright (C) 2011 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#define mpc_sqr mpc_log_sqr
#define mpc_conj mpc_log_conj
#define mpc_neg mpc_log_neg
#define mpc_sqrt mpc_log_sqrt
#define mpc_proj mpc_log_proj
#define mpc_exp mpc_log_exp
#define mpc_log mpc_log_log
#define mpc_sin mpc_log_sin
#define mpc_cos mpc_log_cos
#define mpc_tan mpc_log_tan
#define mpc_sinh mpc_log_sinh
#define mpc_cosh mpc_log_cosh
#define mpc_tanh mpc_log_tanh
#define mpc_asin mpc_log_asin
#define mpc_acos mpc_log_acos
#define mpc_atan mpc_log_atan
#define mpc_asinh mpc_log_asinh
#define mpc_acosh mpc_log_acosh
#define mpc_atanh mpc_log_atanh

#define mpc_add mpc_log_add
#define mpc_sub mpc_log_sub
#define mpc_mul mpc_log_mul
#define mpc_div mpc_log_div
#define mpc_pow mpc_log_pow

#define mpc_fma mpc_log_fma

#define mpc_sin_cos mpc_log_sin_cos

#include "mpc.h"
