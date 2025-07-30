## src/Makefile.am -- Process this file with automake to produce Makefile.in
##
## Copyright (C) 2008, 2009, 2010, 2011, 2012 INRIA
##
## This file is part of GNU MPC.
##
## GNU MPC is free software; you can redistribute it and/or modify it under
## the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation; either version 3 of the License, or (at your
## option) any later version.
##
## GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
## WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
## FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
## more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with this program. If not, see http://www.gnu.org/licenses/ .

lib_LTLIBRARIES = libmpc.la
libmpc_la_LDFLAGS = $(MPC_LDFLAGS) -version-info 3:0:0
libmpc_la_SOURCES = mpc-impl.h abs.c acos.c acosh.c add.c add_fr.c         \
  add_si.c add_ui.c arg.c asin.c asinh.c atan.c atanh.c clear.c cmp.c      \
  cmp_si_si.c conj.c cos.c cosh.c div_2si.c div_2ui.c div.c div_fr.c       \
  div_ui.c exp.c fma.c fr_div.c fr_sub.c get_prec2.c get_prec.c            \
  get_version.c get_x.c imag.c init2.c init3.c inp_str.c log.c log10.c     \
  mem.c mul_2si.c mul_2ui.c mul.c mul_fr.c mul_i.c mul_si.c mul_ui.c       \
  neg.c norm.c out_str.c pow.c pow_fr.c \
  pow_ld.c pow_d.c pow_si.c pow_ui.c pow_z.c proj.c real.c urandom.c set.c \
  set_prec.c set_str.c set_x.c set_x_x.c sin.c sin_cos.c sinh.c sqr.c      \
  sqrt.c strtoc.c sub.c sub_fr.c sub_ui.c swap.c tan.c tanh.c uceil_log2.c \
  ui_div.c ui_ui_sub.c

libmpc_la_LIBADD = @LTLIBOBJS@
