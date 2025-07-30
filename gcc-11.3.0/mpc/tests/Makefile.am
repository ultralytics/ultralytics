## tests/Makefile.am -- Process this file with automake to produce Makefile.in
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

AM_CPPFLAGS = -I$(top_srcdir)/src
LDADD = libmpc-tests.la $(top_builddir)/src/libmpc.la
# let libtool create an executable instead of a shell script
# useful for tests with valgrind
AM_LDFLAGS = -no-install
# LOADLIBES (documented in the "GNU make" manual and equivalent to LDLIBS)
# enables to compile a program foo.c in the test directory by simply doing
# "make foo".
LOADLIBES=$(DEFS) -I$(top_srcdir)/src -I$(top_builddir) $(CPPFLAGS) $(CFLAGS) -L$(top_builddir)/tests/.libs -lmpc-tests -lmpc -lm $(LIBS)

check_PROGRAMS = tabs tacos tacosh tadd tadd_fr tadd_si tadd_ui targ tasin    \
tasinh tatan tatanh tconj tcos tcosh tdiv tdiv_2si tdiv_2ui tdiv_fr tdiv_ui   \
texp tfma tfr_div tfr_sub timag tio_str tlog tlog10                           \
tmul tmul_2si tmul_2ui tmul_fr tmul_i tmul_si tmul_ui                         \
tneg tnorm tpow tpow_ld tpow_d tpow_fr tpow_si tpow_ui tpow_z                 \
tprec tproj treal treimref tset tsin tsin_cos tsinh tsqr tsqrt tstrtoc tsub   \
tsub_fr tsub_ui tswap ttan ttanh tui_div tui_ui_sub tget_version

check_LTLIBRARIES=libmpc-tests.la
libmpc_tests_la_SOURCES=mpc-tests.h random.c tgeneric.c read_data.c           \
comparisons.c

EXTRA_DIST = abs.dat acos.dat acosh.dat asin.dat asinh.dat atan.dat atanh.dat \
add.dat add_fr.dat arg.dat conj.dat cos.dat cosh.dat div.dat div_fr.dat       \
exp.dat fma.dat fr_div.dat fr_sub.dat inp_str.dat log.dat log10.dat mul.dat   \
mul_fr.dat neg.dat norm.dat pow.dat pow_fr.dat pow_si.dat pow_ui.dat proj.dat \
sin.dat sinh.dat sqr.dat sqrt.dat strtoc.dat sub.dat sub_fr.dat tan.dat       \
tanh.dat

TESTS_ENVIRONMENT = $(VALGRIND)
TESTS = $(check_PROGRAMS)
CLEANFILES = mpc_test

