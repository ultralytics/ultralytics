## Process this file with automake to generate Makefile.in

# Copyright 2001-2004 Free Software Foundation, Inc.
#
# This file is part of the GNU MP Library test suite.
#
# The GNU MP Library test suite is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the License,
# or (at your option) any later version.
#
# The GNU MP Library test suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.


# LDADD has an explicit -L of $(top_builddir)/.libs for the benefit of gcc
# 3.2 on itanium2-hp-hpux11.22.  Without this option, the libgmp.sl.6
# required by libgmpxx.sl (ie. in its NEEDED records) is not found by the
# linker.  FIXME: Presumably libtool should do something about this itself.
# -lm is needed for t-ops2 which compares the results of trunc and mpf_trunc.
#
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/tests
LDADD = -L$(top_builddir)/.libs \
  $(top_builddir)/tests/libtests.la \
  $(top_builddir)/libgmpxx.la \
  $(top_builddir)/libgmp.la \
  -lm

if WANT_CXX
check_PROGRAMS = t-binary t-cast t-cxx11 \
  t-headers t-iostream t-istream t-locale t-misc t-mix \
  t-ops t-ops2 t-ops3 t-ostream t-prec \
  t-ternary t-unary \
  t-do-exceptions-work-at-all-with-this-compiler \
  t-assign t-constr t-rand
TESTS = $(check_PROGRAMS)
endif

t_assign_SOURCES  = t-assign.cc
t_binary_SOURCES  = t-binary.cc
t_cast_SOURCES    = t-cast.cc
t_constr_SOURCES  = t-constr.cc
t_cxx11_SOURCES   = t-cxx11.cc
t_headers_SOURCES = t-headers.cc
t_iostream_SOURCES= t-iostream.cc
t_istream_SOURCES = t-istream.cc
t_locale_SOURCES  = t-locale.cc clocale.c
t_misc_SOURCES    = t-misc.cc
t_mix_SOURCES     = t-mix.cc
t_ops_SOURCES     = t-ops.cc
t_ops2_SOURCES    = t-ops2.cc
t_ops3_SOURCES    = t-ops3.cc
t_ostream_SOURCES = t-ostream.cc
t_prec_SOURCES    = t-prec.cc
t_rand_SOURCES    = t-rand.cc
t_ternary_SOURCES = t-ternary.cc
t_unary_SOURCES   = t-unary.cc
t_do_exceptions_work_at_all_with_this_compiler_SOURCES = \
  t-do-exceptions-work-at-all-with-this-compiler.cc

$(top_builddir)/tests/libtests.la:
	cd $(top_builddir)/tests; $(MAKE) $(AM_MAKEFLAGS) libtests.la


# Libtool (1.5) somehow botches its uninstalled shared library setups on
# OpenBSD 3.2, making the C++ test programs here fail.  libgmpxx.so ends up
# with a NEEDED record asking for ./.libs/libgmp.so.N, but the loader can't
# find that unless it exists in the current directory.
#
# FIXME: Clearly libtool ought to handle this itself, in which case the hack
# here can be removed.
#
# Note this fix applies only when running "make check".  The cp here should
# be done manually if just one program is to be built and run.
#
TESTS_ENVIRONMENT = cp $(top_builddir)/.libs/libgmp.so.* .libs 2>/dev/null || true;
