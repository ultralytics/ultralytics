## Process this file with automake to generate Makefile.in

# Copyright 2000-2004, 2013 Free Software Foundation, Inc.
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


SUBDIRS = . devel mpn mpz mpq mpf rand misc cxx

include ../mpn/Makeasm.am

AM_CPPFLAGS = -I$(top_srcdir)
LDADD = libtests.la $(top_builddir)/libgmp.la

check_LTLIBRARIES = libtests.la

EXTRA_libtests_la_SOURCES = amd64call.asm amd64check.c x86call.asm x86check.c \
    arm32call.asm arm32check.c
libtests_la_SOURCES = tests.h \
  memory.c misc.c refmpf.c refmpn.c refmpq.c refmpz.c spinner.c trace.c
libtests_la_DEPENDENCIES = @CALLING_CONVENTIONS_OBJS@
libtests_la_LIBADD = $(libtests_la_DEPENDENCIES) $(top_builddir)/libgmp.la

check_PROGRAMS = t-bswap t-constants t-count_zeros t-hightomask \
  t-modlinv t-popc t-parity t-sub
TESTS = $(check_PROGRAMS)
