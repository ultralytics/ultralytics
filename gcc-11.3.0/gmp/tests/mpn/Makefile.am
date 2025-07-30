## Process this file with automake to generate Makefile.in

# Copyright 2001-2003, 2009-2014 Free Software Foundation, Inc.
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


AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/tests
LDADD = $(top_builddir)/tests/libtests.la $(top_builddir)/libgmp.la

check_PROGRAMS = t-asmtype t-aors_1 t-divrem_1 t-mod_1 t-fat t-get_d	\
  t-instrument t-iord_u t-mp_bases t-perfsqr t-scan logic		\
  t-toom22 t-toom32 t-toom33 t-toom42 t-toom43 t-toom44			\
  t-toom52 t-toom53 t-toom54 t-toom62 t-toom63 t-toom6h t-toom8h	\
  t-toom2-sqr t-toom3-sqr t-toom4-sqr t-toom6-sqr t-toom8-sqr		\
  t-div t-mul t-mullo t-sqrlo t-mulmod_bnm1 t-sqrmod_bnm1 t-mulmid	\
  t-hgcd t-hgcd_appr t-matrix22 t-invert t-bdiv				\
  t-broot t-brootinv t-minvert t-sizeinbase

EXTRA_DIST = toom-shared.h toom-sqr-shared.h

TESTS = $(check_PROGRAMS)

$(top_builddir)/tests/libtests.la:
	cd $(top_builddir)/tests; $(MAKE) $(AM_MAKEFLAGS) libtests.la
