## Process this file with automake to generate Makefile.in

# Copyright 1996, 1999-2004, 2015 Free Software Foundation, Inc.
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

check_PROGRAMS = t-dm2exp t-conv t-add t-sub t-sqrt t-sqrt_ui t-muldiv reuse \
  t-cmp_d t-cmp_si t-div t-fits t-get_d t-get_d_2exp \
  t-get_si t-get_ui t-gsprec t-inp_str t-int_p t-mul_ui \
  t-set t-set_q t-set_si t-set_ui t-trunc t-ui_div t-eq t-pow_ui
TESTS = $(check_PROGRAMS)

$(top_builddir)/tests/libtests.la:
	cd $(top_builddir)/tests; $(MAKE) $(AM_MAKEFLAGS) libtests.la
