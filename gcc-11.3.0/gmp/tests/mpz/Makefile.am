## Process this file with automake to generate Makefile.in

# Copyright 1996, 1997, 1999-2003, 2009, 2012 Free Software Foundation, Inc.
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

check_PROGRAMS = reuse t-addsub t-cmp t-mul t-mul_i t-tdiv t-tdiv_ui t-fdiv \
  t-fdiv_ui t-cdiv_ui t-gcd t-gcd_ui t-lcm t-invert dive dive_ui t-sqrtrem \
  convert io t-inp_str logic bit t-powm t-powm_ui t-pow t-div_2exp      \
  t-root t-perfsqr t-perfpow t-jac t-bin t-get_d t-get_d_2exp t-get_si	\
  t-set_d t-set_si							\
  t-fac_ui t-mfac_uiui t-primorial_ui t-fib_ui t-lucnum_ui t-scan t-fits   \
  t-divis t-divis_2exp t-cong t-cong_2exp t-sizeinbase t-set_str        \
  t-aorsmul t-cmp_d t-cmp_si t-hamdist t-oddeven t-popcount t-set_f     \
  t-io_raw t-import t-export t-pprime_p t-nextprime t-remove t-limbs

TESTS = $(check_PROGRAMS)

# Temporary files used by the tests.  Removed automatically if the tests
# pass, but ensure they're cleaned if they fail.
#
CLEANFILES = *.tmp

$(top_builddir)/tests/libtests.la:
	cd $(top_builddir)/tests; $(MAKE) $(AM_MAKEFLAGS) libtests.la
