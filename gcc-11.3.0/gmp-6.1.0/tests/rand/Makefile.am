## Process this file with automake to generate Makefile.in

# Copyright 2000-2003 Free Software Foundation, Inc.
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

check_PROGRAMS = t-iset t-lc2exp t-mt t-rand t-urbui t-urmui t-urndmm
TESTS = $(check_PROGRAMS)

EXTRA_PROGRAMS = findlc gen gen.static spect stat
gen_static_SOURCES = gen.c
gen_static_LDFLAGS = -static
findlc_LDADD = libstat.la
spect_LDADD = libstat.la
stat_LDADD = libstat.la

EXTRA_LTLIBRARIES = libstat.la
libstat_la_SOURCES = gmpstat.h statlib.c zdiv_round.c
libstat_la_LIBADD = $(top_builddir)/libgmp.la $(LIBM)

CLEANFILES = $(EXTRA_PROGRAMS) $(EXTRA_LTLIBRARIES)

allprogs: $(EXTRA_PROGRAMS)

$(top_builddir)/tests/libtests.la:
	cd $(top_builddir)/tests; $(MAKE) $(AM_MAKEFLAGS) libtests.la

manual-test: gen$(EXEEXT) stat$(EXEEXT)
	@(echo -n '16i: '; ./gen -f mpz_urandomb -z 16 1000 \
		| ./stat -i 0xffff | grep '^[0-9]')
	@(echo -n '32i: '; ./gen -f mpz_urandomb -z 32 1000 \
		| ./stat -i 0xffffffff  | grep '^[0-9]')
	@(echo -n '33i: '; ./gen -f mpz_urandomb -z 33 1000 \
		| ./stat -i 0x1ffffffff  | grep '^[0-9]')
	@(echo -n '64i: '; ./gen -f mpz_urandomb -z 64 1000 \
		| ./stat -i 0xffffffffffffffff  | grep '^[0-9]')
	@(echo -n '128i: '; ./gen -f mpz_urandomb -z 128 1000 \
		| ./stat -i 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF | grep '^[0-9]')

	@(echo -n '16f: '; ./gen -f mpf_urandomb -z 16 1000 \
		| ./stat | grep '^[0-9]')
	@(echo -n '32f: '; ./gen -f mpf_urandomb -z 32 1000 \
		| ./stat | grep '^[0-9]')
	@(echo -n '33f: '; ./gen -f mpf_urandomb -z 33 1000 \
		| ./stat | grep '^[0-9]')
	@(echo -n '64f: '; ./gen -f mpf_urandomb -z 64 1000 \
		| ./stat | grep '^[0-9]')
	@(echo -n '128f: '; ./gen -f mpf_urandomb -z 128 1000 \
		| ./stat | grep '^[0-9]')

manual-bigtest: gen$(EXEEXT) stat$(EXEEXT)
	@(echo '16i: '; ./gen -f mpz_urandomb -z 16 50000 \
		| ./stat -2 1000 -i 0xffff | grep '^K[mp]')
	@(echo '32i: '; ./gen -f mpz_urandomb -z 32 50000 \
		| ./stat -2 1000 -i 0xffffffff | grep '^K[mp]')
	@(echo '33i: '; ./gen -f mpz_urandomb -z 33 50000 \
		| ./stat -2 1000 -i 0x1ffffffff | grep '^K[mp]')
	@(echo '64i: '; ./gen -f mpz_urandomb -z 64 50000 \
		| ./stat -2 1000 -i 0xffffffffffffffff  | grep '^K[mp]')
	@(echo '128i: '; ./gen -f mpz_urandomb -z 128 50000 \
		| ./stat -2 1000 -i 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF | grep '^K[mp]')

	@(echo '16f: '; ./gen -f mpf_urandomb -z 16 50000 \
		| ./stat -2 1000 | grep '^K[mp]')
	@(echo '32f: '; ./gen -f mpf_urandomb -z 32 50000 \
		| ./stat -2 1000 | grep '^K[mp]')
	@(echo '33f: '; ./gen -f mpf_urandomb -z 33 50000 \
		| ./stat -2 1000 | grep '^K[mp]')
	@(echo '64f: '; ./gen -f mpf_urandomb -z 64 50000 \
		| ./stat -2 1000 | grep '^K[mp]')
	@(echo '128f: '; ./gen -f mpf_urandomb -z 128 50000 \
		| ./stat -2 1000 | grep '^K[mp]')
