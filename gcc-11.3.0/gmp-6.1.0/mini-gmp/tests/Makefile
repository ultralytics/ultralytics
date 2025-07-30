# Note: Requires GNU make

# Copyright 2011, 2012, 2014 Free Software Foundation, Inc.
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

srcdir=.
MINI_GMP_DIR=..

CC = gcc
EXTRA_CFLAGS = -O -Wall -g
CFLAGS = $(EXTRA_CFLAGS) -I$(MINI_GMP_DIR)
LDFLAGS =

LIBS = -lgmp -lm -lmcheck

CHECK_PROGRAMS = t-add t-sub t-mul t-invert t-div t-div_2exp \
	t-double t-cmp_d t-gcd t-lcm t-import t-comb t-signed \
	t-sqrt t-root t-powm t-logops t-bitops t-scan t-str \
	t-reuse t-aorsmul t-limbs t-cong t-pprime_p

MISC_OBJS = hex-random.o mini-random.o testutils.o

all:

clean:
	rm -f *.o $(CHECK_PROGRAMS)

%: %.c
.c:

# Keep object files
.PRECIOUS: %.o

%.o: %.c $(MINI_GMP_DIR)/mini-gmp.h hex-random.h mini-random.h
	$(CC) $(CFLAGS) -c $< -o $@

testutils.o: $(MINI_GMP_DIR)/mini-gmp.c

%: %.o $(MISC_OBJS)
	$(CC) $(LDFLAGS) $^ $(LIBS) -o $@

# Missing tests include:
#   mpz_cmp_d, mpz_popcount, mpz_hamdist, mpz_ui_pow_ui

check: $(CHECK_PROGRAMS)
	$(srcdir)/run-tests $(CHECK_PROGRAMS)
