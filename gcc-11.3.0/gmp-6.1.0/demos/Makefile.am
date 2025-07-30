## Process this file with automake to generate Makefile.in

# Copyright 2000-2002, 2012 Free Software Foundation, Inc.
#
#  This file is part of the GNU MP Library.
#
#  The GNU MP Library is free software; you can redistribute it and/or modify
#  it under the terms of either:
#
#    * the GNU Lesser General Public License as published by the Free
#      Software Foundation; either version 3 of the License, or (at your
#      option) any later version.
#
#  or
#
#    * the GNU General Public License as published by the Free Software
#      Foundation; either version 2 of the License, or (at your option) any
#      later version.
#
#  or both in parallel, as here.
#
#  The GNU MP Library is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received copies of the GNU General Public License and the
#  GNU Lesser General Public License along with the GNU MP Library.  If not,
#  see https://www.gnu.org/licenses/.


SUBDIRS = calc expr
EXTRA_DIST = perl primes.h

AM_CPPFLAGS = -I$(top_srcdir)
LDADD = $(top_builddir)/libgmp.la

qcn_LDADD = $(LDADD) $(LIBM)
primes_LDADD = $(LDADD) $(LIBM)

# None of these programs are built by default, but "make <whatever>" will
# build them once libgmp.la is built.
#
EXTRA_PROGRAMS = factorize isprime pexpr primes qcn

CLEANFILES = $(EXTRA_PROGRAMS)

allprogs: $(EXTRA_PROGRAMS)
	cd calc; $(MAKE) $(AM_MAKEFLAGS) allprogs
	cd expr; $(MAKE) $(AM_MAKEFLAGS) allprogs
