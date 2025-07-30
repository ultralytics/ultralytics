## Process this file with automake to generate Makefile.in

# Copyright 2000-2004 Free Software Foundation, Inc.
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


AM_CPPFLAGS = -I$(top_srcdir)

# $(LEXLIB) is not actually needed for flex (which means the distributed
# calclex.c), but it's included here for the benefit of anyone rebuilding
# with some other lex.
#
LDADD = $(top_builddir)/libgmp.la $(LIBREADLINE) $(LIBCURSES) $(LEXLIB)

EXTRA_PROGRAMS = calc
AM_YFLAGS = -d
calc_SOURCES = calc.y calclex.l calcread.c calc-common.h
BUILT_SOURCES = calc.h

CLEANFILES = $(EXTRA_PROGRAMS)

allprogs: $(EXTRA_PROGRAMS)
