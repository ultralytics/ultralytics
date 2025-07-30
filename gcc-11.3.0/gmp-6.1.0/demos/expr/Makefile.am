## Process this file with automake to generate Makefile.in

# Copyright 2001-2004 Free Software Foundation, Inc.
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


AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/tests

# FIXME: This is a workaround for a bug in automake 1.8.4.  When the only
# library is in EXTRA_LIBRARIES, $(ARFLAGS) is used but no default setting
# for that variable is established.  We give an explicit ARFLAGS=cru the
# same as generated for lib_LIBRARIES or noinst_LIBRARIES.
#
ARFLAGS = cru

EXTRA_LIBRARIES = libexpr.a
libexpr_a_SOURCES = expr.h expr-impl.h \
  expr.c exprv.c exprz.c exprza.c exprq.c exprqa.c exprf.c exprfa.c

EXTRA_PROGRAMS = run-expr t-expr
LDADD = libexpr.a $(top_builddir)/libgmp.la
t_expr_LDADD = $(top_builddir)/tests/libtests.la $(LDADD)

CLEANFILES = $(EXTRA_PROGRAMS) $(EXTRA_LIBRARIES)

allprogs: $(EXTRA_PROGRAMS)

$(top_builddir)/tests/libtests.la:
	cd $(top_builddir)/tests; $(MAKE) $(AM_MAKEFLAGS) libtests.la
