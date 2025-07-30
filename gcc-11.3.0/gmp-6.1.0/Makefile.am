## Process this file with automake to generate Makefile.in


# Copyright 1991, 1993, 1994, 1996, 1997, 1999-2004, 2006-2009, 2011-2015 Free
# Software Foundation, Inc.
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


# The following options are the same as AM_INIT_AUTOMAKE in configure.in,
# except no $(top_builddir) on ansi2knr.  That directory is wanted for the
# Makefiles in subdirectories, but here we must omit it so automake gives
# the actual ansi2knr build rule, not "cd $(top_builddir) && make ansi2knr".
#
# AUTOMAKE_OPTIONS = 1.8 gnu no-dependencies


# Libtool -version-info for libgmp.la and libmp.la.  See "Versioning" in the
# libtool manual.
#
#	CURRENT:REVISION:AGE
#
# 1. No interfaces changed, only implementations (good): Increment REVISION.
#
# 2. Interfaces added, none removed (good): Increment CURRENT, increment
#    AGE, set REVISION to 0.
#
# 3. Interfaces removed (BAD, breaks upward compatibility): Increment
#    CURRENT, set AGE and REVISION to 0.
#
# Do this separately for libgmp, libgmpxx and libmp, and only for releases.
#
#	  GMP	   -version-info
#       release   libgmp  libgmpxx libmp
#        2.0.x      -        -       -
#        3.0      3:0:0      -     3:0:0
#        3.0.1    3:1:0      -     3:0:0
#        3.1      4:0:1      -     4:0:1
#        3.1.1    4:1:1      -     4:1:1
#        4.0      5:0:2    3:0:0   4:2:1
#        4.0.1    5:1:2    3:1:0   4:3:1
#        4.1      6:0:3    3:2:0   4:4:1
#        4.1.1    6:1:3    3:3:0   4:5:1
#        4.1.2    6:2:3    3:4:0   4:6:1
#        4.1.3    6:3:3    3:5:0   4:7:1
#        4.1.4    6:3:3    3:5:0   4:7:1	WRONG, same as 4.1.3!
#        4.2      6:0:3    3:2:0   4:4:1	REALLY WRONG, same as 4.1!
#        4.2.1    7:1:4    4:1:1   4:10:1	WRONG for libgmpxx
#        4.2.2    7:2:4    4:2:0   4:11:1
#        4.2.3    7:3:4    4:3:0   4:12:1
#        4.2.4    7:4:4    4:4:0   4:13:1
#        4.3.0    8:0:5    5:0:1   4:14:1
#        4.3.1    8:1:5    5:1:1   4:15:1	WRONG Really used same as 4.3.0
#        4.3.2    8:2:5    5:2:1   4:16:1
#        5.0.0    9:0:6    6:0:2   4:20:1	Should have been 10:0:0
#        5.0.1   10:1:0    6:1:2   4:21:1
#        5.0.2   10:2:0    6:2:2   4:22:1
#        5.0.3   10:3:0    6:3:2   4:23:1
#        5.0.4   10:4:0    6:4:2   4:24:1
#        5.0.5   10:5:0    6:5:2   4:25:1
#        5.1.0   11:0:1    7:0:3     -
#        5.1.1   11:1:1    7:1:3     -
#        5.1.2   11:2:1    7:2:3     -
#        6.0.0   12:0:2    8:0:4     -
#        6.1.0   13:0:3    9:0:5     -
#
# Starting at 3:0:0 is a slight abuse of the versioning system, but it
# ensures we're past soname libgmp.so.2, which was used on Debian GNU/Linux
# packages of gmp 2.  Pretend gmp 2 was 2:0:0, so the interface changes for
# gmp 3 mean 3:0:0 is right.
#
# We interpret "implementation changed" in item "1." above as meaning any
# release, ie. the REVISION is incremented every time (if nothing else).
# Even if we thought the code generated will be identical on all systems,
# it's still good to get the shared library filename (like
# libgmpxx.so.3.0.4) incrementing, to make it clear which GMP it's from.

LIBGMP_LT_CURRENT    = 13
LIBGMP_LT_REVISION   = 0
LIBGMP_LT_AGE        = 3

LIBGMPXX_LT_CURRENT  = 9
LIBGMPXX_LT_REVISION = 0
LIBGMPXX_LT_AGE      = 5


SUBDIRS = tests mpn mpz mpq mpf printf scanf rand cxx demos tune doc

EXTRA_DIST = configfsf.guess configfsf.sub .gdbinit INSTALL.autoconf \
	     COPYING.LESSERv3 COPYINGv2 COPYINGv3


if WANT_CXX
GMPXX_HEADERS_OPTION = gmpxx.h
endif
EXTRA_DIST += gmpxx.h

# gmp.h and mp.h are architecture dependent, mainly since they encode the
# limb size used in libgmp.  For that reason they belong under $exec_prefix
# not $prefix, strictly speaking.
#
# $exec_prefix/include is not in the default include path for gcc built to
# the same $prefix and $exec_prefix, which might mean gmp.h is not found,
# but anyone knowledgeable enough to be playing with exec_prefix will be able
# to address that.
#
includeexecdir = $(exec_prefix)/include
include_HEADERS = $(GMPXX_HEADERS_OPTION)
nodist_includeexec_HEADERS = gmp.h
lib_LTLIBRARIES = libgmp.la $(GMPXX_LTLIBRARIES_OPTION)

BUILT_SOURCES = gmp.h

DISTCLEANFILES = $(BUILT_SOURCES) config.m4 @gmp_srclinks@

# Tell gmp.h it's building gmp, not an application, used by windows DLL stuff.
AM_CPPFLAGS=-D__GMP_WITHIN_GMP


MPF_OBJECTS = mpf/init$U.lo mpf/init2$U.lo mpf/inits$U.lo mpf/set$U.lo	    \
  mpf/set_ui$U.lo mpf/set_si$U.lo mpf/set_str$U.lo mpf/set_d$U.lo	    \
  mpf/set_z$U.lo mpf/iset$U.lo mpf/iset_ui$U.lo mpf/iset_si$U.lo	    \
  mpf/iset_str$U.lo mpf/iset_d$U.lo mpf/clear$U.lo mpf/clears$U.lo	    \
  mpf/get_str$U.lo mpf/dump$U.lo mpf/size$U.lo mpf/eq$U.lo mpf/reldiff$U.lo \
  mpf/sqrt$U.lo mpf/random2$U.lo mpf/inp_str$U.lo mpf/out_str$U.lo	    \
  mpf/add$U.lo mpf/add_ui$U.lo mpf/sub$U.lo mpf/sub_ui$U.lo mpf/ui_sub$U.lo \
  mpf/mul$U.lo mpf/mul_ui$U.lo mpf/div$U.lo mpf/div_ui$U.lo mpf/cmp_z$U.lo  \
  mpf/cmp$U.lo mpf/cmp_d$U.lo mpf/cmp_ui$U.lo mpf/cmp_si$U.lo		    \
  mpf/mul_2exp$U.lo mpf/div_2exp$U.lo mpf/abs$U.lo mpf/neg$U.lo		    \
  mpf/set_q$U.lo mpf/get_d$U.lo mpf/get_d_2exp$U.lo mpf/set_dfl_prec$U.lo   \
  mpf/set_prc$U.lo mpf/set_prc_raw$U.lo mpf/get_dfl_prec$U.lo               \
  mpf/get_prc$U.lo mpf/ui_div$U.lo mpf/sqrt_ui$U.lo                         \
  mpf/ceilfloor$U.lo mpf/trunc$U.lo mpf/pow_ui$U.lo			    \
  mpf/urandomb$U.lo mpf/swap$U.lo					    \
  mpf/fits_sint$U.lo mpf/fits_slong$U.lo mpf/fits_sshort$U.lo		    \
  mpf/fits_uint$U.lo mpf/fits_ulong$U.lo mpf/fits_ushort$U.lo		    \
  mpf/get_si$U.lo mpf/get_ui$U.lo					    \
  mpf/int_p$U.lo

MPZ_OBJECTS = mpz/abs$U.lo mpz/add$U.lo mpz/add_ui$U.lo			\
  mpz/aorsmul$U.lo mpz/aorsmul_i$U.lo mpz/and$U.lo mpz/array_init$U.lo	\
  mpz/bin_ui$U.lo mpz/bin_uiui$U.lo					\
  mpz/cdiv_q$U.lo mpz/cdiv_q_ui$U.lo					\
  mpz/cdiv_qr$U.lo mpz/cdiv_qr_ui$U.lo					\
  mpz/cdiv_r$U.lo mpz/cdiv_r_ui$U.lo mpz/cdiv_ui$U.lo			\
  mpz/cfdiv_q_2exp$U.lo mpz/cfdiv_r_2exp$U.lo				\
  mpz/clear$U.lo mpz/clears$U.lo mpz/clrbit$U.lo			\
  mpz/cmp$U.lo mpz/cmp_d$U.lo mpz/cmp_si$U.lo mpz/cmp_ui$U.lo		\
  mpz/cmpabs$U.lo mpz/cmpabs_d$U.lo mpz/cmpabs_ui$U.lo			\
  mpz/com$U.lo mpz/combit$U.lo						\
  mpz/cong$U.lo mpz/cong_2exp$U.lo mpz/cong_ui$U.lo			\
  mpz/divexact$U.lo mpz/divegcd$U.lo mpz/dive_ui$U.lo			\
  mpz/divis$U.lo mpz/divis_ui$U.lo mpz/divis_2exp$U.lo mpz/dump$U.lo	\
  mpz/export$U.lo mpz/mfac_uiui$U.lo					\
  mpz/2fac_ui$U.lo mpz/fac_ui$U.lo mpz/oddfac_1$U.lo mpz/prodlimbs$U.lo	\
  mpz/fdiv_q_ui$U.lo mpz/fdiv_qr$U.lo mpz/fdiv_qr_ui$U.lo		\
  mpz/fdiv_r$U.lo mpz/fdiv_r_ui$U.lo mpz/fdiv_q$U.lo			\
  mpz/fdiv_ui$U.lo mpz/fib_ui$U.lo mpz/fib2_ui$U.lo mpz/fits_sint$U.lo	\
  mpz/fits_slong$U.lo mpz/fits_sshort$U.lo mpz/fits_uint$U.lo		\
  mpz/fits_ulong$U.lo mpz/fits_ushort$U.lo mpz/gcd$U.lo			\
  mpz/gcd_ui$U.lo mpz/gcdext$U.lo mpz/get_d$U.lo mpz/get_d_2exp$U.lo	\
  mpz/get_si$U.lo mpz/get_str$U.lo mpz/get_ui$U.lo mpz/getlimbn$U.lo	\
  mpz/hamdist$U.lo							\
  mpz/import$U.lo mpz/init$U.lo mpz/init2$U.lo mpz/inits$U.lo		\
  mpz/inp_raw$U.lo mpz/inp_str$U.lo mpz/invert$U.lo			\
  mpz/ior$U.lo mpz/iset$U.lo mpz/iset_d$U.lo mpz/iset_si$U.lo		\
  mpz/iset_str$U.lo mpz/iset_ui$U.lo mpz/jacobi$U.lo mpz/kronsz$U.lo	\
  mpz/kronuz$U.lo mpz/kronzs$U.lo mpz/kronzu$U.lo			\
  mpz/lcm$U.lo mpz/lcm_ui$U.lo mpz/limbs_finish$U.lo			\
  mpz/limbs_modify$U.lo mpz/limbs_read$U.lo mpz/limbs_write$U.lo	\
  mpz/lucnum_ui$U.lo mpz/lucnum2_ui$U.lo				\
  mpz/millerrabin$U.lo mpz/mod$U.lo mpz/mul$U.lo mpz/mul_2exp$U.lo	\
  mpz/mul_si$U.lo mpz/mul_ui$U.lo					\
  mpz/n_pow_ui$U.lo mpz/neg$U.lo mpz/nextprime$U.lo			\
  mpz/out_raw$U.lo mpz/out_str$U.lo mpz/perfpow$U.lo mpz/perfsqr$U.lo	\
  mpz/popcount$U.lo mpz/pow_ui$U.lo mpz/powm$U.lo mpz/powm_sec$U.lo	\
  mpz/powm_ui$U.lo mpz/primorial_ui$U.lo				\
  mpz/pprime_p$U.lo mpz/random$U.lo mpz/random2$U.lo			\
  mpz/realloc$U.lo mpz/realloc2$U.lo mpz/remove$U.lo mpz/roinit_n$U.lo  \
  mpz/root$U.lo mpz/rootrem$U.lo mpz/rrandomb$U.lo mpz/scan0$U.lo	\
  mpz/scan1$U.lo mpz/set$U.lo mpz/set_d$U.lo mpz/set_f$U.lo		\
  mpz/set_q$U.lo mpz/set_si$U.lo mpz/set_str$U.lo mpz/set_ui$U.lo	\
  mpz/setbit$U.lo							\
  mpz/size$U.lo mpz/sizeinbase$U.lo mpz/sqrt$U.lo			\
  mpz/sqrtrem$U.lo mpz/sub$U.lo mpz/sub_ui$U.lo mpz/swap$U.lo		\
  mpz/tdiv_ui$U.lo mpz/tdiv_q$U.lo mpz/tdiv_q_2exp$U.lo			\
  mpz/tdiv_q_ui$U.lo mpz/tdiv_qr$U.lo mpz/tdiv_qr_ui$U.lo		\
  mpz/tdiv_r$U.lo mpz/tdiv_r_2exp$U.lo mpz/tdiv_r_ui$U.lo		\
  mpz/tstbit$U.lo mpz/ui_pow_ui$U.lo mpz/ui_sub$U.lo mpz/urandomb$U.lo	\
  mpz/urandomm$U.lo mpz/xor$U.lo

MPQ_OBJECTS = mpq/abs$U.lo mpq/aors$U.lo				\
  mpq/canonicalize$U.lo mpq/clear$U.lo mpq/clears$U.lo			\
  mpq/cmp$U.lo mpq/cmp_si$U.lo mpq/cmp_ui$U.lo mpq/div$U.lo		\
  mpq/get_d$U.lo mpq/get_den$U.lo mpq/get_num$U.lo mpq/get_str$U.lo	\
  mpq/init$U.lo mpq/inits$U.lo mpq/inp_str$U.lo mpq/inv$U.lo		\
  mpq/md_2exp$U.lo mpq/mul$U.lo mpq/neg$U.lo mpq/out_str$U.lo		\
  mpq/set$U.lo mpq/set_den$U.lo mpq/set_num$U.lo			\
  mpq/set_si$U.lo mpq/set_str$U.lo mpq/set_ui$U.lo			\
  mpq/equal$U.lo mpq/set_z$U.lo mpq/set_d$U.lo				\
  mpq/set_f$U.lo mpq/swap$U.lo

MPN_OBJECTS = mpn/fib_table$U.lo mpn/mp_bases$U.lo

PRINTF_OBJECTS =							\
  printf/asprintf$U.lo printf/asprntffuns$U.lo				\
  printf/doprnt$U.lo printf/doprntf$U.lo printf/doprnti$U.lo		\
  printf/fprintf$U.lo							\
  printf/obprintf$U.lo printf/obvprintf$U.lo printf/obprntffuns$U.lo	\
  printf/printf$U.lo printf/printffuns$U.lo				\
  printf/snprintf$U.lo printf/snprntffuns$U.lo				\
  printf/sprintf$U.lo printf/sprintffuns$U.lo				\
  printf/vasprintf$U.lo printf/vfprintf$U.lo printf/vprintf$U.lo	\
  printf/vsnprintf$U.lo printf/vsprintf$U.lo				\
  printf/repl-vsnprintf$U.lo

SCANF_OBJECTS =							\
  scanf/doscan$U.lo scanf/fscanf$U.lo scanf/fscanffuns$U.lo	\
  scanf/scanf$U.lo scanf/sscanf$U.lo scanf/sscanffuns$U.lo	\
  scanf/vfscanf$U.lo scanf/vscanf$U.lo scanf/vsscanf$U.lo

RANDOM_OBJECTS =							\
  rand/rand$U.lo rand/randclr$U.lo rand/randdef$U.lo rand/randiset$U.lo	\
  rand/randlc2s$U.lo rand/randlc2x$U.lo rand/randmt$U.lo		\
  rand/randmts$U.lo rand/rands$U.lo rand/randsd$U.lo rand/randsdui$U.lo	\
  rand/randbui$U.lo rand/randmui$U.lo

# no $U for C++ files
CXX_OBJECTS =								\
  cxx/isfuns.lo cxx/ismpf.lo cxx/ismpq.lo cxx/ismpz.lo cxx/ismpznw.lo	\
  cxx/limits.lo cxx/osdoprnti.lo cxx/osfuns.lo				\
  cxx/osmpf.lo cxx/osmpq.lo cxx/osmpz.lo

# In libtool 1.5 it doesn't work to build libgmp.la from the convenience
# libraries like mpz/libmpz.la.  Or rather it works, but it ends up putting
# PIC objects into libgmp.a if shared and static are both built.  (The PIC
# objects go into mpz/.libs/libmpz.a, and thence into .libs/libgmp.a.)
#
# For now the big lists of objects above are used.  Something like mpz/*.lo
# would probably work, but might risk missing something out or getting
# something extra.  The source files for each .lo are listed in the
# Makefile.am's in the subdirectories.
#
# Currently, for libgmp, unlike libmp below, we're not using
# -export-symbols, since the tune and speed programs, and perhaps some of
# the test programs, want to access undocumented symbols.

libgmp_la_SOURCES = gmp-impl.h longlong.h				\
  assert.c compat.c errno.c extract-dbl.c invalid.c memory.c		\
  mp_bpl.c mp_clz_tab.c mp_dv_tab.c mp_minv_tab.c mp_get_fns.c mp_set_fns.c \
  version.c nextprime.c primesieve.c
EXTRA_libgmp_la_SOURCES = tal-debug.c tal-notreent.c tal-reent.c
libgmp_la_DEPENDENCIES = @TAL_OBJECT@		\
  $(MPF_OBJECTS) $(MPZ_OBJECTS) $(MPQ_OBJECTS)	\
  $(MPN_OBJECTS) @mpn_objs_in_libgmp@		\
  $(PRINTF_OBJECTS)  $(SCANF_OBJECTS) $(RANDOM_OBJECTS)
libgmp_la_LIBADD = $(libgmp_la_DEPENDENCIES)
libgmp_la_LDFLAGS = $(GMP_LDFLAGS) $(LIBGMP_LDFLAGS) \
  -version-info $(LIBGMP_LT_CURRENT):$(LIBGMP_LT_REVISION):$(LIBGMP_LT_AGE)


# We need at least one .cc file in $(libgmpxx_la_SOURCES) so automake will
# use $(CXXLINK) rather than the plain C $(LINK).  cxx/dummy.cc is that
# file.

if WANT_CXX
GMPXX_LTLIBRARIES_OPTION = libgmpxx.la
endif
libgmpxx_la_SOURCES = cxx/dummy.cc
libgmpxx_la_DEPENDENCIES = $(CXX_OBJECTS) libgmp.la
libgmpxx_la_LIBADD = $(libgmpxx_la_DEPENDENCIES)
libgmpxx_la_LDFLAGS = $(GMP_LDFLAGS) $(LIBGMPXX_LDFLAGS) \
  -version-info $(LIBGMPXX_LT_CURRENT):$(LIBGMPXX_LT_REVISION):$(LIBGMPXX_LT_AGE)



install-data-hook:
	@echo ''
	@echo '+-------------------------------------------------------------+'
	@echo '| CAUTION:                                                    |'
	@echo '|                                                             |'
	@echo '| If you have not already run "make check", then we strongly  |'
	@echo '| recommend you do so.                                        |'
	@echo '|                                                             |'
	@echo '| GMP has been carefully tested by its authors, but compilers |'
	@echo '| are all too often released with serious bugs.  GMP tends to |'
	@echo '| explore interesting corners in compilers and has hit bugs   |'
	@echo '| on quite a few occasions.                                   |'
	@echo '|                                                             |'
	@echo '+-------------------------------------------------------------+'
	@echo ''


# The "test -f" support for srcdir!=builddir is similar to the automake .c.o
# etc rules, but with each foo.c explicitly, since $< is not portable
# outside an inference rule.
#
# A quoted 'foo.c' is used with the "test -f"'s to avoid Sun make rewriting
# it as part of its VPATH support.  See the autoconf manual "Limitations of
# Make".
#
# Generated .h files which are used by gmp-impl.h are BUILT_SOURCES since
# they must exist before anything can be compiled.
#
# Other generated .h files are also BUILT_SOURCES so as to get all the
# build-system stuff over and done with at the start.  Also, dependencies on
# the .h files are not properly expressed for the various objects that use
# them.

EXTRA_DIST += bootstrap.c

fac_table.h: gen-fac$(EXEEXT_FOR_BUILD)
	./gen-fac $(GMP_LIMB_BITS) $(GMP_NAIL_BITS) >fac_table.h || (rm -f fac_table.h; exit 1)
BUILT_SOURCES += fac_table.h

gen-fac$(EXEEXT_FOR_BUILD): gen-fac$(U_FOR_BUILD).c bootstrap.c
	$(CC_FOR_BUILD) `test -f 'gen-fac$(U_FOR_BUILD).c' || echo '$(srcdir)/'`gen-fac$(U_FOR_BUILD).c -o gen-fac$(EXEEXT_FOR_BUILD)
DISTCLEANFILES += gen-fac$(EXEEXT_FOR_BUILD)
EXTRA_DIST += gen-fac.c


fib_table.h: gen-fib$(EXEEXT_FOR_BUILD)
	./gen-fib header $(GMP_LIMB_BITS) $(GMP_NAIL_BITS) >fib_table.h || (rm -f fib_table.h; exit 1)
BUILT_SOURCES += fib_table.h

mpn/fib_table.c: gen-fib$(EXEEXT_FOR_BUILD)
	./gen-fib table $(GMP_LIMB_BITS) $(GMP_NAIL_BITS) >mpn/fib_table.c || (rm -f mpn/fib_table.c; exit 1)
BUILT_SOURCES += mpn/fib_table.c

gen-fib$(EXEEXT_FOR_BUILD): gen-fib$(U_FOR_BUILD).c bootstrap.c
	$(CC_FOR_BUILD) `test -f 'gen-fib$(U_FOR_BUILD).c' || echo '$(srcdir)/'`gen-fib$(U_FOR_BUILD).c -o gen-fib$(EXEEXT_FOR_BUILD)
DISTCLEANFILES += gen-fib$(EXEEXT_FOR_BUILD)
EXTRA_DIST += gen-fib.c


mp_bases.h: gen-bases$(EXEEXT_FOR_BUILD)
	./gen-bases header $(GMP_LIMB_BITS) $(GMP_NAIL_BITS) >mp_bases.h || (rm -f mp_bases.h; exit 1)
BUILT_SOURCES += mp_bases.h

mpn/mp_bases.c: gen-bases$(EXEEXT_FOR_BUILD)
	./gen-bases table $(GMP_LIMB_BITS) $(GMP_NAIL_BITS) >mpn/mp_bases.c || (rm -f mpn/mp_bases.c; exit 1)
BUILT_SOURCES += mpn/mp_bases.c

gen-bases$(EXEEXT_FOR_BUILD): gen-bases$(U_FOR_BUILD).c bootstrap.c
	$(CC_FOR_BUILD) `test -f 'gen-bases$(U_FOR_BUILD).c' || echo '$(srcdir)/'`gen-bases$(U_FOR_BUILD).c -o gen-bases$(EXEEXT_FOR_BUILD) $(LIBM_FOR_BUILD)
DISTCLEANFILES += gen-bases$(EXEEXT_FOR_BUILD)
EXTRA_DIST += gen-bases.c


trialdivtab.h: gen-trialdivtab$(EXEEXT_FOR_BUILD)
	./gen-trialdivtab $(GMP_LIMB_BITS) 8000 >trialdivtab.h || (rm -f trialdivtab.h; exit 1)
BUILT_SOURCES += trialdivtab.h

gen-trialdivtab$(EXEEXT_FOR_BUILD): gen-trialdivtab$(U_FOR_BUILD).c bootstrap.c
	$(CC_FOR_BUILD) `test -f 'gen-trialdivtab$(U_FOR_BUILD).c' || echo '$(srcdir)/'`gen-trialdivtab$(U_FOR_BUILD).c -o gen-trialdivtab$(EXEEXT_FOR_BUILD) $(LIBM_FOR_BUILD)
DISTCLEANFILES += gen-trialdivtab$(EXEEXT_FOR_BUILD)
EXTRA_DIST += gen-trialdivtab.c


mpn/jacobitab.h: gen-jacobitab$(EXEEXT_FOR_BUILD)
	./gen-jacobitab >mpn/jacobitab.h || (rm -f mpn/jacobitab.h; exit 1)
BUILT_SOURCES += mpn/jacobitab.h

gen-jacobitab$(EXEEXT_FOR_BUILD): gen-jacobitab$(U_FOR_BUILD).c
	$(CC_FOR_BUILD) `test -f 'gen-jacobitab$(U_FOR_BUILD).c' || echo '$(srcdir)/'`gen-jacobitab$(U_FOR_BUILD).c -o gen-jacobitab$(EXEEXT_FOR_BUILD)
DISTCLEANFILES += gen-jacobitab$(EXEEXT_FOR_BUILD)
EXTRA_DIST += gen-jacobitab.c


mpn/perfsqr.h: gen-psqr$(EXEEXT_FOR_BUILD)
	./gen-psqr $(GMP_LIMB_BITS) $(GMP_NAIL_BITS) >mpn/perfsqr.h || (rm -f mpn/perfsqr.h; exit 1)
BUILT_SOURCES += mpn/perfsqr.h

gen-psqr$(EXEEXT_FOR_BUILD): gen-psqr$(U_FOR_BUILD).c bootstrap.c
	$(CC_FOR_BUILD) `test -f 'gen-psqr$(U_FOR_BUILD).c' || echo '$(srcdir)/'`gen-psqr$(U_FOR_BUILD).c -o gen-psqr$(EXEEXT_FOR_BUILD) $(LIBM_FOR_BUILD)
DISTCLEANFILES += gen-psqr$(EXEEXT_FOR_BUILD)
EXTRA_DIST += gen-psqr.c

# Distribute mini-gmp. Test sources copied by dist-hook.
EXTRA_DIST += mini-gmp/README mini-gmp/mini-gmp.c mini-gmp/mini-gmp.h \
	      mini-gmp/tests/Makefile mini-gmp/tests/run-tests

# Avoid: CVS - cvs directories
#        *~  - emacs backups
#        .#* - cvs merge originals
#
# *~ and .#* only occur when a whole directory without it's own Makefile.am
# is distributed, like "doc" or the mpn cpu subdirectories.
#
dist-hook:
	-find $(distdir) \( -name CVS -type d \) -o -name "*~" -o -name ".#*" \
		| xargs rm -rf
	cp "$(srcdir)"/mini-gmp/tests/*.[ch] "$(distdir)/mini-gmp/tests"
#	grep -F $(VERSION) $(srcdir)/Makefile.am \
#		| grep -q "^# *$(VERSION) *$(LIBGMP_LT_CURRENT):$(LIBGMP_LT_REVISION):$(LIBGMP_LT_AGE) *$(LIBGMPXX_LT_CURRENT):$(LIBGMPXX_LT_REVISION):$(LIBGMPXX_LT_AGE)"
#	test -z "`sed -n 's/^# *[0-9]*\.[0-9]*\.[0-9]* *\([0-9]*:[0-9]*:[0-9]*\) *\([0-9]*:[0-9]*:[0-9]*\) *\([0-9]*:[0-9]*:[0-9]*\).*/A\1\nB\2\nC\3/p' $(srcdir)/Makefile.am | grep -v 'A6:3:3\|B3:5:0\|C4:7:1' | sort | uniq -d`"

.PHONY: check-mini-gmp clean-mini-gmp

check-mini-gmp:
	abs_srcdir="`cd $(srcdir) && pwd`" ; \
	$(MKDIR_P) mini-gmp/tests \
	&& cd mini-gmp/tests \
	&& LD_LIBRARY_PATH="../../.libs:$$LD_LIBRARY_PATH" \
	   DYLD_LIBRARY_PATH="../../.libs:$$DYLD_LIBRARY_PATH" \
	   $(MAKE) -f "$$abs_srcdir/mini-gmp/tests/Makefile" \
		VPATH="$$abs_srcdir/mini-gmp/tests" \
		srcdir="$$abs_srcdir/mini-gmp/tests" \
		MINI_GMP_DIR="$$abs_srcdir/mini-gmp" \
		LDFLAGS="-L../../.libs" \
		LIBS="-lgmp -lm" \
		CC="$(CC_FOR_BUILD)" EXTRA_CFLAGS="-g -I../.." check

clean-mini-gmp:
	if [ -d mini-gmp/tests ] ; then \
	  abs_srcdir="`cd $(srcdir) && pwd`" ; \
	  cd mini-gmp/tests \
	  && $(MAKE) -f "$$abs_srcdir/mini-gmp/tests/Makefile" clean ; \
	fi

clean-local: clean-mini-gmp
distclean-local: clean-mini-gmp
