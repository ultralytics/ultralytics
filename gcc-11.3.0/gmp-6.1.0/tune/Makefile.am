## Process this file with automake to generate Makefile.in

# Copyright 2000-2003, 2005-2011 Free Software Foundation, Inc.
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

EXTRA_DIST = alpha.asm pentium.asm sparcv9.asm hppa.asm hppa2.asm hppa2w.asm \
  ia64.asm powerpc.asm powerpc64.asm x86_64.asm many.pl
noinst_HEADERS = speed.h

# Prefer -static on the speed and tune programs, since that can avoid
# overheads of shared library linkages on some systems.  Libtool tends to
# botch -static if configured with --disable-static, perhaps reasonably
# enough.  In any event under --disable-static the only choice is a dynamic
# link so there's no point in -static.
#
if ENABLE_STATIC
STATIC = -static
else
STATIC =
endif


EXTRA_LTLIBRARIES = libspeed.la

libspeed_la_SOURCES =							\
  common.c divrem1div.c divrem1inv.c divrem2div.c divrem2inv.c		\
  div_qr_1n_pi1_1.c div_qr_1n_pi1_2.c div_qr_1_tune.c			\
  freq.c								\
  gcdext_single.c gcdext_double.c gcdextod.c gcdextos.c			\
  hgcd_lehmer.c hgcd_appr_lehmer.c hgcd_reduce_1.c hgcd_reduce_2.c	\
  jacbase1.c jacbase2.c jacbase3.c jacbase4.c				\
  mod_1_div.c mod_1_inv.c mod_1_1-1.c mod_1_1-2.c modlinv.c		\
  noop.c powm_mod.c powm_redc.c pre_divrem_1.c				\
  set_strb.c set_strs.c set_strp.c time.c

libspeed_la_DEPENDENCIES = $(SPEED_CYCLECOUNTER_OBJ) \
  $(top_builddir)/tests/libtests.la $(top_builddir)/libgmp.la
libspeed_la_LIBADD = $(libspeed_la_DEPENDENCIES) $(LIBM)
libspeed_la_LDFLAGS = $(STATIC)

$(top_builddir)/tests/libtests.la:
	cd $(top_builddir)/tests; $(MAKE) $(AM_MAKEFLAGS) libtests.la


# The library code is faster static than shared on some systems, so do
# tuning and measuring with static, since users who care about maximizing
# speed will be using that.  speed-dynamic exists to show the difference.
#
# On Solaris 8, gcc 2.95.2 -static is somehow broken (it creates executables
# that immediately seg fault), so -all-static is not used.  The only thing
# -all-static does is make libc static linked as well as libgmp, and that
# makes a difference only when measuring malloc and friends in the speed
# program.  This can always be forced with "make speed_LDFLAGS=-all-static
# ..." if desired, see tune/README.

EXTRA_PROGRAMS = speed speed-dynamic speed-ext tuneup tune-gcd-p

DEPENDENCIES = libspeed.la
LDADD = $(DEPENDENCIES) $(TUNE_LIBS)

speed_SOURCES = speed.c
speed_LDFLAGS = $(STATIC)

speed_dynamic_SOURCES = speed.c

speed_ext_SOURCES = speed-ext.c
speed_ext_LDFLAGS = $(STATIC)

tuneup_SOURCES = tuneup.c
nodist_tuneup_SOURCES = sqr_basecase.c fac_ui.c $(TUNE_MPN_SRCS)
tuneup_DEPENDENCIES = $(TUNE_SQR_OBJ) libspeed.la
tuneup_LDADD = $(tuneup_DEPENDENCIES) $(TUNE_LIBS)
tuneup_LDFLAGS = $(STATIC)

tune_gcd_p_SOURCES = tune-gcd-p.c
tune_gcd_p_DEPENDENCIES = ../mpn/gcd.c
tune_gcd_p_LDFLAGS = $(STATIC)


tune:
	$(MAKE) $(AM_MAKEFLAGS) tuneup$(EXEEXT)
	./tuneup

allprogs: $(EXTRA_PROGRAMS)

# $(MANY_CLEAN) and $(MANY_DISTCLEAN) are hooks for many.pl
CLEANFILES = $(EXTRA_PROGRAMS) $(EXTRA_LTLIBRARIES) \
	$(TUNE_MPN_SRCS) fac_ui.c sqr_asm.asm \
	stg.gnuplot stg.data \
	mtg.gnuplot mtg.data \
	fibg.gnuplot fibg.data \
	graph.gnuplot graph.data \
	$(MANY_CLEAN)
DISTCLEANFILES = sqr_basecase.c  $(MANY_DISTCLEAN)


# Generating these little files at build time seems better than including
# them in the distribution, since the list can be changed more easily.
#
# mpn/generic/tdiv_qr.c uses mpn_divrem_1 and mpn_divrem_2, but only for 1
# and 2 limb divisors, which are never used during tuning, so it doesn't
# matter whether it picks up a tuned or untuned version of those.
#
# divrem_1 and mod_1 are recompiled renamed to "_tune" to avoid a linking
# problem.  If a native divrem_1 provides an mpn_divrem_1c entrypoint then
# common.c will want that, but the generic divrem_1 doesn't provide it,
# likewise for mod_1.  The simplest way around this is to have the tune
# build versions renamed suitably.
#
# FIXME: Would like say mul_n.c to depend on $(top_builddir)/mul_n.c so the
# recompiled object will be rebuilt if that file changes.

TUNE_MPN_SRCS = $(TUNE_MPN_SRCS_BASIC) divrem_1.c mod_1.c
TUNE_MPN_SRCS_BASIC = div_qr_2.c bdiv_q.c bdiv_qr.c			\
  dcpi1_div_qr.c dcpi1_divappr_q.c dcpi1_bdiv_qr.c dcpi1_bdiv_q.c	\
  invertappr.c invert.c binvert.c divrem_2.c gcd.c gcdext.c		\
  get_str.c set_str.c matrix22_mul.c					\
  hgcd.c hgcd_appr.c hgcd_reduce.c					\
  mul_n.c sqr.c sec_powm.c						\
  mullo_n.c mul_fft.c mul.c tdiv_qr.c mulmod_bnm1.c sqrmod_bnm1.c	\
  mulmid.c mulmid_n.c toom42_mulmid.c sqrlo.c sqrlo_basecase.c		\
  nussbaumer_mul.c toom6h_mul.c toom8h_mul.c toom6_sqr.c toom8_sqr.c	\
  toom22_mul.c toom2_sqr.c toom33_mul.c toom3_sqr.c toom44_mul.c toom4_sqr.c

$(TUNE_MPN_SRCS_BASIC):
	for i in $(TUNE_MPN_SRCS_BASIC); do \
	  echo "#define TUNE_PROGRAM_BUILD 1" >$$i; \
	  echo "#include \"mpn/generic/$$i\"" >>$$i; \
	done

divrem_1.c:
	echo "#define TUNE_PROGRAM_BUILD 1"                >divrem_1.c
	echo "#define __gmpn_divrem_1  mpn_divrem_1_tune" >>divrem_1.c
	echo "#include \"mpn/generic/divrem_1.c\""        >>divrem_1.c

mod_1.c:
	echo "#define TUNE_PROGRAM_BUILD 1"          >mod_1.c
	echo "#define __gmpn_mod_1  mpn_mod_1_tune" >>mod_1.c
	echo "#include \"mpn/generic/mod_1.c\""     >>mod_1.c

sqr_asm.asm: $(top_builddir)/mpn/sqr_basecase.asm
	echo 'define(SQR_TOOM2_THRESHOLD_OVERRIDE,SQR_TOOM2_THRESHOLD_MAX)' >sqr_asm.asm
	echo 'include(../mpn/sqr_basecase.asm)' >>sqr_asm.asm

# FIXME: Should it depend on $(top_builddir)/fac_ui.h too?
fac_ui.c: $(top_builddir)/mpz/fac_ui.c
	echo "#define TUNE_PROGRAM_BUILD 1"          >fac_ui.c
	echo "#define __gmpz_fac_ui mpz_fac_ui_tune" >>fac_ui.c
	echo "#define __gmpz_oddfac_1 mpz_oddfac_1_tune" >>fac_ui.c
	echo "#include \"mpz/oddfac_1.c\""           >>fac_ui.c
	echo "#include \"mpz/fac_ui.c\""             >>fac_ui.c

include ../mpn/Makeasm.am
