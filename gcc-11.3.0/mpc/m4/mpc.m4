# mpc.m4
#
# Copyright (C) 2008, 2009, 2010, 2011, 2012 INRIA
#
# This file is part of GNU MPC.
#
# GNU MPC is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/ .


#
# SYNOPSIS
#
#
# MPC_COMPLEX_H
#
# DESCRIPTION
#
# Check whether complex.h is usable; if yes, define HAVE_COMPLEX_H.
#
AC_DEFUN([MPC_COMPLEX_H], [
   AC_CHECK_HEADER(
      [complex.h],
      [
         m4_define(
            [MPC_CONFTEST],
            [
               AC_LANG_PROGRAM(
                  [[#include <complex.h>]],
                  [[complex double x = 1.0 + 2.0 * I; return (creal (x) + cimag (x));]]
               )
            ]
         )

         AC_SEARCH_LIBS([creal], [m])
#           needed on Solaris
         AC_MSG_CHECKING([whether creal, cimag and I can be used])
         AC_LINK_IFELSE(
            [MPC_CONFTEST],
            [
               AC_MSG_RESULT([yes])
               AC_DEFINE([HAVE_COMPLEX_H], [1], [complex.h present and usable])
            ],
            [
               AC_MSG_RESULT([no, build without support for C complex numbers])
            ]
         )
      ]
   )
])


#
# SYNOPSIS
#
#
# MPC_C_CHECK_FLAG([FLAG,ACCUMULATOR])
#
# DESCRIPTION
#
# Checks if the C compiler accepts the flag FLAG
# If yes, adds it to CFLAGS.

AC_DEFUN([MPC_C_CHECK_FLAG], [
   AX_C_CHECK_FLAG($1,,,[CFLAGS="$CFLAGS $1"])
])


#
# SYNOPSIS
#
#
# MPC_C_CHECK_WARNINGFLAGS
#
# DESCRIPTION
#
# For development version only: Checks if gcc accepts warning flags.
# Adds accepted ones to CFLAGS.
#
AC_DEFUN([MPC_C_CHECK_WARNINGCFLAGS], [
  AC_REQUIRE([AC_PROG_GREP])
  if echo $VERSION | grep -c dev >/dev/null 2>&1 ; then
    if test "x$GCC" = "xyes" -a "x$compiler" != "xicc" -a "x$compiler" != "xg++"; then
      # enable -Werror for myself (Andreas Enge)
      if test "x$USER" = "xenge"; then
         MPC_C_CHECK_FLAG(-Werror)
      fi
      MPC_C_CHECK_FLAG(-g)
      MPC_C_CHECK_FLAG(-std=c99)
      MPC_C_CHECK_FLAG(-Wno-long-long)
      MPC_C_CHECK_FLAG(-Wall)
      MPC_C_CHECK_FLAG(-Wextra)
      MPC_C_CHECK_FLAG(-Wdeclaration-after-statement)
      MPC_C_CHECK_FLAG(-Wshadow)
      MPC_C_CHECK_FLAG(-Wstrict-prototypes)
      MPC_C_CHECK_FLAG(-Wmissing-prototypes)
      MPC_C_CHECK_FLAG(-Wno-unused-value)
    fi
  fi
])


#
# SYNOPSIS
#
#
# MPC_GMP_CC_CFLAGS
#
# DESCRIPTION
#
# Checks if CC and CFLAGS can be extracted from gmp.h
# essentially copied from mpfr
#
AC_DEFUN([MPC_GMP_CC_CFLAGS], [
   AC_MSG_CHECKING(for CC and CFLAGS in gmp.h)
   # AC_PROG_CPP triggers the search for a C compiler; use hack instead
   for cpp in /lib/cpp gcc cc c99
   do
      test $cpp = /lib/cpp || cpp="$cpp -E"
      echo foo > conftest.c
      if $cpp $CPPFLAGS conftest.c > /dev/null 2> /dev/null ; then
         # Get CC
         echo "#include \"gmp.h\"" >  conftest.c
         echo "MPFR_OPTION __GMP_CC"           >> conftest.c
         GMP_CC=`$cpp $CPPFLAGS conftest.c 2> /dev/null | $EGREP MPFR_OPTION | $SED -e 's/MPFR_OPTION //g;s/ *" *//g'`
         # Get CFLAGS
         echo "#include \"gmp.h\"" >  conftest.c
         echo "MPFR_OPTION __GMP_CFLAGS"           >> conftest.c
         GMP_CFLAGS=`$cpp $CPPFLAGS conftest.c 2> /dev/null | $EGREP MPFR_OPTION | $SED -e 's/MPFR_OPTION //g;s/ *" *//g'`
         break
      fi
   done

   if test "x$GMP_CFLAGS" = "x__GMP_CFLAGS" -o "x$GMP_CC" = "x__GMP_CC" ; then
      AC_MSG_RESULT(no)
      GMP_CC=
      GMP_CFLAGS=
   else
      AC_MSG_RESULT(yes [CC=$GMP_CC CFLAGS=$GMP_CFLAGS])
   fi

   # Check for validity of CC and CFLAGS obtained from gmp.h
   if test -n "$GMP_CC$GMP_CFLAGS" ; then
      AC_MSG_CHECKING(for CC=$GMP_CC and CFLAGS=$GMP_CFLAGS)
      echo "int main (void) { return 0; }" > conftest.c
      if $GMP_CC $GMP_CFLAGS -o conftest conftest.c 2> /dev/null ; then
         AC_MSG_RESULT(yes)
         CC=$GMP_CC
         CFLAGS=$GMP_CFLAGS
      else
         AC_MSG_RESULT(no)
      fi
   fi

   rm -f conftest*
])


#
# SYNOPSIS
#
#
# MPC_WINDOWS
#
# DESCRIPTION
#
# Additional checks on windows
# libtool requires "-no-undefined" for win32 dll
# It also disables the tests involving the linking with LIBGMP if DLL
#
AC_DEFUN([MPC_WINDOWS], [
   if test "$enable_shared" = yes; then
     MPC_LDFLAGS="$MPC_LDFLAGS -no-undefined"
     AC_MSG_CHECKING(for DLL/static gmp)
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include "gmp.h"
#if !__GMP_LIBGMP_DLL
#error
error
#endif
     ]], [[]])],[AC_MSG_RESULT(DLL)],[
  AC_MSG_RESULT(static)
  AC_MSG_ERROR([gmp is not available as a DLL: use --enable-static --disable-shared]) ])
     AC_MSG_CHECKING(for DLL/static mpfr)
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include "mpfr.h"
#if !__GMP_LIBGMP_DLL
#error
error
#endif
     ]], [[]])],[AC_MSG_RESULT(DLL)],[
  AC_MSG_RESULT(static)
  AC_MSG_ERROR([mpfr is not available as a DLL: use --enable-static --disable-shared]) ])
   else
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include "gmp.h"
#if __GMP_LIBGMP_DLL
#error
error
#endif
     ]], [[]])],[AC_MSG_RESULT(static)],[
  AC_MSG_RESULT(DLL)
  AC_MSG_ERROR([gmp is only available as a DLL: use --disable-static --enable-shared]) ])
  fi
  ;;
])


#
# SYNOPSIS
#
#
# MPC_GITVERSION
#
# DESCRIPTION
#
# If current version string contains "dev", substitutes the short git hash
# into GITVERSION
#
AC_DEFUN([MPC_GITVERSION], [
   if echo $VERSION | grep -c dev >/dev/null 2>&1 ; then
      AC_CHECK_PROG([HASGIT], [git], [yes], [no])
      AS_IF([test "x$HASGIT" = "xyes"], [
         AC_MSG_CHECKING([for current git version])
         GITVERSION=esyscmd([git rev-parse --short HEAD])
         AC_SUBST([GITVERSION])
         AC_MSG_RESULT([$GITVERSION])
      ])
   fi
])
