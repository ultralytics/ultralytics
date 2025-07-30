# ===========================================================================
#    http://www.nongnu.org/autoconf-archive/ax_gcc_warn_unused_result.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_GCC_WARN_UNUSED_RESULT
#
# DESCRIPTION
#
#   The macro will compile a test program to see whether the compiler does
#   understand the per-function postfix pragma.
#
# LICENSE
#
#   Copyright (c) 2008 Guido U. Draheim <guidod@gmx.de>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

AC_DEFUN([AX_GCC_WARN_UNUSED_RESULT],[dnl
AC_CACHE_CHECK(
 [whether the compiler supports function __attribute__((__warn_unused_result__))],
 ax_cv_gcc_warn_unused_result,[
 AC_TRY_COMPILE([__attribute__((__warn_unused_result__))
 int f(int i) { return i; }],
 [],
 ax_cv_gcc_warn_unused_result=yes, ax_cv_gcc_warn_unused_result=no)])
 if test "$ax_cv_gcc_warn_unused_result" = yes; then
   AC_DEFINE([GCC_WARN_UNUSED_RESULT],[__attribute__((__warn_unused_result__))],
    [most gcc compilers know a function __attribute__((__warn_unused_result__))])
 fi
])
