/* Various Thresholds of MPFR, not exported.  -*- mode: C -*-

Copyright 2005-2017 Free Software Foundation, Inc.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#ifndef __MPFR_IMPL_H__
# error "MPFR Internal not included"
#endif

/* Note: the different macros used here are those defined by gcc,
   for example with gcc -dM -E -xc /dev/null
   As of gcc 4.2, you can also use: -march=native or -mtune=native */

#if defined (__tune_pentium4__) /* Threshold for Pentium 4 */
#define MPFR_TUNE_CASE "src/x86_64/pentium4/mparam.h"
#include "x86_64/pentium4/mparam.h"

#elif (defined (__tune_core2__) || defined (__tune_nocona__)) && defined (__x86_64) /* 64-bit Core 2 or Xeon */
#define MPFR_TUNE_CASE "src/x86_64/core2/mparam.h"
#include "x86_64/core2/mparam.h"

#elif defined (__tune_core2__) && defined (__i386) /* 32-bit Core 2,
      for example a 64-bit machine with gmp/mpfr compiled with ABI=32 */
#define MPFR_TUNE_CASE "src/x86/core2/mparam.h"
#include "x86/core2/mparam.h"

#elif defined (__tune_k8__) /* Threshold for AMD 64 */
#define MPFR_TUNE_CASE "src/amd/k8/mparam.h"
#include "amd/k8/mparam.h"

#elif defined (__tune_athlon__) /* Threshold for Athlon */
#define MPFR_TUNE_CASE "src/amd/athlon/mparam.h"
#include "amd/athlon/mparam.h"

#elif defined (__tune_pentiumpro__) || defined (__tune_i686__) || defined (__i386) /* we consider all other 386's here */
#define MPFR_TUNE_CASE "src/x86/mparam.h"
#include "x86/mparam.h"

#elif defined (__ia64) || defined (__itanium__) || defined (__tune_ia64__)
/* Threshold for IA64 */
#define MPFR_TUNE_CASE "src/ia64/mparam.h"
#include "ia64/mparam.h"

#elif defined (__arm__) /* Threshold for ARM */
#define MPFR_TUNE_CASE "src/arm/mparam.h"
#include "arm/mparam.h"

#elif defined (__PPC64__) /* Threshold for 64-bit PowerPC, test it before
                             32-bit PowerPC since _ARCH_PPC is also defined
                             on 64-bit PowerPC */
#define MPFR_TUNE_CASE "src/powerpc64/mparam.h"
#include "powerpc64/mparam.h"

#elif defined (_ARCH_PPC) /* Threshold for 32-bit PowerPC */
#define MPFR_TUNE_CASE "src/powerpc32/mparam.h"
#include "powerpc32/mparam.h"

#elif defined (__sparc_v9__) /* Threshold for 64-bits Sparc */
#define MPFR_TUNE_CASE "src/sparc64/mparam.h"
#include "sparc64/mparam.h"

#elif defined (__hppa__) /* Threshold for HPPA */
#define MPFR_TUNE_CASE "src/hppa/mparam.h"
#include "hppa/mparam.h"

#else
#define MPFR_TUNE_CASE "default"
#endif

/****************************************************************
 * Default values of Threshold.                                 *
 * Must be included in any case: it checks, for every constant, *
 * if it has been defined, and it sets it to a default value if *
 * it was not previously defined.                               *
 ****************************************************************/
#include "generic/mparam.h"
