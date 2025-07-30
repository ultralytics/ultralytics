dnl  GMP specific autoconf macros


dnl  Copyright 2000-2006, 2009, 2011, 2013-2015 Free Software Foundation, Inc.
dnl
dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.


dnl  Some tests use, or must delete, the default compiler output.  The
dnl  possible filenames are based on what autoconf looks for, namely
dnl
dnl    a.out - normal unix style
dnl    b.out - i960 systems, including gcc there
dnl    a.exe - djgpp
dnl    a_out.exe - OpenVMS DEC C called via GNV wrapper (gnv.sourceforge.net)
dnl    conftest.exe - various DOS compilers


define(IA64_PATTERN,
[[ia64*-*-* | itanium-*-* | itanium2-*-*]])

dnl  Need to be careful not to match m6811, m6812, m68hc11 and m68hc12, all
dnl  of which config.sub accepts.  (Though none of which are likely to work
dnl  with GMP.)
dnl
define(M68K_PATTERN,
[[m68k-*-* | m68[0-9][0-9][0-9]-*-*]])

define(POWERPC64_PATTERN,
[[powerpc64-*-* | powerpc64le-*-* | powerpc620-*-* | powerpc630-*-* | powerpc970-*-* | power[3-9]-*-*]])

define(S390_PATTERN,
[[s390-*-* | z900esa-*-* | z990esa-*-* | z9esa-*-* | z10esa-*-* | z196esa-*-*]])

define(S390X_PATTERN,
[[s390x-*-* | z900-*-* | z990-*-* | z9-*-* | z10-*-* | z196-*-*]])

define(X86_PATTERN,
[[i?86*-*-* | k[5-8]*-*-* | pentium*-*-* | athlon-*-* | viac3*-*-* | geode*-*-* | atom-*-*]])

define(X86_64_PATTERN,
[[athlon64-*-* | k8-*-* | k10-*-* | bobcat-*-* | jaguar*-*-* | bulldozer*-*-* | piledriver*-*-* | steamroller*-*-* | excavator*-*-* | pentium4-*-* | atom-*-* | silvermont-*-* | goldmont-*-* | core2-*-* | corei*-*-* | x86_64-*-* | nano-*-* | nehalem*-*-* | westmere*-*-* | sandybridge*-*-* | ivybridge*-*-* | haswell*-*-* | broadwell*-*-* | skylake*-*-* | cabylake*-*-*]])

dnl  GMP_FAT_SUFFIX(DSTVAR, DIRECTORY)
dnl  ---------------------------------
dnl  Emit code to set shell variable DSTVAR to the suffix for a fat binary
dnl  routine from DIRECTORY.  DIRECTORY can be a shell expression like $foo
dnl  etc.
dnl
dnl  The suffix is directory separators / or \ changed to underscores, and
dnl  if there's more than one directory part, then the first is dropped.
dnl
dnl  For instance,
dnl
dnl      x86         ->  x86
dnl      x86/k6      ->  k6
dnl      x86/k6/mmx  ->  k6_mmx

define(GMP_FAT_SUFFIX,
[[$1=`echo $2 | sed -e '/\//s:^[^/]*/::' -e 's:[\\/]:_:g'`]])


dnl  GMP_REMOVE_FROM_LIST(listvar,item)
dnl  ----------------------------------
dnl  Emit code to remove any occurrence of ITEM from $LISTVAR.  ITEM can be a
dnl  shell expression like $foo if desired.

define(GMP_REMOVE_FROM_LIST,
[remove_from_list_tmp=
for remove_from_list_i in $[][$1]; do
  if test $remove_from_list_i = [$2]; then :;
  else
     remove_from_list_tmp="$remove_from_list_tmp $remove_from_list_i"
  fi
done
[$1]=$remove_from_list_tmp
])


dnl  GMP_STRIP_PATH(subdir)
dnl  ----------------------
dnl  Strip entries */subdir from $path and $fat_path.

define(GMP_STRIP_PATH,
[GMP_STRIP_PATH_VAR(path, [$1])
GMP_STRIP_PATH_VAR(fat_path, [$1])
])

define(GMP_STRIP_PATH_VAR,
[tmp_path=
for i in $[][$1]; do
  case $i in
    */[$2]) ;;
    *) tmp_path="$tmp_path $i" ;;
  esac
done
[$1]="$tmp_path"
])


dnl  GMP_INCLUDE_GMP_H
dnl  -----------------
dnl  Expand to the right way to #include gmp-h.in.  This must be used
dnl  instead of gmp.h, since that file isn't generated until the end of the
dnl  configure.
dnl
dnl  Dummy value for GMP_LIMB_BITS is enough
dnl  for all current configure-time uses of gmp.h.

define(GMP_INCLUDE_GMP_H,
[[#define __GMP_WITHIN_CONFIGURE 1   /* ignore template stuff */
#define GMP_NAIL_BITS $GMP_NAIL_BITS
#define GMP_LIMB_BITS 123
$DEFN_LONG_LONG_LIMB
#include "$srcdir/gmp-h.in"]
])


dnl  GMP_HEADER_GETVAL(NAME,FILE)
dnl  ----------------------------
dnl  Expand at autoconf time to the value of a "#define NAME" from the given
dnl  FILE.  The regexps here aren't very rugged, but are enough for gmp.
dnl  /dev/null as a parameter prevents a hang if $2 is accidentally omitted.

define(GMP_HEADER_GETVAL,
[patsubst(patsubst(
esyscmd([grep "^#define $1 " $2 /dev/null 2>/dev/null]),
[^.*$1[ 	]+],[]),
[[
 	]*$],[])])


dnl  GMP_VERSION
dnl  -----------
dnl  The gmp version number, extracted from the #defines in gmp-h.in at
dnl  autoconf time.  Two digits like 3.0 if patchlevel <= 0, or three digits
dnl  like 3.0.1 if patchlevel > 0.

define(GMP_VERSION,
[GMP_HEADER_GETVAL(__GNU_MP_VERSION,gmp-h.in)[]dnl
.GMP_HEADER_GETVAL(__GNU_MP_VERSION_MINOR,gmp-h.in)[]dnl
.GMP_HEADER_GETVAL(__GNU_MP_VERSION_PATCHLEVEL,gmp-h.in)])


dnl  GMP_SUBST_CHECK_FUNCS(func,...)
dnl  ------------------------------
dnl  Setup an AC_SUBST of HAVE_FUNC_01 for each argument.

AC_DEFUN([GMP_SUBST_CHECK_FUNCS],
[m4_if([$1],,,
[_GMP_SUBST_CHECK_FUNCS(ac_cv_func_[$1],HAVE_[]m4_translit([$1],[a-z],[A-Z])_01)
GMP_SUBST_CHECK_FUNCS(m4_shift($@))])])

dnl  Called: _GMP_SUBST_CHECK_FUNCS(cachevar,substvar)
AC_DEFUN([_GMP_SUBST_CHECK_FUNCS],
[case $[$1] in
yes) AC_SUBST([$2],1) ;;
no)  [$2]=0 ;;
esac
])


dnl  GMP_SUBST_CHECK_HEADERS(foo.h,...)
dnl  ----------------------------------
dnl  Setup an AC_SUBST of HAVE_FOO_H_01 for each argument.

AC_DEFUN([GMP_SUBST_CHECK_HEADERS],
[m4_if([$1],,,
[_GMP_SUBST_CHECK_HEADERS(ac_cv_header_[]m4_translit([$1],[./],[__]),
HAVE_[]m4_translit([$1],[a-z./],[A-Z__])_01)
GMP_SUBST_CHECK_HEADERS(m4_shift($@))])])

dnl  Called: _GMP_SUBST_CHECK_HEADERS(cachevar,substvar)
AC_DEFUN([_GMP_SUBST_CHECK_HEADERS],
[case $[$1] in
yes) AC_SUBST([$2],1) ;;
no)  [$2]=0 ;;
esac
])


dnl  GMP_COMPARE_GE(A1,B1, A2,B2, ...)
dnl  ---------------------------------
dnl  Compare two version numbers A1.A2.etc and B1.B2.etc.  Set
dnl  $gmp_compare_ge to yes or no according to the result.  The A parts
dnl  should be variables, the B parts fixed numbers.  As many parts as
dnl  desired can be included.  An empty string in an A part is taken to be
dnl  zero, the B parts should be non-empty and non-zero.
dnl
dnl  For example,
dnl
dnl      GMP_COMPARE($major,10, $minor,3, $subminor,1)
dnl
dnl  would test whether $major.$minor.$subminor is greater than or equal to
dnl  10.3.1.

AC_DEFUN([GMP_COMPARE_GE],
[gmp_compare_ge=no
GMP_COMPARE_GE_INTERNAL($@)
])

AC_DEFUN([GMP_COMPARE_GE_INTERNAL],
[ifelse(len([$3]),0,
[if test -n "$1" && test "$1" -ge $2; then
  gmp_compare_ge=yes
fi],
[if test -n "$1"; then
  if test "$1" -gt $2; then
    gmp_compare_ge=yes
  else
    if test "$1" -eq $2; then
      GMP_COMPARE_GE_INTERNAL(m4_shift(m4_shift($@)))
    fi
  fi
fi])
])


dnl  GMP_PROG_AR
dnl  -----------
dnl  GMP additions to $AR.
dnl
dnl  A cross-"ar" may be necessary when cross-compiling since the build
dnl  system "ar" might try to interpret the object files to build a symbol
dnl  table index, hence the use of AC_CHECK_TOOL.
dnl
dnl  A user-selected $AR is always left unchanged.  AC_CHECK_TOOL is still
dnl  run to get the "checking" message printed though.
dnl
dnl  If extra flags are added to AR, then ac_cv_prog_AR and
dnl  ac_cv_prog_ac_ct_AR are set too, since libtool (cvs 2003-03-31 at
dnl  least) will do an AC_CHECK_TOOL and that will AR from one of those two
dnl  cached variables.  (ac_cv_prog_AR is used if there's an ac_tool_prefix,
dnl  or ac_cv_prog_ac_ct_AR is used otherwise.)  FIXME: This is highly
dnl  dependent on autoconf internals, perhaps it'd work to put our extra
dnl  flags into AR_FLAGS instead.
dnl
dnl  $AR_FLAGS is set to "cq" rather than leaving it to libtool "cru".  The
dnl  latter fails when libtool goes into piecewise mode and is unlucky
dnl  enough to have two same-named objects in separate pieces, as happens
dnl  for instance to random.o (and others) on vax-dec-ultrix4.5.  Naturally
dnl  a user-selected $AR_FLAGS is left unchanged.
dnl
dnl  For reference, $ARFLAGS is used by automake (1.8) for its ".a" archive
dnl  file rules.  This doesn't get used by the piecewise linking, so we
dnl  leave it at the default "cru".
dnl
dnl  FIXME: Libtool 1.5.2 has its own arrangements for "cq", but that version
dnl  is broken in other ways.  When we can upgrade, remove the forcible
dnl  AR_FLAGS=cq.

AC_DEFUN([GMP_PROG_AR],
[dnl  Want to establish $AR before libtool initialization.
AC_BEFORE([$0],[AC_PROG_LIBTOOL])
gmp_user_AR=$AR
AC_CHECK_TOOL(AR, ar, ar)
if test -z "$gmp_user_AR"; then
                        eval arflags=\"\$ar${abi1}_flags\"
  test -n "$arflags" || eval arflags=\"\$ar${abi2}_flags\"
  if test -n "$arflags"; then
    AC_MSG_CHECKING([for extra ar flags])
    AR="$AR $arflags"
    ac_cv_prog_AR="$AR $arflags"
    ac_cv_prog_ac_ct_AR="$AR $arflags"
    AC_MSG_RESULT([$arflags])
  fi
fi
if test -z "$AR_FLAGS"; then
  AR_FLAGS=cq
fi
])


dnl  GMP_PROG_M4
dnl  -----------
dnl  Find a working m4, either in $PATH or likely locations, and setup $M4
dnl  and an AC_SUBST accordingly.  If $M4 is already set then it's a user
dnl  choice and is accepted with no checks.  GMP_PROG_M4 is like
dnl  AC_PATH_PROG or AC_CHECK_PROG, but tests each m4 found to see if it's
dnl  good enough.
dnl
dnl  See mpn/asm-defs.m4 for details on the known bad m4s.

AC_DEFUN([GMP_PROG_M4],
[AC_ARG_VAR(M4,[m4 macro processor])
AC_CACHE_CHECK([for suitable m4],
                gmp_cv_prog_m4,
[if test -n "$M4"; then
  gmp_cv_prog_m4="$M4"
else
  cat >conftest.m4 <<\EOF
dnl  Must protect this against being expanded during autoconf m4!
dnl  Dont put "dnl"s in this as autoconf will flag an error for unexpanded
dnl  macros.
[define(dollarhash,``$][#'')ifelse(dollarhash(x),1,`define(t1,Y)',
``bad: $][# not supported (SunOS /usr/bin/m4)
'')ifelse(eval(89),89,`define(t2,Y)',
`bad: eval() doesnt support 8 or 9 in a constant (OpenBSD 2.6 m4)
')ifelse(eval(9,9),10,`define(t3,Y)',
`bad: eval() doesnt support radix in eval (FreeBSD 8.x,9.0,9.1,9.2 m4)
')ifelse(t1`'t2`'t3,YYY,`good
')]
EOF
dnl ' <- balance the quotes for emacs sh-mode
  echo "trying m4" >&AC_FD_CC
  gmp_tmp_val=`(m4 conftest.m4) 2>&AC_FD_CC`
  echo "$gmp_tmp_val" >&AC_FD_CC
  if test "$gmp_tmp_val" = good; then
    gmp_cv_prog_m4="m4"
  else
    IFS="${IFS= 	}"; ac_save_ifs="$IFS"; IFS=":"
dnl $ac_dummy forces splitting on constant user-supplied paths.
dnl POSIX.2 word splitting is done only on the output of word expansions,
dnl not every word.  This closes a longstanding sh security hole.
    ac_dummy="$PATH:/usr/5bin"
    for ac_dir in $ac_dummy; do
      test -z "$ac_dir" && ac_dir=.
      echo "trying $ac_dir/m4" >&AC_FD_CC
      gmp_tmp_val=`($ac_dir/m4 conftest.m4) 2>&AC_FD_CC`
      echo "$gmp_tmp_val" >&AC_FD_CC
      if test "$gmp_tmp_val" = good; then
        gmp_cv_prog_m4="$ac_dir/m4"
        break
      fi
    done
    IFS="$ac_save_ifs"
    if test -z "$gmp_cv_prog_m4"; then
      AC_MSG_ERROR([No usable m4 in \$PATH or /usr/5bin (see config.log for reasons).])
    fi
  fi
  rm -f conftest.m4
fi])
M4="$gmp_cv_prog_m4"
AC_SUBST(M4)
])


dnl  GMP_M4_M4WRAP_SPURIOUS
dnl  ----------------------
dnl  Check for spurious output from m4wrap(), as described in mpn/asm-defs.m4.
dnl
dnl  The following systems have been seen with the problem.
dnl
dnl  - Unicos alpha, but its assembler doesn't seem to mind.
dnl  - MacOS X Darwin, its assembler fails.
dnl  - NetBSD 1.4.1 m68k, and gas 1.92.3 there gives a warning and ignores
dnl    the bad last line since it doesn't have a newline.
dnl  - NetBSD 1.4.2 alpha, but its assembler doesn't seem to mind.
dnl  - HP-UX ia64.
dnl
dnl  Enhancement: Maybe this could be in GMP_PROG_M4, and attempt to prefer
dnl  an m4 with a working m4wrap, if it can be found.

AC_DEFUN([GMP_M4_M4WRAP_SPURIOUS],
[AC_REQUIRE([GMP_PROG_M4])
AC_CACHE_CHECK([if m4wrap produces spurious output],
               gmp_cv_m4_m4wrap_spurious,
[# hide the d-n-l from autoconf's error checking
tmp_d_n_l=d""nl
cat >conftest.m4 <<EOF
[changequote({,})define(x,)m4wrap({x})$tmp_d_n_l]
EOF
echo test input is >&AC_FD_CC
cat conftest.m4 >&AC_FD_CC
tmp_chars=`$M4 conftest.m4 | wc -c`
echo produces $tmp_chars chars output >&AC_FD_CC
rm -f conftest.m4
if test $tmp_chars = 0; then
  gmp_cv_m4_m4wrap_spurious=no
else
  gmp_cv_m4_m4wrap_spurious=yes
fi
])
GMP_DEFINE_RAW(["define(<M4WRAP_SPURIOUS>,<$gmp_cv_m4_m4wrap_spurious>)"])
])


dnl  GMP_PROG_NM
dnl  -----------
dnl  GMP additions to libtool AC_PROG_NM.
dnl
dnl  Note that if AC_PROG_NM can't find a working nm it still leaves
dnl  $NM set to "nm", so $NM can't be assumed to actually work.
dnl
dnl  A user-selected $NM is always left unchanged.  AC_PROG_NM is still run
dnl  to get the "checking" message printed though.
dnl
dnl  Perhaps it'd be worthwhile checking that nm works, by running it on an
dnl  actual object file.  For instance on sparcv9 solaris old versions of
dnl  GNU nm don't recognise 64-bit objects.  Checking would give a better
dnl  error message than just a failure in later tests like GMP_ASM_W32 etc.
dnl
dnl  On the other hand it's not really normal autoconf practice to take too
dnl  much trouble over detecting a broken set of tools.  And libtool doesn't
dnl  do anything at all for say ranlib or strip.  So for now we're inclined
dnl  to just demand that the user provides a coherent environment.

AC_DEFUN([GMP_PROG_NM],
[dnl  Make sure we're the first to call AC_PROG_NM, so our extra flags are
dnl   used by everyone.
AC_BEFORE([$0],[AC_PROG_NM])
gmp_user_NM=$NM
AC_PROG_NM

# FIXME: When cross compiling (ie. $ac_tool_prefix not empty), libtool
# defaults to plain "nm" if a "${ac_tool_prefix}nm" is not found.  In this
# case run it again to try the native "nm", firstly so that likely locations
# are searched, secondly so that -B or -p are added if necessary for BSD
# format.  This is necessary for instance on OSF with "./configure
# --build=alphaev5-dec-osf --host=alphaev6-dec-osf".
#
if test -z "$gmp_user_NM" && test -n "$ac_tool_prefix" && test "$NM" = nm; then
  $as_unset lt_cv_path_NM
  gmp_save_ac_tool_prefix=$ac_tool_prefix
  ac_tool_prefix=
  NM=
  AC_PROG_NM
  ac_tool_prefix=$gmp_save_ac_tool_prefix
fi

if test -z "$gmp_user_NM"; then
                        eval nmflags=\"\$nm${abi1}_flags\"
  test -n "$nmflags" || eval nmflags=\"\$nm${abi2}_flags\"
  if test -n "$nmflags"; then
    AC_MSG_CHECKING([for extra nm flags])
    NM="$NM $nmflags"
    AC_MSG_RESULT([$nmflags])
  fi
fi
])


dnl  GMP_PROG_CC_WORKS(cc+cflags,[ACTION-IF-WORKS][,ACTION-IF-NOT-WORKS])
dnl  --------------------------------------------------------------------
dnl  Check if cc+cflags can compile and link.
dnl
dnl  This test is designed to be run repeatedly with different cc+cflags
dnl  selections, so the result is not cached.
dnl
dnl  For a native build, meaning $cross_compiling == no, we require that the
dnl  generated program will run.  This is the same as AC_PROG_CC does in
dnl  _AC_COMPILER_EXEEXT_WORKS, and checking here will ensure we don't pass
dnl  a CC/CFLAGS combination that it rejects.
dnl
dnl  sparc-*-solaris2.7 can compile ABI=64 but won't run it if the kernel
dnl  was booted in 32-bit mode.  The effect of requiring the compiler output
dnl  will run is that a plain native "./configure" falls back on ABI=32, but
dnl  ABI=64 is still available as a cross-compile.
dnl
dnl  The various specific problems we try to detect are done in separate
dnl  compiles.  Although this is probably a bit slower than one test
dnl  program, it makes it easy to indicate the problem in AC_MSG_RESULT,
dnl  hence giving the user a clue about why we rejected the compiler.

AC_DEFUN([GMP_PROG_CC_WORKS],
[AC_MSG_CHECKING([compiler $1])
gmp_prog_cc_works=yes

# first see a simple "main()" works, then go on to other checks
GMP_PROG_CC_WORKS_PART([$1], [])

GMP_PROG_CC_WORKS_PART([$1], [function pointer return],
[/* The following provokes an internal error from gcc 2.95.2 -mpowerpc64
   (without -maix64), hence detecting an unusable compiler */
void *g() { return (void *) 0; }
void *f() { return g(); }
])

GMP_PROG_CC_WORKS_PART([$1], [cmov instruction],
[/* The following provokes an invalid instruction syntax from i386 gcc
   -march=pentiumpro on Solaris 2.8.  The native sun assembler
   requires a non-standard syntax for cmov which gcc (as of 2.95.2 at
   least) doesn't know.  */
int n;
int cmov () { return (n >= 0 ? n : 0); }
])

GMP_PROG_CC_WORKS_PART([$1], [double -> ulong conversion],
[/* The following provokes a linker invocation problem with gcc 3.0.3
   on AIX 4.3 under "-maix64 -mpowerpc64 -mcpu=630".  The -mcpu=630
   option causes gcc to incorrectly select the 32-bit libgcc.a, not
   the 64-bit one, and consequently it misses out on the __fixunsdfdi
   helper (double -> uint64 conversion).  */
double d;
unsigned long gcc303 () { return (unsigned long) d; }
])

GMP_PROG_CC_WORKS_PART([$1], [double negation],
[/* The following provokes an error from hppa gcc 2.95 under -mpa-risc-2-0 if
   the assembler doesn't know hppa 2.0 instructions.  fneg is a 2.0
   instruction, and a negation like this comes out using it.  */
double fneg_data;
unsigned long fneg () { return -fneg_data; }
])

GMP_PROG_CC_WORKS_PART([$1], [double -> float conversion],
[/* The following makes gcc 3.3 -march=pentium4 generate an SSE2 xmm insn
   (cvtsd2ss) which will provoke an error if the assembler doesn't recognise
   those instructions.  Not sure how much of the gmp code will come out
   wanting sse2, but it's easiest to reject an option we know is bad.  */
double ftod_data;
float ftod () { return (float) ftod_data; }
])

GMP_PROG_CC_WORKS_PART([$1], [gnupro alpha ev6 char spilling],
[/* The following provokes an internal compiler error from gcc version
   "2.9-gnupro-99r1" under "-O2 -mcpu=ev6", apparently relating to char
   values being spilled into floating point registers.  The problem doesn't
   show up all the time, but has occurred enough in GMP for us to reject
   this compiler+flags.  */
#include <string.h>  /* for memcpy */
struct try_t
{
 char dst[2];
 char size;
 long d0, d1, d2, d3, d4, d5, d6;
 char overlap;
};
struct try_t param[6];
int
param_init ()
{
 struct try_t *p;
 memcpy (p, &param[ 2 ], sizeof (*p));
 memcpy (p, &param[ 2 ], sizeof (*p));
 p->size = 2;
 memcpy (p, &param[ 1 ], sizeof (*p));
 p->dst[0] = 1;
 p->overlap = 2;
 memcpy (p, &param[ 3 ], sizeof (*p));
 p->dst[0] = 1;
 p->overlap = 8;
 memcpy (p, &param[ 4 ], sizeof (*p));
 memcpy (p, &param[ 4 ], sizeof (*p));
 p->overlap = 8;
 memcpy (p, &param[ 5 ], sizeof (*p));
 memcpy (p, &param[ 5 ], sizeof (*p));
 memcpy (p, &param[ 5 ], sizeof (*p));
 return 0;
}
])

# __builtin_alloca is not available everywhere, check it exists before
# seeing that it works
GMP_PROG_CC_WORKS_PART_TEST([$1],[__builtin_alloca availability],
[int k; int foo () { __builtin_alloca (k); }],
  [GMP_PROG_CC_WORKS_PART([$1], [alloca array],
[/* The following provokes an internal compiler error from Itanium HP-UX cc
    under +O2 or higher.  We use this sort of code in mpn/generic/mul_fft.c. */
int k;
int foo ()
{
  int i, **a;
  a = __builtin_alloca (k);
  for (i = 0; i <= k; i++)
    a[i] = __builtin_alloca (1 << i);
}
])])

GMP_PROG_CC_WORKS_PART([$1], [abs int -> double conversion],
[/* The following provokes an internal error from the assembler on
   power2-ibm-aix4.3.1.0.  gcc -mrios2 compiles to nabs+fcirz, and this
   results in "Internal error related to the source program domain".

   For reference it seems to be the combination of nabs+fcirz which is bad,
   not either alone.  This sort of thing occurs in mpz/get_str.c with the
   way double chars_per_bit_exactly is applied in MPN_SIZEINBASE.  Perhaps
   if that code changes to a scaled-integer style then we won't need this
   test.  */

double fp[1];
int x;
int f ()
{
  int a;
  a = (x >= 0 ? x : -x);
  return a * fp[0];
}
])

GMP_PROG_CC_WORKS_PART([$1], [long long reliability test 1],
[/* The following provokes a segfault in the compiler on powerpc-apple-darwin.
   Extracted from tests/mpn/t-iord_u.c.  Causes Apple's gcc 3.3 build 1640 and
   1666 to segfault with e.g., -O2 -mpowerpc64.  */

#if defined (__GNUC__) && ! defined (__cplusplus)
typedef unsigned long long t1;typedef t1*t2;
void g(){}
void h(){}
static __inline__ t1 e(t2 rp,t2 up,int n,t1 v0)
{t1 c,x,r;int i;if(v0){c=1;for(i=1;i<n;i++){x=up[i];r=x+1;rp[i]=r;}}return c;}
void f(){static const struct{t1 n;t1 src[9];t1 want[9];}d[]={{1,{0},{1}},};t1 got[9];int i;
for(i=0;i<1;i++){if(e(got,got,9,d[i].n)==0)h();g(i,d[i].src,d[i].n,got,d[i].want,9);if(d[i].n)h();}}
#else
int dummy;
#endif
])

GMP_PROG_CC_WORKS_PART([$1], [long long reliability test 2],
[/* The following provokes an internal compiler error on powerpc-apple-darwin.
   Extracted from mpz/cfdiv_q_2exp.c.  Causes Apple's gcc 3.3 build 1640 and
   1666 to get an ICE with -O1 -mpowerpc64.  */

#if defined (__GNUC__) && ! defined (__cplusplus)
int g();
void f(int u){int i;long long x;x=u?~0:0;if(x)for(i=0;i<9;i++);x&=g();if(x)g();}
int g(){return 0;}
#else
int dummy;
#endif
])

GMP_PROG_CC_WORKS_PART([$1], [freebsd hacked gcc],
[/* Provokes an ICE on i386-freebsd with the FreeBSD-hacked gcc, under
   -O2 -march=amdfam10.  We call helper functions here "open" and "close" in
   order for linking to succeed.  */

#if defined (__GNUC__) && ! defined (__cplusplus)
int open(int*,int*,int);void*close(int);void g(int*rp,int*up,int un){
__builtin_expect(un<=0x7f00,1)?__builtin_alloca(un):close(un);if(__builtin_clzl
(up[un])){open(rp,up,un);while(1){if(rp[un-1]!=0)break;un--;}}}
#else
int dummy;
#endif
])

GMP_PROG_CC_WORKS_PART_MAIN([$1], [mpn_lshift_com optimization],
[/* The following is mis-compiled by HP ia-64 cc version
        cc: HP aC++/ANSI C B3910B A.05.55 [Dec 04 2003]
   under "cc +O3", both in +DD32 and +DD64 modes.  The mpn_lshift_com gets
   inlined and its return value somehow botched to be 0 instead of 1.  This
   arises in the real mpn_lshift_com in mul_fft.c.  A lower optimization
   level, like +O2 seems ok.  This code needs to be run to show the problem,
   but that's fine, the offending cc is a native-only compiler so we don't
   have to worry about cross compiling.  */

#if ! defined (__cplusplus)
unsigned long
lshift_com (rp, up, n, cnt)
  unsigned long *rp;
  unsigned long *up;
  long n;
  unsigned cnt;
{
  unsigned long retval, high_limb, low_limb;
  unsigned tnc;
  long i;
  tnc = 8 * sizeof (unsigned long) - cnt;
  low_limb = *up++;
  retval = low_limb >> tnc;
  high_limb = low_limb << cnt;
  for (i = n - 1; i != 0; i--)
    {
      low_limb = *up++;
      *rp++ = ~(high_limb | (low_limb >> tnc));
      high_limb = low_limb << cnt;
    }
  return retval;
}
int
main ()
{
  unsigned long cy, rp[2], up[2];
  up[0] = ~ 0L;
  up[1] = 0;
  cy = lshift_com (rp, up, 2L, 1);
  if (cy != 1L)
    return 1;
  return 0;
}
#else
int
main ()
{
  return 0;
}
#endif
])

GMP_PROG_CC_WORKS_PART_MAIN([$1], [mpn_lshift_com optimization 2],
[/* The following is mis-compiled by Intel ia-64 icc version 1.8 under
    "icc -O3",  After several calls, the function writes partial garbage to
    the result vector.  Perhaps relates to the chk.a.nc insn.  This code needs
    to be run to show the problem, but that's fine, the offending cc is a
    native-only compiler so we don't have to worry about cross compiling.  */

#if ! defined (__cplusplus)
#include <stdlib.h>
void
lshift_com (rp, up, n, cnt)
  unsigned long *rp;
  unsigned long *up;
  long n;
  unsigned cnt;
{
  unsigned long high_limb, low_limb;
  unsigned tnc;
  long i;
  up += n;
  rp += n;
  tnc = 8 * sizeof (unsigned long) - cnt;
  low_limb = *--up;
  high_limb = low_limb << cnt;
  for (i = n - 1; i != 0; i--)
    {
      low_limb = *--up;
      *--rp = ~(high_limb | (low_limb >> tnc));
      high_limb = low_limb << cnt;
    }
  *--rp = ~high_limb;
}
int
main ()
{
  unsigned long *r, *r2;
  unsigned long a[88 + 1];
  long i;
  for (i = 0; i < 88 + 1; i++)
    a[i] = ~0L;
  r = malloc (10000 * sizeof (unsigned long));
  r2 = r;
  for (i = 0; i < 528; i += 22)
    {
      lshift_com (r2, a,
		  i / (8 * sizeof (unsigned long)) + 1,
		  i % (8 * sizeof (unsigned long)));
      r2 += 88 + 1;
    }
  if (r[2048] != 0 || r[2049] != 0 || r[2050] != 0 || r[2051] != 0 ||
      r[2052] != 0 || r[2053] != 0 || r[2054] != 0)
    abort ();
  return 0;
}
#else
int
main ()
{
  return 0;
}
#endif
])


# A certain _GLOBAL_OFFSET_TABLE_ problem in past versions of gas, tickled
# by recent versions of gcc.
#
if test "$gmp_prog_cc_works" = yes; then
  case $host in
    X86_PATTERN)
      # this problem only arises in PIC code, so don't need to test when
      # --disable-shared.  We don't necessarily have $enable_shared set to
      # yes at this point, it will still be unset for the default (which is
      # yes); hence the use of "!= no".
      if test "$enable_shared" != no; then
        GMP_PROG_CC_X86_GOT_EAX_EMITTED([$1],
          [GMP_ASM_X86_GOT_EAX_OK([$1],,
            [gmp_prog_cc_works="no, bad gas GOT with eax"])])
      fi
      ;;
  esac
fi

AC_MSG_RESULT($gmp_prog_cc_works)
case $gmp_prog_cc_works in
  yes)
    [$2]
    ;;
  *)
    [$3]
    ;;
esac
])

dnl  Called: GMP_PROG_CC_WORKS_PART(CC+CFLAGS,FAIL-MESSAGE [,CODE])
dnl  A dummy main() is appended to the CODE given.
dnl
AC_DEFUN([GMP_PROG_CC_WORKS_PART],
[GMP_PROG_CC_WORKS_PART_MAIN([$1],[$2],
[$3]
[int main () { return 0; }])
])

dnl  Called: GMP_PROG_CC_WORKS_PART_MAIN(CC+CFLAGS,FAIL-MESSAGE,CODE)
dnl  CODE must include a main().
dnl
AC_DEFUN([GMP_PROG_CC_WORKS_PART_MAIN],
[GMP_PROG_CC_WORKS_PART_TEST([$1],[$2],[$3],
  [],
  gmp_prog_cc_works="no[]m4_if([$2],,,[[, ]])[$2]",
  gmp_prog_cc_works="no[]m4_if([$2],,,[[, ]])[$2][[, program does not run]]")
])

dnl  Called: GMP_PROG_CC_WORKS_PART_TEST(CC+CFLAGS,TITLE,[CODE],
dnl            [ACTION-GOOD],[ACTION-BAD][ACTION-NORUN])
dnl
AC_DEFUN([GMP_PROG_CC_WORKS_PART_TEST],
[if test "$gmp_prog_cc_works" = yes; then
  # remove anything that might look like compiler output to our "||" expression
  rm -f conftest* a.out b.out a.exe a_out.exe
  cat >conftest.c <<EOF
[$3]
EOF
  echo "Test compile: [$2]" >&AC_FD_CC
  gmp_compile="$1 conftest.c >&AC_FD_CC"
  if AC_TRY_EVAL(gmp_compile); then
    cc_works_part=yes
    if test "$cross_compiling" = no; then
      if AC_TRY_COMMAND([./a.out || ./b.out || ./a.exe || ./a_out.exe || ./conftest]); then :;
      else
        cc_works_part=norun
      fi
    fi
  else
    cc_works_part=no
  fi
  if test "$cc_works_part" != yes; then
    echo "failed program was:" >&AC_FD_CC
    cat conftest.c >&AC_FD_CC
  fi
  rm -f conftest* a.out b.out a.exe a_out.exe
  case $cc_works_part in
    yes)
      $4
      ;;
    no)
      $5
      ;;
    norun)
      $6
      ;;
  esac
fi
])


dnl  GMP_PROG_CC_WORKS_LONGLONG(cc+cflags,[ACTION-YES][,ACTION-NO])
dnl  --------------------------------------------------------------
dnl  Check that cc+cflags accepts "long long".
dnl
dnl  This test is designed to be run repeatedly with different cc+cflags
dnl  selections, so the result is not cached.

AC_DEFUN([GMP_PROG_CC_WORKS_LONGLONG],
[AC_MSG_CHECKING([compiler $1 has long long])
cat >conftest.c <<EOF
long long  foo;
long long  bar () { return foo; }
int main () { return 0; }
EOF
gmp_prog_cc_works=no
gmp_compile="$1 -c conftest.c >&AC_FD_CC"
if AC_TRY_EVAL(gmp_compile); then
  gmp_prog_cc_works=yes
else
  echo "failed program was:" >&AC_FD_CC
  cat conftest.c >&AC_FD_CC
fi
rm -f conftest* a.out b.out a.exe a_out.exe
AC_MSG_RESULT($gmp_prog_cc_works)
if test $gmp_prog_cc_works = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_C_TEST_SIZEOF(cc/cflags,test,[ACTION-GOOD][,ACTION-BAD])
dnl  ------------------------------------------------------------
dnl  The given cc/cflags compiler is run to check the size of a type
dnl  specified by the "test" argument.  "test" can either be a string, or a
dnl  variable like $foo.  The value should be for instance "sizeof-long-4",
dnl  to test that sizeof(long)==4.
dnl
dnl  This test is designed to be run for different compiler and/or flags
dnl  combinations, so the result is not cached.
dnl
dnl  The idea for making an array that has a negative size if the desired
dnl  condition test is false comes from autoconf AC_CHECK_SIZEOF.  The cast
dnl  to "long" in the array dimension also follows autoconf, apparently it's
dnl  a workaround for a HP compiler bug.

AC_DEFUN([GMP_C_TEST_SIZEOF],
[echo "configure: testlist $2" >&AC_FD_CC
[gmp_sizeof_type=`echo "$2" | sed 's/sizeof-\([a-z]*\).*/\1/'`]
[gmp_sizeof_want=`echo "$2" | sed 's/sizeof-[a-z]*-\([0-9]*\).*/\1/'`]
AC_MSG_CHECKING([compiler $1 has sizeof($gmp_sizeof_type)==$gmp_sizeof_want])
cat >conftest.c <<EOF
[int
main ()
{
  static int test_array [1 - 2 * (long) (sizeof ($gmp_sizeof_type) != $gmp_sizeof_want)];
  test_array[0] = 0;
  return 0;
}]
EOF
gmp_c_testlist_sizeof=no
gmp_compile="$1 -c conftest.c >&AC_FD_CC"
if AC_TRY_EVAL(gmp_compile); then
  gmp_c_testlist_sizeof=yes
fi
rm -f conftest*
AC_MSG_RESULT($gmp_c_testlist_sizeof)
if test $gmp_c_testlist_sizeof = yes; then
  ifelse([$3],,:,[$3])
else
  ifelse([$4],,:,[$4])
fi
])


dnl  GMP_PROG_CC_IS_GNU(CC,[ACTIONS-IF-YES][,ACTIONS-IF-NO])
dnl  -------------------------------------------------------
dnl  Determine whether the given compiler is GNU C.
dnl
dnl  This test is the same as autoconf _AC_LANG_COMPILER_GNU, but doesn't
dnl  cache the result.  The same "ifndef" style test is used, to avoid
dnl  problems with syntax checking cpp's used on NeXT and Apple systems.

AC_DEFUN([GMP_PROG_CC_IS_GNU],
[cat >conftest.c <<EOF
#if ! defined (__GNUC__) || defined (__INTEL_COMPILER)
  choke me
#endif
EOF
gmp_compile="$1 -c conftest.c >&AC_FD_CC"
if AC_TRY_EVAL(gmp_compile); then
  rm -f conftest*
  AC_MSG_CHECKING([whether $1 is gcc])
  AC_MSG_RESULT(yes)
  ifelse([$2],,:,[$2])
else
  rm -f conftest*
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_PROG_CC_IS_XLC(CC,[ACTIONS-IF-YES][,ACTIONS-IF-NO])
dnl  -------------------------------------------------------
dnl  Determine whether the given compiler is IBM xlc (on AIX).
dnl
dnl  There doesn't seem to be a preprocessor symbol to test for this, or if
dnl  there is one then it's well hidden in xlc 3.1 on AIX 4.3, so just grep
dnl  the man page printed when xlc is invoked with no arguments.

AC_DEFUN([GMP_PROG_CC_IS_XLC],
[gmp_command="$1 2>&1 | grep xlc >/dev/null"
if AC_TRY_EVAL(gmp_command); then
  AC_MSG_CHECKING([whether $1 is xlc])
  AC_MSG_RESULT(yes)
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_PROG_CC_X86_GOT_EAX_EMITTED(CC+CFLAGS, [ACTION-YES] [, ACTION-NO])
dnl  ----------------------------------------------------------------------
dnl  Determine whether CC+CFLAGS emits instructions using %eax with
dnl  _GLOBAL_OFFSET_TABLE_.  This test is for use on x86 systems.
dnl
dnl  Recent versions of gcc will use %eax for the GOT in leaf functions, for
dnl  instance gcc 3.3.3 with -O3.  This avoids having to save and restore
dnl  %ebx which otherwise usually holds the GOT, and is what gcc used in the
dnl  past.
dnl
dnl  %ecx and %edx are also candidates for this sort of optimization, and
dnl  are used under lesser optimization levels, like -O2 in 3.3.3.  FIXME:
dnl  It's not quite clear what the conditions for using %eax are, we might
dnl  need more test code to provoke it.
dnl
dnl  The motivation for this test is that past versions of gas have bugs
dnl  affecting this usage, see GMP_ASM_X86_GOT_EAX_OK.
dnl
dnl  This test is not specific to gcc, other compilers might emit %eax GOT
dnl  insns like this, though we've not investigated that.
dnl
dnl  This is for use by compiler probing in GMP_PROG_CC_WORKS, so we doesn't
dnl  cache the result.
dnl
dnl  -fPIC is hard coded here, because this test is for use before libtool
dnl  has established the pic options.  It's right for gcc, but perhaps not
dnl  other compilers.

AC_DEFUN([GMP_PROG_CC_X86_GOT_EAX_EMITTED],
[echo "Testing gcc GOT with eax emitted" >&AC_FD_CC
cat >conftest.c <<\EOF
[int foo;
int bar () { return foo; }
]EOF
tmp_got_emitted=no
gmp_compile="$1 -fPIC -S conftest.c >&AC_FD_CC 2>&1"
if AC_TRY_EVAL(gmp_compile); then
  if grep "addl.*_GLOBAL_OFFSET_TABLE_.*eax" conftest.s >/dev/null; then
    tmp_got_emitted=yes
  fi
fi
rm -f conftest.*
echo "Result: $tmp_got_emitted" >&AC_FD_CC
if test "$tmp_got_emitted" = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_HPC_HPPA_2_0(cc,[ACTION-IF-GOOD][,ACTION-IF-BAD])
dnl  ---------------------------------------------------------
dnl  Find out whether a HP compiler is good enough to generate hppa 2.0.
dnl
dnl  This test might be repeated for different compilers, so the result is
dnl  not cached.

AC_DEFUN([GMP_HPC_HPPA_2_0],
[AC_MSG_CHECKING([whether HP compiler $1 is good for 64-bits])
# Bad compiler output:
#   ccom: HP92453-01 G.10.32.05 HP C Compiler
# Good compiler output:
#   ccom: HP92453-01 A.10.32.30 HP C Compiler
# Let A.10.32.30 or higher be ok.
echo >conftest.c
gmp_tmp_vs=`$1 $2 -V -c -o conftest.$OBJEXT conftest.c 2>&1 | grep "^ccom:"`
echo "Version string: $gmp_tmp_vs" >&AC_FD_CC
rm conftest*
gmp_tmp_v1=`echo $gmp_tmp_vs | sed 's/.* .\.\([[0-9]]*\).*/\1/'`
gmp_tmp_v2=`echo $gmp_tmp_vs | sed 's/.* .\..*\.\(.*\)\..* HP C.*/\1/'`
gmp_tmp_v3=`echo $gmp_tmp_vs | sed 's/.* .\..*\..*\.\(.*\) HP C.*/\1/'`
echo "Version number: $gmp_tmp_v1.$gmp_tmp_v2.$gmp_tmp_v3" >&AC_FD_CC
if test -z "$gmp_tmp_v1"; then
  gmp_hpc_64bit=not-applicable
else
  GMP_COMPARE_GE($gmp_tmp_v1, 10, $gmp_tmp_v2, 32, $gmp_tmp_v3, 30)
  gmp_hpc_64bit=$gmp_compare_ge
fi
AC_MSG_RESULT($gmp_hpc_64bit)
if test $gmp_hpc_64bit = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_GCC_ARM_UMODSI(CC,[ACTIONS-IF-GOOD][,ACTIONS-IF-BAD])
dnl  ---------------------------------------------------------
dnl  gcc 2.95.3 and earlier on arm has a bug in the libgcc __umodsi routine
dnl  making "%" give wrong results for some operands, eg. "0x90000000 % 3".
dnl  We're hoping it'll be fixed in 2.95.4, and we know it'll be fixed in
dnl  gcc 3.
dnl
dnl  There's only a couple of places gmp cares about this, one is the
dnl  size==1 case in mpn/generic/mode1o.c, and this shows up in
dnl  tests/mpz/t-jac.c as a wrong result from mpz_kronecker_ui.

AC_DEFUN([GMP_GCC_ARM_UMODSI],
[AC_MSG_CHECKING([whether ARM gcc unsigned division works])
tmp_version=`$1 --version`
echo "$tmp_version" >&AC_FD_CC
case $tmp_version in
  [2.95 | 2.95.[123]])
    ifelse([$3],,:,[$3])
    gmp_gcc_arm_umodsi_result=["no, gcc 2.95.[0123]"] ;;
  *)
    ifelse([$2],,:,[$2])
    gmp_gcc_arm_umodsi_result=yes ;;
esac
AC_MSG_RESULT([$gmp_gcc_arm_umodsi_result])
])


dnl  GMP_GCC_MIPS_O32(gcc,[actions-yes][,[actions-no]])
dnl  -------------------------------------------------
dnl  Test whether gcc supports o32.
dnl
dnl  gcc 2.7.2.2 only does o32, and doesn't accept -mabi=32.
dnl
dnl  gcc 2.95 accepts -mabi=32 but it only works on irix5, on irix6 it gives
dnl  "cc1: The -mabi=32 support does not work yet".

AC_DEFUN([GMP_GCC_MIPS_O32],
[AC_MSG_CHECKING([whether gcc supports o32])
echo 'int x;' >conftest.c
echo "$1 -mabi=32 -c conftest.c" >&AC_FD_CC
if $1 -mabi=32 -c conftest.c >conftest.out 2>&1; then
  result=yes
else
  cat conftest.out >&AC_FD_CC
  if grep "cc1: Invalid option \`abi=32'" conftest.out >/dev/null; then
    result=yes
  else
    result=no
  fi
fi
rm -f conftest.*
AC_MSG_RESULT($result)
if test $result = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_GCC_NO_CPP_PRECOMP(CCBASE,CC,CFLAGS,[ACTIONS-YES][,ACTIONS-NO])
dnl  -------------------------------------------------------------------
dnl  Check whether -no-cpp-precomp should be used on this compiler, and
dnl  execute the corresponding ACTIONS-YES or ACTIONS-NO.
dnl
dnl  -no-cpp-precomp is only meant for Apple's hacked version of gcc found
dnl  on powerpc*-*-darwin*, but we can give it a try on any gcc.  Normal gcc
dnl  (as of 3.0 at least) only gives a warning, not an actual error, and we
dnl  watch for that and decide against the option in that case, to avoid
dnl  confusing the user.

AC_DEFUN([GMP_GCC_NO_CPP_PRECOMP],
[if test "$ccbase" = gcc; then
  AC_MSG_CHECKING([compiler $2 $3 -no-cpp-precomp])
  result=no
  cat >conftest.c <<EOF
int main () { return 0; }
EOF
  gmp_compile="$2 $3 -no-cpp-precomp conftest.c >conftest.out 2>&1"
  if AC_TRY_EVAL(gmp_compile); then
    if grep "unrecognized option.*-no-cpp-precomp" conftest.out >/dev/null; then : ;
    else
      result=yes
    fi
  fi
  cat conftest.out >&AC_FD_CC
  rm -f conftest* a.out b.out a.exe a_out.exe
  AC_MSG_RESULT($result)
  if test "$result" = yes; then
      ifelse([$4],,:,[$4])
  else
      ifelse([$5],,:,[$5])
  fi
fi
])


dnl  GMP_GCC_PENTIUM4_SSE2(CC+CFLAGS,[ACTION-IF-YES][,ACTION-IF-NO])
dnl  ---------------------------------------------------------------
dnl  Determine whether gcc CC+CFLAGS is a good enough version for
dnl  -march=pentium4 with sse2.
dnl
dnl  Gcc 3.2.1 was seen generating incorrect code for raw double -> int
dnl  conversions through a union.  We believe the problem is in all 3.1 and
dnl  3.2 versions, but that it's fixed in 3.3.

AC_DEFUN([GMP_GCC_PENTIUM4_SSE2],
[AC_MSG_CHECKING([whether gcc is good for sse2])
case `$1 -dumpversion` in
  [3.[012] | 3.[012].*]) result=no ;;
  *)                     result=yes ;;
esac
AC_MSG_RESULT($result)
if test "$result" = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_GCC_WA_MCPU(CC+CFLAGS, NEWFLAG [,ACTION-YES [,ACTION-NO]])
dnl  --------------------------------------------------------------
dnl  Check whether gcc (or gas rather) accepts a flag like "-Wa,-mev67".
dnl
dnl  Gas doesn't give an error for an unknown cpu, it only prints a warning
dnl  like "Warning: Unknown CPU identifier `ev78'".
dnl
dnl  This is intended for use on alpha, since only recent versions of gas
dnl  accept -mev67, but there's nothing here that's alpha specific.

AC_DEFUN([GMP_GCC_WA_MCPU],
[AC_MSG_CHECKING([assembler $1 $2])
result=no
cat >conftest.c <<EOF
int main () {}
EOF
gmp_compile="$1 $2 -c conftest.c >conftest.out 2>&1"
if AC_TRY_EVAL(gmp_compile); then
  if grep "Unknown CPU identifier" conftest.out >/dev/null; then : ;
  else
    result=yes
  fi
fi
cat conftest.out >&AC_FD_CC
rm -f conftest*
AC_MSG_RESULT($result)
if test "$result" = yes; then
  ifelse([$3],,:,[$3])
else
  ifelse([$4],,:,[$4])
fi
])


dnl  GMP_GCC_WA_OLDAS(CC+CFLAGS [,ACTION-YES [,ACTION-NO]])
dnl  ------------------------------------------------------
dnl  Check whether gcc should be run with "-Wa,-oldas".
dnl
dnl  On systems alpha*-*-osf* (or maybe just osf5), apparently there's a
dnl  newish Compaq "as" which doesn't work with the gcc mips-tfile.
dnl  Compiling an empty file with "gcc -c foo.c" produces for instance
dnl
dnl      mips-tfile, /tmp/ccaqUNnF.s:7 Segmentation fault
dnl
dnl  The fix is to pass "-oldas" to that assembler, as noted by
dnl
dnl      http://gcc.gnu.org/install/specific.html#alpha*-dec-osf*
dnl
dnl  The test here tries to compile an empty file, and if that fails but
dnl  adding -Wa,-oldas makes it succeed, then that flag is considered
dnl  necessary.
dnl
dnl  We look for the failing case specifically, since it may not be a good
dnl  idea to use -Wa,-oldas in other circumstances.  For instance gas takes
dnl  "-oldas" to mean the "-o" option and will write a file called "ldas" as
dnl  its output.  Normally gcc puts its own "-o" after any -Wa options, so
dnl  -oldas ends up being harmless, but clearly that's only through good
dnl  luck.
dnl
dnl  This macro is designed for use while probing for a good compiler, and
dnl  so doesn't cache it's result.

AC_DEFUN([GMP_GCC_WA_OLDAS],
[AC_MSG_CHECKING([for $1 -Wa,-oldas])
result=no
cat >conftest.c <<EOF
EOF
echo "with empty conftest.c" >&AC_FD_CC
gmp_compile="$1 -c conftest.c >&AC_FD_CC 2>&1"
if AC_TRY_EVAL(gmp_compile); then : ;
else
  # empty fails
  gmp_compile="$1 -Wa,-oldas -c conftest.c >&AC_FD_CC 2>&1"
  if AC_TRY_EVAL(gmp_compile); then
    # but with -Wa,-oldas it works
    result=yes
  fi
fi
rm -f conftest*
AC_MSG_RESULT($result)
if test "$result" = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_OS_X86_XMM(CC+CFLAGS,[ACTION-IF-YES][,ACTION-IF-NO])
dnl  --------------------------------------------------------
dnl  Determine whether the operating system supports XMM registers.
dnl
dnl  If build==host then a test program is run, executing an SSE2
dnl  instruction using an XMM register.  This will give a SIGILL if the
dnl  system hasn't set the OSFXSR bit in CR4 to say it knows it must use
dnl  fxsave/fxrestor in a context switch (to save xmm registers).
dnl
dnl  If build!=host, we can fallback on:
dnl
dnl      - FreeBSD version 4 is the first supporting xmm.
dnl
dnl      - Linux kernel 2.4 might be the first stable series supporting xmm
dnl        (not sure).  But there's no version number in the GNU/Linux
dnl        config tuple to test anyway.
dnl
dnl  The default is to allow xmm.  This might seem rash, but it's likely
dnl  most systems know xmm by now, so this will normally be what's wanted.
dnl  And cross compiling is a bit hairy anyway, so hopefully anyone doing it
dnl  will be smart enough to know what to do.
dnl
dnl  In the test program, .text and .globl are hard coded because this macro
dnl  is wanted before GMP_ASM_TEXT and GMP_ASM_GLOBL are run.  A .byte
dnl  sequence is used (for xorps %xmm0, %xmm0) to make us independent of
dnl  tests for whether the assembler supports sse2/xmm.  Obviously we need
dnl  both assembler and OS support, but this means we don't force the order
dnl  in which we test.
dnl
dnl  FIXME: Maybe we should use $CCAS to assemble, if it's set.  (Would
dnl  still want $CC/$CFLAGS for the link.)  But this test is used before
dnl  AC_PROG_CC sets $OBJEXT, so we'd need to check for various object file
dnl  suffixes ourselves.

AC_DEFUN([GMP_OS_X86_XMM],
[AC_CACHE_CHECK([whether the operating system supports XMM registers],
		gmp_cv_os_x86_xmm,
[if test "$build" = "$host"; then
  # remove anything that might look like compiler output to our "||" expression
  rm -f conftest* a.out b.out a.exe a_out.exe
  cat >conftest.s <<EOF
	.text
main:
_main:
	.globl	main
	.globl	_main
	.byte	0x0f, 0x57, 0xc0
	xorl	%eax, %eax
	ret
EOF
  gmp_compile="$1 conftest.s -o conftest >&AC_FD_CC"
  if AC_TRY_EVAL(gmp_compile); then
    if AC_TRY_COMMAND([./a.out || ./b.out || ./a.exe || ./a_out.exe || ./conftest]); then
      gmp_cv_os_x86_xmm=yes
    else
      gmp_cv_os_x86_xmm=no
    fi
  else
    AC_MSG_WARN([Oops, cannot compile test program])
  fi
  rm -f conftest*
fi

if test -z "$gmp_cv_os_x86_xmm"; then
  case $host_os in
    [freebsd[123] | freebsd[123].*])
      gmp_cv_os_x86_xmm=no ;;
    freebsd*)
      gmp_cv_os_x86_xmm=yes ;;
    *)
      gmp_cv_os_x86_xmm=probably ;;
  esac
fi
])

if test "$gmp_cv_os_x86_xmm" = probably; then
  AC_MSG_WARN([Not certain of OS support for xmm when cross compiling.])
  AC_MSG_WARN([Will assume it's ok, expect a SIGILL if this is wrong.])
fi

case $gmp_cv_os_x86_xmm in
no)
  $3
  ;;
*)
  $2
  ;;
esac
])


dnl  GMP_CRAY_HOST_TYPES(C90/T90-IEEE, C90/T90-CFP, J90/SV1)
dnl  -------------------------------------------------------
dnl  Execute the actions in the arguments on the respective Cray vector
dnl  systems.  For other hosts, do nothing.
dnl
dnl  This macro should be used after the C compiler has been chosen, since
dnl  on c90 and t90 we ask the compiler whether we're in IEEE or CFP float
dnl  mode.
dnl
dnl  This code is in a macro so that any AC_REQUIRE pre-requisites of
dnl  AC_EGREP_CPP will be expanded at the top-level, ie. for all hosts not
dnl  merely c90 and t90.  In autoconf 2.57 for instance this means
dnl  AC_PROG_EGREP, which is needed by various other macros.

AC_DEFUN([GMP_CRAY_OPTIONS],
[case $host_cpu in
  c90 | t90)
    AC_EGREP_CPP(yes,
[#ifdef _CRAYIEEE
yes
#endif],
    [$1],
    [$2])
    ;;
  j90 | sv1)
    [$3]
    ;;
esac
])


dnl  GMP_HPPA_LEVEL_20(cc/cflags [, ACTION-GOOD [,ACTION-BAD]])
dnl  ----------------------------------------------------------
dnl  Check that the given cc/cflags accepts HPPA 2.0n assembler code.
dnl
dnl  Old versions of gas don't know 2.0 instructions.  It rejects ".level
dnl  2.0" for a start, so just test that.
dnl
dnl  This test is designed to be run for various different compiler and
dnl  flags combinations, and hence doesn't cache its result.

AC_DEFUN([GMP_HPPA_LEVEL_20],
[AC_MSG_CHECKING([$1 assembler knows hppa 2.0])
result=no
cat >conftest.s <<EOF
	.level 2.0
EOF
gmp_compile="$1 -c conftest.s >&AC_FD_CC 2>&1"
if AC_TRY_EVAL(gmp_compile); then
  result=yes
else
  echo "failed program was" >&AC_FD_CC
  cat conftest.s >&AC_FD_CC
fi
rm -f conftest*
AC_MSG_RESULT($result)
if test "$result" = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_PROG_CXX_WORKS(cxx/cxxflags [, ACTION-YES [,ACTION-NO]])
dnl  ------------------------------------------------------------
dnl  Check whether cxx/cxxflags can compile and link.
dnl
dnl  This test is designed to be run repeatedly with different cxx/cxxflags
dnl  selections, so the result is not cached.
dnl
dnl  For a native build, we insist on being able to run the program, so as
dnl  to detect any problems with the standard C++ library.  During
dnl  development various systems with broken or incomplete C++ installations
dnl  were seen.
dnl
dnl  The various features and problems we try to detect are done in separate
dnl  compiles.  Although this is probably a bit slower than one test
dnl  program, it makes it easy to indicate the problem in AC_MSG_RESULT,
dnl  hence giving the user a clue about why we rejected the compiler.

AC_DEFUN([GMP_PROG_CXX_WORKS],
[AC_MSG_CHECKING([C++ compiler $1])
gmp_prog_cxx_works=yes

# start with a plain "main()", then go on to further checks
GMP_PROG_CXX_WORKS_PART([$1], [])

GMP_PROG_CXX_WORKS_PART([$1], [namespace],
[namespace foo { }
using namespace foo;
])

# GMP requires the standard C++ iostream classes
GMP_PROG_CXX_WORKS_PART([$1], [std iostream],
[/* This test rejects g++ 2.7.2 which doesn't have <iostream>, only a
    pre-standard iostream.h. */
#include <iostream>

/* This test rejects OSF 5.1 Compaq C++ in its default pre-standard iostream
   mode, since that mode puts cout in the global namespace, not "std".  */
void someoutput (void) { std::cout << 123; }
])

AC_MSG_RESULT($gmp_prog_cxx_works)
case $gmp_prog_cxx_works in
  yes)
    [$2]
    ;;
  *)
    [$3]
    ;;
esac
])

dnl  Called: GMP_PROG_CXX_WORKS_PART(CXX+CXXFLAGS, FAIL-MESSAGE [,CODE])
dnl
AC_DEFUN([GMP_PROG_CXX_WORKS_PART],
[if test "$gmp_prog_cxx_works" = yes; then
  # remove anything that might look like compiler output to our "||" expression
  rm -f conftest* a.out b.out a.exe a_out.exe
  cat >conftest.cc <<EOF
[$3]
int main (void) { return 0; }
EOF
  echo "Test compile: [$2]" >&AC_FD_CC
  gmp_cxxcompile="$1 conftest.cc >&AC_FD_CC"
  if AC_TRY_EVAL(gmp_cxxcompile); then
    if test "$cross_compiling" = no; then
      if AC_TRY_COMMAND([./a.out || ./b.out || ./a.exe || ./a_out.exe || ./conftest]); then :;
      else
        gmp_prog_cxx_works="no[]m4_if([$2],,,[, ])[$2], program does not run"
      fi
    fi
  else
    gmp_prog_cxx_works="no[]m4_if([$2],,,[, ])[$2]"
  fi
  case $gmp_prog_cxx_works in
    no*)
      echo "failed program was:" >&AC_FD_CC
      cat conftest.cc >&AC_FD_CC
      ;;
  esac
  rm -f conftest* a.out b.out a.exe a_out.exe
fi
])


dnl  GMP_INIT([M4-DEF-FILE])
dnl  -----------------------
dnl  Initializations for GMP config.m4 generation.
dnl
dnl  FIXME: The generated config.m4 doesn't get recreated by config.status.
dnl  Maybe the relevant "echo"s should go through AC_CONFIG_COMMANDS.

AC_DEFUN([GMP_INIT],
[ifelse([$1], , gmp_configm4=config.m4, gmp_configm4="[$1]")
gmp_tmpconfigm4=cnfm4.tmp
gmp_tmpconfigm4i=cnfm4i.tmp
gmp_tmpconfigm4p=cnfm4p.tmp
rm -f $gmp_tmpconfigm4 $gmp_tmpconfigm4i $gmp_tmpconfigm4p

# CONFIG_TOP_SRCDIR is a path from the mpn builddir to the top srcdir.
# The pattern here tests for an absolute path the same way as
# _AC_OUTPUT_FILES in autoconf acgeneral.m4.
case $srcdir in
[[\\/]]* | ?:[[\\/]]* )  tmp="$srcdir"    ;;
*)                       tmp="../$srcdir" ;;
esac
echo ["define(<CONFIG_TOP_SRCDIR>,<\`$tmp'>)"] >>$gmp_tmpconfigm4

# All CPUs use asm-defs.m4
echo ["include][(CONFIG_TOP_SRCDIR\`/mpn/asm-defs.m4')"] >>$gmp_tmpconfigm4i
])


dnl  GMP_FINISH
dnl  ----------
dnl  Create config.m4 from its accumulated parts.
dnl
dnl  __CONFIG_M4_INCLUDED__ is used so that a second or subsequent include
dnl  of config.m4 is harmless.
dnl
dnl  A separate ifdef on the angle bracket quoted part ensures the quoting
dnl  style there is respected.  The basic defines from gmp_tmpconfigm4 are
dnl  fully quoted but are still put under an ifdef in case any have been
dnl  redefined by one of the m4 include files.
dnl
dnl  Doing a big ifdef within asm-defs.m4 and/or other macro files wouldn't
dnl  work, since it'd interpret parentheses and quotes in dnl comments, and
dnl  having a whole file as a macro argument would overflow the string space
dnl  on BSD m4.

AC_DEFUN([GMP_FINISH],
[AC_REQUIRE([GMP_INIT])
echo "creating $gmp_configm4"
echo ["d""nl $gmp_configm4.  Generated automatically by configure."] > $gmp_configm4
if test -f $gmp_tmpconfigm4; then
  echo ["changequote(<,>)"] >> $gmp_configm4
  echo ["ifdef(<__CONFIG_M4_INCLUDED__>,,<"] >> $gmp_configm4
  cat $gmp_tmpconfigm4 >> $gmp_configm4
  echo [">)"] >> $gmp_configm4
  echo ["changequote(\`,')"] >> $gmp_configm4
  rm $gmp_tmpconfigm4
fi
echo ["ifdef(\`__CONFIG_M4_INCLUDED__',,\`"] >> $gmp_configm4
if test -f $gmp_tmpconfigm4i; then
  cat $gmp_tmpconfigm4i >> $gmp_configm4
  rm $gmp_tmpconfigm4i
fi
if test -f $gmp_tmpconfigm4p; then
  cat $gmp_tmpconfigm4p >> $gmp_configm4
  rm $gmp_tmpconfigm4p
fi
echo ["')"] >> $gmp_configm4
echo ["define(\`__CONFIG_M4_INCLUDED__')"] >> $gmp_configm4
])


dnl  GMP_INCLUDE_MPN(FILE)
dnl  ---------------------
dnl  Add an include_mpn(`FILE') to config.m4.  FILE should be a path
dnl  relative to the mpn source directory, for example
dnl
dnl      GMP_INCLUDE_MPN(`x86/x86-defs.m4')
dnl

AC_DEFUN([GMP_INCLUDE_MPN],
[AC_REQUIRE([GMP_INIT])
echo ["include_mpn(\`$1')"] >> $gmp_tmpconfigm4i
])


dnl  GMP_DEFINE(MACRO, DEFINITION [, LOCATION])
dnl  ------------------------------------------
dnl  Define M4 macro MACRO as DEFINITION in temporary file.
dnl
dnl  If LOCATION is `POST', the definition will appear after any include()
dnl  directives inserted by GMP_INCLUDE.  Mind the quoting!  No shell
dnl  variables will get expanded.  Don't forget to invoke GMP_FINISH to
dnl  create file config.m4.  config.m4 uses `<' and '>' as quote characters
dnl  for all defines.

AC_DEFUN([GMP_DEFINE],
[AC_REQUIRE([GMP_INIT])
echo ['define(<$1>, <$2>)'] >>ifelse([$3], [POST],
                              $gmp_tmpconfigm4p, $gmp_tmpconfigm4)
])


dnl  GMP_DEFINE_RAW(STRING [, LOCATION])
dnl  ------------------------------------
dnl  Put STRING into config.m4 file.
dnl
dnl  If LOCATION is `POST', the definition will appear after any include()
dnl  directives inserted by GMP_INCLUDE.  Don't forget to invoke GMP_FINISH
dnl  to create file config.m4.

AC_DEFUN([GMP_DEFINE_RAW],
[AC_REQUIRE([GMP_INIT])
echo [$1] >> ifelse([$2], [POST], $gmp_tmpconfigm4p, $gmp_tmpconfigm4)
])


dnl  GMP_TRY_ASSEMBLE(asm-code,[action-success][,action-fail])
dnl  ----------------------------------------------------------
dnl  Attempt to assemble the given code.
dnl  Do "action-success" if this succeeds, "action-fail" if not.
dnl
dnl  conftest.o and conftest.out are available for inspection in
dnl  "action-success".  If either action does a "break" out of a loop then
dnl  an explicit "rm -f conftest*" will be necessary.
dnl
dnl  This is not unlike AC_TRY_COMPILE, but there's no default includes or
dnl  anything in "asm-code", everything wanted must be given explicitly.

AC_DEFUN([GMP_TRY_ASSEMBLE],
[cat >conftest.s <<EOF
[$1]
EOF
gmp_assemble="$CCAS $CFLAGS $CPPFLAGS conftest.s >conftest.out 2>&1"
if AC_TRY_EVAL(gmp_assemble); then
  cat conftest.out >&AC_FD_CC
  ifelse([$2],,:,[$2])
else
  cat conftest.out >&AC_FD_CC
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.s >&AC_FD_CC
  ifelse([$3],,:,[$3])
fi
rm -f conftest*
])


dnl Checks whether the stack can be marked nonexecutable by passing an option
dnl to the C-compiler when acting on .s files. Appends that option to ASMFLAGS.
dnl This macro is adapted from one found in GLIBC-2.3.5.
dnl FIXME: This test looks broken. It tests that a file with .note.GNU-stack...
dnl can be compiled/assembled with -Wa,--noexecstack.  It does not determine
dnl if that command-line option has any effect on general asm code.
AC_DEFUN([CL_AS_NOEXECSTACK],[
dnl AC_REQUIRE([AC_PROG_CC]) GMP uses something else
AC_CACHE_CHECK([whether assembler supports --noexecstack option],
cl_cv_as_noexecstack, [dnl
  cat > conftest.c <<EOF
void foo() {}
EOF
  if AC_TRY_COMMAND([${CC} $CFLAGS $CPPFLAGS
                     -S -o conftest.s conftest.c >/dev/null]) \
     && grep .note.GNU-stack conftest.s >/dev/null \
     && AC_TRY_COMMAND([${CC} $CFLAGS $CPPFLAGS -Wa,--noexecstack
                       -c -o conftest.o conftest.s >/dev/null])
  then
    cl_cv_as_noexecstack=yes
  else
    cl_cv_as_noexecstack=no
  fi
  rm -f conftest*])
  if test "$cl_cv_as_noexecstack" = yes; then
    ASMFLAGS="$ASMFLAGS -Wa,--noexecstack"
  fi
  AC_SUBST(ASMFLAGS)
])


dnl  GMP_ASM_LABEL_SUFFIX
dnl  --------------------
dnl  : - is usual.
dnl  empty - hppa on HP-UX doesn't use a :, just the label name
dnl
dnl  Note that it's necessary to test the empty case first, since HP "as"
dnl  will accept "somelabel:", and take it to mean a label with a name that
dnl  happens to end in a colon.

AC_DEFUN([GMP_ASM_LABEL_SUFFIX],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([for assembler label suffix],
                gmp_cv_asm_label_suffix,
[gmp_cv_asm_label_suffix=unknown
for i in "" ":"; do
  echo "trying $i" >&AC_FD_CC
  GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
somelabel$i],
    [gmp_cv_asm_label_suffix=$i
     rm -f conftest*
     break],
    [cat conftest.out >&AC_FD_CC])
done
if test "$gmp_cv_asm_label_suffix" = "unknown"; then
  AC_MSG_ERROR([Cannot determine label suffix])
fi
])
echo ["define(<LABEL_SUFFIX>, <$gmp_cv_asm_label_suffix>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_UNDERSCORE
dnl  ------------------
dnl  Determine whether global symbols need to be prefixed with an underscore.
dnl  The output from "nm" is grepped to see what a typical symbol looks like.
dnl
dnl  This test used to grep the .o file directly, but that failed with greps
dnl  that don't like binary files (eg. SunOS 4).
dnl
dnl  This test also used to construct an assembler file with and without an
dnl  underscore and try to link that to a C file, to see which worked.
dnl  Although that's what will happen in the real build we don't really want
dnl  to depend on creating asm files within configure for every possible CPU
dnl  (or at least we don't want to do that more than we have to).
dnl
dnl  The fallback on no underscore is based on the assumption that the world
dnl  is moving towards non-underscore systems.  There should actually be no
dnl  good reason for nm to fail though.

AC_DEFUN([GMP_ASM_UNDERSCORE],
[AC_REQUIRE([GMP_PROG_NM])
AC_CACHE_CHECK([if globals are prefixed by underscore],
               gmp_cv_asm_underscore,
[gmp_cv_asm_underscore="unknown"
cat >conftest.c <<EOF
int gurkmacka;
EOF
gmp_compile="$CC $CFLAGS $CPPFLAGS -c conftest.c >&AC_FD_CC"
if AC_TRY_EVAL(gmp_compile); then
  $NM conftest.$OBJEXT >conftest.out
  if grep _gurkmacka conftest.out >/dev/null; then
    gmp_cv_asm_underscore=yes
  elif grep gurkmacka conftest.out >/dev/null; then
    gmp_cv_asm_underscore=no
  else
    echo "configure: $NM doesn't have gurkmacka:" >&AC_FD_CC
    cat conftest.out >&AC_FD_CC
  fi
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.c >&AC_FD_CC
fi
rm -f conftest*
])
case $gmp_cv_asm_underscore in
  yes)
    GMP_DEFINE(GSYM_PREFIX, [_]) ;;
  no)
    GMP_DEFINE(GSYM_PREFIX, []) ;;
  *)
    AC_MSG_WARN([+----------------------------------------------------------])
    AC_MSG_WARN([| Cannot determine global symbol prefix.])
    AC_MSG_WARN([| $NM output doesn't contain a global data symbol.])
    AC_MSG_WARN([| Will proceed with no underscore.])
    AC_MSG_WARN([| If this is wrong then you'll get link errors referring])
    AC_MSG_WARN([| to ___gmpn_add_n (note three underscores).])
    AC_MSG_WARN([| In this case do a fresh build with an override,])
    AC_MSG_WARN([|     ./configure gmp_cv_asm_underscore=yes])
    AC_MSG_WARN([+----------------------------------------------------------])
    GMP_DEFINE(GSYM_PREFIX, [])
    ;;
esac
])


dnl  GMP_ASM_ALIGN_LOG
dnl  -----------------
dnl  Is parameter to `.align' logarithmic?

AC_DEFUN([GMP_ASM_ALIGN_LOG],
[AC_REQUIRE([GMP_ASM_GLOBL])
AC_REQUIRE([GMP_ASM_BYTE])
AC_REQUIRE([GMP_ASM_DATA])
AC_REQUIRE([GMP_ASM_LABEL_SUFFIX])
AC_REQUIRE([GMP_PROG_NM])
AC_CACHE_CHECK([if .align assembly directive is logarithmic],
               gmp_cv_asm_align_log,
[GMP_TRY_ASSEMBLE(
[      	$gmp_cv_asm_data
      	.align  4
	$gmp_cv_asm_globl	foo
	$gmp_cv_asm_byte	1
	.align	4
foo$gmp_cv_asm_label_suffix
	$gmp_cv_asm_byte	2],
  [gmp_tmp_val=[`$NM conftest.$OBJEXT | grep foo | \
     sed -e 's;[[][0-9][]]\(.*\);\1;' -e 's;[^1-9]*\([0-9]*\).*;\1;'`]
  if test "$gmp_tmp_val" = "10" || test "$gmp_tmp_val" = "16"; then
    gmp_cv_asm_align_log=yes
  else
    gmp_cv_asm_align_log=no
  fi],
  [AC_MSG_ERROR([cannot assemble alignment test])])])

GMP_DEFINE_RAW(["define(<ALIGN_LOGARITHMIC>,<$gmp_cv_asm_align_log>)"])
])


dnl  GMP_ASM_ALIGN_FILL_0x90
dnl  -----------------------
dnl  Determine whether a ",0x90" suffix works on a .align directive.
dnl  This is only meant for use on x86, 0x90 being a "nop".
dnl
dnl  Old gas, eg. 1.92.3
dnl       Needs ",0x90" or else the fill is 0x00, which can't be executed
dnl       across.
dnl
dnl  New gas, eg. 2.91
dnl       Generates multi-byte nop fills even when ",0x90" is given.
dnl
dnl  Solaris 2.6 as
dnl       ",0x90" is not allowed, causes a fatal error.
dnl
dnl  Solaris 2.8 as
dnl       ",0x90" does nothing, generates a warning that it's being ignored.
dnl
dnl  SCO OpenServer 5 as
dnl       Second parameter is max bytes to fill, not a fill pattern.
dnl       ",0x90" is an error due to being bigger than the first parameter.
dnl       Multi-byte nop fills are generated in text segments.
dnl
dnl  Note that both solaris "as"s only care about ",0x90" if they actually
dnl  have to use it to fill something, hence the .byte in the test.  It's
dnl  the second .align which provokes the error or warning.
dnl
dnl  The warning from solaris 2.8 is suppressed to stop anyone worrying that
dnl  something might be wrong.

AC_DEFUN([GMP_ASM_ALIGN_FILL_0x90],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([if the .align directive accepts an 0x90 fill in .text],
               gmp_cv_asm_align_fill_0x90,
[GMP_TRY_ASSEMBLE(
[      	$gmp_cv_asm_text
      	.align  4, 0x90
	.byte   0
      	.align  4, 0x90],
[if grep "Warning: Fill parameter ignored for executable section" conftest.out >/dev/null; then
  echo "Suppressing this warning by omitting 0x90" 1>&AC_FD_CC
  gmp_cv_asm_align_fill_0x90=no
else
  gmp_cv_asm_align_fill_0x90=yes
fi],
[gmp_cv_asm_align_fill_0x90=no])])

GMP_DEFINE_RAW(["define(<ALIGN_FILL_0x90>,<$gmp_cv_asm_align_fill_0x90>)"])
])


dnl  GMP_ASM_BYTE
dnl  ------------
dnl  .byte - is usual.
dnl  data1 - required by ia64 (on hpux at least).
dnl
dnl  This macro is just to support other configure tests, not any actual asm
dnl  code.

AC_DEFUN([GMP_ASM_BYTE],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_REQUIRE([GMP_ASM_LABEL_SUFFIX])
AC_CACHE_CHECK([for assembler byte directive],
                gmp_cv_asm_byte,
[for i in .byte data1; do
  echo "trying $i" >&AC_FD_CC
  GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_data
	$i	0
],
    [gmp_cv_asm_byte=$i
     rm -f conftest*
     break],
    [cat conftest.out >&AC_FD_CC])
done
if test -z "$gmp_cv_asm_byte"; then
  AC_MSG_ERROR([Cannot determine how to emit a data byte])
fi
])
])


dnl  GMP_ASM_TEXT
dnl  ------------
dnl  .text - is usual.
dnl  .code - is needed by the hppa on HP-UX (but ia64 HP-UX uses .text)
dnl  .csect .text[PR] - is for AIX.

AC_DEFUN([GMP_ASM_TEXT],
[AC_CACHE_CHECK([how to switch to text section],
                gmp_cv_asm_text,
[for i in ".text" ".code" [".csect .text[PR]"]; do
  echo "trying $i" >&AC_FD_CC
  GMP_TRY_ASSEMBLE([	$i],
    [gmp_cv_asm_text=$i
     rm -f conftest*
     break])
done
if test -z "$gmp_cv_asm_text"; then
  AC_MSG_ERROR([Cannot determine text section directive])
fi
])
echo ["define(<TEXT>, <$gmp_cv_asm_text>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_DATA
dnl  ------------
dnl  Can we say `.data'?

AC_DEFUN([GMP_ASM_DATA],
[AC_CACHE_CHECK([how to switch to data section],
                gmp_cv_asm_data,
[case $host in
  *-*-aix*) gmp_cv_asm_data=[".csect .data[RW]"] ;;
  *)        gmp_cv_asm_data=".data" ;;
esac
])
echo ["define(<DATA>, <$gmp_cv_asm_data>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_RODATA
dnl  --------------
dnl  Find out how to switch to the read-only data section.
dnl
dnl  The compiler output is grepped for the right directive.  It's not
dnl  considered wise to just probe for ".section .rodata" or whatever works,
dnl  since arbitrary section names might be accepted, but not necessarily do
dnl  the right thing when they get to the linker.
dnl
dnl  Only a few asm files use RODATA, so this code is perhaps a bit
dnl  excessive right now, but should find more uses in the future.
dnl
dnl  FIXME: gcc on aix generates something like ".csect _foo.ro_c[RO],3"
dnl  where foo is the object file.  Might need to check for that if we use
dnl  RODATA there.

AC_DEFUN([GMP_ASM_RODATA],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_REQUIRE([GMP_ASM_DATA])
AC_REQUIRE([GMP_ASM_LABEL_SUFFIX])
AC_REQUIRE([GMP_ASM_UNDERSCORE])
AC_CACHE_CHECK([how to switch to read-only data section],
               gmp_cv_asm_rodata,
[
dnl Default to DATA on CPUs with split code/data caching, and TEXT
dnl elsewhere.  i386 means generic x86, so use DATA on it.
case $host in
X86_PATTERN | x86_64-*-*)
  gmp_cv_asm_rodata="$gmp_cv_asm_data" ;;
*)
  gmp_cv_asm_rodata="$gmp_cv_asm_text" ;;
esac

cat >conftest.c <<EOF
extern const int foo[[]];		/* Suppresses C++'s suppression of foo */
const int foo[[]] = {1,2,3};
EOF
echo "Test program:" >&AC_FD_CC
cat conftest.c >&AC_FD_CC
gmp_compile="$CC $CFLAGS $CPPFLAGS -S conftest.c >&AC_FD_CC"
if AC_TRY_EVAL(gmp_compile); then
  echo "Compiler output:" >&AC_FD_CC
  cat conftest.s >&AC_FD_CC
  if test $gmp_cv_asm_underscore = yes; then
    tmp_gsym_prefix=_
  else
    tmp_gsym_prefix=
  fi
  # must see our label
  if grep "^${tmp_gsym_prefix}foo$gmp_cv_asm_label_suffix" conftest.s >/dev/null 2>&AC_FD_CC; then
    # take the last directive before our label (hence skipping segments
    # getting debugging info etc)
    tmp_match=`sed -n ["/^${tmp_gsym_prefix}foo$gmp_cv_asm_label_suffix/q
                        /^[. 	]*data/p
                        /^[. 	]*rdata/p
                        /^[. 	]*text/p
                        /^[. 	]*section/p
                        /^[. 	]*csect/p
                        /^[. 	]*CSECT/p"] conftest.s | sed -n '$p'`
    echo "Match: $tmp_match" >&AC_FD_CC
    if test -n "$tmp_match"; then
      gmp_cv_asm_rodata=$tmp_match
    fi
  else
    echo "Couldn't find label: ^${tmp_gsym_prefix}foo$gmp_cv_asm_label_suffix" >&AC_FD_CC
  fi
fi
rm -f conftest*
])
echo ["define(<RODATA>, <$gmp_cv_asm_rodata>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_GLOBL
dnl  -------------
dnl  The assembler directive to mark a label as a global symbol.
dnl
dnl  ia64 - .global is standard, according to the Intel documentation.
dnl
dnl  hppa - ".export foo,entry" is demanded by HP hppa "as".  ".global" is a
dnl      kind of import.
dnl
dnl  other - .globl is usual.
dnl
dnl  "gas" tends to accept .globl everywhere, in addition to .export or
dnl  .global or whatever the system assembler demands.

AC_DEFUN([GMP_ASM_GLOBL],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([for assembler global directive],
                gmp_cv_asm_globl,
[case $host in
  hppa*-*-*)     gmp_cv_asm_globl=.export ;;
  IA64_PATTERN)  gmp_cv_asm_globl=.global ;;
  *)             gmp_cv_asm_globl=.globl  ;;
esac
])
echo ["define(<GLOBL>, <$gmp_cv_asm_globl>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_GLOBL_ATTR
dnl  ------------------
dnl  Do we need something after `GLOBL symbol'?

AC_DEFUN([GMP_ASM_GLOBL_ATTR],
[AC_REQUIRE([GMP_ASM_GLOBL])
AC_CACHE_CHECK([for assembler global directive attribute],
                gmp_cv_asm_globl_attr,
[case $gmp_cv_asm_globl in
  .export) gmp_cv_asm_globl_attr=",entry" ;;
  *)       gmp_cv_asm_globl_attr="" ;;
esac
])
echo ["define(<GLOBL_ATTR>, <$gmp_cv_asm_globl_attr>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_TYPE
dnl  ------------
dnl  Can we say ".type", and how?
dnl
dnl  For i386 GNU/Linux ELF systems, and very likely other ELF systems,
dnl  .type and .size are important on functions in shared libraries.  If
dnl  .type is omitted and the mainline program references that function then
dnl  the code will be copied down to the mainline at load time like a piece
dnl  of data.  If .size is wrong or missing (it defaults to 4 bytes or some
dnl  such) then incorrect bytes will be copied and a segv is the most likely
dnl  result.  In any case such copying is not what's wanted, a .type
dnl  directive will ensure a PLT entry is used.
dnl
dnl  In GMP the assembler functions are normally only used from within the
dnl  library (since most programs are not interested in the low level
dnl  routines), and in those circumstances a missing .type isn't fatal,
dnl  letting the problem go unnoticed.  tests/mpn/t-asmtype.c aims to check
dnl  for it.

AC_DEFUN([GMP_ASM_TYPE],
[AC_CACHE_CHECK([for assembler .type directive],
                gmp_cv_asm_type,
[gmp_cv_asm_type=
for gmp_tmp_prefix in @ \# %; do
  GMP_TRY_ASSEMBLE([	.type	sym,${gmp_tmp_prefix}function],
    [if grep "\.type pseudo-op used outside of \.def/\.endef ignored" conftest.out >/dev/null; then : ;
    else
      gmp_cv_asm_type=".type	\$][1,${gmp_tmp_prefix}\$][2"
      break
    fi])
done
rm -f conftest*
])
echo ["define(<TYPE>, <$gmp_cv_asm_type>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_SIZE
dnl  ------------
dnl  Can we say `.size'?

AC_DEFUN([GMP_ASM_SIZE],
[AC_CACHE_CHECK([for assembler .size directive],
                gmp_cv_asm_size,
[gmp_cv_asm_size=
GMP_TRY_ASSEMBLE([	.size	sym,1],
  [if grep "\.size pseudo-op used outside of \.def/\.endef ignored" conftest.out >/dev/null; then : ;
  else
    gmp_cv_asm_size=".size	\$][1,\$][2"
  fi])
])
echo ["define(<SIZE>, <$gmp_cv_asm_size>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_COFF_TYPE
dnl  -----------------
dnl  Determine whether the assembler supports COFF type information.
dnl
dnl  Currently this is only needed for mingw (and cygwin perhaps) and so is
dnl  run only on the x86s, but it ought to work anywhere.
dnl
dnl  On MINGW, recent versions of the linker have an automatic import scheme
dnl  for data in a DLL which is referenced by a mainline but without
dnl  __declspec (__dllimport__) on the prototype.  It seems functions
dnl  without type information are treated as data, or something, and calls
dnl  to them from the mainline will crash.  gcc puts type information on the
dnl  C functions it generates, we need to do the same for assembler
dnl  functions.
dnl
dnl  This applies only to functions without __declspec(__dllimport__),
dnl  ie. without __GMP_DECLSPEC in the case of libgmp, so it also works just
dnl  to ensure all assembler functions used from outside libgmp have
dnl  __GMP_DECLSPEC on their prototypes.  But this isn't an ideal situation,
dnl  since we don't want perfectly valid calls going wrong just because
dnl  there wasn't a prototype in scope.
dnl
dnl  When an auto-import takes place, the following warning is given by the
dnl  linker.  This shouldn't be seen for any functions.
dnl
dnl      Info: resolving _foo by linking to __imp__foo (auto-import)
dnl
dnl
dnl  COFF type directives look like the following
dnl
dnl      .def    _foo
dnl      .scl    2
dnl      .type   32
dnl      .endef
dnl
dnl  _foo is the symbol with GSYM_PREFIX (_).  .scl is the storage class, 2
dnl  for external, 3 for static.  .type is the object type, 32 for a
dnl  function.
dnl
dnl  On an ELF system, this is (correctly) rejected due to .def, .endef and
dnl  .scl being invalid, and .type not having enough arguments.

AC_DEFUN([GMP_ASM_COFF_TYPE],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_REQUIRE([GMP_ASM_GLOBL])
AC_REQUIRE([GMP_ASM_GLOBL_ATTR])
AC_REQUIRE([GMP_ASM_LABEL_SUFFIX])
AC_REQUIRE([GMP_ASM_UNDERSCORE])
AC_CACHE_CHECK([for assembler COFF type directives],
		gmp_cv_asm_x86_coff_type,
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	$gmp_cv_asm_globl ${tmp_gsym_prefix}foo$gmp_cv_asm_globl_attr
	.def	${tmp_gsym_prefix}foo
	.scl	2
	.type	32
	.endef
${tmp_gsym_prefix}foo$gmp_cv_asm_label_suffix
],
  [gmp_cv_asm_x86_coff_type=yes],
  [gmp_cv_asm_x86_coff_type=no])
])
echo ["define(<HAVE_COFF_TYPE>, <$gmp_cv_asm_x86_coff_type>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_ASM_LSYM_PREFIX
dnl  -------------------
dnl  What is the prefix for a local label?
dnl
dnl  The prefixes tested are,
dnl
dnl      L  - usual for underscore systems
dnl      .L - usual for non-underscore systems
dnl      $  - alpha (gas and OSF system assembler)
dnl      L$ - hppa (gas and HP-UX system assembler)
dnl
dnl  The default is "L" if the tests fail for any reason.  There's a good
dnl  chance this will be adequate, since on most systems labels are local
dnl  anyway unless given a ".globl", and an "L" will avoid clashes with
dnl  other identifers.
dnl
dnl  For gas, ".L" is normally purely local to the assembler, it doesn't get
dnl  put into the object file at all.  This style is preferred, to keep the
dnl  object files nice and clean.
dnl
dnl  BSD format nm produces a line like
dnl
dnl      00000000 t Lgurkmacka
dnl
dnl  The symbol code is normally "t" for text, but any lower case letter
dnl  indicates a local definition.
dnl
dnl  Code "n" is for a debugging symbol, OSF "nm -B" gives that as an upper
dnl  case "N" for a local.
dnl
dnl  HP-UX nm prints an error message (though seems to give a 0 exit) if
dnl  there's no symbols at all in an object file, hence the use of "dummy".

AC_DEFUN([GMP_ASM_LSYM_PREFIX],
[AC_REQUIRE([GMP_ASM_LABEL_SUFFIX])
AC_REQUIRE([GMP_ASM_TEXT])
AC_REQUIRE([GMP_PROG_NM])
AC_CACHE_CHECK([for assembler local label prefix],
               gmp_cv_asm_lsym_prefix,
[gmp_tmp_pre_appears=yes
for gmp_tmp_pre in L .L $L $ L$; do
  echo "Trying $gmp_tmp_pre" >&AC_FD_CC
  GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
dummy${gmp_cv_asm_label_suffix}
${gmp_tmp_pre}gurkmacka${gmp_cv_asm_label_suffix}],
  [if $NM conftest.$OBJEXT >conftest.nm 2>&AC_FD_CC; then : ; else
    cat conftest.nm >&AC_FD_CC
    AC_MSG_WARN(["$NM" failure])
    break
  fi
  cat conftest.nm >&AC_FD_CC
  if grep gurkmacka conftest.nm >/dev/null; then : ; else
    # no mention of the symbol, this is good
    echo "$gmp_tmp_pre label doesn't appear in object file at all (good)" >&AC_FD_CC
    gmp_cv_asm_lsym_prefix="$gmp_tmp_pre"
    gmp_tmp_pre_appears=no
    break
  fi
  if grep [' [a-zN] .*gurkmacka'] conftest.nm >/dev/null; then
    # symbol mentioned as a local, use this if nothing better
    echo "$gmp_tmp_pre label is local but still in object file" >&AC_FD_CC
    if test -z "$gmp_cv_asm_lsym_prefix"; then
      gmp_cv_asm_lsym_prefix="$gmp_tmp_pre"
    fi
  else
    echo "$gmp_tmp_pre label is something unknown" >&AC_FD_CC
  fi
  ])
done
rm -f conftest*
if test -z "$gmp_cv_asm_lsym_prefix"; then
  gmp_cv_asm_lsym_prefix=L
  AC_MSG_WARN([cannot determine local label, using default $gmp_cv_asm_lsym_prefix])
fi
# for development purposes, note whether we got a purely temporary local label
echo "Local label appears in object files: $gmp_tmp_pre_appears" >&AC_FD_CC
])
echo ["define(<LSYM_PREFIX>, <${gmp_cv_asm_lsym_prefix}>)"] >> $gmp_tmpconfigm4
AC_DEFINE_UNQUOTED(LSYM_PREFIX, "$gmp_cv_asm_lsym_prefix",
                   [Assembler local label prefix])
])


dnl  GMP_ASM_W32
dnl  -----------
dnl  How to define a 32-bit word.
dnl
dnl  FIXME: This test is not right for ia64-*-hpux*.  The directive should
dnl  be "data4", but the W32 macro is not currently used by the mpn/ia64 asm
dnl  files.

AC_DEFUN([GMP_ASM_W32],
[AC_REQUIRE([GMP_ASM_DATA])
AC_REQUIRE([GMP_ASM_BYTE])
AC_REQUIRE([GMP_ASM_GLOBL])
AC_REQUIRE([GMP_ASM_LABEL_SUFFIX])
AC_REQUIRE([GMP_PROG_NM])
AC_CACHE_CHECK([how to define a 32-bit word],
	       gmp_cv_asm_w32,
[case $host in
  *-*-hpux*)
    # FIXME: HPUX puts first symbol at 0x40000000, breaking our assumption
    # that it's at 0x0.  We'll have to declare another symbol before the
    # .long/.word and look at the distance between the two symbols.  The
    # only problem is that the sed expression(s) barfs (on Solaris, for
    # example) for the symbol with value 0.  For now, HPUX uses .word.
    gmp_cv_asm_w32=".word"
    ;;
  *-*-*)
    gmp_tmp_val=
    for gmp_tmp_op in .long .word data4; do
      GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_data
	$gmp_cv_asm_globl	foo
	$gmp_tmp_op	0
foo$gmp_cv_asm_label_suffix
	$gmp_cv_asm_byte	0],
        [gmp_tmp_val=[`$NM conftest.$OBJEXT | grep foo | \
          sed -e 's;[[][0-9][]]\(.*\);\1;' -e 's;[^1-9]*\([0-9]*\).*;\1;'`]
        if test "$gmp_tmp_val" = 4; then
          gmp_cv_asm_w32="$gmp_tmp_op"
          break
        fi])
    done
    rm -f conftest*
    ;;
esac
if test -z "$gmp_cv_asm_w32"; then
  AC_MSG_ERROR([cannot determine how to define a 32-bit word])
fi
])
echo ["define(<W32>, <$gmp_cv_asm_w32>)"] >> $gmp_tmpconfigm4
])


dnl  GMP_X86_ASM_GOT_UNDERSCORE
dnl  --------------------------
dnl  Determine whether i386 _GLOBAL_OFFSET_TABLE_ needs an additional
dnl  underscore prefix.
dnl
dnl    SVR4      - the standard is _GLOBAL_OFFSET_TABLE_
dnl    GNU/Linux - follows SVR4
dnl    OpenBSD   - an a.out underscore system, uses __GLOBAL_OFFSET_TABLE_
dnl    NetBSD    - also an a.out underscore system, but _GLOBAL_OFFSET_TABLE_
dnl
dnl  The test attempts to link a program using _GLOBAL_OFFSET_TABLE_ or
dnl  __GLOBAL_OFFSET_TABLE_ to see which works.
dnl
dnl  $lt_prog_compiler_pic is included in the compile because old versions
dnl  of gas wouldn't accept PIC idioms without the right option (-K).  This
dnl  is the same as what libtool and mpn/Makeasm.am will do.
dnl
dnl  $lt_prog_compiler_pic is also included in the link because OpenBSD ld
dnl  won't accept an R_386_GOTPC relocation without the right options.  This
dnl  is not what's done by the Makefiles when building executables, but
dnl  let's hope it's ok (it works fine with gcc).
dnl
dnl  The fallback is no additional underscore, on the basis that this will
dnl  suit SVR4/ELF style systems, which should be much more common than
dnl  a.out systems with shared libraries.
dnl
dnl  Note that it's not an error for the tests to fail, since for instance
dnl  cygwin, mingw and djgpp don't have a _GLOBAL_OFFSET_TABLE_ scheme at
dnl  all.
dnl
dnl  Perhaps $CCAS could be asked to do the linking as well as the
dnl  assembling, but in the Makefiles it's only used for assembling, so lets
dnl  keep it that way.
dnl
dnl  The test here is run even under --disable-shared, so that PIC objects
dnl  can be built and tested by the tune/many.pl development scheme.  The
dnl  tests will be reasonably quick and won't give a fatal error, so this
dnl  arrangement is ok.  AC_LIBTOOL_PROG_COMPILER_PIC does its
dnl  $lt_prog_compiler_pic setups even for --disable-shared too.

AC_DEFUN([GMP_ASM_X86_GOT_UNDERSCORE],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_REQUIRE([GMP_ASM_GLOBL])
AC_REQUIRE([GMP_ASM_GLOBL_ATTR])
AC_REQUIRE([GMP_ASM_LABEL_SUFFIX])
AC_REQUIRE([GMP_ASM_UNDERSCORE])
AC_REQUIRE([AC_LIBTOOL_PROG_COMPILER_PIC])
AC_CACHE_CHECK([if _GLOBAL_OFFSET_TABLE_ is prefixed by underscore],
               gmp_cv_asm_x86_got_underscore,
[gmp_cv_asm_x86_got_underscore="not applicable"
if test $gmp_cv_asm_underscore = yes; then
  tmp_gsym_prefix=_
else
  tmp_gsym_prefix=
fi
for tmp_underscore in "" "_"; do
  cat >conftest.s <<EOF
	$gmp_cv_asm_text
	$gmp_cv_asm_globl ${tmp_gsym_prefix}main$gmp_cv_asm_globl_attr
${tmp_gsym_prefix}main$gmp_cv_asm_label_suffix
	addl	$ ${tmp_underscore}_GLOBAL_OFFSET_TABLE_, %ebx
EOF
  gmp_compile="$CCAS $CFLAGS $CPPFLAGS $lt_prog_compiler_pic conftest.s >&AC_FD_CC && $CC $CFLAGS $CPPFLAGS $lt_prog_compiler_pic conftest.$OBJEXT >&AC_FD_CC"
  if AC_TRY_EVAL(gmp_compile); then
    if test "$tmp_underscore" = "_"; then
      gmp_cv_asm_x86_got_underscore=yes
    else
      gmp_cv_asm_x86_got_underscore=no
    fi
    break
  fi
done
rm -f conftest* a.out b.out a.exe a_out.exe
])
if test "$gmp_cv_asm_x86_got_underscore" = "yes"; then
  GMP_DEFINE(GOT_GSYM_PREFIX, [_])
else
  GMP_DEFINE(GOT_GSYM_PREFIX, [])
fi
])


dnl  GMP_ASM_X86_GOT_EAX_OK(CC+CFLAGS, [ACTION-YES] [, ACTION-NO])
dnl  -------------------------------------------------------------
dnl  Determine whether _GLOBAL_OFFSET_TABLE_ used with %eax is ok.
dnl
dnl  An instruction
dnl
dnl          addl  $_GLOBAL_OFFSET_TABLE_, %eax
dnl
dnl  is incorrectly assembled by gas 2.12 (or thereabouts) and earlier.  It
dnl  puts an addend 2 into the R_386_GOTPC relocation, but it should be 1
dnl  for this %eax form being a 1 byte opcode (with other registers it's 2
dnl  opcode bytes).  See note about this in mpn/x86/README too.
dnl
dnl  We assemble this, surrounded by some unlikely byte sequences as
dnl  delimiters, and check for the bad output.
dnl
dnl  This is for use by compiler probing in GMP_PROG_CC_WORKS, so the result
dnl  is not cached.
dnl
dnl  This test is not specific to gas, but old gas is the only assembler we
dnl  know of with this problem.  The Solaris has been seen coming out ok.
dnl
dnl  ".text" is hard coded because this macro is wanted before GMP_ASM_TEXT.
dnl  This should be fine, ".text" is normal on x86 systems, and certainly
dnl  will be fine with the offending gas.
dnl
dnl  If an error occurs when assembling, we consider the assembler ok, since
dnl  the bad output does not occur.  This happens for instance on mingw,
dnl  where _GLOBAL_OFFSET_TABLE_ results in a bfd error, since there's no
dnl  GOT etc in PE object files.
dnl
dnl  This test is used before the object file extension has been determined,
dnl  so we force output to conftest.o.  Using -o with -c is not portable,
dnl  but we think all x86 compilers will accept -o with -c, certainly gcc
dnl  does.
dnl
dnl  -fPIC is hard coded here, because this test is for use before libtool
dnl  has established the pic options.  It's right for gcc, but perhaps not
dnl  other compilers.

AC_DEFUN([GMP_ASM_X86_GOT_EAX_OK],
[echo "Testing gas GOT with eax good" >&AC_FD_CC
cat >conftest.awk <<\EOF
[BEGIN {
  want[0]  = "001"
  want[1]  = "043"
  want[2]  = "105"
  want[3]  = "147"
  want[4]  = "211"
  want[5]  = "253"
  want[6]  = "315"
  want[7]  = "357"

  want[8]  = "005"
  want[9]  = "002"
  want[10] = "000"
  want[11] = "000"
  want[12] = "000"

  want[13] = "376"
  want[14] = "334"
  want[15] = "272"
  want[16] = "230"
  want[17] = "166"
  want[18] = "124"
  want[19] = "062"
  want[20] = "020"

  result = "yes"
}
{
  for (f = 2; f <= NF; f++)
    {
      for (i = 0; i < 20; i++)
        got[i] = got[i+1];
      got[20] = $f;

      found = 1
      for (i = 0; i < 21; i++)
        if (got[i] != want[i])
          {
            found = 0
            break
          }
      if (found)
        {
          result = "no"
          exit
        }
    }
}
END {
  print result
}
]EOF
cat >conftest.s <<\EOF
[	.text
	.byte	1, 35, 69, 103, 137, 171, 205, 239
	addl	$_GLOBAL_OFFSET_TABLE_, %eax
	.byte	254, 220, 186, 152, 118, 84, 50, 16
]EOF
tmp_got_good=yes
gmp_compile="$1 -fPIC -o conftest.o -c conftest.s >&AC_FD_CC 2>&1"
if AC_TRY_EVAL(gmp_compile); then
  tmp_got_good=`od -b conftest.o | $AWK -f conftest.awk`
fi
rm -f conftest.*
echo "Result: $tmp_got_good" >&AC_FD_CC
if test "$tmp_got_good" = no; then
  ifelse([$3],,:,[$3])
else
  ifelse([$2],,:,[$2])
fi
])


dnl  GMP_ASM_X86_MMX([ACTION-IF-YES][,ACTION-IF-NO])
dnl  -----------------------------------------------
dnl  Determine whether the assembler supports MMX instructions.
dnl
dnl  This macro is wanted before GMP_ASM_TEXT, so ".text" is hard coded
dnl  here.  ".text" is believed to be correct on all x86 systems.  Actually
dnl  ".text" probably isn't needed at all, at least for just checking
dnl  instruction syntax.
dnl
dnl  "movq %mm0, %mm1" should assemble to "0f 6f c8", but Solaris 2.6 and
dnl  2.7 wrongly assemble it to "0f 6f c1" (that being the reverse "movq
dnl  %mm1, %mm0").  It seems more trouble than it's worth to work around
dnl  this in the code, so just detect and reject.

AC_DEFUN([GMP_ASM_X86_MMX],
[AC_CACHE_CHECK([if the assembler knows about MMX instructions],
		gmp_cv_asm_x86_mmx,
[GMP_TRY_ASSEMBLE(
[	.text
	movq	%mm0, %mm1],
[gmp_cv_asm_x86_mmx=yes
case $host in
*-*-solaris*)
  if (dis conftest.$OBJEXT >conftest.out) 2>/dev/null; then
    if grep "0f 6f c1" conftest.out >/dev/null; then
      gmp_cv_asm_x86_mmx=movq-bug
    fi
  else
    AC_MSG_WARN(["dis" not available to check for "as" movq bug])
  fi
esac],
[gmp_cv_asm_x86_mmx=no])])

case $gmp_cv_asm_x86_mmx in
movq-bug)
  AC_MSG_WARN([+----------------------------------------------------------])
  AC_MSG_WARN([| WARNING WARNING WARNING])
  AC_MSG_WARN([| Host CPU has MMX code, but the assembler])
  AC_MSG_WARN([|     $CCAS $CFLAGS $CPPFLAGS])
  AC_MSG_WARN([| has the Solaris 2.6 and 2.7 bug where register to register])
  AC_MSG_WARN([| movq operands are reversed.])
  AC_MSG_WARN([| Non-MMX replacements will be used.])
  AC_MSG_WARN([| This will be an inferior build.])
  AC_MSG_WARN([+----------------------------------------------------------])
  ;;
no)
  AC_MSG_WARN([+----------------------------------------------------------])
  AC_MSG_WARN([| WARNING WARNING WARNING])
  AC_MSG_WARN([| Host CPU has MMX code, but it can't be assembled by])
  AC_MSG_WARN([|     $CCAS $CFLAGS $CPPFLAGS])
  AC_MSG_WARN([| Non-MMX replacements will be used.])
  AC_MSG_WARN([| This will be an inferior build.])
  AC_MSG_WARN([+----------------------------------------------------------])
  ;;
esac
if test "$gmp_cv_asm_x86_mmx" = yes; then
  ifelse([$1],,:,[$1])
else
  ifelse([$2],,:,[$2])
fi
])


dnl  GMP_ASM_X86_SHLDL_CL
dnl  --------------------

AC_DEFUN([GMP_ASM_X86_SHLDL_CL],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([if the assembler takes cl with shldl],
		gmp_cv_asm_x86_shldl_cl,
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	shldl	%cl, %eax, %ebx],
  gmp_cv_asm_x86_shldl_cl=yes,
  gmp_cv_asm_x86_shldl_cl=no)
])
if test "$gmp_cv_asm_x86_shldl_cl" = "yes"; then
  GMP_DEFINE(WANT_SHLDL_CL,1)
else
  GMP_DEFINE(WANT_SHLDL_CL,0)
fi
])


dnl  GMP_ASM_X86_SSE2([ACTION-IF-YES][,ACTION-IF-NO])
dnl  ------------------------------------------------
dnl  Determine whether the assembler supports SSE2 instructions.
dnl
dnl  This macro is wanted before GMP_ASM_TEXT, so ".text" is hard coded
dnl  here.  ".text" is believed to be correct on all x86 systems, certainly
dnl  it's all GMP_ASM_TEXT gives currently.  Actually ".text" probably isn't
dnl  needed at all, at least for just checking instruction syntax.

AC_DEFUN([GMP_ASM_X86_SSE2],
[AC_CACHE_CHECK([if the assembler knows about SSE2 instructions],
		gmp_cv_asm_x86_sse2,
[GMP_TRY_ASSEMBLE(
[	.text
	paddq	%mm0, %mm1],
  [gmp_cv_asm_x86_sse2=yes],
  [gmp_cv_asm_x86_sse2=no])
])
case $gmp_cv_asm_x86_sse2 in
yes)
  ifelse([$1],,:,[$1])
  ;;
*)
  AC_MSG_WARN([+----------------------------------------------------------])
  AC_MSG_WARN([| WARNING WARNING WARNING])
  AC_MSG_WARN([| Host CPU has SSE2 code, but it can't be assembled by])
  AC_MSG_WARN([|     $CCAS $CFLAGS $CPPFLAGS])
  AC_MSG_WARN([| Non-SSE2 replacements will be used.])
  AC_MSG_WARN([| This will be an inferior build.])
  AC_MSG_WARN([+----------------------------------------------------------])
  ifelse([$2],,:,[$2])
  ;;
esac
])


dnl  GMP_ASM_X86_MULX([ACTION-IF-YES][,ACTION-IF-NO])
dnl  ------------------------------------------------
dnl  Determine whether the assembler supports the mulx instruction which debut
dnl  with Haswell.
dnl
dnl  This macro is wanted before GMP_ASM_TEXT, so ".text" is hard coded
dnl  here.  ".text" is believed to be correct on all x86 systems, certainly
dnl  it's all GMP_ASM_TEXT gives currently.  Actually ".text" probably isn't
dnl  needed at all, at least for just checking instruction syntax.

AC_DEFUN([GMP_ASM_X86_MULX],
[AC_CACHE_CHECK([if the assembler knows about the mulx instruction],
		gmp_cv_asm_x86_mulx,
[GMP_TRY_ASSEMBLE(
[	.text
	mulx	%r8, %r9, %r10],
  [gmp_cv_asm_x86_mulx=yes],
  [gmp_cv_asm_x86_mulx=no])
])
case $gmp_cv_asm_x86_mulx in
yes)
  ifelse([$1],,:,[$1])
  ;;
*)
  AC_MSG_WARN([+----------------------------------------------------------])
  AC_MSG_WARN([| WARNING WARNING WARNING])
  AC_MSG_WARN([| Host CPU has the mulx instruction, but it can't be])
  AC_MSG_WARN([| assembled by])
  AC_MSG_WARN([|     $CCAS $CFLAGS $CPPFLAGS])
  AC_MSG_WARN([| Older x86 instructions will be used.])
  AC_MSG_WARN([| This will be an inferior build.])
  AC_MSG_WARN([+----------------------------------------------------------])
  ifelse([$2],,:,[$2])
  ;;
esac
])


dnl  GMP_ASM_X86_ADX([ACTION-IF-YES][,ACTION-IF-NO])
dnl  ------------------------------------------------
dnl  Determine whether the assembler supports the adcx and adox instructions
dnl  which debut with the Haswell shrink Broadwell.
dnl
dnl  This macro is wanted before GMP_ASM_TEXT, so ".text" is hard coded
dnl  here.  ".text" is believed to be correct on all x86 systems, certainly
dnl  it's all GMP_ASM_TEXT gives currently.  Actually ".text" probably isn't
dnl  needed at all, at least for just checking instruction syntax.

AC_DEFUN([GMP_ASM_X86_ADX],
[AC_CACHE_CHECK([if the assembler knows about the adox instruction],
		gmp_cv_asm_x86_adx,
[GMP_TRY_ASSEMBLE(
[	.text
	adox	%r8, %r9
	adcx	%r8, %r9],
  [gmp_cv_asm_x86_adx=yes],
  [gmp_cv_asm_x86_adx=no])
])
case $gmp_cv_asm_x86_adx in
yes)
  ifelse([$1],,:,[$1])
  ;;
*)
  AC_MSG_WARN([+----------------------------------------------------------])
  AC_MSG_WARN([| WARNING WARNING WARNING])
  AC_MSG_WARN([| Host CPU has the adcx and adox instructions, but they])
  AC_MSG_WARN([| can't be assembled by])
  AC_MSG_WARN([|     $CCAS $CFLAGS $CPPFLAGS])
  AC_MSG_WARN([| Older x86 instructions will be used.])
  AC_MSG_WARN([| This will be an inferior build.])
  AC_MSG_WARN([+----------------------------------------------------------])
  ifelse([$2],,:,[$2])
  ;;
esac
])


dnl  GMP_ASM_X86_MCOUNT
dnl  ------------------
dnl  Find out how to call mcount for profiling on an x86 system.
dnl
dnl  A dummy function is compiled and the ".s" output examined.  The pattern
dnl  matching might be a bit fragile, but should work at least with gcc on
dnl  sensible systems.  Certainly it's better than hard coding a table of
dnl  conventions.
dnl
dnl  For non-PIC, any ".data" is taken to mean a counter might be passed.
dnl  It's assumed a movl will set it up, and the right register is taken
dnl  from that movl.  Any movl involving %esp is ignored (a frame pointer
dnl  setup normally).
dnl
dnl  For PIC, any ".data" is similarly interpreted, but a GOTOFF identifies
dnl  the line setting up the right register.
dnl
dnl  In both cases a line with "mcount" identifies the call and that line is
dnl  used literally.
dnl
dnl  On some systems (eg. FreeBSD 3.5) gcc emits ".data" but doesn't use it,
dnl  so it's not an error to have .data but then not find a register.
dnl
dnl  Variations in mcount conventions on different x86 systems can be found
dnl  in gcc config/i386.  mcount can have a "_" prefix or be .mcount or
dnl  _mcount_ptr, and for PIC it can be called through a GOT entry, or via
dnl  the PLT.  If a pointer to a counter is required it's passed in %eax or
dnl  %edx.
dnl
dnl  Flags to specify PIC are taken from $lt_prog_compiler_pic set by
dnl  AC_PROG_LIBTOOL.
dnl
dnl  Enhancement: Cache the values determined here. But what's the right way
dnl  to get two variables (mcount_nonpic_reg and mcount_nonpic_call say) set
dnl  from one block of commands?

AC_DEFUN([GMP_ASM_X86_MCOUNT],
[AC_REQUIRE([AC_ENABLE_SHARED])
AC_REQUIRE([AC_PROG_LIBTOOL])
AC_MSG_CHECKING([how to call x86 mcount])
cat >conftest.c <<EOF
foo(){bar();}
EOF

if test "$enable_static" = yes; then
  gmp_asmout_compile="$CC $CFLAGS $CPPFLAGS -S conftest.c 1>&AC_FD_CC"
  if AC_TRY_EVAL(gmp_asmout_compile); then
    if grep '\.data' conftest.s >/dev/null; then
      mcount_nonpic_reg=`sed -n ['/esp/!s/.*movl.*,\(%[a-z]*\).*$/\1/p'] conftest.s`
    else
      mcount_nonpic_reg=
    fi
    mcount_nonpic_call=`grep 'call.*mcount' conftest.s`
    if test -z "$mcount_nonpic_call"; then
      AC_MSG_ERROR([Cannot find mcount call for non-PIC])
    fi
  else
    AC_MSG_ERROR([Cannot compile test program for non-PIC])
  fi
fi

if test "$enable_shared" = yes; then
  gmp_asmout_compile="$CC $CFLAGS $CPPFLAGS $lt_prog_compiler_pic -S conftest.c 1>&AC_FD_CC"
  if AC_TRY_EVAL(gmp_asmout_compile); then
    if grep '\.data' conftest.s >/dev/null; then
      case $lt_prog_compiler_pic in
        *-DDLL_EXPORT*)
          # Windows DLLs have non-PIC style mcount
          mcount_pic_reg=`sed -n ['/esp/!s/.*movl.*,\(%[a-z]*\).*$/\1/p'] conftest.s`
          ;;
        *)
          mcount_pic_reg=`sed -n ['s/.*GOTOFF.*,\(%[a-z]*\).*$/\1/p'] conftest.s`
          ;;
      esac
    else
      mcount_pic_reg=
    fi
    mcount_pic_call=`grep 'call.*mcount' conftest.s`
    if test -z "$mcount_pic_call"; then
      AC_MSG_ERROR([Cannot find mcount call for PIC])
    fi
  else
    AC_MSG_ERROR([Cannot compile test program for PIC])
  fi
fi

GMP_DEFINE_RAW(["define(<MCOUNT_NONPIC_REG>, <\`$mcount_nonpic_reg'>)"])
GMP_DEFINE_RAW(["define(<MCOUNT_NONPIC_CALL>,<\`$mcount_nonpic_call'>)"])
GMP_DEFINE_RAW(["define(<MCOUNT_PIC_REG>,    <\`$mcount_pic_reg'>)"])
GMP_DEFINE_RAW(["define(<MCOUNT_PIC_CALL>,   <\`$mcount_pic_call'>)"])

rm -f conftest.*
AC_MSG_RESULT([determined])
])


dnl  GMP_ASM_IA64_ALIGN_OK
dnl  ---------------------
dnl  Determine whether .align correctly pads with nop instructions in a text
dnl  segment.
dnl
dnl  gas 2.14 and earlier byte swaps its padding bundle on big endian
dnl  systems, which is incorrect (endianness only changes data).  What
dnl  should be "nop.m / nop.f / nop.i" comes out as "break" instructions.
dnl
dnl  The test here detects the bad case, and assumes anything else is ok
dnl  (there are many sensible nop bundles, so it'd be impractical to try to
dnl  match everything good).

AC_DEFUN([GMP_ASM_IA64_ALIGN_OK],
[AC_CACHE_CHECK([whether assembler .align padding is good],
		gmp_cv_asm_ia64_align_ok,
[cat >conftest.awk <<\EOF
[BEGIN {
  want[0]  = "011"
  want[1]  = "160"
  want[2]  = "074"
  want[3]  = "040"
  want[4]  = "000"
  want[5]  = "040"
  want[6]  = "020"
  want[7]  = "221"
  want[8]  = "114"
  want[9]  = "000"
  want[10] = "100"
  want[11] = "200"
  want[12] = "122"
  want[13] = "261"
  want[14] = "000"
  want[15] = "200"

  want[16] = "000"
  want[17] = "004"
  want[18] = "000"
  want[19] = "000"
  want[20] = "000"
  want[21] = "000"
  want[22] = "002"
  want[23] = "000"
  want[24] = "000"
  want[25] = "000"
  want[26] = "000"
  want[27] = "001"
  want[28] = "000"
  want[29] = "000"
  want[30] = "000"
  want[31] = "014"

  want[32] = "011"
  want[33] = "270"
  want[34] = "140"
  want[35] = "062"
  want[36] = "000"
  want[37] = "040"
  want[38] = "240"
  want[39] = "331"
  want[40] = "160"
  want[41] = "000"
  want[42] = "100"
  want[43] = "240"
  want[44] = "343"
  want[45] = "371"
  want[46] = "000"
  want[47] = "200"

  result = "yes"
}
{
  for (f = 2; f <= NF; f++)
    {
      for (i = 0; i < 47; i++)
        got[i] = got[i+1];
      got[47] = $f;

      found = 1
      for (i = 0; i < 48; i++)
        if (got[i] != want[i])
          {
            found = 0
            break
          }
      if (found)
        {
          result = "no"
          exit
        }
    }
}
END {
  print result
}
]EOF
GMP_TRY_ASSEMBLE(
[	.text
	.align	32
{ .mmi;	add	r14 = r15, r16
	add	r17 = r18, r19
	add	r20 = r21, r22 ;; }
	.align	32
{ .mmi;	add	r23 = r24, r25
	add	r26 = r27, r28
	add	r29 = r30, r31 ;; }
],
  [gmp_cv_asm_ia64_align_ok=`od -b conftest.$OBJEXT | $AWK -f conftest.awk`],
  [AC_MSG_WARN([oops, cannot compile test program])
   gmp_cv_asm_ia64_align_ok=yes])
])
GMP_DEFINE_RAW(["define(<IA64_ALIGN_OK>, <\`$gmp_cv_asm_ia64_align_ok'>)"])
])




dnl  GMP_ASM_M68K_INSTRUCTION
dnl  ------------------------
dnl  Not sure if ".l" and "%" are independent settings, but it doesn't hurt
dnl  to try all four possibilities.  Note that the % ones must be first, so
dnl  "d0" won't be interpreted as a label.
dnl
dnl  gas 1.92.3 on NetBSD 1.4 needs to be tested with a two operand
dnl  instruction.  It takes registers without "%", but a single operand
dnl  "clrl %d0" only gives a warning, not an error.

AC_DEFUN([GMP_ASM_M68K_INSTRUCTION],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([assembler instruction and register style],
		gmp_cv_asm_m68k_instruction,
[for i in "addl %d0,%d1" "add.l %d0,%d1" "addl d0,d1" "add.l d0,d1"; do
  GMP_TRY_ASSEMBLE(
    [	$gmp_cv_asm_text
	$i],
    [gmp_cv_asm_m68k_instruction=$i
    rm -f conftest*
    break])
done
if test -z "$gmp_cv_asm_m68k_instruction"; then
  AC_MSG_ERROR([cannot determine assembler instruction and register style])
fi
])
case $gmp_cv_asm_m68k_instruction in
"addl d0,d1")    want_dot_size=no;  want_register_percent=no  ;;
"addl %d0,%d1")  want_dot_size=no;  want_register_percent=yes ;;
"add.l d0,d1")   want_dot_size=yes; want_register_percent=no  ;;
"add.l %d0,%d1") want_dot_size=yes; want_register_percent=yes ;;
*) AC_MSG_ERROR([oops, unrecognised instruction and register style]) ;;
esac
GMP_DEFINE_RAW(["define(<WANT_REGISTER_PERCENT>, <\`$want_register_percent'>)"])
GMP_DEFINE_RAW(["define(<WANT_DOT_SIZE>, <\`$want_dot_size'>)"])
])


dnl  GMP_ASM_M68K_ADDRESSING
dnl  -----------------------

AC_DEFUN([GMP_ASM_M68K_ADDRESSING],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_REQUIRE([GMP_ASM_M68K_INSTRUCTION])
AC_CACHE_CHECK([assembler addressing style],
		gmp_cv_asm_m68k_addressing,
[case $gmp_cv_asm_m68k_instruction in
addl*)  movel=movel ;;
add.l*) movel=move.l ;;
*) AC_MSG_ERROR([oops, unrecognised gmp_cv_asm_m68k_instruction]) ;;
esac
case $gmp_cv_asm_m68k_instruction in
*"%d0,%d1") dreg=%d0; areg=%a0 ;;
*"d0,d1")   dreg=d0;  areg=a0  ;;
*) AC_MSG_ERROR([oops, unrecognised gmp_cv_asm_m68k_instruction]) ;;
esac
GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	$movel	$dreg, $areg@-],
  [gmp_cv_asm_m68k_addressing=mit],
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	$movel	$dreg, -($areg)],
  [gmp_cv_asm_m68k_addressing=motorola],
[AC_MSG_ERROR([cannot determine assembler addressing style])])])
])
GMP_DEFINE_RAW(["define(<WANT_ADDRESSING>, <\`$gmp_cv_asm_m68k_addressing'>)"])
])


dnl  GMP_ASM_M68K_BRANCHES
dnl  ---------------------
dnl  "bra" is the standard branch instruction.  "jra" or "jbra" are
dnl  preferred where available, since on gas for instance they give a
dnl  displacement only as big as it needs to be, whereas "bra" is always
dnl  16-bits.  This applies to the conditional branches "bcc" etc too.
dnl  However "dbcc" etc on gas are already only as big as they need to be.

AC_DEFUN([GMP_ASM_M68K_BRANCHES],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([assembler shortest branches],
		gmp_cv_asm_m68k_branches,
[for i in jra jbra bra; do
  GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
foo$gmp_cv_asm_label_suffix
	$i	foo],
  [gmp_cv_asm_m68k_branches=$i
  rm -f conftest*
  break])
done
if test -z "$gmp_cv_asm_m68k_branches"; then
  AC_MSG_ERROR([cannot determine assembler branching style])
fi
])
GMP_DEFINE_RAW(["define(<WANT_BRANCHES>, <\`$gmp_cv_asm_m68k_branches'>)"])
])


dnl  GMP_ASM_POWERPC_PIC_ALWAYS
dnl  --------------------------
dnl  Determine whether PIC is the default compiler output.
dnl
dnl  SVR4 style "foo@ha" addressing is interpreted as non-PIC, and anything
dnl  else is assumed to require PIC always (Darwin or AIX).  SVR4 is the
dnl  only non-PIC addressing syntax the asm files have at the moment anyway.
dnl
dnl  Libtool does this by taking "*-*-aix* | *-*-darwin* | *-*-rhapsody*" to
dnl  mean PIC always, but it seems more reliable to grep the compiler
dnl  output.
dnl
dnl The next paragraph is untrue for Tiger.  Was it ever true?  For tiger,
dnl "cc -fast" makes non-PIC the default (and the binaries do run).
dnl  On Darwin "cc -static" is non-PIC with syntax "ha16(_foo)", but that's
dnl  apparently only for use in the kernel, which we're not attempting to
dnl  target at the moment, so don't look for that.

AC_DEFUN([GMP_ASM_POWERPC_PIC_ALWAYS],
[AC_REQUIRE([AC_PROG_CC])
AC_CACHE_CHECK([whether compiler output is PIC by default],
               gmp_cv_asm_powerpc_pic,
[gmp_cv_asm_powerpc_pic=yes
cat >conftest.c <<EOF
int foo;
int *bar() { return &foo; }
EOF
echo "Test program:" >&AC_FD_CC
cat conftest.c >&AC_FD_CC
gmp_compile="$CC $CFLAGS $CPPFLAGS -S conftest.c >&AC_FD_CC"
if AC_TRY_EVAL(gmp_compile); then
  echo "Compiler output:" >&AC_FD_CC
  cat conftest.s >&AC_FD_CC
  if grep 'foo@ha' conftest.s >/dev/null 2>&AC_FD_CC; then
    gmp_cv_asm_powerpc_pic=no
  fi
  if grep 'ha16(_foo)' conftest.s >/dev/null 2>&AC_FD_CC; then
    gmp_cv_asm_powerpc_pic=no
  fi
fi
rm -f conftest*
])
GMP_DEFINE_RAW(["define(<PIC_ALWAYS>,<$gmp_cv_asm_powerpc_pic>)"])
])


dnl  GMP_ASM_POWERPC_R_REGISTERS
dnl  ---------------------------
dnl  Determine whether the assembler takes powerpc registers with an "r" as
dnl  in "r6", or as plain "6".  The latter is standard, but NeXT, Rhapsody,
dnl  and MacOS-X require the "r" forms.
dnl
dnl  See also mpn/powerpc32/powerpc-defs.m4 which uses the result of this
dnl  test.

AC_DEFUN([GMP_ASM_POWERPC_R_REGISTERS],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([if the assembler needs r on registers],
               gmp_cv_asm_powerpc_r_registers,
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	mtctr	6],
[gmp_cv_asm_powerpc_r_registers=no],
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	mtctr	r6],
[gmp_cv_asm_powerpc_r_registers=yes],
[AC_MSG_ERROR([neither "mtctr 6" nor "mtctr r6" works])])])])

GMP_DEFINE_RAW(["define(<WANT_R_REGISTERS>,<$gmp_cv_asm_powerpc_r_registers>)"])
])


dnl  GMP_ASM_SPARC_REGISTER
dnl  ----------------------
dnl  Determine whether the assembler accepts the ".register" directive.
dnl  Old versions of solaris "as" don't.
dnl
dnl  See also mpn/sparc32/sparc-defs.m4 which uses the result of this test.

AC_DEFUN([GMP_ASM_SPARC_REGISTER],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([if the assembler accepts ".register"],
               gmp_cv_asm_sparc_register,
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	.register	%g2,#scratch
],
[gmp_cv_asm_sparc_register=yes],
[gmp_cv_asm_sparc_register=no])])

GMP_DEFINE_RAW(["define(<HAVE_REGISTER>,<$gmp_cv_asm_sparc_register>)"])
])


dnl  GMP_ASM_SPARC_GOTDATA
dnl  ----------------------
dnl  Determine whether the assembler accepts gotdata relocations.
dnl
dnl  See also mpn/sparc32/sparc-defs.m4 which uses the result of this test.

AC_DEFUN([GMP_ASM_SPARC_GOTDATA],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([if the assembler accepts gotdata relocations],
               gmp_cv_asm_sparc_gotdata,
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	.text
	sethi	%gdop_hix22(symbol), %g1
	or	%g1, %gdop_lox10(symbol), %g1
],
[gmp_cv_asm_sparc_gotdata=yes],
[gmp_cv_asm_sparc_gotdata=no])])

GMP_DEFINE_RAW(["define(<HAVE_GOTDATA>,<$gmp_cv_asm_sparc_gotdata>)"])
])


dnl  GMP_ASM_SPARC_SHARED_THUNKS
dnl  ----------------------
dnl  Determine whether the assembler supports all of the features
dnl  necessary in order to emit shared PIC thunks on sparc.
dnl
dnl  See also mpn/sparc32/sparc-defs.m4 which uses the result of this test.

AC_DEFUN([GMP_ASM_SPARC_SHARED_THUNKS],
[AC_REQUIRE([GMP_ASM_TEXT])
AC_CACHE_CHECK([if the assembler can support shared PIC thunks],
               gmp_cv_asm_sparc_shared_thunks,
[GMP_TRY_ASSEMBLE(
[	$gmp_cv_asm_text
	.section	.text.__sparc_get_pc_thunk.l7,"axG",@progbits,__sparc_get_pc_thunk.l7,comdat
	.weak	__sparc_get_pc_thunk.l7
	.hidden	__sparc_get_pc_thunk.l7
	.type	__sparc_get_pc_thunk.l7, #function
__sparc_get_pc_thunk.l7:
	jmp	%o7+8
	 add	%o7, %l7, %l7
],
[gmp_cv_asm_sparc_shared_thunks=yes],
[gmp_cv_asm_sparc_shared_thunks=no])])

GMP_DEFINE_RAW(["define(<HAVE_SHARED_THUNKS>,<$gmp_cv_asm_sparc_shared_thunks>)"])
])


dnl  GMP_C_ATTRIBUTE_CONST
dnl  ---------------------

AC_DEFUN([GMP_C_ATTRIBUTE_CONST],
[AC_CACHE_CHECK([whether gcc __attribute__ ((const)) works],
                gmp_cv_c_attribute_const,
[AC_TRY_COMPILE([int foo (int x) __attribute__ ((const));], ,
  gmp_cv_c_attribute_const=yes, gmp_cv_c_attribute_const=no)
])
if test $gmp_cv_c_attribute_const = yes; then
  AC_DEFINE(HAVE_ATTRIBUTE_CONST, 1,
  [Define to 1 if the compiler accepts gcc style __attribute__ ((const))])
fi
])


dnl  GMP_C_ATTRIBUTE_MALLOC
dnl  ----------------------
dnl  gcc 2.95.x accepts __attribute__ ((malloc)) but with a warning that
dnl  it's ignored.  Pretend it doesn't exist in this case, to avoid that
dnl  warning.

AC_DEFUN([GMP_C_ATTRIBUTE_MALLOC],
[AC_CACHE_CHECK([whether gcc __attribute__ ((malloc)) works],
                gmp_cv_c_attribute_malloc,
[cat >conftest.c <<EOF
void *foo (int x) __attribute__ ((malloc));
EOF
gmp_compile="$CC $CFLAGS $CPPFLAGS -c conftest.c >conftest.out 2>&1"
if AC_TRY_EVAL(gmp_compile); then
  if grep "attribute directive ignored" conftest.out >/dev/null; then
    gmp_cv_c_attribute_malloc=no
  else
    gmp_cv_c_attribute_malloc=yes
  fi
else
  gmp_cv_c_attribute_malloc=no
fi
cat conftest.out >&AC_FD_CC
rm -f conftest*
])
if test $gmp_cv_c_attribute_malloc = yes; then
  AC_DEFINE(HAVE_ATTRIBUTE_MALLOC, 1,
  [Define to 1 if the compiler accepts gcc style __attribute__ ((malloc))])
fi
])


dnl  GMP_C_ATTRIBUTE_MODE
dnl  --------------------
dnl  Introduced in gcc 2.2, but perhaps not in all Apple derived versions.

AC_DEFUN([GMP_C_ATTRIBUTE_MODE],
[AC_CACHE_CHECK([whether gcc __attribute__ ((mode (XX))) works],
                gmp_cv_c_attribute_mode,
[AC_TRY_COMPILE([typedef int SItype __attribute__ ((mode (SI)));], ,
  gmp_cv_c_attribute_mode=yes, gmp_cv_c_attribute_mode=no)
])
if test $gmp_cv_c_attribute_mode = yes; then
  AC_DEFINE(HAVE_ATTRIBUTE_MODE, 1,
  [Define to 1 if the compiler accepts gcc style __attribute__ ((mode (XX)))])
fi
])


dnl  GMP_C_ATTRIBUTE_NORETURN
dnl  ------------------------

AC_DEFUN([GMP_C_ATTRIBUTE_NORETURN],
[AC_CACHE_CHECK([whether gcc __attribute__ ((noreturn)) works],
                gmp_cv_c_attribute_noreturn,
[AC_TRY_COMPILE([void foo (int x) __attribute__ ((noreturn));], ,
  gmp_cv_c_attribute_noreturn=yes, gmp_cv_c_attribute_noreturn=no)
])
if test $gmp_cv_c_attribute_noreturn = yes; then
  AC_DEFINE(HAVE_ATTRIBUTE_NORETURN, 1,
  [Define to 1 if the compiler accepts gcc style __attribute__ ((noreturn))])
fi
])

dnl  GMP_C_HIDDEN_ALIAS
dnl  ------------------------

AC_DEFUN([GMP_C_HIDDEN_ALIAS],
[AC_CACHE_CHECK([whether gcc hidden aliases work],
                gmp_cv_c_hidden_alias,
[AC_TRY_COMPILE(
[void hid() __attribute__ ((visibility("hidden")));
void hid() {}
void pub() __attribute__ ((alias("hid")));],
, gmp_cv_c_hidden_alias=yes, gmp_cv_c_hidden_alias=no)
])
if test $gmp_cv_c_hidden_alias = yes; then
  AC_DEFINE(HAVE_HIDDEN_ALIAS, 1,
  [Define to 1 if the compiler accepts gcc style __attribute__ ((visibility))
and __attribute__ ((alias))])
fi
])

dnl  GMP_C_DOUBLE_FORMAT
dnl  -------------------
dnl  Determine the floating point format.
dnl
dnl  The object file is grepped, in order to work when cross compiling.  A
dnl  start and end sequence is included to avoid false matches, and allowance
dnl  is made for the desired data crossing an "od -b" line boundary.  The test
dnl  number is a small integer so it should appear exactly, no rounding or
dnl  truncation etc.
dnl
dnl  "od -b", incidentally, is supported even by Unix V7, and the awk script
dnl  used doesn't have functions or anything, so even an "old" awk should
dnl  suffice.
dnl
dnl  The C code here declares the variable foo as extern; without that, some
dnl  C++ compilers will not put foo in the object file.

AC_DEFUN([GMP_C_DOUBLE_FORMAT],
[AC_REQUIRE([AC_PROG_CC])
AC_REQUIRE([AC_PROG_AWK])
AC_CACHE_CHECK([format of `double' floating point],
                gmp_cv_c_double_format,
[gmp_cv_c_double_format=unknown
cat >conftest.c <<\EOF
[struct foo {
  char    before[8];
  double  x;
  char    after[8];
};
extern struct foo foo;
struct foo foo = {
  { '\001', '\043', '\105', '\147', '\211', '\253', '\315', '\357' },
  -123456789.0,
  { '\376', '\334', '\272', '\230', '\166', '\124', '\062', '\020' },
};]
EOF
gmp_compile="$CC $CFLAGS $CPPFLAGS -c conftest.c >&AC_FD_CC 2>&1"
if AC_TRY_EVAL(gmp_compile); then
cat >conftest.awk <<\EOF
[
BEGIN {
  found = 0
}

{
  for (f = 2; f <= NF; f++)
    {
      for (i = 0; i < 23; i++)
        got[i] = got[i+1];
      got[23] = $f;

      # match the special begin and end sequences
      if (got[0] != "001") continue
      if (got[1] != "043") continue
      if (got[2] != "105") continue
      if (got[3] != "147") continue
      if (got[4] != "211") continue
      if (got[5] != "253") continue
      if (got[6] != "315") continue
      if (got[7] != "357") continue
      if (got[16] != "376") continue
      if (got[17] != "334") continue
      if (got[18] != "272") continue
      if (got[19] != "230") continue
      if (got[20] != "166") continue
      if (got[21] != "124") continue
      if (got[22] != "062") continue
      if (got[23] != "020") continue

      saw = " (" got[8] " " got[9] " " got[10] " " got[11] " " got[12] " " got[13] " " got[14] " " got[15] ")"

      if (got[8]  == "000" &&  \
          got[9]  == "000" &&  \
          got[10] == "000" &&  \
          got[11] == "124" &&  \
          got[12] == "064" &&  \
          got[13] == "157" &&  \
          got[14] == "235" &&  \
          got[15] == "301")
        {
          print "IEEE little endian"
          found = 1
          exit
        }

      # Little endian with the two 4-byte halves swapped, as used by ARM
      # when the chip is in little endian mode.
      #
      if (got[8]  == "064" &&  \
          got[9]  == "157" &&  \
          got[10] == "235" &&  \
          got[11] == "301" &&  \
          got[12] == "000" &&  \
          got[13] == "000" &&  \
          got[14] == "000" &&  \
          got[15] == "124")
        {
          print "IEEE little endian, swapped halves"
          found = 1
          exit
        }

      # gcc 2.95.4 on one GNU/Linux ARM system was seen generating 000 in
      # the last byte, whereas 124 is correct.  Not sure where the bug
      # actually lies, but a running program didn't seem to get a full
      # mantissa worth of working bits.
      #
      # We match this case explicitly so we can give a nice result message,
      # but we deliberately exclude it from the normal IEEE double setups
      # since it's too broken.
      #
      if (got[8]  == "064" &&  \
          got[9]  == "157" &&  \
          got[10] == "235" &&  \
          got[11] == "301" &&  \
          got[12] == "000" &&  \
          got[13] == "000" &&  \
          got[14] == "000" &&  \
          got[15] == "000")
        {
          print "bad ARM software floats"
          found = 1
          exit
        }

      if (got[8]  == "301" &&  \
          got[9]  == "235" &&  \
          got[10] == "157" &&  \
          got[11] == "064" &&  \
          got[12] == "124" &&  \
          got[13] == "000" &&  \
          got[14] == "000" &&  \
	  got[15] == "000")
        {
          print "IEEE big endian"
          found = 1
          exit
        }

      if (got[8]  == "353" &&  \
          got[9]  == "315" &&  \
          got[10] == "242" &&  \
          got[11] == "171" &&  \
          got[12] == "000" &&  \
          got[13] == "240" &&  \
          got[14] == "000" &&  \
          got[15] == "000")
        {
          print "VAX D"
          found = 1
          exit
        }

      if (got[8]  == "275" &&  \
          got[9]  == "301" &&  \
          got[10] == "064" &&  \
          got[11] == "157" &&  \
          got[12] == "000" &&  \
          got[13] == "124" &&  \
          got[14] == "000" &&  \
          got[15] == "000")
        {
          print "VAX G"
          found = 1
          exit
        }

      if (got[8]  == "300" &&  \
          got[9]  == "033" &&  \
          got[10] == "353" &&  \
          got[11] == "171" &&  \
          got[12] == "242" &&  \
          got[13] == "240" &&  \
          got[14] == "000" &&  \
          got[15] == "000")
        {
          print "Cray CFP"
          found = 1
          exit
        }
    }
}

END {
  if (! found)
    print "unknown", saw
}
]
EOF
  gmp_cv_c_double_format=`od -b conftest.$OBJEXT | $AWK -f conftest.awk`
  case $gmp_cv_c_double_format in
  unknown*)
    echo "cannot match anything, conftest.$OBJEXT contains" >&AC_FD_CC
    od -b conftest.$OBJEXT >&AC_FD_CC
    ;;
  esac
else
  AC_MSG_WARN([oops, cannot compile test program])
fi
rm -f conftest*
])

AH_VERBATIM([HAVE_DOUBLE],
[/* Define one of the following to 1 for the format of a `double'.
   If your format is not among these choices, or you don't know what it is,
   then leave all undefined.
   IEEE_LITTLE_SWAPPED means little endian, but with the two 4-byte halves
   swapped, as used by ARM CPUs in little endian mode.  */
#undef HAVE_DOUBLE_IEEE_BIG_ENDIAN
#undef HAVE_DOUBLE_IEEE_LITTLE_ENDIAN
#undef HAVE_DOUBLE_IEEE_LITTLE_SWAPPED
#undef HAVE_DOUBLE_VAX_D
#undef HAVE_DOUBLE_VAX_G
#undef HAVE_DOUBLE_CRAY_CFP])

case $gmp_cv_c_double_format in
  "IEEE big endian")
    AC_DEFINE(HAVE_DOUBLE_IEEE_BIG_ENDIAN, 1)
    GMP_DEFINE_RAW("define_not_for_expansion(\`HAVE_DOUBLE_IEEE_BIG_ENDIAN')", POST)
    ;;
  "IEEE little endian")
    AC_DEFINE(HAVE_DOUBLE_IEEE_LITTLE_ENDIAN, 1)
    GMP_DEFINE_RAW("define_not_for_expansion(\`HAVE_DOUBLE_IEEE_LITTLE_ENDIAN')", POST)
    ;;
  "IEEE little endian, swapped halves")
    AC_DEFINE(HAVE_DOUBLE_IEEE_LITTLE_SWAPPED, 1) ;;
  "VAX D")
    AC_DEFINE(HAVE_DOUBLE_VAX_D, 1) ;;
  "VAX G")
    AC_DEFINE(HAVE_DOUBLE_VAX_G, 1) ;;
  "Cray CFP")
    AC_DEFINE(HAVE_DOUBLE_CRAY_CFP, 1) ;;
  "bad ARM software floats")
    ;;
  unknown*)
    AC_MSG_WARN([Could not determine float format.])
    AC_MSG_WARN([Conversions to and from "double" may be slow.])
    ;;
  *)
    AC_MSG_WARN([oops, unrecognised float format: $gmp_cv_c_double_format])
    ;;
esac
])


dnl  GMP_C_STDARG
dnl  ------------
dnl  Test whether to use <stdarg.h>.
dnl
dnl  Notice the AC_DEFINE here is HAVE_STDARG to avoid clashing with
dnl  HAVE_STDARG_H which could arise from AC_CHECK_HEADERS.
dnl
dnl  This test might be slight overkill, after all there's really only going
dnl  to be ANSI or K&R and the two can be differentiated by AC_PROG_CC_STDC
dnl  or very likely by the setups for _PROTO in gmp.h.  On the other hand
dnl  this test is nice and direct, being what we're going to actually use.

dnl  AC_DEFUN([GMP_C_STDARG],
dnl  [AC_CACHE_CHECK([whether <stdarg.h> exists and works],
dnl                  gmp_cv_c_stdarg,
dnl  [AC_TRY_COMPILE(
dnl  [#include <stdarg.h>
dnl  int foo (int x, ...)
dnl  {
dnl    va_list  ap;
dnl    int      y;
dnl    va_start (ap, x);
dnl    y = va_arg (ap, int);
dnl    va_end (ap);
dnl    return y;
dnl  }],,
dnl  gmp_cv_c_stdarg=yes, gmp_cv_c_stdarg=no)
dnl  ])
dnl  if test $gmp_cv_c_stdarg = yes; then
dnl    AC_DEFINE(HAVE_STDARG, 1, [Define to 1 if <stdarg.h> exists and works])
dnl  fi
dnl  ])


dnl  GMP_FUNC_ALLOCA
dnl  ---------------
dnl  Determine whether "alloca" is available.  This is AC_FUNC_ALLOCA from
dnl  autoconf, but changed so it doesn't use alloca.c if alloca() isn't
dnl  available, and also to use gmp-impl.h for the conditionals detecting
dnl  compiler builtin alloca's.

AC_DEFUN([GMP_FUNC_ALLOCA],
[AC_REQUIRE([GMP_HEADER_ALLOCA])
AC_CACHE_CHECK([for alloca (via gmp-impl.h)],
               gmp_cv_func_alloca,
[AC_TRY_LINK(
GMP_INCLUDE_GMP_H
[#include "$srcdir/gmp-impl.h"
],
  [char *p = (char *) alloca (1);],
  gmp_cv_func_alloca=yes,
  gmp_cv_func_alloca=no)])
if test $gmp_cv_func_alloca = yes; then
  AC_DEFINE(HAVE_ALLOCA, 1, [Define to 1 if alloca() works (via gmp-impl.h).])
fi
])

AC_DEFUN([GMP_HEADER_ALLOCA],
[# The Ultrix 4.2 mips builtin alloca declared by alloca.h only works
# for constant arguments.  Useless!
AC_CACHE_CHECK([for working alloca.h],
               gmp_cv_header_alloca,
[AC_TRY_LINK([#include <alloca.h>],
  [char *p = (char *) alloca (2 * sizeof (int));],
  gmp_cv_header_alloca=yes,
  gmp_cv_header_alloca=no)])
if test $gmp_cv_header_alloca = yes; then
  AC_DEFINE(HAVE_ALLOCA_H, 1,
  [Define to 1 if you have <alloca.h> and it should be used (not on Ultrix).])
fi
])


dnl  GMP_OPTION_ALLOCA
dnl  -----------------
dnl  Decide what to do about --enable-alloca from the user.
dnl  This is a macro so it can require GMP_FUNC_ALLOCA.

AC_DEFUN([GMP_OPTION_ALLOCA],
[AC_REQUIRE([GMP_FUNC_ALLOCA])
AC_CACHE_CHECK([how to allocate temporary memory],
               gmp_cv_option_alloca,
[case $enable_alloca in
  yes)
    gmp_cv_option_alloca=alloca
    ;;
  no)
    gmp_cv_option_alloca=malloc-reentrant
    ;;
  reentrant | notreentrant)
    case $gmp_cv_func_alloca in
    yes)  gmp_cv_option_alloca=alloca ;;
    *)    gmp_cv_option_alloca=malloc-$enable_alloca ;;
    esac
    ;;
  *)
    gmp_cv_option_alloca=$enable_alloca
    ;;
esac
])

AH_VERBATIM([WANT_TMP],
[/* Define one of these to 1 for the desired temporary memory allocation
   method, per --enable-alloca. */
#undef WANT_TMP_ALLOCA
#undef WANT_TMP_REENTRANT
#undef WANT_TMP_NOTREENTRANT
#undef WANT_TMP_DEBUG])

case $gmp_cv_option_alloca in
  alloca)
    if test $gmp_cv_func_alloca = no; then
      AC_MSG_ERROR([--enable-alloca=alloca specified, but alloca not available])
    fi
    AC_DEFINE(WANT_TMP_ALLOCA)
    TAL_OBJECT=tal-reent$U.lo
    ;;
  malloc-reentrant)
    AC_DEFINE(WANT_TMP_REENTRANT)
    TAL_OBJECT=tal-reent$U.lo
    ;;
  malloc-notreentrant)
    AC_DEFINE(WANT_TMP_NOTREENTRANT)
    TAL_OBJECT=tal-notreent$U.lo
    ;;
  debug)
    AC_DEFINE(WANT_TMP_DEBUG)
    TAL_OBJECT=tal-debug$U.lo
    ;;
  *)
    # checks at the start of configure.in should protect us
    AC_MSG_ERROR([unrecognised --enable-alloca=$gmp_cv_option_alloca])
    ;;
esac
AC_SUBST(TAL_OBJECT)
])


dnl  GMP_FUNC_SSCANF_WRITABLE_INPUT
dnl  ------------------------------
dnl  Determine whether sscanf requires a writable input string.
dnl
dnl  It might be nicer to run a program to determine this when doing a
dnl  native build, but the systems afflicted are few and far between these
dnl  days, so it seems good enough just to list them.

AC_DEFUN([GMP_FUNC_SSCANF_WRITABLE_INPUT],
[AC_CACHE_CHECK([whether sscanf needs writable input],
                 gmp_cv_func_sscanf_writable_input,
[case $host in
  *-*-hpux9 | *-*-hpux9.*)
     gmp_cv_func_sscanf_writable_input=yes ;;
  *) gmp_cv_func_sscanf_writable_input=no  ;;
esac
])
case $gmp_cv_func_sscanf_writable_input in
  yes) AC_DEFINE(SSCANF_WRITABLE_INPUT, 1,
                 [Define to 1 if sscanf requires writable inputs]) ;;
  no)  ;;
  *)   AC_MSG_ERROR([unrecognised \$gmp_cv_func_sscanf_writable_input]) ;;
esac
])


dnl  GMP_FUNC_VSNPRINTF
dnl  ------------------
dnl  Check whether vsnprintf exists, and works properly.
dnl
dnl  Systems without vsnprintf include mingw32, OSF 4.
dnl
dnl  Sparc Solaris 2.7 in 64-bit mode doesn't always truncate, making
dnl  vsnprintf like vsprintf, and hence completely useless.  On one system a
dnl  literal string is enough to provoke the problem, on another a "%n" was
dnl  needed.  There seems to be something weird going on with the optimizer
dnl  or something, since on the first system adding a second check with
dnl  "%n", or even just an initialized local variable, makes it work.  In
dnl  any case, without bothering to get to the bottom of this, the two
dnl  program runs in the code below end up successfully detecting the
dnl  problem.
dnl
dnl  glibc 2.0.x returns either -1 or bufsize-1 for an overflow (both seen,
dnl  not sure which 2.0.x does which), but still puts the correct null
dnl  terminated result into the buffer.

AC_DEFUN([GMP_FUNC_VSNPRINTF],
[AC_CHECK_FUNC(vsnprintf,
              [gmp_vsnprintf_exists=yes],
              [gmp_vsnprintf_exists=no])
if test "$gmp_vsnprintf_exists" = no; then
  gmp_cv_func_vsnprintf=no
else
  AC_CACHE_CHECK([whether vsnprintf works],
                 gmp_cv_func_vsnprintf,
  [gmp_cv_func_vsnprintf=yes
   for i in 'return check ("hello world");' 'int n; return check ("%nhello world", &n);'; do
     AC_TRY_RUN([
#include <string.h>  /* for strcmp */
#include <stdio.h>   /* for vsnprintf */

#include <stdarg.h>

int
check (const char *fmt, ...)
{
  static char  buf[128];
  va_list  ap;
  int      ret;

  va_start (ap, fmt);

  ret = vsnprintf (buf, 4, fmt, ap);

  if (strcmp (buf, "hel") != 0)
    return 1;

  /* allowed return values */
  if (ret != -1 && ret != 3 && ret != 11)
    return 2;

  return 0;
}

int
main ()
{
$i
}
],
      [:],
      [gmp_cv_func_vsnprintf=no; break],
      [gmp_cv_func_vsnprintf=probably; break])
  done
  ])
  if test "$gmp_cv_func_vsnprintf" = probably; then
    AC_MSG_WARN([cannot check for properly working vsnprintf when cross compiling, will assume it's ok])
  fi
  if test "$gmp_cv_func_vsnprintf" != no; then
    AC_DEFINE(HAVE_VSNPRINTF,1,
    [Define to 1 if you have the `vsnprintf' function and it works properly.])
  fi
fi
])


dnl  GMP_H_EXTERN_INLINE
dnl  -------------------
dnl  If the compiler has an "inline" of some sort, check whether the
dnl  #ifdef's in gmp.h recognise it.

AC_DEFUN([GMP_H_EXTERN_INLINE],
[AC_REQUIRE([AC_C_INLINE])
case $ac_cv_c_inline in
no) ;;
*)
  AC_TRY_COMPILE(
[#define __GMP_WITHIN_CONFIGURE_INLINE 1
]GMP_INCLUDE_GMP_H[
#ifndef __GMP_EXTERN_INLINE
die die die
#endif
],,,
  [case $ac_cv_c_inline in
  yes) tmp_inline=inline ;;
  *)   tmp_inline=$ac_cv_c_inline ;;
  esac
  AC_MSG_WARN([gmp.h doesnt recognise compiler "$tmp_inline", inlines will be unavailable])])
  ;;
esac
])


dnl  GMP_H_HAVE_FILE
dnl  ---------------
dnl  Check whether the #ifdef's in gmp.h recognise when stdio.h has been
dnl  included to get FILE.

AC_DEFUN([GMP_H_HAVE_FILE],
[AC_TRY_COMPILE(
[#include <stdio.h>]
GMP_INCLUDE_GMP_H
[#if ! _GMP_H_HAVE_FILE
die die die
#endif
],,,
  [AC_MSG_WARN([gmp.h doesnt recognise <stdio.h>, FILE prototypes will be unavailable])])
])


dnl  GMP_PROG_CC_FOR_BUILD
dnl  ---------------------
dnl  Establish CC_FOR_BUILD, a C compiler for the build system.
dnl
dnl  If CC_FOR_BUILD is set then it's expected to work, likewise the old
dnl  style HOST_CC, otherwise some likely candidates are tried, the same as
dnl  configfsf.guess.

AC_DEFUN([GMP_PROG_CC_FOR_BUILD],
[AC_REQUIRE([AC_PROG_CC])
if test -n "$CC_FOR_BUILD"; then
  GMP_PROG_CC_FOR_BUILD_WORKS($CC_FOR_BUILD,,
    [AC_MSG_ERROR([Specified CC_FOR_BUILD doesn't seem to work])])
elif test -n "$HOST_CC"; then
  GMP_PROG_CC_FOR_BUILD_WORKS($HOST_CC,
    [CC_FOR_BUILD=$HOST_CC],
    [AC_MSG_ERROR([Specified HOST_CC doesn't seem to work])])
else
  for i in "$CC" "$CC $CFLAGS $CPPFLAGS" cc gcc c89 c99; do
    GMP_PROG_CC_FOR_BUILD_WORKS($i,
      [CC_FOR_BUILD=$i
       break])
  done
  if test -z "$CC_FOR_BUILD"; then
    AC_MSG_ERROR([Cannot find a build system compiler])
  fi
fi

AC_ARG_VAR(CC_FOR_BUILD,[build system C compiler])
AC_SUBST(CC_FOR_BUILD)
])


dnl  GMP_PROG_CC_FOR_BUILD_WORKS(cc/cflags[,[action-if-good][,action-if-bad]])
dnl  -------------------------------------------------------------------------
dnl  See if the given cc/cflags works on the build system.
dnl
dnl  It seems easiest to just use the default compiler output, rather than
dnl  figuring out the .exe or whatever at this stage.

AC_DEFUN([GMP_PROG_CC_FOR_BUILD_WORKS],
[AC_MSG_CHECKING([build system compiler $1])
# remove anything that might look like compiler output to our "||" expression
rm -f conftest* a.out b.out a.exe a_out.exe
cat >conftest.c <<EOF
int
main ()
{
  return 0;
}
EOF
gmp_compile="$1 conftest.c"
cc_for_build_works=no
if AC_TRY_EVAL(gmp_compile); then
  if (./a.out || ./b.out || ./a.exe || ./a_out.exe || ./conftest) >&AC_FD_CC 2>&1; then
    cc_for_build_works=yes
  fi
fi
rm -f conftest* a.out b.out a.exe a_out.exe
AC_MSG_RESULT($cc_for_build_works)
if test "$cc_for_build_works" = yes; then
  ifelse([$2],,:,[$2])
else
  ifelse([$3],,:,[$3])
fi
])


dnl  GMP_PROG_CPP_FOR_BUILD
dnl  ---------------------
dnl  Establish CPP_FOR_BUILD, the build system C preprocessor.
dnl  The choices tried here are the same as AC_PROG_CPP, but with
dnl  CC_FOR_BUILD.

AC_DEFUN([GMP_PROG_CPP_FOR_BUILD],
[AC_REQUIRE([GMP_PROG_CC_FOR_BUILD])
AC_MSG_CHECKING([for build system preprocessor])
if test -z "$CPP_FOR_BUILD"; then
  AC_CACHE_VAL(gmp_cv_prog_cpp_for_build,
  [cat >conftest.c <<EOF
#define FOO BAR
EOF
  for i in "$CC_FOR_BUILD -E" "$CC_FOR_BUILD -E -traditional-cpp" "/lib/cpp"; do
    gmp_compile="$i conftest.c"
    if AC_TRY_EVAL(gmp_compile) >&AC_FD_CC 2>&1; then
      gmp_cv_prog_cpp_for_build=$i
      break
    fi
  done
  rm -f conftest* a.out b.out a.exe a_out.exe
  if test -z "$gmp_cv_prog_cpp_for_build"; then
    AC_MSG_ERROR([Cannot find build system C preprocessor.])
  fi
  ])
  CPP_FOR_BUILD=$gmp_cv_prog_cpp_for_build
fi
AC_MSG_RESULT([$CPP_FOR_BUILD])

AC_ARG_VAR(CPP_FOR_BUILD,[build system C preprocessor])
AC_SUBST(CPP_FOR_BUILD)
])


dnl  GMP_PROG_EXEEXT_FOR_BUILD
dnl  -------------------------
dnl  Determine EXEEXT_FOR_BUILD, the build system executable suffix.
dnl
dnl  The idea is to find what "-o conftest$foo" will make it possible to run
dnl  the program with ./conftest.  On Unix-like systems this is of course
dnl  nothing, for DOS it's ".exe", or for a strange RISC OS foreign file
dnl  system cross compile it can be ",ff8" apparently.  Not sure if the
dnl  latter actually applies to a build-system executable, maybe it doesn't,
dnl  but it won't hurt to try.

AC_DEFUN([GMP_PROG_EXEEXT_FOR_BUILD],
[AC_REQUIRE([GMP_PROG_CC_FOR_BUILD])
AC_CACHE_CHECK([for build system executable suffix],
               gmp_cv_prog_exeext_for_build,
[cat >conftest.c <<EOF
int
main ()
{
  return 0;
}
EOF
for i in .exe ,ff8 ""; do
  gmp_compile="$CC_FOR_BUILD conftest.c -o conftest$i"
  if AC_TRY_EVAL(gmp_compile); then
    if (./conftest) 2>&AC_FD_CC; then
      gmp_cv_prog_exeext_for_build=$i
      break
    fi
  fi
done
rm -f conftest*
if test "${gmp_cv_prog_exeext_for_build+set}" != set; then
  AC_MSG_ERROR([Cannot determine executable suffix])
fi
])
AC_SUBST(EXEEXT_FOR_BUILD,$gmp_cv_prog_exeext_for_build)
])


dnl  GMP_C_FOR_BUILD_ANSI
dnl  --------------------
dnl  Determine whether CC_FOR_BUILD is ANSI, and establish U_FOR_BUILD
dnl  accordingly.
dnl
dnl  FIXME: Use AC_PROG_CC sets ac_cv_prog_cc_c89 which could be used instead

AC_DEFUN([GMP_C_FOR_BUILD_ANSI],
[AC_REQUIRE([GMP_PROG_CC_FOR_BUILD])
AC_CACHE_CHECK([whether build system compiler is ANSI],
               gmp_cv_c_for_build_ansi,
[cat >conftest.c <<EOF
int
main (int argc, char **argv)
{
  return 0;
}
EOF
gmp_compile="$CC_FOR_BUILD conftest.c"
if AC_TRY_EVAL(gmp_compile); then
  gmp_cv_c_for_build_ansi=yes
else
  gmp_cv_c_for_build_ansi=no
fi
rm -f conftest* a.out b.out a.exe a_out.exe
])
if test "$gmp_cv_c_for_build_ansi" = yes; then
  U_FOR_BUILD=
else
  AC_SUBST(U_FOR_BUILD,_)
fi
])


dnl  GMP_CHECK_LIBM_FOR_BUILD
dnl  ------------------------
dnl  Establish LIBM_FOR_BUILD as -lm, if that seems to work.
dnl
dnl  Libtool AC_CHECK_LIBM also uses -lmw on *-ncr-sysv4.3*, if it works.
dnl  Don't know what that does, lets assume it's not needed just for log().

AC_DEFUN([GMP_CHECK_LIBM_FOR_BUILD],
[AC_REQUIRE([GMP_PROG_CC_FOR_BUILD])
AC_CACHE_CHECK([for build system compiler math library],
               gmp_cv_check_libm_for_build,
[cat >conftest.c <<EOF
#include <math.h>
int
main ()
{
  return 0;
}
double d;
double
foo ()
{
  return log (d);
}
EOF
gmp_compile="$CC_FOR_BUILD conftest.c -lm"
if AC_TRY_EVAL(gmp_compile); then
  gmp_cv_check_libm_for_build=-lm
else
  gmp_cv_check_libm_for_build=no
fi
rm -f conftest* a.out b.out a.exe a_out.exe
])
case $gmp_cv_check_libm_for_build in
  yes) AC_SUBST(LIBM_FOR_BUILD,-lm) ;;
  no)  LIBM_FOR_BUILD= ;;
  *)   LIBM_FOR_BUILD=$gmp_cv_check_libm_for_build ;;
esac
])
