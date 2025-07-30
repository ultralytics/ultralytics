#! /usr/bin/perl -w

# Copyright 2000-2002 Free Software Foundation, Inc.
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


# Usage:  cd $builddir/tune
#	  perl $srcdir/tune/many.pl [-t] <files/dirs>...
#
# Output: speed-many.c
#         try-many.c
#         Makefile.many
#
# Make alternate versions of various mpn routines available for measuring
# and testing.
#
# The $srcdir and $builddir in the invocation above just means the script
# lives in the tune source directory, but should be run in the tune build
# directory.  When not using a separate object directory this just becomes
#
#	cd tune
#	perl many.pl [-t] <files/dirs>...
#
#
# SINGLE FILES
#
# Suppose $HOME/newcode/mul_1_experiment.asm is a new implementation of
# mpn_mul_1, then
#
#	cd $builddir/tune
#	perl $srcdir/tune/many.pl $HOME/newcode/mul_1_experiment.asm
#
# will produce rules and renaming so that a speed program incorporating it
# can be built,
#
#	make -f Makefile.many speed-many
#
# then for example it can be compared to the standard mul_1,
#
#	./speed-many -s 1-30 mpn_mul_1 mpn_mul_1_experiment
#
# An expanded try program can be used to check correctness,
#
#	make -f Makefile.many try-many
#
# and run
#
#	./try-many mpn_mul_1_experiment
#
# Files can be ".c", ".S" or ".asm".  ".s" files can't be used because they
# don't get any preprocessing so there's no way to do renaming of their
# functions.
#
#
# WHOLE DIRECTORIES
#
# If a directory is given, then all files in it will be made available.
# For example,
#
#	cd $builddir/tune
#	perl $srcdir/tune/many.pl $HOME/newcode
#
# Each file should have a suffix, like "_experiment" above.
#
#
# MPN DIRECTORIES
#
# mpn directories from the GMP source tree can be included, and this is a
# convenient way to compare multiple implementations suiting different chips
# in a CPU family.  For example the following would make all x86 routines
# available,
#
#	cd $builddir/tune
#	perl $srcdir/tune/many.pl `find $srcdir/mpn/x86 -type d`
#
# On a new x86 chip a comparison could then be made to see how existing code
# runs.  For example,
#
#	make -f Makefile.many speed-many
#	./speed-many -s 1-30 -c \
#		mpn_add_n_x86 mpn_add_n_pentium mpn_add_n_k6 mpn_add_n_k7
#
# Files in "mpn" subdirectories don't need the "_experiment" style suffix
# described above, instead a suffix is constructed from the subdirectory.
# For example "mpn/x86/k7/mmx/mod_1.asm" will generate a function
# mpn_mod_1_k7_mmx.  The rule is to take the last directory name after the
# "mpn", or the last two if there's three or more.  (Check the generated
# speed-many.c if in doubt.)
#
#
# GENERIC C
#
# The mpn/generic directory can be included too, just like any processor
# specific directory.  This is a good way to compare assembler and generic C
# implementations.  For example,
#
#	cd $builddir/tune
#	perl $srcdir/tune/many.pl $srcdir/mpn/generic
#
# or if just a few routines are of interest, then for example
#
#	cd $builddir/tune
#	perl $srcdir/tune/many.pl \
#		$srcdir/mpn/generic/lshift.c \
#		$srcdir/mpn/generic/mod_1.c \
#		$srcdir/mpn/generic/aorsmul_1.c
#
# giving mpn_lshift_generic etc.
#
#
# TESTS/DEVEL PROGRAMS
#
# Makefile.many also has rules to build the tests/devel programs with suitable
# renaming, and with some parameters for correctness or speed.  This is less
# convenient than the speed and try programs, but provides an independent
# check.  For example,
#
#	make -f Makefile.many tests_mul_1_experimental
#	./tests_mul_1_experimental
#
# and for speed
#
#	make -f Makefile.many tests_mul_1_experimental_sp
#	./tests_mul_1_experimental_sp
#
# Not all the programs support speed measuring, in which case only the
# correctness test will be useful.
#
# The parameters for repetitions and host clock speed are -D defines.  Some
# defaults are provided at the end of Makefile.many, but probably these will
# want to be overridden.  For example,
#
#	rm tests_mul_1_experimental.o
#	make -f Makefile.many \
#	   CFLAGS_TESTS="-DSIZE=50 -DTIMES=1000 -DRANDOM -DCLOCK=175000000" \
#	   tests_mul_1_experimental
#	./tests_mul_1_experimental
#
#
# OTHER NOTES
#
# The mappings of file names to functions, and the macros to then use for
# speed measuring etc are driven by @table below.  The scheme isn't
# completely general, it's only got as many variations as have been needed
# so far.
#
# Some functions are only made available in speed-many, or others only in
# try-many.  An @table entry speed=>none means no speed measuring is
# available, or try=>none no try program testing.  These can be removed
# if/when the respective programs get the necessary support.
#
# If a file has "1c" or "nc" carry-in entrypoints, they're renamed and made
# available too.  These are recognised from PROLOGUE or MULFUNC_PROLOGUE in
# .S and .asm files, or from a line starting with "mpn_foo_1c" in a .c file
# (possibly via a #define), and on that basis are entirely optional.  This
# entrypoint matching is done for the standard entrypoints too, but it would
# be very unusual to have for instance a mul_1c without a mul_1.
#
# Some mpz files are recognized.  For example an experimental copy of
# mpz/powm.c could be included as powm_new.c and would be called
# mpz_powm_new.  So far only speed measuring is available for these.
#
# For the ".S" and ".asm" files, both PIC and non-PIC objects are built.
# The PIC functions have a "_pic" suffix, for example "mpn_mod_1_k7_mmx_pic".
# This can be ignored for routines that don't differ for PIC, or for CPUs
# where everything is PIC anyway.
#
# K&R compilers are supported via the same ansi2knr mechanism used by
# automake, though it's hard to believe anyone will have much interest in
# measuring a compiler so old that it doesn't even have an ANSI mode.
#
# The "-t" option can be used to print a trace of the files found and what's
# done with them.  A great deal of obscure output is produced, but it can
# indicate where or why some files aren't being recognised etc.  For
# example,
#
#	cd $builddir/tune
#	perl $srcdir/tune/many.pl -t $HOME/newcode/add_n_weird.asm
#
# In general, when including new code, all that's really necessary is that
# it will compile or assemble under the current configuration.  It's fine if
# some code doesn't actually run due to bugs, or to needing a newer CPU or
# whatever, simply don't ask for the offending routines when invoking
# speed-many or try-many, or don't try to run them on sizes they don't yet
# support, or whatever.
#
#
# CPU SPECIFICS
#
# x86 - All the x86 code will assemble on any system, but code for newer
#       chips might not run on older chips.  Expect SIGILLs from new
#       instructions on old chips.
#
#       A few "new" instructions, like cmov for instance, are done as macros
#       and will generate some equivalent plain i386 code when HAVE_HOST_CPU
#       in config.m4 indicates an old CPU.  It won't run fast, but it does
#       make it possible to test correctness.
#
#
# INTERNALS
#
# The nonsense involving $ENV is some hooks used during development to add
# additional functions temporarily.
#
#
# FUTURE
#
# Maybe the C files should be compiled pic and non-pic too.  Wait until
# there's a difference that might be of interest.
#
# Warn if a file provides no functions.
#
# Allow mpz and mpn files of the same name.  Currently the mpn fib2_ui
# matching hides the mpz version of that.  Will need to check the file
# contents to see which it is.  Would be worth allowing an "mpz_" or "mpn_"
# prefix on the filenames to have working versions of both in one directory.
#
#
# LIMITATIONS
#
# Some of the command lines can become very long when a lot of files are
# included.  If this is a problem on a given system the only suggestion is
# to run many.pl for just those that are actually wanted at a particular
# time.
#
# DOS 8.3 or SysV 14 char filesystems won't work, since the long filenames
# generated will almost certainly fail to be unique.


use strict;
use File::Basename;
use Getopt::Std;

my %opt;
getopts('t', \%opt);

my @DIRECTORIES = @ARGV;
if (defined $ENV{directories}) { push @DIRECTORIES, @{$ENV{directories}} }


# regexp - matched against the start of the filename.  If a grouping "(...)"
#          is present then only the first such part is used.
#
# mulfunc - filenames to be generated from a multi-function file.
#
# funs - functions provided by the file, defaulting to the filename with mpn
#          (or mpX).
#
# mpX - prefix like "mpz", defaulting to "mpn".
#
# ret - return value type.
#
# args, args_<fun> - arguments for the given function.  If an args_<fun> is
#          set then it's used, otherwise plain args is used.  "mp_limb_t
#          carry" is appended for carry-in variants.
#
# try - try.c TYPE_ to use, defaulting to TYPE_fun with the function name
#          in upper case.  "C" is appended for carry-in variants.  Can be
#          'none' for no try program entry.
#
# speed - SPEED_ROUTINE_ to use, handled like "try".
#
# speed_flags - SPEED_ROUTINE_ to use, handled like "try".


my @table =
    (
     {
       'regexp'=> 'add_n|sub_n|addlsh1_n|sublsh1_n|rsh1add_n|rsh1sub_n',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size',
       'speed' => 'SPEED_ROUTINE_MPN_BINARY_N',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
     },
     {
       'regexp'=> 'aors_n',
       'mulfunc'=> ['add_n','sub_n'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size',
       'speed' => 'SPEED_ROUTINE_MPN_BINARY_N',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
     },

     {
       'regexp'=> 'addmul_1|submul_1',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_limb_t mult',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_1',
       'speed_flags'=> 'FLAG_R',
     },
     {
       'regexp'=> 'aorsmul_1',
       'mulfunc'=> ['addmul_1','submul_1'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_limb_t mult',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_1',
       'speed_flags'=> 'FLAG_R',
     },

     {
       'regexp'=> 'addmul_2|submul_2',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr yp',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_2',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try-minsize' => 2,
     },
     {
       'regexp'=> 'addmul_3|submul_3',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr yp',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_3',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try-minsize' => 3,
     },
     {
       'regexp'=> 'addmul_4|submul_4',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr yp',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_4',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try-minsize' => 4,
     },
     {
       'regexp'=> 'addmul_5|submul_5',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr yp',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_5',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try-minsize' => 5,
     },
     {
       'regexp'=> 'addmul_6|submul_6',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr yp',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_6',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try-minsize' => 6,
     },
     {
       'regexp'=> 'addmul_7|submul_7',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr yp',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_7',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try-minsize' => 7,
     },
     {
       'regexp'=> 'addmul_8|submul_8',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr yp',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_8',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try-minsize' => 8,
     },

     {
       'regexp'=> 'add_n_sub_n',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr sum, mp_ptr diff, mp_srcptr xp, mp_srcptr yp, mp_size_t size',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
     },

     {
       'regexp'=> 'com|copyi|copyd',
       'ret'   => 'void',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size',
       'speed' => 'SPEED_ROUTINE_MPN_COPY',
     },

     {
       'regexp'=> 'dive_1',
       'funs'  => ['divexact_1'],
       'ret'   => 'void',
       'args'  => 'mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor',
       'speed_flags'=> 'FLAG_R',
     },
     {
       'regexp'=> 'diveby3',
       'funs'  => ['divexact_by3c'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr dst, mp_srcptr src, mp_size_t size',
       'carrys'=> [''],
       'speed' => 'SPEED_ROUTINE_MPN_COPY',
     },

     # mpn_preinv_divrem_1 is an optional extra entrypoint
     {
       'regexp'=> 'divrem_1',
       'funs'  => ['divrem_1', 'preinv_divrem_1'],
       'ret'   => 'mp_limb_t',
       'args_divrem_1' => 'mp_ptr rp, mp_size_t xsize, mp_srcptr sp, mp_size_t size, mp_limb_t divisor',
       'args_preinv_divrem_1' => 'mp_ptr rp, mp_size_t xsize, mp_srcptr sp, mp_size_t size, mp_limb_t divisor, mp_limb_t inverse, unsigned shift',
       'speed_flags'=> 'FLAG_R',
       'speed_suffixes' => ['f'],
     },
     {
       'regexp'=> 'pre_divrem_1',
       'funs'  => ['preinv_divrem_1'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr qp, mp_size_t qxn, mp_srcptr ap, mp_size_t asize, mp_limb_t divisor, mp_limb_t inverse, int shift',
       'speed_flags' => 'FLAG_R',
     },

     {
       'regexp'=> 'divrem_2',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr qp, mp_size_t qxn, mp_srcptr np, mp_size_t nsize, mp_srcptr dp',
       'try'   => 'none',
     },

     {
       'regexp'=> 'sb_divrem_mn',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr qp, mp_ptr np, mp_size_t nsize, mp_srcptr dp, mp_size_t dsize',
       'speed' => 'SPEED_ROUTINE_MPN_DC_DIVREM_SB',
       'try-minsize' => 3,
     },
     {
       'regexp'=> 'tdiv_qr',
       'ret'   => 'void',
       'args'  => 'mp_ptr qp, mp_size_t qxn, mp_ptr np, mp_size_t nsize, mp_srcptr dp, mp_size_t dsize',
       'speed' => 'none',
     },

     {
       'regexp'=> 'get_str',
       'ret'   => 'size_t',
       'args'  => 'unsigned char *str, int base, mp_ptr mptr, mp_size_t msize',
       'speed_flags' => 'FLAG_R_OPTIONAL',
       'try'   => 'none',
     },
     {
       'regexp'=> 'set_str',
       'ret'   => 'mp_size_t',
       'args'  => 'mp_ptr xp, const unsigned char *str, size_t str_len, int base',
       'speed_flags' => 'FLAG_R_OPTIONAL',
       'try'   => 'none',
     },

     {
       'regexp'=> 'fac_ui',
       'mpX'   => 'mpz',
       'ret'   => 'void',
       'args'  => 'mpz_ptr r, unsigned long n',
       'speed_flags' => 'FLAG_NODATA',
       'try'   => 'none',
     },

     {
       'regexp'=> 'fib2_ui',
       'ret'   => 'void',
       'args'  => 'mp_ptr fp, mp_ptr f1p, unsigned long n',
       'rename'=> ['__gmp_fib_table'],
       'speed_flags' => 'FLAG_NODATA',
       'try'   => 'none',
     },
     {
       'regexp'=> 'fib_ui',
       'mpX'   => 'mpz',
       'ret'   => 'void',
       'args'  => 'mpz_ptr fn, unsigned long n',
       'speed_flags' => 'FLAG_NODATA',
       'try'   => 'none',
     },
     {
       'regexp'=> 'fib2_ui',
       'mpX'   => 'mpz',
       'ret'   => 'void',
       'args'  => 'mpz_ptr fn, mpz_ptr fnsub1, unsigned long n',
       'speed_flags' => 'FLAG_NODATA',
       'try'   => 'none',
     },

     {
       'regexp'=> 'lucnum_ui',
       'mpX'   => 'mpz',
       'ret'   => 'void',
       'args'  => 'mpz_ptr ln, unsigned long n',
       'speed_flags' => 'FLAG_NODATA',
       'try'   => 'none',
     },
     {
       'regexp'=> 'lucnum2_ui',
       'mpX'   => 'mpz',
       'ret'   => 'void',
       'args'  => 'mpz_ptr ln, mpz_ptr lnsub1, unsigned long n',
       'speed_flags' => 'FLAG_NODATA',
       'try'   => 'none',
     },

     {
       'regexp'=> 'gcd_1',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr xp, mp_size_t xsize, mp_limb_t y',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'speed_suffixes' => ['N'],
     },
     {
       'regexp'=> '(gcd)(?!(_1|ext|_finda))',
       'ret'   => 'mp_size_t',
       'args'  => 'mp_ptr gp, mp_ptr up, mp_size_t usize, mp_ptr vp, mp_size_t vsize',
     },
     {
       'regexp'=> 'gcd_finda',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_srcptr cp',
     },


     {
       'regexp'=> 'jacobi',
       'funs'  => ['jacobi', 'legendre', 'kronecker'],
       'mpX'   => 'mpz',
       'ret'   => 'int',
       'args'  => 'mpz_srcptr a, mpz_srcptr b',
       'try-legendre' => 'TYPE_MPZ_JACOBI',
     },
     {
       'regexp'=> 'jacbase',
       'funs'  => ['jacobi_base'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_limb_t a, mp_limb_t b, int bit1',
       'speed' => 'SPEED_ROUTINE_MPN_JACBASE',
       'try'   => 'none',
     },

     {
       'regexp'=> 'logops_n',
       'mulfunc'=> ['and_n','andn_n','nand_n','ior_n','iorn_n','nior_n','xor_n','xnor_n'],
       'ret'   => 'void',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size',
       'speed' => 'SPEED_ROUTINE_MPN_BINARY_N',
     },

     {
       'regexp'=> '[lr]shift',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, unsigned shift',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_1',
       'speed_flags'=> 'FLAG_R',
     },

     # mpn_preinv_mod_1 is an optional extra entrypoint
     {
       'regexp'=> '(mod_1)(?!_rs)',
       'funs'  => ['mod_1','preinv_mod_1'],
       'ret'   => 'mp_limb_t',
       'args_mod_1'       => 'mp_srcptr xp, mp_size_t size, mp_limb_t divisor',
       'args_preinv_mod_1'=> 'mp_srcptr xp, mp_size_t size, mp_limb_t divisor, mp_limb_t inverse',
       'speed_flags'=> 'FLAG_R',
     },
     {
       'regexp'=> 'pre_mod_1',
       'funs'  => ['preinv_mod_1'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_srcptr xp, mp_size_t size, mp_limb_t divisor, mp_limb_t inverse',
       'speed_flags'=> 'FLAG_R',
     },
     {
       'regexp'=> 'mod_34lsub1',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_srcptr src, mp_size_t len',
     },
     {
       'regexp'=> 'invert_limb',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_limb_t divisor',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try'   => 'none',
     },

     {
       # not for use with hppa reversed argument versions of mpn_umul_ppmm
       'regexp'=> 'udiv',
       'funs'  => ['udiv_qrnnd','udiv_qrnnd_r'],
       'ret'   => 'mp_limb_t',
       'args_udiv_qrnnd'   => 'mp_limb_t *, mp_limb_t, mp_limb_t, mp_limb_t',
       'args_udiv_qrnnd_r' => 'mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t *',
       'speed' => 'none',
       'try-minsize' => 2,
     },

     {
       'regexp'=> 'mode1o',
       'funs'  => ['modexact_1_odd'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_srcptr src, mp_size_t size, mp_limb_t divisor',
       'speed_flags'=> 'FLAG_R',
     },
     {
       'regexp'=> 'modlinv',
       'funs'  => ['modlimb_invert'],
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_limb_t v',
       'carrys'=> [''],
       'try'   => 'none',
     },

     {
       'regexp'=> 'mul_1',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_limb_t mult',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_1',
       'speed_flags'=> 'FLAG_R',
     },
     {
       'regexp'=> 'mul_2',
       'ret'   => 'mp_limb_t',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size, mp_srcptr mult',
       'speed' => 'SPEED_ROUTINE_MPN_UNARY_2',
       'speed_flags'=> 'FLAG_R',
     },

     {
       'regexp'=> 'mul_basecase',
       'ret'   => 'void',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t xsize, mp_srcptr yp, mp_size_t ysize',
       'speed_flags' => 'FLAG_R_OPTIONAL | FLAG_RSIZE',
     },
     {
       'regexp'=> '(mul_n)[_.]',
       'ret'   => 'void',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size',
       'rename'=> ['kara_mul_n','kara_sqr_n','toom3_mul_n','toom3_sqr_n'],
     },
     {
       'regexp'=> 'umul',
       'funs'  => ['umul_ppmm','umul_ppmm_r'],
       'ret'   => 'mp_limb_t',
       'args_umul_ppmm'   => 'mp_limb_t *lowptr, mp_limb_t m1, mp_limb_t m2',
       'args_umul_ppmm_r' => 'mp_limb_t m1, mp_limb_t m2, mp_limb_t *lowptr',
       'speed' => 'none',
       'try-minsize' => 3,
     },


     {
       'regexp'=> 'popham',
       'mulfunc'=> ['popcount','hamdist'],
       'ret'   => 'unsigned long',
       'args_popcount'=> 'mp_srcptr xp, mp_size_t size',
       'args_hamdist' => 'mp_srcptr xp, mp_srcptr yp, mp_size_t size',
     },
     {
       'regexp'=> 'popcount',
       'ret'   => 'unsigned long',
       'args'  => 'mp_srcptr xp, mp_size_t size',
     },
     {
       'regexp'=> 'hamdist',
       'ret'   => 'unsigned long',
       'args'  => 'mp_srcptr xp, mp_srcptr yp, mp_size_t size',
       # extra renaming to support sharing a data table with mpn_popcount
       'rename'=> ['popcount'],
     },

     {
       'regexp'=> 'sqr_basecase',
       'ret'   => 'void',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size',
       'speed' => 'SPEED_ROUTINE_MPN_SQR',
       'try'   => 'TYPE_SQR',
     },
     {
       'regexp'=> 'sqr_diagonal',
       'ret'   => 'void',
       'args'  => 'mp_ptr wp, mp_srcptr xp, mp_size_t size',
       'try'   => 'none',
     },

     {
       'regexp'=> 'sqrtrem',
       'ret'   => 'mp_size_t',
       'args'  => 'mp_ptr root, mp_ptr rem, mp_srcptr src, mp_size_t size',
       'try'   => 'none',
     },

     {
       'regexp'=> 'cntlz',
       'funs'  => ['count_leading_zeros'],
       'ret'   => 'unsigned',
       'args'  => 'mp_limb_t',
       'macro-before' => "#undef COUNT_LEADING_ZEROS_0",
       'macro-speed'  =>
'#ifdef COUNT_LEADING_ZEROS_0
#define COUNT_LEADING_ZEROS_0_ALLOWED   1
#else
#define COUNT_LEADING_ZEROS_0_ALLOWED   0
#endif
  SPEED_ROUTINE_COUNT_ZEROS_A (1, COUNT_LEADING_ZEROS_0_ALLOWED);
  $fun (c, n);
  SPEED_ROUTINE_COUNT_ZEROS_B ()',
       'speed_flags'=> 'FLAG_R_OPTIONAL',
       'try'   => 'none',
     },
     {
       'regexp'=> 'cnttz',
       'funs'  => ['count_trailing_zeros'],
       'ret'   => 'unsigned',
       'args'  => 'mp_limb_t',
       'macro-speed' => '
  SPEED_ROUTINE_COUNT_ZEROS_A (0, 0);
  $fun (c, n);
  SPEED_ROUTINE_COUNT_ZEROS_B ()',
       'speed_flags' => 'FLAG_R_OPTIONAL',
       'try'   => 'none',
     },

     {
       'regexp'=> 'zero',
       'ret'   => 'void',
       'args'  => 'mp_ptr ptr, mp_size_t size',
     },

     {
       'regexp'=> '(powm)(?!_ui)',
       'mpX'   => 'mpz',
       'ret'   => 'void',
       'args'  => 'mpz_ptr r, mpz_srcptr b, mpz_srcptr e, mpz_srcptr m',
       'try'   => 'none',
     },
     {
       'regexp'=> 'powm_ui',
       'mpX'   => 'mpz',
       'ret'   => 'void',
       'args'  => 'mpz_ptr r, mpz_srcptr b, unsigned long e, mpz_srcptr m',
       'try'   => 'none',
     },

     # special for use during development
     {
       'regexp'=> 'back',
       'funs'  => ['back_to_back'],
       'ret'   => 'void',
       'args'  => 'void',
       'pic'   => 'no',
       'try'   => 'none',
       'speed_flags'=> 'FLAG_NODATA',
     },
     );

if (defined $ENV{table2}) {
  my @newtable = @{$ENV{table2}};
  push @newtable, @table;
  @table = @newtable;
}


my %pictable =
    (
     'yes' => {
       'suffix' =>  '_pic',
       'asmflags'=> '$(ASMFLAGS_PIC)',
       'cflags' =>  '$(CFLAGS_PIC)',
     },
     'no' => {
       'suffix' =>  '',
       'asmflags'=> '',
       'cflags' =>  '',
     },
     );


my $builddir = $ENV{builddir};
$builddir = "." if (! defined $builddir);

my $top_builddir = "${builddir}/..";


open(MAKEFILE, "<${builddir}/Makefile")
  or die "Cannot open ${builddir}/Makefile: $!\n"
       . "Is this a tune build directory?";
my ($srcdir, $top_srcdir);
while (<MAKEFILE>) {
  if (/^srcdir = (.*)/) {     $srcdir = $1;     }
  if (/^top_srcdir = (.*)/) { $top_srcdir = $1; }
}
die "Cannot find \$srcdir in Makefile\n" if (! defined $srcdir);
die "Cannot find \$top_srcdir in Makefile\n" if (! defined $top_srcdir);
print "srcdir $srcdir\n" if $opt{'t'};
print "top_srcdir $top_srcdir\n" if $opt{'t'};
close(MAKEFILE);


open(SPEED, ">speed-many.c") or die;
print SPEED
"/* speed-many.c generated by many.pl - DO NOT EDIT, CHANGES WILL BE LOST */

";
my $SPEED_EXTRA_ROUTINES = "#define SPEED_EXTRA_ROUTINES \\\n";
my $SPEED_EXTRA_PROTOS = "#define SPEED_EXTRA_PROTOS \\\n";
my $SPEED_CODE = "";

open(TRY, ">try-many.c") or die;
print TRY
    "/* try-many.c generated by many.pl - DO NOT EDIT, CHANGES WILL BE LOST */\n" .
    "\n";
my $TRY_EXTRA_ROUTINES = "#define EXTRA_ROUTINES \\\n";
my $TRY_EXTRA_PROTOS = "#define EXTRA_PROTOS \\\n";

open(FD,"<${top_builddir}/libtool") or die "Cannot open \"${top_builddir}/libtool\": $!\n";
my $pic_flag;
while (<FD>) {
  if (/^pic_flag="?([^"]*)"?$/) {
    $pic_flag=$1;
    last;
  }
}
close FD;
if (! defined $pic_flag) {
  die "Cannot find pic_flag in ${top_builddir}/libtool";
}

my $CFLAGS_PIC = $pic_flag;

my $ASMFLAGS_PIC = "";
foreach (split /[ \t]/, $pic_flag) {
  if (/^-D/) {
    $ASMFLAGS_PIC .= " " . $_;
  }
}

open(MAKEFILE, ">Makefile.many") or die;
print MAKEFILE
    "# Makefile.many generated by many.pl - DO NOT EDIT, CHANGES WILL BE LOST\n" .
    "\n" .
    "all: speed-many try-many\n" .
    "\n" .
    "#--------- begin included copy of basic Makefile ----------\n" .
    "\n";
open(FD,"<${builddir}/Makefile") or die "Cannot open \"${builddir}/Makefile\": $!\n";
print MAKEFILE <FD>;
close FD;
print MAKEFILE
    "\n" .
    "#--------- end included copy of basic Makefile ----------\n" .
    "\n" .
    "CFLAGS_PIC = $CFLAGS_PIC\n" .
    "ASMFLAGS_PIC = $ASMFLAGS_PIC\n" .
    "\n";

my $CLEAN="";
my $MANY_OBJS="";


sub print_ansi2knr {
  my ($base,$file,$includes) = @_;
  if (! defined $file)     { $file = "$base.c"; }
  if (! defined $includes) { $includes = ""; }

  print MAKEFILE <<EOF;
${base}_.c: $file \$(ANSI2KNR)
	\$(CPP) \$(DEFS) \$(INCLUDES) $includes \$(AM_CPPFLAGS) \$(CPPFLAGS) $file | sed 's/^# \([0-9]\)/#line \\1/' | \$(ANSI2KNR) >${base}_.c

EOF
}


# Spawning a glob is a touch slow when there's lots of files.
my @files = ();
foreach my $dir (@DIRECTORIES) {
  print "dir $dir\n" if $opt{'t'};
  if (-f $dir) {
    push @files,$dir;
  } else {
    if (! opendir DD,$dir) {
      print "Cannot open $dir: $!\n";
    } else {
      push @files, map {$_="$dir/$_"} grep /\.(c|asm|S|h)$/, readdir DD;
      closedir DD;
    }
  }
}
@files = sort @files;
print "@files ",join(" ",@files),"\n" if $opt{'t'};

my $count_files = 0;
my $count_functions = 0;
my %seen_obj;
my %seen_file;

foreach my $file_full (@files) {
  if (! -f $file_full) {
    print "Not a file: $file_full\n";
    next;
  }
  if (defined $seen_file{$file_full}) {
    print "Skipping duplicate file: $file_full\n";
    next;
  }
  $seen_file{$file_full} = 1;

  my ($FILE,$path,$lang) = fileparse($file_full,"\.[a-zA-Z]+");
  $path =~ s/\/$//;
  print "file $FILE path $path lang $lang\n" if $opt{'t'};

  my @pic_choices;
  if ($lang eq '.asm')  { @pic_choices=('no','yes'); }
  elsif ($lang eq '.c') { @pic_choices=('no'); }
  elsif ($lang eq '.S') { @pic_choices=('no','yes'); }
  elsif ($lang eq '.h') { @pic_choices=('no'); }
  else { next };

  my ($t, $file_match);
  foreach my $p (@table) {
    # print " ",$p->{'regexp'},"\n" if $opt{'t'};
    if ($FILE =~ "^($p->{'regexp'})") {
      $t = $p;
      $file_match = $1;
      $file_match = $2 if defined $2;
      last;
    }
  }
  next if ! defined $t;
  print "match $t->{'regexp'} $FILE ($file_full)\n" if $opt{'t'};

  if (! open FD,"<$file_full") { print "Can't open $file_full: $!\n"; next }
  my @file_contents = <FD>;
  close FD;

  my $objs;
  if (defined $t->{'mulfunc'}) { $objs = $t->{'mulfunc'}; }
  else                         { $objs = [$file_match]; }
  print "objs @$objs\n" if $opt{'t'};

  my $ret = $t->{'ret'};
  if (! defined $ret && $lang eq '.h') { $ret = ''; }
  if (! defined $ret) { die "$FILE return type not defined\n" };
  print "ret $ret\n" if $opt{'t'};

  my $mpX = $t->{'mpX'};
  if (! defined $mpX) { $mpX = ($lang eq '.h' ? '' : 'mpn'); }
  $mpX = "${mpX}_" if $mpX ne '';
  print "mpX $mpX\n" if $opt{'t'};

  my $carrys;
  if (defined $t->{'carrys'}) { $carrys = $t->{'carrys'}; }
  else                        { $carrys = ['','c'];       }
  print "carrys $carrys @$carrys\n" if $opt{'t'};

  # some restriction functions are implemented, but they're not very useful
  my $restriction='';

  my $suffix;
  if ($FILE =~ ("${file_match}_(.+)")) {
    $suffix = $1;
  } elsif ($path =~ /\/mp[zn]\/(.*)$/) {
    # derive the suffix from the path
    $suffix = $1;
    $suffix =~ s/\//_/g;
    # use last directory name, or if there's 3 or more then the last two
    if ($suffix =~ /([^_]*_)+([^_]+_[^_]+)$/) {
      $suffix = $2;
    } elsif ($suffix =~ /([^_]*_)*([^_]+)$/) {
      $suffix = $2;
    }
  } else {
    die "Can't determine suffix for: $file_full (path $path)\n";
  }
  print "suffix $suffix\n" if $opt{'t'};

  $count_files++;

  foreach my $obj (@{$objs}) {
    print "obj $obj\n" if $opt{'t'};

    my $obj_with_suffix = "${obj}_$suffix";
    if (defined $seen_obj{$obj_with_suffix}) {
      print "Skipping duplicate object: $obj_with_suffix\n";
      print "   first from: $seen_obj{$obj_with_suffix}\n";
      print "   now from:   $file_full\n";
      next;
    }
    $seen_obj{$obj_with_suffix} = $file_full;

    my $funs = $t->{'funs'};
    $funs = [$obj] if ! defined $funs;
    print "funs @$funs\n" if $opt{'t'};

    if (defined $t->{'pic'}) { @pic_choices = ('no'); }

    foreach my $pic (map {$pictable{$_}} @pic_choices) {
      print "pic $pic->{'suffix'}\n" if $opt{'t'};

      my $objbase = "${obj}_$suffix$pic->{'suffix'}";
      print "objbase $objbase\n" if $opt{'t'};

      if ($path !~ "." && -f "${objbase}.c") {
	die "Already have ${objbase}.c";
      }

      my $tmp_file = "tmp-$objbase.c";

      my $renaming;
      foreach my $fun (@{$funs}) {
        if ($mpX eq 'mpn_' && $lang eq '.c') {
          $renaming .= "\t\t-DHAVE_NATIVE_mpn_$fun=1 \\\n";
        }

        # The carry-in variant is with a "c" appended, unless there's a "_1"
        # somewhere, eg. "modexact_1_odd", in which case that becomes "_1c".
	my $fun_carry = $fun;
	if (! ($fun_carry =~ s/_1/_1c/)) { $fun_carry = "${fun}c"; }

	$renaming .=
	    "\t\t-D__g$mpX$fun=$mpX${fun}_$suffix$pic->{'suffix'} \\\n" .
	    "\t\t-D__g$mpX$fun_carry=$mpX${fun_carry}_$suffix$pic->{'suffix'} \\\n";
      }
      foreach my $r (@{$t->{'rename'}}) {
	if ($r =~ /^__gmp/) {
	  $renaming .= "\\\n" .
	      "\t\t-D$r=${r}_$suffix$pic->{'suffix'}";
	} else {
	  $renaming .= "\\\n" .
	      "\t\t-D__g$mpX$r=$mpX${r}_$suffix$pic->{'suffix'}";
	}
      }
      print "renaming $renaming\n" if $opt{'t'};

      print MAKEFILE "\n";
      if ($lang eq '.asm') {
	print MAKEFILE
	    "$objbase.o: $file_full \$(ASM_HEADERS)\n" .
	    "	\$(M4) \$(M4FLAGS) -DOPERATION_$obj $pic->{'asmflags'} \\\n" .
  	    "$renaming" .
	    "		$file_full >tmp-$objbase.s\n" .
            "	\$(CCAS) \$(COMPILE_FLAGS) $pic->{'cflags'} tmp-$objbase.s -o $objbase.o\n" .
            "	\$(RM_TMP) tmp-$objbase.s\n";
	$MANY_OBJS .= " $objbase.o";

      } elsif ($lang eq '.c') {
	print MAKEFILE
	    "$objbase.o: $file_full\n" .
	    "	\$(COMPILE) -DOPERATION_$obj $pic->{'cflags'} \\\n" .
  	    "$renaming" .
	    "		-c $file_full -o $objbase.o\n";
	print_ansi2knr($objbase,
		       $file_full,
		       " -DOPERATION_$obj\\\n$renaming\t\t");
	$MANY_OBJS .= " $objbase\$U.o";

      } elsif ($lang eq '.S') {
	print MAKEFILE
	    "$objbase.o: $file_full\n" .
            "	\$(COMPILE) -g $pic->{'asmflags'} \\\n" .
  	    "$renaming" .
            "	-c $file_full -o $objbase.o\n";
	$MANY_OBJS .= " $objbase.o";

      } elsif ($lang eq '.h') {
	print MAKEFILE
	    "$objbase.o: tmp-$objbase.c $file_full\n" .
	    "	\$(COMPILE) -DOPERATION_$obj $pic->{'cflags'} \\\n" .
  	    "$renaming" .
	    "		-c tmp-$objbase.c -o $objbase.o\n";
	print_ansi2knr($objbase,
		       "tmp-$objbase.c",
		       " -DOPERATION_$obj\\\n$renaming\t\t");
	$MANY_OBJS .= " $objbase\$U.o";

        $CLEAN .= " tmp-$objbase.c";
	open(TMP_C,">tmp-$objbase.c")
	    or die "Can't create tmp-$objbase.c: $!\n";
	print TMP_C
"/* tmp-$objbase.c generated by many.pl - DO NOT EDIT, CHANGES WILL BE LOST */

#include \"gmp.h\"
#include \"gmp-impl.h\"
#include \"longlong.h\"
#include \"speed.h\"

";
      }

      my $tests_program = "$top_srcdir/tests/devel/$obj.c";
      if (-f $tests_program) {
	$tests_program = "\$(top_srcdir)/tests/devel/$obj.c";
	print_ansi2knr("tests_${objbase}",
		       $tests_program,
		       "\\\n$renaming\t\t\$(CFLAGS_TESTS_SP)");
	print_ansi2knr("tests_${objbase}_sp",
		       $tests_program,
		       "\\\n$renaming\t\t\$(CFLAGS_TESTS_SP)");

	print MAKEFILE <<EOF;
tests_$objbase.o: $tests_program
	\$(COMPILE) \$(CFLAGS_TESTS) \\
$renaming		-c $tests_program -o tests_$objbase.o

tests_$objbase: $objbase\$U.o tests_$objbase\$U.o ../libgmp.la
	\$(LINK) tests_$objbase\$U.o $objbase\$U.o ../libgmp.la -o tests_$objbase

tests_${objbase}_sp.o: $tests_program
	\$(COMPILE) \$(CFLAGS_TESTS_SP) \\
$renaming		-c $tests_program -o tests_${objbase}_sp.o

tests_${objbase}_sp: $objbase\$U.o tests_${objbase}_sp\$U.o ../libgmp.la
	\$(LINK) tests_${objbase}_sp\$U.o $objbase\$U.o ../libgmp.la -o tests_${objbase}_sp

EOF
        $CLEAN .= " tests_$objbase tests_${objbase}_sp";
      }

      foreach my $fun (@{$funs}) {
	print "fun $fun\n" if $opt{'t'};

	if ($lang eq '.h') {
          my $macro_before = $t->{'macro_before'};
          $macro_before = "" if ! defined $macro_before;
	  print TMP_C
"$macro_before
#undef $fun
#include \"$file_full\"

";
	}

	my $args = $t->{"args_$fun"};
	if (! defined $args) { $args = $t->{'args'}; }
	if (! defined $args) { die "Need args for $fun\n"; }
	print "args $args\n" if $opt{'t'};

	foreach my $carry (@$carrys) {
	  print "carry $carry\n" if $opt{'t'};

	  my $fun_carry = $fun;
	  if (! ($fun_carry =~ s/_1/_1$carry/)) { $fun_carry = "$fun$carry"; }
          print "fun_carry $fun_carry\n" if $opt{'t'};

	  if ($lang =~ /\.(asm|S)/
	      && ! grep(m"PROLOGUE\((.* )?$mpX$fun_carry[ ,)]",@file_contents)) {
	    print "no PROLOGUE $mpX$fun_carry\n" if $opt{'t'};
	    next;
	  }
	  if ($lang eq '.c'
	      && ! grep(m"^(#define FUNCTION\s+)?$mpX$fun_carry\W", @file_contents)) {
	    print "no mention of $mpX$fun_carry\n" if $opt{'t'};
	    next;
	  }
	  if ($lang eq '.h'
	      && ! grep(m"^#define $fun_carry\W", @file_contents)) {
	    print "no mention of #define $fun_carry\n" if $opt{'t'};
	    next;
	  }

	  $count_functions++;

	  my $carryarg;
	  if (defined $t->{'carryarg'}) { $carryarg = $t->{'carryarg'}; }
	  if ($carry eq '')             { $carryarg = ''; }
	  else                          { $carryarg = ', mp_limb_t carry'; }
	  print "carryarg $carryarg\n" if $opt{'t'};

	  my $funfull="$mpX${fun_carry}_$suffix$pic->{'suffix'}";
	  print "funfull $funfull\n" if $opt{'t'};

	  if ($lang ne '.h') {
	    my $proto = "$t->{'ret'} $funfull _PROTO (($args$carryarg)); \\\n";
	    $SPEED_EXTRA_PROTOS .= $proto;
	    $TRY_EXTRA_PROTOS .= $proto;
	  }

	  my $try_type = $t->{"try-$fun"};
	  $try_type = $t->{'try'} if ! defined $try_type;
	  if (! defined $try_type) {
	    if ($mpX eq 'mpn_') {
	      $try_type = "TYPE_\U$fun_carry";
	    } else {
	      $try_type = "TYPE_\U$mpX\U$fun_carry";
	    }
	  }
	  print "try_type $try_type\n" if $opt{'t'};

	  my $try_minsize = $t->{'try-minsize'};
	  if (defined $try_minsize) {
	    $try_minsize = ", " . $try_minsize;
	  } else {
	    $try_minsize = "";
	  }
	  print "try_minsize $try_minsize\n" if $opt{'t'};

	  if ($try_type ne 'none') {
	    $TRY_EXTRA_ROUTINES .=
		"  { TRY($mpX${fun_carry}_$suffix$pic->{'suffix'}), $try_type$try_minsize }, \\\n";
	  }

	  my $speed_flags = $t->{'speed_flags'};
	  $speed_flags = '0' if ! defined $speed_flags;
	  print "speed_flags $speed_flags\n" if $opt{'t'};

	  my $speed_routine = $t->{'speed'};
	  $speed_routine = "SPEED_ROUTINE_\U$mpX\U$fun"
	      if !defined $speed_routine;
	  if (! ($speed_routine =~ s/_1/_1\U$carry/)) {
	    $speed_routine = "$speed_routine\U$carry";
	  }
	  print "speed_routine $speed_routine\n" if $opt{'t'};

	  my @speed_suffixes = ();
	  push (@speed_suffixes, '') if $speed_routine ne 'none';
	  push (@speed_suffixes, @{$t->{'speed_suffixes'}})
	      if defined $t->{'speed_suffixes'};

          my $macro_speed = $t->{'macro-speed'};
          $macro_speed = "$speed_routine ($fun_carry)" if ! defined $macro_speed;
          $macro_speed =~ s/\$fun/$fun_carry/g;

	  foreach my $S (@speed_suffixes) {
	    my $Sfunfull="$mpX${fun_carry}${S}_$suffix$pic->{'suffix'}";

	    $SPEED_EXTRA_PROTOS .=
	      "double speed_$Sfunfull _PROTO ((struct speed_params *s)); \\\n";
	    $SPEED_EXTRA_ROUTINES .=
	      "  { \"$Sfunfull\", speed_$Sfunfull, $speed_flags }, \\\n";
	    if ($lang eq '.h') {
              print TMP_C
"double
speed_$Sfunfull (struct speed_params *s)
{
$macro_speed
}

";
            } else {
	      $SPEED_CODE .=
	        "double\n" .
	        "speed_$Sfunfull (struct speed_params *s)\n" .
                "{\n" .
                "$restriction" .
	        "  $speed_routine\U$S\E ($funfull)\n" .
                "}\n";
            }
	  }
	}
      }
    }
  }
}


print SPEED $SPEED_EXTRA_PROTOS . "\n";
print SPEED $SPEED_EXTRA_ROUTINES . "\n";
if (defined $ENV{speedinc}) { print SPEED $ENV{speedinc} . "\n"; }
print SPEED
    "#include \"speed.c\"\n" .
    "\n";
print SPEED $SPEED_CODE;

print TRY $TRY_EXTRA_ROUTINES . "\n";
print TRY $TRY_EXTRA_PROTOS . "\n";
my $tryinc = "";
if (defined $ENV{tryinc}) {
  $tryinc = $ENV{tryinc};
  print TRY "#include \"$tryinc\"\n";
}
print "tryinc $tryinc\n" if $opt{'t'};
print TRY
    "#include \"try.c\"\n" .
    "\n";

my $extra_libraries = "";
if (defined $ENV{extra_libraries}) { $extra_libraries = $ENV{extra_libraries};}

my $trydeps = "";
if (defined $ENV{trydeps}) { $trydeps = $ENV{trydeps}; }
$trydeps .= " $tryinc";
print "trydeps $trydeps\n" if $opt{'t'};

print MAKEFILE <<EOF;

MANY_OBJS = $MANY_OBJS
MANY_CLEAN = \$(MANY_OBJS) \\
	speed-many.c speed-many\$U.o speed-many\$(EXEEXT) \\
	try-many.c try-many\$U.o try-many \\
	$CLEAN
MANY_DISTCLEAN = Makefile.many

speed-many: \$(MANY_OBJS) speed-many\$U.o libspeed.la $extra_libraries
	\$(LINK) \$(LDFLAGS) speed-many\$U.o \$(MANY_OBJS) \$(LDADD) \$(LIBS) $extra_libraries

try-many: \$(MANY_OBJS) try-many\$U.o libspeed.la $extra_libraries
	\$(LINK) \$(LDFLAGS) try-many\$U.o \$(MANY_OBJS)  \$(LDADD) \$(LIBS) $extra_libraries

try-many.o: try-many.c \$(top_srcdir)/tests/devel/try.c $trydeps
	\$(COMPILE) -I\$(top_srcdir)/tests/devel -c try-many.c

EOF

print_ansi2knr("speed-many");
print_ansi2knr("try-many",
	       "\$(top_srcdir)/tests/devel/try.c",
	       "-I\$(top_srcdir)/tests/devel");

print MAKEFILE <<EOF;
RM_TMP = rm -f
CFLAGS_TESTS = -DSIZE=50 -DTIMES=1 -DRANDOM -DCLOCK=333000000
CFLAGS_TESTS_SP = -DSIZE=1024 -DNOCHECK -DOPS=200000000 -DCLOCK=333000000
EOF

close MAKEFILE or die;

print "Total $count_files files, $count_functions functions\n";



# Local variables:
# perl-indent-level: 2
# End:
