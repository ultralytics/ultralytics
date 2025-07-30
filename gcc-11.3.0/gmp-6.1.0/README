Copyright 1991, 1996, 1999, 2000, 2007 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.






			THE GNU MP LIBRARY


GNU MP is a library for arbitrary precision arithmetic, operating on signed
integers, rational numbers, and floating point numbers.  It has a rich set of
functions, and the functions have a regular interface.

GNU MP is designed to be as fast as possible, both for small operands and huge
operands.  The speed is achieved by using fullwords as the basic arithmetic
type, by using fast algorithms, with carefully optimized assembly code for the
most common inner loops for lots of CPUs, and by a general emphasis on speed
(instead of simplicity or elegance).

GNU MP is believed to be faster than any other similar library.  Its advantage
increases with operand sizes for certain operations, since GNU MP in many
cases has asymptotically faster algorithms.

GNU MP is free software and may be freely copied on the terms contained in the
files COPYING* (see the manual for information on which license(s) applies to
which components of GNU MP).



			OVERVIEW OF GNU MP

There are four classes of functions in GNU MP.

 1. Signed integer arithmetic functions (mpz).  These functions are intended
    to be easy to use, with their regular interface.  The associated type is
    `mpz_t'.

 2. Rational arithmetic functions (mpq).  For now, just a small set of
    functions necessary for basic rational arithmetics.  The associated type
    is `mpq_t'.

 3. Floating-point arithmetic functions (mpf).  If the C type `double'
    doesn't give enough precision for your application, declare your
    variables as `mpf_t' instead, set the precision to any number desired,
    and call the functions in the mpf class for the arithmetic operations.

 4. Positive-integer, hard-to-use, very low overhead functions are in the
    mpn class.  No memory management is performed.  The caller must ensure
    enough space is available for the results.  The set of functions is not
    regular, nor is the calling interface.  These functions accept input
    arguments in the form of pairs consisting of a pointer to the least
    significant word, and an integral size telling how many limbs (= words)
    the pointer points to.

    Almost all calculations, in the entire package, are made by calling these
    low-level functions.

For more information on how to use GNU MP, please refer to the documentation.
It is composed from the file doc/gmp.texi, and can be displayed on the screen
or printed.  How to do that, as well how to build the library, is described in
the INSTALL file in this directory.



			REPORTING BUGS

If you find a bug in the library, please make sure to tell us about it!

You should first check the GNU MP web pages at https://gmplib.org/, under
"Status of the current release".  There will be patches for all known serious
bugs there.

Report bugs to gmp-bugs@gmplib.org.  What information is needed in a useful bug
report is described in the manual.  The same address can be used for suggesting
modifications and enhancements.




----------------
Local variables:
mode: text
fill-column: 78
End:
