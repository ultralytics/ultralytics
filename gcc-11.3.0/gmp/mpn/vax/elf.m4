divert(-1)

dnl  m4 macros for VAX assembler.

dnl  Copyright 2001, 2012 Free Software Foundation, Inc.

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


defreg(r0,`%r``''0')
defreg(r1,`%r``''1')
defreg(r2,`%r``''2')
defreg(r3,`%r``''3')
defreg(r4,`%r``''4')
defreg(r5,`%r``''5')
defreg(r6,`%r``''6')
defreg(r7,`%r``''7')
defreg(r8,`%r``''8')
defreg(r9,`%r``''9')
defreg(r10,`%r``''10')
defreg(r11,`%r``''11')
defreg(r12,`%r``''12')
defreg(r13,`%r``''13')
defreg(r14,`%r``''14')
defreg(r15,`%r``''15')
defreg(ap,`%a``''p')

define(`foo', blablabla)

divert
