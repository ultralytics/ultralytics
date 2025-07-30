dnl  x86 pentium time stamp counter access routine.

dnl  Copyright 1999, 2000, 2003-2005 Free Software Foundation, Inc.

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


include(`../config.m4')


C void speed_cyclecounter (unsigned p[2]);
C
C Get the pentium rdtsc cycle counter, storing the least significant word in
C p[0] and the most significant in p[1].
C
C cpuid is used to serialize execution.  On big measurements this won't be
C significant but it may help make small single measurements more accurate.

PROLOGUE(speed_cyclecounter)

	C rdi	p

	movq	%rbx, %r10
	xorl	%eax, %eax
	cpuid
	rdtsc
	movl	%eax, (%rdi)
	movl	%edx, 4(%rdi)
	movq	%r10, %rbx
	ret
EPILOGUE()
