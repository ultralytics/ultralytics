dnl  AMD64 mpn_hamdist -- hamming distance.

dnl  Copyright 2008, 2010-2012 Free Software Foundation, Inc.

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

C		    cycles/limb
C AMD K8,K9		 n/a
C AMD K10		 2
C Intel P4		 n/a
C Intel core2		 n/a
C Intel corei		 2.05
C Intel atom		 n/a
C VIA nano		 n/a

C This is very straightforward 2-way unrolled code.

C TODO
C  * Write something less basic.  It should not be hard to reach 1.5 c/l with
C    4-way unrolling.

define(`ap',		`%rdi')
define(`bp',		`%rsi')
define(`n',		`%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_hamdist)
	FUNC_ENTRY(3)
	mov	(ap), %r8
	xor	(bp), %r8

	lea	(ap,n,8), ap			C point at A operand end
	lea	(bp,n,8), bp			C point at B operand end
	neg	n

	bt	$0, R32(n)
	jnc	L(2)

L(1):	.byte	0xf3,0x49,0x0f,0xb8,0xc0	C popcnt %r8, %rax
	xor	R32(%r10), R32(%r10)
	add	$1, n
	js	L(top)
	FUNC_EXIT()
	ret

	ALIGN(16)
L(2):	mov	8(ap,n,8), %r9
	.byte	0xf3,0x49,0x0f,0xb8,0xc0	C popcnt %r8, %rax
	xor	8(bp,n,8), %r9
	.byte	0xf3,0x4d,0x0f,0xb8,0xd1	C popcnt %r9, %r10
	add	$2, n
	js	L(top)
	lea	(%r10, %rax), %rax
	FUNC_EXIT()
	ret

	ALIGN(16)
L(top):	mov	(ap,n,8), %r8
	lea	(%r10, %rax), %rax
	mov	8(ap,n,8), %r9
	xor	(bp,n,8), %r8
	xor	8(bp,n,8), %r9
	.byte	0xf3,0x49,0x0f,0xb8,0xc8	C popcnt %r8, %rcx
	lea	(%rcx, %rax), %rax
	.byte	0xf3,0x4d,0x0f,0xb8,0xd1	C popcnt %r9, %r10
	add	$2, n
	js	L(top)

	lea	(%r10, %rax), %rax
	FUNC_EXIT()
	ret
EPILOGUE()
