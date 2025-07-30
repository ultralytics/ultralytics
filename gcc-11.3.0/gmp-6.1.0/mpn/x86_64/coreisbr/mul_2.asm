dnl  AMD64 mpn_mul_2 optimised for Intel Sandy Bridge.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2003-2005, 2007, 2008, 2011-2013 Free Software Foundation, Inc.

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

C	     cycles/limb	best
C AMD K8,K9
C AMD K10
C AMD bull
C AMD pile
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core
C Intel NHM
C Intel SBR	 2.57		 2.52 using 4-way code
C Intel IBR	 2.35		 2.32 using 4-way code
C Intel HWL	 2.02		 1.86
C Intel BWL
C Intel atom
C VIA nano

C This code is the result of running a code generation and optimisation tool
C suite written by David Harvey and Torbjorn Granlund.

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`vp',      `%rcx')   C r9

define(`n',	  `%rcx')
define(`v0',      `%rbx')
define(`v1',      `%rbp')

define(`w0',	`%r8')
define(`w1',	`%r9')
define(`w2',	`%r10')
define(`w3',	`%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mul_2)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp

	mov	(vp), v0
	mov	8(vp), v1

	mov	(up), %rax
	lea	(up,n_param,8), up
	lea	(rp,n_param,8), rp

	test	$1, R8(n_param)
	jnz	L(b1)

L(b0):	mov	$0, R32(n)
	sub	n_param, n
	xor	w0, w0
	mul	v0
	mov	%rax, w2
	mov	%rdx, w1
	mov	(up,n,8), %rax
	jmp	L(lo0)

L(b1):	mov	$1, R32(n)
	sub	n_param, n
	xor	w2, w2
	mul	v0
	mov	%rax, w0
	mov	%rdx, w3
	mov	-8(up,n,8), %rax
	mul	v1
	jmp	L(lo1)

	ALIGN(32)
L(top):	mul	v0
	add	%rax, w0		C 1
	mov	%rdx, w3		C 2
	adc	$0, w3			C 2
	mov	-8(up,n,8), %rax
	mul	v1
	add	w1, w0			C 1
	adc	$0, w3			C 2
L(lo1):	add	%rax, w2		C 2
	mov	w0, -8(rp,n,8)		C 1
	mov	%rdx, w0		C 3
	adc	$0, w0			C 3
	mov	(up,n,8), %rax
	mul	v0
	add	%rax, w2		C 2
	mov	%rdx, w1		C 3
	adc	$0, w1			C 3
	add	w3, w2			C 2
	mov	(up,n,8), %rax
	adc	$0, w1			C 1
L(lo0):	mul	v1
	mov	w2, (rp,n,8)		C 2
	add	%rax, w0		C 3
	mov	%rdx, w2		C 4
	mov	8(up,n,8), %rax
	adc	$0, w2			C 4
	add	$2, n
	jnc	L(top)

L(end):	mul	v0
	add	%rax, w0
	mov	%rdx, w3
	adc	$0, w3
	mov	I(-8(up),-8(up,n,8)), %rax
	mul	v1
	add	w1, w0
	adc	$0, w3
	add	%rax, w2
	mov	w0, I(-8(rp),-8(rp,n,8))
	adc	$0, %rdx
	add	w3, w2
	mov	w2, I((rp),(rp,n,8))
	adc	$0, %rdx
	mov	%rdx, %rax

	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
