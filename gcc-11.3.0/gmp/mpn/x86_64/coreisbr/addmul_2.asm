dnl  AMD64 mpn_addmul_2 optimised for Intel Sandy Bridge.

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
C Intel SBR	 2.93		this
C Intel IBR	 2.66		this
C Intel HWL	 2.5		 2.15
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
define(`w0',      `%r8')
define(`w1',      `%r9')
define(`w2',      `%r10')
define(`w3',      `%r11')
define(`X0',      `%r12')
define(`X1',      `%r13')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_addmul_2)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13

	mov	(vp), v0
	mov	8(vp), v1

	mov	(up), %rax

	mov	n_param, n
	neg	n

	lea	(up,n_param,8), up
	lea	8(rp,n_param,8), rp
	mul	v0

	test	$1, R8(n)
	jnz	L(bx1)

L(bx0):	mov	-8(rp,n,8), X0
	mov	%rdx, w1
	add	%rax, X0
	adc	$0, w1
	mov	(up,n,8), %rax
	xor	w0, w0
	xor	w3, w3
	test	$2, R8(n)
	jnz	L(b10)

L(b00):	nop				C this nop make loop go faster on SBR!
	mul	v1
	mov	(rp,n,8), X1
	jmp	L(lo0)

L(b10):	lea	-2(n), n
	jmp	L(lo2)

L(bx1):	mov	-8(rp,n,8), X1
	mov	%rdx, w3
	add	%rax, X1
	adc	$0, w3
	mov	(up,n,8), %rax
	xor	w1, w1
	xor	w2, w2
	test	$2, R8(n)
	jz	L(b11)

L(b01):	mov	(rp,n,8), X0
	inc	n
	jmp	L(lo1)

L(b11):	dec	n
	jmp	L(lo3)

	ALIGN(32)
L(top):
L(lo1):	mul	v1
	mov	%rdx, w0		C 1
	add	%rax, X0		C 0
	adc	$0, w0			C 1
	add	w1, X1			C 3
	adc	$0, w3			C 0
	add	w2, X0			C 0
	adc	$0, w0			C 1
	mov	(up,n,8), %rax
	mul	v0
	add	%rax, X0		C 0
	mov	%rdx, w1		C 1
	adc	$0, w1			C 1
	mov	(up,n,8), %rax
	mul	v1
	mov	X1, -16(rp,n,8)		C 3
	mov	(rp,n,8), X1		C 1
	add	w3, X0			C 0
	adc	$0, w1			C 1
L(lo0):	mov	%rdx, w2		C 2
	mov	X0, -8(rp,n,8)		C 0
	add	%rax, X1		C 1
	adc	$0, w2			C 2
	mov	8(up,n,8), %rax
	add	w0, X1			C 1
	adc	$0, w2			C 2
	mul	v0
	add	%rax, X1		C 1
	mov	%rdx, w3		C 2
	adc	$0, w3			C 2
	mov	8(up,n,8), %rax
L(lo3):	mul	v1
	add	w1, X1			C 1
	mov	8(rp,n,8), X0		C 2
	adc	$0, w3			C 2
	mov	%rdx, w0		C 3
	add	%rax, X0		C 2
	adc	$0, w0			C 3
	mov	16(up,n,8), %rax
	mul	v0
	add	w2, X0			C 2
	mov	X1, (rp,n,8)		C 1
	mov	%rdx, w1		C 3
	adc	$0, w0			C 3
	add	%rax, X0		C 2
	adc	$0, w1			C 3
	mov	16(up,n,8), %rax
	add	w3, X0			C 2
	adc	$0, w1			C 3
L(lo2):	mul	v1
	mov	16(rp,n,8), X1		C 3
	add	%rax, X1		C 3
	mov	%rdx, w2		C 4
	adc	$0, w2			C 4
	mov	24(up,n,8), %rax
	mov	X0, 8(rp,n,8)		C 2
	mul	v0
	add	w0, X1			C 3
	mov	%rdx, w3		C 4
	adc	$0, w2			C 4
	add	%rax, X1		C 3
	mov	24(up,n,8), %rax
	mov	24(rp,n,8), X0		C 0	useless but harmless final read
	adc	$0, w3			C 4
	add	$4, n
	jnc	L(top)

L(end):	mul	v1
	add	w1, X1
	adc	$0, w3
	add	w2, %rax
	adc	$0, %rdx
	mov	X1, I(-16(rp),-16(rp,n,8))
	add	w3, %rax
	adc	$0, %rdx
	mov	%rax, I(-8(rp),-8(rp,n,8))
	mov	%rdx, %rax

	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
