dnl  AMD64 mpn_addmul_2 optimised for Intel Atom.

dnl  Copyright 2008, 2011-2013 Free Software Foundation, Inc.

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
C AMD bd1
C AMD bd2
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel PNR
C Intel NHM
C Intel SBR
C Intel IBR
C Intel HWL
C Intel BWL
C Intel atom	18.8		this
C VIA nano

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`vp',      `%rcx')   C r9

define(`v0', `%r8')
define(`v1', `%r9')
define(`w0', `%rbx')
define(`w1', `%rcx')
define(`w2', `%rbp')
define(`w3', `%r10')
define(`n',  `%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_addmul_2)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp

	mov	(up), %rax

	mov	(vp), v0
	mov	8(vp), v1

	mov	n_param, n
	mul	v0

	test	$1, R8(n)
	jnz	L(bx1)

L(bx0):	test	$2, R8(n)
	jnz	L(b10)

L(b00):	mov	%rax, w0
	mov	(up), %rax
	mov	%rdx, w1
	xor	R32(w2), R32(w2)
	lea	-8(rp), rp
	jmp	L(lo0)

L(b10):	mov	%rax, w2
	mov	(up), %rax
	mov	%rdx, w3
	xor	R32(w0), R32(w0)
	lea	-16(up), up
	lea	-24(rp), rp
	jmp	L(lo2)

L(bx1):	test	$2, R8(n)
	jnz	L(b11)

L(b01):	mov	%rax, w3
	mov	%rdx, w0
	mov	(up), %rax
	xor	R32(w1), R32(w1)
	lea	8(up), up
	dec	n
	jmp	L(lo1)

L(b11):	mov	%rax, w1
	mov	(up), %rax
	mov	%rdx, w2
	xor	R32(w3), R32(w3)
	lea	-8(up), up
	lea	-16(rp), rp
	jmp	L(lo3)

	ALIGN(16)
L(top):
L(lo1):	mul	v1
	add	w3, (rp)
	mov	$0, R32(w2)
	adc	%rax, w0
	mov	(up), %rax
	adc	%rdx, w1
	mul	v0
	add	%rax, w0
	mov	(up), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
L(lo0):	mul	v1
	add	w0, 8(rp)
	adc	%rax, w1
	mov	8(up), %rax
	mov	$0, R32(w3)
	adc	%rdx, w2
	mul	v0
	add	%rax, w1
	mov	8(up), %rax
	adc	%rdx, w2
	adc	$0, R32(w3)
L(lo3):	mul	v1
	add	w1, 16(rp)
	adc	%rax, w2
	mov	16(up), %rax
	mov	$0, R32(w0)
	adc	%rdx, w3
	mul	v0
	add	%rax, w2
	mov	16(up), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
L(lo2):	mul	v1
	add	w2, 24(rp)
	adc	%rax, w3
	mov	24(up), %rax
	adc	%rdx, w0
	mov	$0, R32(w1)
	lea	32(rp), rp
	mul	v0
	lea	32(up), up
	add	%rax, w3
	adc	%rdx, w0
	mov	-8(up), %rax
	adc	$0, R32(w1)
	sub	$4, n
	ja	L(top)

L(end):	mul	v1
	add	w3, (rp)
	adc	%rax, w0
	adc	%rdx, w1
	mov	w0, 8(rp)
	mov	w1, %rax
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
