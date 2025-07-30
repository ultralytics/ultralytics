dnl  AMD64 mpn_mul_2 -- Multiply an n-limb vector with a 2-limb vector and
dnl  store the result in a third limb vector.

dnl  Copyright 2008, 2011, 2012 Free Software Foundation, Inc.

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

C	     cycles/limb
C AMD K8,K9	 2.275
C AMD K10	 2.275
C Intel P4	13.5
C Intel core2	 4.0
C Intel corei	 3.8
C Intel atom	 ?
C VIA nano	 ?

C This code is the result of running a code generation and optimization tool
C suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * Work on feed-in and wind-down code.
C  * Convert "mov $0" to "xor".
C  * Adjust initial lea to save some bytes.
C  * Perhaps adjust n from n_param&3 value?
C  * Replace with 2.25 c/l sequence.

C INPUT PARAMETERS
define(`rp',	 `%rdi')
define(`up',	 `%rsi')
define(`n_param',`%rdx')
define(`vp',	 `%rcx')

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
PROLOGUE(mpn_mul_2)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp

	mov	(vp), v0
	mov	8(vp), v1

	mov	(up), %rax

	mov	n_param, n
	neg	n
	lea	-8(up,n_param,8), up
	lea	-8(rp,n_param,8), rp

	and	$3, R32(n_param)
	jz	L(m2p0)
	cmp	$2, R32(n_param)
	jc	L(m2p1)
	jz	L(m2p2)
L(m2p3):
	mul	v0
	xor	R32(w3), R32(w3)
	mov	%rax, w1
	mov	%rdx, w2
	mov	8(up,n,8), %rax
	add	$-1, n
	mul	v1
	add	%rax, w2
	jmp	L(m23)
L(m2p0):
	mul	v0
	xor	R32(w2), R32(w2)
	mov	%rax, w0
	mov	%rdx, w1
	jmp	L(m20)
L(m2p1):
	mul	v0
	xor	R32(w3), R32(w3)
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	add	$1, n
	jmp	L(m2top)
L(m2p2):
	mul	v0
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	mov	%rax, w2
	mov	%rdx, w3
	mov	8(up,n,8), %rax
	add	$-2, n
	jmp	L(m22)


	ALIGN(32)
L(m2top):
	add	%rax, w3
	adc	%rdx, w0
	mov	0(up,n,8), %rax
	adc	$0, R32(w1)
	mov	$0, R32(w2)
	mul	v1
	add	%rax, w0
	mov	w3, 0(rp,n,8)
	adc	%rdx, w1
	mov	8(up,n,8), %rax
	mul	v0
	add	%rax, w0
	adc	%rdx, w1
	adc	$0, R32(w2)
L(m20):	mov	8(up,n,8), %rax
	mul	v1
	add	%rax, w1
	adc	%rdx, w2
	mov	16(up,n,8), %rax
	mov	$0, R32(w3)
	mul	v0
	add	%rax, w1
	mov	16(up,n,8), %rax
	adc	%rdx, w2
	adc	$0, R32(w3)
	mul	v1
	add	%rax, w2
	mov	w0, 8(rp,n,8)
L(m23):	adc	%rdx, w3
	mov	24(up,n,8), %rax
	mul	v0
	mov	$0, R32(w0)
	add	%rax, w2
	adc	%rdx, w3
	mov	w1, 16(rp,n,8)
	mov	24(up,n,8), %rax
	mov	$0, R32(w1)
	adc	$0, R32(w0)
L(m22):	mul	v1
	add	%rax, w3
	mov	w2, 24(rp,n,8)
	adc	%rdx, w0
	mov	32(up,n,8), %rax
	mul	v0
	add	$4, n
	js	L(m2top)


	add	%rax, w3
	adc	%rdx, w0
	adc	$0, R32(w1)
	mov	(up), %rax
	mul	v1
	mov	w3, (rp)
	add	%rax, w0
	adc	%rdx, w1
	mov	w0, 8(rp)
	mov	w1, %rax

	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
