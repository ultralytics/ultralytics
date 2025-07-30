dnl  AMD64 mpn_mul_1 optimised for AMD bobcat.

dnl  Copyright 2003-2005, 2007, 2008, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 4.5
C AMD K10	 4.5
C AMD bd1	 4.62
C AMD bobcat	 5
C Intel P4	14
C Intel core2	 4.5
C Intel NHM	 4.23
C Intel SBR	 3.0
C Intel atom	21
C VIA nano	 4.94

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

C Standard parameters
define(`rp',              `%rdi')
define(`up',              `%rsi')
define(`n_param',         `%rdx')
define(`v0',              `%rcx')
define(`cy',              `%r8')
C Standard allocations
define(`n',               `%rbx')
define(`w0',              `%r8')
define(`w1',              `%r9')
define(`w2',              `%r10')
define(`w3',              `%r11')

C DOS64 parameters
IFDOS(` define(`rp',      `%rcx')    ') dnl
IFDOS(` define(`up',      `%rsi')    ') dnl
IFDOS(` define(`n_param', `%r8')     ') dnl
IFDOS(` define(`v0',      `%r9')     ') dnl
IFDOS(` define(`cy',      `56(%rsp)')') dnl
C DOS64 allocations
IFDOS(` define(`n',       `%rbx')    ') dnl
IFDOS(` define(`w0',      `%r8')     ') dnl
IFDOS(` define(`w1',      `%rdi')    ') dnl
IFDOS(` define(`w2',      `%r10')    ') dnl
IFDOS(` define(`w3',      `%r11')    ') dnl

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_1c)
IFDOS(`	push	%rsi		')
IFDOS(`	push	%rdi		')
IFDOS(`	mov	%rdx, %rsi	')
	mov	cy, w2
	jmp	L(com)
EPILOGUE()

PROLOGUE(mpn_mul_1)
IFDOS(`	push	%rsi		')
IFDOS(`	push	%rdi		')
IFDOS(`	mov	%rdx, %rsi	')
	xor	w2, w2
L(com):	push	%rbx
	mov	(up), %rax

	lea	-16(rp,n_param,8), rp
	lea	-16(up,n_param,8), up

	mov	n_param, n
	and	$3, R32(n_param)
	jz	L(b0)
	cmp	$2, R32(n_param)
	ja	L(b3)
	jz	L(b2)

L(b1):	mul	v0
	cmp	$1, n
	jz	L(n1)
	neg	n
	add	$3, n
	add	%rax, w2
	mov	%rdx, w3
	jmp	L(L1)
L(n1):	add	%rax, w2
	mov	%rdx, %rax
	mov	w2, 8(rp)
	adc	$0, %rax
	pop	%rbx
IFDOS(`	pop	%rdi		')
IFDOS(`	pop	%rsi		')
	ret

L(b3):	mul	v0
	neg	n
	inc	n
	add	%rax, w2
	mov	%rdx, w3
	jmp	L(L3)

L(b0):	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	neg	n
	add	$2, n
	add	w2, w0
	jmp	L(L0)

L(b2):	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	neg	n
	add	w2, w0
	jmp	L(L2)

	ALIGN(16)
L(top):	mov	w0, -16(rp,n,8)
	add	w1, w2
L(L1):	adc	$0, w3
	mov	0(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, -8(rp,n,8)
	add	w3, w0
L(L0):	adc	$0, w1
	mov	8(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	mov	w0, 0(rp,n,8)
	add	w1, w2
L(L3):	adc	$0, w3
	mov	16(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, 8(rp,n,8)
	add	w3, w0
L(L2):	adc	$0, w1
	mov	24(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	$4, n
	js	L(top)

L(end):	mov	w0, (rp)
	add	w1, w2
	adc	$0, w3
	mov	w2, 8(rp)
	mov	w3, %rax

	pop	%rbx
IFDOS(`	pop	%rdi		')
IFDOS(`	pop	%rsi		')
	ret
EPILOGUE()
