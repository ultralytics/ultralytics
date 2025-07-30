dnl  AMD64 mpn_mul_1.

dnl  Copyright 2003-2005, 2007, 2008, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.5
C AMD K10	 2.5
C AMD bd1	 5.0
C AMD bobcat	 5.5
C Intel P4	12.3
C Intel core2	 4.0
C Intel NHM	 3.75
C Intel SBR	 2.95
C Intel atom	19.8
C VIA nano	 4.25

C The loop of this code is the result of running a code generation and
C optimization tool suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * The loop is great, but the prologue and epilogue code was quickly written.
C    Tune it!

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`vl',      `%rcx')   C r9

define(`n',       `%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

IFDOS(`	define(`up', ``%rsi'')	') dnl
IFDOS(`	define(`rp', ``%rcx'')	') dnl
IFDOS(`	define(`vl', ``%r9'')	') dnl
IFDOS(`	define(`r9', ``rdi'')	') dnl
IFDOS(`	define(`n',  ``%r8'')	') dnl
IFDOS(`	define(`r8', ``r11'')	') dnl

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_1c)
IFDOS(``push	%rsi		'')
IFDOS(``push	%rdi		'')
IFDOS(``mov	%rdx, %rsi	'')
	push	%rbx
IFSTD(`	mov	%r8, %r10')
IFDOS(`	mov	64(%rsp), %r10')	C 40 + 3*8  (3 push insns)
	jmp	L(common)
EPILOGUE()

PROLOGUE(mpn_mul_1)
IFDOS(``push	%rsi		'')
IFDOS(``push	%rdi		'')
IFDOS(``mov	%rdx, %rsi	'')

	push	%rbx
	xor	%r10, %r10
L(common):
	mov	(up), %rax		C read first u limb early
IFSTD(`	mov	n_param, %rbx   ')	C move away n from rdx, mul uses it
IFDOS(`	mov	n, %rbx         ')
	mul	vl
IFSTD(`	mov	%rbx, n         ')

	add	%r10, %rax
	adc	$0, %rdx

	and	$3, R32(%rbx)
	jz	L(b0)
	cmp	$2, R32(%rbx)
	jz	L(b2)
	jg	L(b3)

L(b1):	dec	n
	jne	L(gt1)
	mov	%rax, (rp)
	jmp	L(ret)
L(gt1):	lea	8(up,n,8), up
	lea	-8(rp,n,8), rp
	neg	n
	xor	%r10, %r10
	xor	R32(%rbx), R32(%rbx)
	mov	%rax, %r9
	mov	(up,n,8), %rax
	mov	%rdx, %r8
	jmp	L(L1)

L(b0):	lea	(up,n,8), up
	lea	-16(rp,n,8), rp
	neg	n
	xor	%r10, %r10
	mov	%rax, %r8
	mov	%rdx, %rbx
	jmp	 L(L0)

L(b3):	lea	-8(up,n,8), up
	lea	-24(rp,n,8), rp
	neg	n
	mov	%rax, %rbx
	mov	%rdx, %r10
	jmp	L(L3)

L(b2):	lea	-16(up,n,8), up
	lea	-32(rp,n,8), rp
	neg	n
	xor	%r8, %r8
	xor	R32(%rbx), R32(%rbx)
	mov	%rax, %r10
	mov	24(up,n,8), %rax
	mov	%rdx, %r9
	jmp	L(L2)

	ALIGN(16)
L(top):	mov	%r10, (rp,n,8)
	add	%rax, %r9
	mov	(up,n,8), %rax
	adc	%rdx, %r8
	mov	$0, R32(%r10)
L(L1):	mul	vl
	mov	%r9, 8(rp,n,8)
	add	%rax, %r8
	adc	%rdx, %rbx
L(L0):	mov	8(up,n,8), %rax
	mul	vl
	mov	%r8, 16(rp,n,8)
	add	%rax, %rbx
	adc	%rdx, %r10
L(L3):	mov	16(up,n,8), %rax
	mul	vl
	mov	%rbx, 24(rp,n,8)
	mov	$0, R32(%r8)		C zero
	mov	%r8, %rbx		C zero
	add	%rax, %r10
	mov	24(up,n,8), %rax
	mov	%r8, %r9		C zero
	adc	%rdx, %r9
L(L2):	mul	vl
	add	$4, n
	js	 L(top)

	mov	%r10, (rp,n,8)
	add	%rax, %r9
	adc	%r8, %rdx
	mov	%r9, 8(rp,n,8)
	add	%r8, %rdx
L(ret):	mov	%rdx, %rax

	pop	%rbx
IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret
EPILOGUE()
