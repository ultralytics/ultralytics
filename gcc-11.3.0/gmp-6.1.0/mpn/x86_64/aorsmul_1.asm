dnl  AMD64 mpn_addmul_1 and mpn_submul_1.

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
C AMD K8,K9	 2.5
C AMD K10	 2.5
C AMD bd1	 5.0
C AMD bobcat	 6.17
C Intel P4	14.9
C Intel core2	 5.09
C Intel NHM	 4.9
C Intel SBR	 4.0
C Intel atom	21.3
C VIA nano	 5.0

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

ifdef(`OPERATION_addmul_1',`
      define(`ADDSUB',        `add')
      define(`func',  `mpn_addmul_1')
')
ifdef(`OPERATION_submul_1',`
      define(`ADDSUB',        `sub')
      define(`func',  `mpn_submul_1')
')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

IFDOS(`	define(`up', ``%rsi'')	') dnl
IFDOS(`	define(`rp', ``%rcx'')	') dnl
IFDOS(`	define(`vl', ``%r9'')	') dnl
IFDOS(`	define(`r9', ``rdi'')	') dnl
IFDOS(`	define(`n',  ``%r8'')	') dnl
IFDOS(`	define(`r8', ``r11'')	') dnl

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)

IFDOS(``push	%rsi		'')
IFDOS(``push	%rdi		'')
IFDOS(``mov	%rdx, %rsi	'')

	mov	(up), %rax		C read first u limb early
	push	%rbx
IFSTD(`	mov	n_param, %rbx   ')	C move away n from rdx, mul uses it
IFDOS(`	mov	n, %rbx         ')
	mul	vl
IFSTD(`	mov	%rbx, n         ')

	and	$3, R32(%rbx)
	jz	L(b0)
	cmp	$2, R32(%rbx)
	jz	L(b2)
	jg	L(b3)

L(b1):	dec	n
	jne	L(gt1)
	ADDSUB	%rax, (rp)
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
L(top):	ADDSUB	%r10, (rp,n,8)
	adc	%rax, %r9
	mov	(up,n,8), %rax
	adc	%rdx, %r8
	mov	$0, R32(%r10)
L(L1):	mul	vl
	ADDSUB	%r9, 8(rp,n,8)
	adc	%rax, %r8
	adc	%rdx, %rbx
L(L0):	mov	8(up,n,8), %rax
	mul	vl
	ADDSUB	%r8, 16(rp,n,8)
	adc	%rax, %rbx
	adc	%rdx, %r10
L(L3):	mov	16(up,n,8), %rax
	mul	vl
	ADDSUB	%rbx, 24(rp,n,8)
	mov	$0, R32(%r8)		C zero
	mov	%r8, %rbx		C zero
	adc	%rax, %r10
	mov	24(up,n,8), %rax
	mov	%r8, %r9		C zero
	adc	%rdx, %r9
L(L2):	mul	vl
	add	$4, n
	js	 L(top)

	ADDSUB	%r10, (rp,n,8)
	adc	%rax, %r9
	adc	%r8, %rdx
	ADDSUB	%r9, 8(rp,n,8)
L(ret):	adc	$0, %rdx
	mov	%rdx, %rax

	pop	%rbx
IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret
EPILOGUE()
