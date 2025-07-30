dnl  AMD64 mpn_addmul_1 and mpn_submul_1 optimised for AMD Bulldozer.

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
C AMD K8,K9
C AMD K10
C AMD bd1	 4.5-4.7
C AMD bobcat
C Intel P4
C Intel core2
C Intel NHM
C Intel SBR
C Intel atom
C VIA nano

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * Try to make loop run closer to 4 c/l.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0',      `%rcx')   C r9

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
IFDOS(`	define(`v0', ``%r9'')	') dnl
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
IFSTD(`	mov	n_param, %rbx	')	C move away n from rdx, mul uses it
IFDOS(`	mov	n, %rbx		')
	mul	v0

IFSTD(`	mov	%rbx, n		')

	and	$3, R32(%rbx)
	lea	-16(rp,n,8), rp
	jz	L(b0)
	cmp	$2, R32(%rbx)
	jb	L(b1)
	jz	L(b2)

L(b3):	mov	$0, R32(%r8)
	mov	%rax, %rbx
	mov	$0, R32(%r9)
	mov	8(up), %rax
	mov	%rdx, %r10
	lea	(up,n,8), up
	not	n
	jmp	L(L3)

L(b0):	mov	$0, R32(%r10)
	mov	%rax, %r8
	mov	%rdx, %rbx
	mov	8(up), %rax
	lea	(up,n,8), up
	neg	n
	jmp	L(L0)

L(b1):	cmp	$1, n
	jz	L(n1)
	mov	%rax, %r9
	mov	8(up), %rax
	mov	%rdx, %r8
	mov	$0, R32(%rbx)
	lea	(up,n,8), up
	neg	n
	inc	n
	jmp	L(L1)

L(b2):	mov	$0, R32(%rbx)
	mov	%rax, %r10
	mov	%rdx, %r9
	mov	8(up), %rax
	mov	$0, R32(%r8)
	lea	(up,n,8), up
	neg	n
	add	$2, n
	jns	L(end)

	ALIGN(32)
L(top):	mul	v0
	ADDSUB	%r10, (rp,n,8)
	adc	%rax, %r9
	mov	(up,n,8), %rax
	adc	%rdx, %r8
L(L1):	mul	v0
	mov	$0, R32(%r10)
	ADDSUB	%r9, 8(rp,n,8)
	adc	%rax, %r8
	adc	%rdx, %rbx
	mov	8(up,n,8), %rax
L(L0):	mul	v0
	ADDSUB	%r8, 16(rp,n,8)
	mov	$0, R32(%r8)
	adc	%rax, %rbx
	mov	$0, R32(%r9)
	mov	16(up,n,8), %rax
	adc	%rdx, %r10
L(L3):	mul	v0
	ADDSUB	%rbx, 24(rp,n,8)
	mov	$0, R32(%rbx)
	adc	%rax, %r10
	adc	%rdx, %r9
	mov	24(up,n,8), %rax
	add	$4, n
	js	L(top)

L(end):	mul	v0
	ADDSUB	%r10, (rp)
	adc	%r9, %rax
	adc	%r8, %rdx
L(n1):	ADDSUB	%rax, 8(rp)
	adc	$0, %rdx
	mov	%rdx, %rax

	pop	%rbx
IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret
EPILOGUE()
ASM_END()
