dnl  AMD64 mpn_mul_1 optimised for AMD Bulldozer.

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

C	     cycles/limb
C AMD K8,K9
C AMD K10
C AMD bd1	 4
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
C  * Move loop code into feed-in blocks, to save insn for zeroing regs.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0',      `%rcx')   C r9

define(`n',       `%rbx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

IFDOS(`	define(`up', ``%rsi'')	') dnl
IFDOS(`	define(`rp', ``%rcx'')	') dnl
IFDOS(`	define(`v0', ``%r9'')	') dnl
IFDOS(`	define(`r9', ``rdi'')	') dnl
IFDOS(`	define(`n',  ``%r8'')	') dnl
IFDOS(`	define(`r8', ``rbx'')	') dnl

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_1c)
IFDOS(``push	%rsi		'')
IFDOS(``push	%rdi		'')
IFDOS(``mov	%rdx, %rsi	'')

	mov	(up), %rax		C read first u limb early
	push	%rbx
IFSTD(`	mov	n_param, %r11	')	C move away n from rdx, mul uses it
IFDOS(`	mov	n, %r11		')
	mul	v0

IFSTD(` add	%r8, %rax	')
IFDOS(` add	64(%rsp), %rax	')	C 40 + 3*8  (3 push insns)
	adc	$0, %rdx
	jmp	L(common)

EPILOGUE()

	ALIGN(16)
PROLOGUE(mpn_mul_1)
IFDOS(``push	%rsi		'')
IFDOS(``push	%rdi		'')
IFDOS(``mov	%rdx, %rsi	'')

	mov	(up), %rax		C read first u limb early
	push	%rbx
IFSTD(`	mov	n_param, %r11	')	C move away n from rdx, mul uses it
IFDOS(`	mov	n, %r11		')
	mul	v0

L(common):
IFSTD(`	mov	%r11, n		')

	and	$3, R32(%r11)
	lea	-16(rp,n,8), rp
	jz	L(b0)
	cmp	$2, R32(%r11)
	jb	L(b1)
	jz	L(b2)

L(b3):	mov	%rax, %r10
	mov	%rdx, %r11
	mov	8(up), %rax
	mul	v0
	lea	(up,n,8), up
	not	n
	jmp	L(L3)

L(b0):	mov	%rax, %r9
	mov	%rdx, %r10
	mov	8(up), %rax
	lea	(up,n,8), up
	neg	n
	jmp	L(L0)

L(b1):	mov	%rax, %r8
	cmp	$1, n
	jz	L(n1)
	mov	%rdx, %r9
	lea	(up,n,8), up
	neg	n
	mov	%r8, 16(rp,n,8)
	inc	n
	jmp	L(L1)

L(b2):	mov	%rax, %r11
	mov	%rdx, %r8
	mov	8(up), %rax
	lea	(up,n,8), up
	neg	n
	add	$2, n
	jns	L(end)

	ALIGN(16)
L(top):	mul	v0
	mov	%rdx, %r9
	add	%rax, %r8
	adc	$0, %r9
	mov	%r8, 8(rp,n,8)
	mov	%r11, (rp,n,8)
L(L1):	mov	(up,n,8), %rax
	mul	v0
	add	%rax, %r9
	mov	%rdx, %r10
	mov	8(up,n,8), %rax
	adc	$0, %r10
L(L0):	mul	v0
	add	%rax, %r10
	mov	%rdx, %r11
	mov	16(up,n,8), %rax
	adc	$0, %r11
	mul	v0
	mov	%r9, 16(rp,n,8)
L(L3):	add	%rax, %r11
	mov	%r10, 24(rp,n,8)
	mov	%rdx, %r8
	adc	$0, %r8
	add	$4, n
	mov	-8(up,n,8), %rax
	js	L(top)

L(end):	mul	v0
	add	%rax, %r8
	adc	$0, %rdx
	mov	%r11, (rp)
L(n1):	mov	%r8, 8(rp)
	mov	%rdx, %rax

	pop	%rbx
IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret
EPILOGUE()
ASM_END()
