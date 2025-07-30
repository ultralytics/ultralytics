dnl  X86-64 mpn_mul_1 optimised for Intel Sandy Bridge.

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

C	     cycles/limb
C AMD K8,K9
C AMD K10
C AMD bd1
C AMD bobcat
C Intel P4
C Intel core2
C Intel NHM
C Intel SBR	 2.5
C Intel IBR	 2.4
C Intel atom
C VIA nano

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * The loop is great, but the prologue code was quickly written.  Tune it!
C  * Add mul_1c entry point.
C  * We could preserve one less register under DOS64 calling conventions, using
C    r10 instead of rsi.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0',      `%rcx')   C r9

define(`n',	  `%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

IFDOS(`	define(`up',     ``%rsi'')') dnl
IFDOS(`	define(`rp',     ``%rcx'')') dnl
IFDOS(`	define(`v0',     ``%r9'')') dnl
IFDOS(`	define(`r9',     ``rdi'')') dnl
IFDOS(`	define(`n_param',``%r8'')') dnl
IFDOS(`	define(`n',      ``%r8'')') dnl
IFDOS(`	define(`r8',     ``r11'')') dnl

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_1)

IFDOS(``push	%rsi		'')
IFDOS(``push	%rdi		'')
IFDOS(``mov	%rdx, %rsi	'')

	mov	(up), %rax
	mov	R32(`n_param'), R32(%r10)
IFSTD(`	mov	n_param, n		')

	lea	(up,n_param,8), up
	lea	-8(rp,n_param,8), rp
	neg	n
	mul	v0
	and	$3, R32(%r10)
	jz	L(b0)
	cmp	$2, R32(%r10)
	jb	L(b1)
	jz	L(b2)

L(b3):	add	$-1, n
	mov	%rax, %r9
	mov	%rdx, %r8
	mov	16(up,n,8), %rax
	jmp	L(L3)

L(b1):	mov	%rax, %r9
	mov	%rdx, %r8
	add	$1, n
	jnc	L(L1)
	mov	%rax, (rp)
	mov	%rdx, %rax
IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret

L(b2):	add	$-2, n
	mov	%rax, %r8
	mov	%rdx, %r9
	mov	24(up,n,8), %rax
	jmp	L(L2)

L(b0):	mov	%rax, %r8
	mov	%rdx, %r9
	mov	8(up,n,8), %rax
	jmp	L(L0)

	ALIGN(8)
L(top):	mov	%rdx, %r8
	add	%rax, %r9
L(L1):	mov	0(up,n,8), %rax
	adc	$0, %r8
	mul	v0
	add	%rax, %r8
	mov	%r9, 0(rp,n,8)
	mov	8(up,n,8), %rax
	mov	%rdx, %r9
	adc	$0, %r9
L(L0):	mul	v0
	mov	%r8, 8(rp,n,8)
	add	%rax, %r9
	mov	%rdx, %r8
	mov	16(up,n,8), %rax
	adc	$0, %r8
L(L3):	mul	v0
	mov	%r9, 16(rp,n,8)
	mov	%rdx, %r9
	add	%rax, %r8
	mov	24(up,n,8), %rax
	adc	$0, %r9
L(L2):	mul	v0
	mov	%r8, 24(rp,n,8)
	add	$4, n
	jnc	L(top)

L(end):	add	%rax, %r9
	mov	%rdx, %rax
	adc	$0, %rax
	mov	%r9, (rp)

IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret
EPILOGUE()
