dnl  X86-64 mpn_addmul_1 and mpn_submul_1 optimised for Intel Sandy Bridge.

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
C AMD bull
C AMD pile
C AMD steam
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core
C Intel NHM
C Intel SBR	 3.24 (average, fluctuating in 3.20-3.57)
C Intel IBR	 3.04
C Intel HWL
C Intel BWL
C Intel atom
C VIA nano

C The loop of this code is the result of running a code generation and
C optimization tool suite written by David Harvey and Torbjörn Granlund.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0',      `%rcx')   C r9

define(`n',       `%rbx')

define(`I',`$1')

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

IFDOS(`	define(`up',     ``%rsi'')') dnl
IFDOS(`	define(`rp',     ``%rcx'')') dnl
IFDOS(`	define(`v0',     ``%r9'')') dnl
IFDOS(`	define(`r9',     ``rdi'')') dnl
IFDOS(`	define(`n_param',``%r8'')') dnl

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(func)

IFDOS(``push	%rsi		'')
IFDOS(``push	%rdi		'')
IFDOS(``mov	%rdx, %rsi	'')

	mov	(up), %rax
	push	%rbx
	lea	(up,n_param,8), up
	lea	(rp,n_param,8), rp

	test	$1, R8(n_param)
	jnz	L(b13)

L(b02):	xor	R32(%r11), R32(%r11)
	test	$2, R8(n_param)
	jnz	L(b2)

L(b0):	mov	$1, R32(n)
	sub	n_param, n
	mul	v0
	mov	%rdx, %r9
	mov	-8(rp,n,8), %r8
	jmp	L(e0)

	ALIGN(16)
L(b2):	mov	$-1, n
	sub	n_param, n
	mul	v0
	mov	8(rp,n,8), %r8
	mov	%rdx, %r9
	jmp	L(e2)

	ALIGN(16)
L(b13):	xor	R32(%r9), R32(%r9)
	test	$2, R8(n_param)
	jnz	L(b3)

L(b1):	mov	$2, R32(n)
	sub	n_param, n
	jns	L(1)
	mul	v0
	mov	-16(rp,n,8), %r10
	mov	%rdx, %r11
	jmp	L(e1)

	ALIGN(16)
L(b3):	xor	R32(n), R32(n)
	sub	n_param, n
	mul	v0
	mov	(rp,n,8), %r10
	jmp	L(e3)

	ALIGN(32)
L(top):	mul	v0
	mov	-16(rp,n,8), %r10
	ADDSUB	%r11, %r8
	mov	%rdx, %r11
	adc	$0, %r9
	mov	%r8, -24(rp,n,8)
L(e1):	ADDSUB	%rax, %r10
	mov	-8(up,n,8), %rax
	adc	$0, %r11
	mul	v0
	ADDSUB	%r9, %r10
	mov	%rdx, %r9
	mov	-8(rp,n,8), %r8
	adc	$0, %r11
	mov	%r10, -16(rp,n,8)
L(e0):	ADDSUB	%rax, %r8
	adc	$0, %r9
	mov	(up,n,8), %rax
	mul	v0
	mov	(rp,n,8), %r10
	ADDSUB	%r11, %r8
	mov	%r8, -8(rp,n,8)
	adc	$0, %r9
L(e3):	mov	%rdx, %r11
	ADDSUB	%rax, %r10
	mov	8(up,n,8), %rax
	adc	$0, %r11
	mul	v0
	mov	8(rp,n,8), %r8
	ADDSUB	%r9, %r10
	mov	%rdx, %r9
	mov	%r10, (rp,n,8)
	adc	$0, %r11
L(e2):	ADDSUB	%rax, %r8
	adc	$0, %r9
	mov	16(up,n,8), %rax
	add	$4, n
	jnc	L(top)

L(end):	mul	v0
	mov	I(-8(rp),-16(rp,n,8)), %r10
	ADDSUB	%r11, %r8
	mov	%rdx, %r11
	adc	$0, %r9
	mov	%r8, I(-16(rp),-24(rp,n,8))
	ADDSUB	%rax, %r10
	adc	$0, %r11
	ADDSUB	%r9, %r10
	adc	$0, %r11
	mov	%r10, I(-8(rp),-16(rp,n,8))
	mov	%r11, %rax

	pop	%rbx
IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret

	ALIGN(16)
L(1):	mul	v0
	ADDSUB	%rax, -8(rp)
	mov	%rdx, %rax
	adc	$0, %rax
	pop	%rbx
IFDOS(``pop	%rdi		'')
IFDOS(``pop	%rsi		'')
	ret
EPILOGUE()
ASM_END()
