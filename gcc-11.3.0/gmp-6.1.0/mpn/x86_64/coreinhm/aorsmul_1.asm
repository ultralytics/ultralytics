dnl  AMD64 mpn_addmul_1 and mpn_submul_1 optimised for Intel Nehalem.

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
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core
C Intel NHM	 4.55  with minor fluctuations
C Intel SBR
C Intel IBR
C Intel HWL
C Intel BWL
C Intel atom
C VIA nano

C The loop of this code is the result of running a code generation and
C optimization tool suite written by David Harvey and Torbjorn Granlund.

C N.B.: Be careful if editing, making sure the loop alignment padding does not
C become large, as we currently fall into it.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0',      `%rcx')   C r9

define(`n',       `%rbx')

ifdef(`OPERATION_addmul_1',`
  define(`ADDSUB', `add')
  define(`func',   `mpn_addmul_1')
')
ifdef(`OPERATION_submul_1',`
  define(`ADDSUB', `sub')
  define(`func',   `mpn_submul_1')
')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(func)
	FUNC_ENTRY(4)
	push	%rbx

	mov	(up), %rax
	lea	-8(up,n_param,8), up
	mov	(rp), %r8
	lea	-8(rp,n_param,8), rp

	test	$1, R8(n_param)
	jnz	L(bx1)

L(bx0):	test	$2, R8(n_param)
	jnz	L(b10)

L(b00):	mov	$3, R32(n)
	sub	n_param, n
	mul	v0
	mov	$0, R32(%r11)
	mov	%r8, %r10
	ADDSUB	%rax, %r10
	mov	-8(up,n,8), %rax
	adc	%rdx, %r11
	jmp	L(lo0)

L(b10):	mov	$1, R32(n)
	sub	n_param, n
	mul	v0
	mov	%r8, %r10
	mov	$0, R32(%r11)
	ADDSUB	%rax, %r10
	mov	8(up,n,8), %rax
	adc	%rdx, %r11
	jmp	L(lo2)

L(bx1):	test	$2, R8(n_param)
	jz	L(b01)

L(b11):	mov	$2, R32(n)
	sub	n_param, n
	mul	v0
	ADDSUB	%rax, %r8
	mov	$0, R32(%r9)
	mov	(up,n,8), %rax
	adc	%rdx, %r9
	jmp	L(lo3)

L(b01):	mov	$0, R32(n)
	sub	n_param, n
	xor	%r11, %r11
	add	$4, n
	jc	L(end)

	ALIGN(32)
L(top):	mul	v0
	ADDSUB	%rax, %r8
	mov	$0, R32(%r9)
	mov	-16(up,n,8), %rax
	adc	%rdx, %r9
L(lo1):	mul	v0
	ADDSUB	%r11, %r8
	mov	$0, R32(%r11)
	mov	-16(rp,n,8), %r10
	adc	$0, %r9
	ADDSUB	%rax, %r10
	mov	-8(up,n,8), %rax
	adc	%rdx, %r11
	mov	%r8, -24(rp,n,8)
	ADDSUB	%r9, %r10
	adc	$0, %r11
L(lo0):	mov	-8(rp,n,8), %r8
	mul	v0
	ADDSUB	%rax, %r8
	mov	$0, R32(%r9)
	mov	(up,n,8), %rax
	adc	%rdx, %r9
	mov	%r10, -16(rp,n,8)
	ADDSUB	%r11, %r8
	adc	$0, %r9
L(lo3):	mul	v0
	mov	(rp,n,8), %r10
	mov	$0, R32(%r11)
	ADDSUB	%rax, %r10
	mov	8(up,n,8), %rax
	adc	%rdx, %r11
	mov	%r8, -8(rp,n,8)
	ADDSUB	%r9, %r10
	adc	$0, %r11
L(lo2):	mov	8(rp,n,8), %r8
	mov	%r10, (rp,n,8)
	add	$4, n
	jnc	L(top)

L(end):	mul	v0
	ADDSUB	%rax, %r8
	mov	$0, R32(%rax)
	adc	%rdx, %rax
	ADDSUB	%r11, %r8
	adc	$0, %rax
	mov	%r8, (rp)

	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
ASM_END()
