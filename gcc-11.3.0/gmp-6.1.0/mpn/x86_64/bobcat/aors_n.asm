dnl  AMD64 mpn_add_n, mpn_sub_n optimised for bobcat.

dnl  Copyright 2003-2005, 2007, 2008, 2010-2013 Free Software Foundation, Inc.

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
C AMD bobcat	 2.28
C Intel P4
C Intel core2
C Intel NHM
C Intel SBR
C Intel IBR
C Intel atom
C VIA nano

C The loop of this code is the result of running a code generation and
C optimization tool suite written by David Harvey and Torbjorn Granlund.

C INPUT PARAMETERS
define(`rp',	`%rdi')	C rcx
define(`up',	`%rsi')	C rdx
define(`vp',	`%rdx')	C r8
define(`n',	`%rcx')	C r9
define(`cy',	`%r8')	C rsp+40    (mpn_add_nc and mpn_sub_nc)

ifdef(`OPERATION_add_n', `
	define(ADCSBB,	      adc)
	define(func,	      mpn_add_n)
	define(func_nc,	      mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
	define(ADCSBB,	      sbb)
	define(func,	      mpn_sub_n)
	define(func_nc,	      mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
	xor	%r8, %r8
L(ent):	test	$1, R8(n)
	jnz	L(bx1)

L(bx0):	test	$2, R8(n)
	jnz	L(b10)

L(b00):	shr	$2, n
	neg	%r8
	mov	$3, R32(%rax)
	mov	(up), %r10
	mov	8(up), %r11
	jmp	L(lo0)

L(b10):	shr	$2, n
	neg	%r8
	mov	$1, R32(%rax)
	mov	(up), %r8
	mov	8(up), %r9
	jrcxz	L(cj2)
	jmp	L(top)

L(bx1):	test	$2, R8(n)
	jnz	L(b11)

L(b01):	shr	$2, n
	neg	%r8
	mov	$0, R32(%rax)
	mov	(up), %r9
	jrcxz	L(cj1)
	mov	8(up), %r10
	jmp	L(lo1)

	ALIGN(8)
L(b11):	inc	n
	shr	$2, n
	neg	%r8
	mov	$2, R32(%rax)
	mov	(up), %r11
	jmp	L(lo3)

	ALIGN(4)
L(top):	mov	8(up,%rax,8), %r10
	ADCSBB	-8(vp,%rax,8), %r8
	mov	%r8, -8(rp,%rax,8)
L(lo1):	mov	16(up,%rax,8), %r11
	ADCSBB	(vp,%rax,8), %r9
	lea	4(%rax), %rax
	mov	%r9, -32(rp,%rax,8)
L(lo0):	ADCSBB	-24(vp,%rax,8), %r10
	mov	%r10, -24(rp,%rax,8)
L(lo3):	ADCSBB	-16(vp,%rax,8), %r11
	dec	n
	mov	-8(up,%rax,8), %r8
	mov	%r11, -16(rp,%rax,8)
L(lo2):	mov	(up,%rax,8), %r9
	jnz	L(top)

L(cj2):	ADCSBB	-8(vp,%rax,8), %r8
	mov	%r8, -8(rp,%rax,8)
L(cj1):	ADCSBB	(vp,%rax,8), %r9
	mov	%r9, (rp,%rax,8)

	mov	$0, R32(%rax)
	adc	$0, R32(%rax)

	FUNC_EXIT()
	ret
EPILOGUE()

	ALIGN(16)
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	jmp	L(ent)
EPILOGUE()
