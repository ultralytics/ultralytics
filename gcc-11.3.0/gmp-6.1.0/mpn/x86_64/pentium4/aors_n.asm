dnl  x86-64 mpn_add_n/mpn_sub_n optimized for Pentium 4.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2007, 2008, 2010-2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.8
C AMD K10	 2.8
C Intel P4	 4
C Intel core2	 3.6-5	(fluctuating)
C Intel corei	 ?
C Intel atom	 ?
C VIA nano	 ?


C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')
define(`cy',	`%r8')

ifdef(`OPERATION_add_n', `
	define(ADDSUB,	      add)
	define(func,	      mpn_add_n)
	define(func_nc,	      mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
	define(ADDSUB,	      sub)
	define(func,	      mpn_sub_n)
	define(func_nc,	      mpn_sub_nc)')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)
ASM_START()
	TEXT
PROLOGUE(func)
	FUNC_ENTRY(4)
	xor	%r8, %r8
IFDOS(`	jmp	L(ent)		')
EPILOGUE()
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
L(ent):	push	%rbx
	push	%r12

	mov	(vp), %r9

	mov	R32(n), R32(%rax)
	and	$3, R32(%rax)
	jne	L(n00)		C n = 0, 4, 8, ...
	mov	R32(%r8), R32(%rbx)
	mov	(up), %r8
	mov	8(up), %r10
	ADDSUB	%r9, %r8
	mov	8(vp), %r9
	setc	R8(%rax)
	lea	-16(rp), rp
	jmp	L(L00)

L(n00):	cmp	$2, R32(%rax)
	jnc	L(n01)		C n = 1, 5, 9, ...
	mov	(up), %r11
	mov	R32(%r8), R32(%rax)
	xor	R32(%rbx), R32(%rbx)
	dec	n
	jnz	L(gt1)
	ADDSUB	%r9, %r11
	setc	R8(%rbx)
	ADDSUB	%rax, %r11
	adc	$0, R32(%rbx)
	mov	%r11, (rp)
	jmp	L(ret)
L(gt1):	mov	8(up), %r8
	ADDSUB	%r9, %r11
	mov	8(vp), %r9
	setc	R8(%rbx)
	lea	-8(rp), rp
	lea	8(up), up
	lea	8(vp), vp
	jmp	L(L01)

L(n01):	jne	L(n10)		C n = 2, 6, 10, ...
	mov	(up), %r12
	mov	R32(%r8), R32(%rbx)
	mov	8(up), %r11
	ADDSUB	%r9, %r12
	mov	8(vp), %r9
	setc	R8(%rax)
	lea	-32(rp), rp
	lea	16(up), up
	lea	16(vp), vp
	jmp	L(L10)

L(n10):	mov	(up), %r10	C n = 3, 7, 11, ...
	mov	R32(%r8), R32(%rax)
	xor	R32(%rbx), R32(%rbx)
	mov	8(up), %r12
	ADDSUB	%r9, %r10
	mov	8(vp), %r9
	setc	R8(%rbx)
	lea	-24(rp), rp
	lea	-8(up), up
	lea	-8(vp), vp
	jmp	L(L11)

L(c0):	mov	$1, R8(%rbx)
	jmp	L(rc0)
L(c1):	mov	$1, R8(%rax)
	jmp	L(rc1)
L(c2):	mov	$1, R8(%rbx)
	jmp	L(rc2)
L(c3):	mov	$1, R8(%rax)
	jmp	L(rc3)

	ALIGN(16)
L(top):	mov	(up), %r8	C not on critical path
	ADDSUB	%r9, %r11	C not on critical path
	mov	(vp), %r9	C not on critical path
	setc	R8(%rbx)	C save carry out
	mov	%r12, (rp)
L(L01):	ADDSUB	%rax, %r11	C apply previous carry out
	jc	L(c0)		C jump if ripple
L(rc0):	mov	8(up), %r10
	ADDSUB	%r9, %r8
	mov	8(vp), %r9
	setc	R8(%rax)
	mov	%r11, 8(rp)
L(L00):	ADDSUB	%rbx, %r8
	jc	L(c1)
L(rc1):	mov	16(up), %r12
	ADDSUB	%r9, %r10
	mov	16(vp), %r9
	setc	R8(%rbx)
	mov	%r8, 16(rp)
L(L11):	ADDSUB	%rax, %r10
	jc	L(c2)
L(rc2):	mov	24(up), %r11
	ADDSUB	%r9, %r12
	lea	32(up), up
	mov	24(vp), %r9
	lea	32(vp), vp
	setc	R8(%rax)
	mov	%r10, 24(rp)
L(L10):	ADDSUB	%rbx, %r12
	jc	L(c3)
L(rc3):	lea	32(rp), rp
	sub	$4, n
	ja	L(top)

L(end):	ADDSUB	%r9, %r11
	setc	R8(%rbx)
	mov	%r12, (rp)
	ADDSUB	%rax, %r11
	jnc	L(1)
	mov	$1, R8(%rbx)
L(1):	mov	%r11, 8(rp)

L(ret):	mov	R32(%rbx), R32(%rax)
	pop	%r12
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
