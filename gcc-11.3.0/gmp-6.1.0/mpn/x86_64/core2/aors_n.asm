dnl  Intel mpn_add_n/mpn_sub_n optimised for Conroe, Nehalem.

dnl  Copyright 2006, 2007, 2011-2013 Free Software Foundation, Inc.

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
C AMD K8,K9	 2
C AMD K10	 2
C Intel P4	10
C Intel core2	 2
C Intel NHM	 2
C Intel SBR	 2
C Intel atom	 9
C VIA nano	 3

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')
define(`cy',	`%r8')

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
L(start):
	mov	(up), %r10
	mov	(vp), %r11

	lea	(up,n,8), up
	lea	(vp,n,8), vp
	lea	(rp,n,8), rp
	mov	R32(n), R32(%rax)
	neg	n
	and	$3, R32(%rax)
	je	L(b00)
	add	%rax, n			C clear low rcx bits for jrcxz
	cmp	$2, R32(%rax)
	jl	L(b01)
	je	L(b10)

L(b11):	neg	%r8			C set cy
	jmp	L(e11)

L(b00):	neg	%r8			C set cy
	mov	%r10, %r8
	mov	%r11, %r9
	lea	4(n), n
	jmp	L(e00)

	nop
	nop
	nop
L(b01):	neg	%r8			C set cy
	jmp	L(top)

L(b10):	neg	%r8			C set cy
	mov	%r10, %r8
	mov	%r11, %r9
	jmp	L(e10)

L(end):	ADCSBB	%r11, %r10
	mov	%r10, -8(rp)
	mov	R32(%rcx), R32(%rax)	C clear eax, ecx contains 0
	adc	R32(%rax), R32(%rax)
	FUNC_EXIT()
	ret

	ALIGN(16)
L(top):	jrcxz	L(end)
	mov	(up,n,8), %r8
	mov	(vp,n,8), %r9
	lea	4(n), n
	ADCSBB	%r11, %r10
	mov	%r10, -40(rp,n,8)
L(e00):	mov	-24(up,n,8), %r10
	mov	-24(vp,n,8), %r11
	ADCSBB	%r9, %r8
	mov	%r8, -32(rp,n,8)
L(e11):	mov	-16(up,n,8), %r8
	mov	-16(vp,n,8), %r9
	ADCSBB	%r11, %r10
	mov	%r10, -24(rp,n,8)
L(e10):	mov	-8(up,n,8), %r10
	mov	-8(vp,n,8), %r11
	ADCSBB	%r9, %r8
	mov	%r8, -16(rp,n,8)
	jmp	L(top)
EPILOGUE()

PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	jmp	L(start)
EPILOGUE()

