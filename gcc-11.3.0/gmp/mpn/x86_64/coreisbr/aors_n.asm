dnl  AMD64 mpn_add_n, mpn_sub_n optimised for Sandy bridge, Ivy bridge, and
dnl  Haswell.

dnl  Contributed to the GNU project by Torbjörn Granlund.

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
C AMD bull	 1.82		average over 400-600
C AMD pile	 1.83		average over 400-600
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core
C Intel NHM
C Intel SBR	 1.55		fluctuates
C Intel IBR	 1.55		fluctuates
C Intel HWL	 1.33		fluctuates
C Intel BWL
C Intel atom
C VIA nano

C The loop of this code was manually written.  It runs close to optimally on
C Intel SBR, IBR, and HWL far as we know, except for the fluctuation problems.
C It also runs slightly faster on average on AMD bull and pile.
C
C No micro-optimisation has been done.
C
C N.B.!  The loop alignment padding insns are executed.  If editing the code,
C make sure the padding does not become excessive.  It is now a 4-byte nop.

define(`rp',	`%rdi')	C rcx
define(`up',	`%rsi')	C rdx
define(`vp',	`%rdx')	C r8
define(`n',	`%rcx')	C r9
define(`cy',	`%r8')	C rsp+40    (mpn_add_nc and mpn_sub_nc)

ifdef(`OPERATION_add_n', `
  define(ADCSBB,    adc)
  define(func,      mpn_add_n)
  define(func_nc,   mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
  define(ADCSBB,    sbb)
  define(func,      mpn_sub_n)
  define(func_nc,   mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(func)
	FUNC_ENTRY(4)
	xor	%r8, %r8

L(ent):	mov	R32(n), R32(%rax)
	shr	$2, n

	test	$1, R8(%rax)
	jnz	L(bx1)

L(bx0):	test	$2, R8(%rax)
	jnz	L(b10)

L(b00):	neg	%r8
	mov	(up), %r8
	mov	8(up), %r9
	ADCSBB	(vp), %r8
	ADCSBB	8(vp), %r9
	mov	16(up), %r10
	mov	24(up), %r11
	lea	32(up), up
	ADCSBB	16(vp), %r10
	ADCSBB	24(vp), %r11
	lea	32(vp), vp
	lea	-16(rp), rp
	jmp	L(lo0)

L(b10):	neg	%r8
	mov	(up), %r10
	mov	8(up), %r11
	ADCSBB	0(vp), %r10
	ADCSBB	8(vp), %r11
	jrcxz	L(e2)
	mov	16(up), %r8
	mov	24(up), %r9
	lea	16(up), up
	ADCSBB	16(vp), %r8
	ADCSBB	24(vp), %r9
	lea	16(vp), vp
	lea	(rp), rp
	jmp	L(lo2)

L(e2):	mov	%r10, (rp)
	mov	%r11, 8(rp)
	setc	R8(%rax)
	FUNC_EXIT()
	ret

L(bx1):	test	$2, R8(%rax)
	jnz	L(b11)

L(b01):	neg	%r8
	mov	(up), %r11
	ADCSBB	(vp), %r11
	jrcxz	L(e1)
	mov	8(up), %r8
	mov	16(up), %r9
	lea	8(up), up
	lea	-8(rp), rp
	ADCSBB	8(vp), %r8
	ADCSBB	16(vp), %r9
	lea	8(vp), vp
	jmp	L(lo1)

L(e1):	mov	%r11, (rp)
	setc	R8(%rax)
	FUNC_EXIT()
	ret

L(b11):	neg	%r8
	mov	(up), %r9
	ADCSBB	(vp), %r9
	mov	8(up), %r10
	mov	16(up), %r11
	lea	24(up), up
	ADCSBB	8(vp), %r10
	ADCSBB	16(vp), %r11
	lea	24(vp), vp
	mov	%r9, (rp)
	lea	8(rp), rp
	jrcxz	L(end)

	ALIGN(32)
L(top):	mov	(up), %r8
	mov	8(up), %r9
	ADCSBB	(vp), %r8
	ADCSBB	8(vp), %r9
L(lo2):	mov	%r10, (rp)
L(lo1):	mov	%r11, 8(rp)
	mov	16(up), %r10
	mov	24(up), %r11
	lea	32(up), up
	ADCSBB	16(vp), %r10
	ADCSBB	24(vp), %r11
	lea	32(vp), vp
L(lo0):	mov	%r8, 16(rp)
L(lo3):	mov	%r9, 24(rp)
	lea	32(rp), rp
	dec	n
	jnz	L(top)

L(end):	mov	R32(n), R32(%rax)	C zero rax
	mov	%r10, (rp)
	mov	%r11, 8(rp)
	setc	R8(%rax)
	FUNC_EXIT()
	ret
EPILOGUE()
	ALIGN(16)
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	jmp	L(ent)
EPILOGUE()
