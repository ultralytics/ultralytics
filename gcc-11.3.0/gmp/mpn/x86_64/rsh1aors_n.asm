dnl  AMD64 mpn_rsh1add_n -- rp[] = (up[] + vp[]) >> 1
dnl  AMD64 mpn_rsh1sub_n -- rp[] = (up[] - vp[]) >> 1

dnl  Copyright 2003, 2005, 2009, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.14	(mpn_add_n + mpn_rshift need 4.125)
C AMD K10	 2.14	(mpn_add_n + mpn_rshift need 4.125)
C Intel P4	12.75
C Intel core2	 3.75
C Intel NMH	 4.4
C Intel SBR	 ?
C Intel atom	 ?
C VIA nano	 3.25

C TODO
C  * Rewrite to use indexed addressing, like addlsh1.asm and sublsh1.asm.

C INPUT PARAMETERS
define(`rp', `%rdi')
define(`up', `%rsi')
define(`vp', `%rdx')
define(`n',`  %rcx')

ifdef(`OPERATION_rsh1add_n', `
	define(ADDSUB,	      add)
	define(ADCSBB,	      adc)
	define(func_n,	      mpn_rsh1add_n)
	define(func_nc,	      mpn_rsh1add_nc)')
ifdef(`OPERATION_rsh1sub_n', `
	define(ADDSUB,	      sub)
	define(ADCSBB,	      sbb)
	define(func_n,	      mpn_rsh1sub_n)
	define(func_nc,	      mpn_rsh1sub_nc)')

MULFUNC_PROLOGUE(mpn_rsh1add_n mpn_rsh1add_nc mpn_rsh1sub_n mpn_rsh1sub_nc)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%rbx

	xor	R32(%rax), R32(%rax)
	neg	%r8			C set C flag from parameter
	mov	(up), %rbx
	ADCSBB	(vp), %rbx
	jmp	L(ent)
EPILOGUE()

	ALIGN(16)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
	push	%rbx

	xor	R32(%rax), R32(%rax)
	mov	(up), %rbx
	ADDSUB	(vp), %rbx
L(ent):
	rcr	%rbx			C rotate, save acy
	adc	R32(%rax), R32(%rax)	C return value

	mov	R32(n), R32(%r11)
	and	$3, R32(%r11)

	cmp	$1, R32(%r11)
	je	L(do)			C jump if n = 1 5 9 ...

L(n1):	cmp	$2, R32(%r11)
	jne	L(n2)			C jump unless n = 2 6 10 ...
	add	%rbx, %rbx		C rotate carry limb, restore acy
	mov	8(up), %r10
	ADCSBB	8(vp), %r10
	lea	8(up), up
	lea	8(vp), vp
	lea	8(rp), rp
	rcr	%r10
	rcr	%rbx
	mov	%rbx, -8(rp)
	jmp	L(cj1)

L(n2):	cmp	$3, R32(%r11)
	jne	L(n3)			C jump unless n = 3 7 11 ...
	add	%rbx, %rbx		C rotate carry limb, restore acy
	mov	8(up), %r9
	mov	16(up), %r10
	ADCSBB	8(vp), %r9
	ADCSBB	16(vp), %r10
	lea	16(up), up
	lea	16(vp), vp
	lea	16(rp), rp
	rcr	%r10
	rcr	%r9
	rcr	%rbx
	mov	%rbx, -16(rp)
	jmp	L(cj2)

L(n3):	dec	n			C come here for n = 4 8 12 ...
	add	%rbx, %rbx		C rotate carry limb, restore acy
	mov	8(up), %r8
	mov	16(up), %r9
	ADCSBB	8(vp), %r8
	ADCSBB	16(vp), %r9
	mov	24(up), %r10
	ADCSBB	24(vp), %r10
	lea	24(up), up
	lea	24(vp), vp
	lea	24(rp), rp
	rcr	%r10
	rcr	%r9
	rcr	%r8
	rcr	%rbx
	mov	%rbx, -24(rp)
	mov	%r8, -16(rp)
L(cj2):	mov	%r9, -8(rp)
L(cj1):	mov	%r10, %rbx

L(do):
	shr	$2, n			C				4
	je	L(end)			C				2
	ALIGN(16)
L(top):	add	%rbx, %rbx		C rotate carry limb, restore acy

	mov	8(up), %r8
	mov	16(up), %r9
	ADCSBB	8(vp), %r8
	ADCSBB	16(vp), %r9
	mov	24(up), %r10
	mov	32(up), %r11
	ADCSBB	24(vp), %r10
	ADCSBB	32(vp), %r11

	lea	32(up), up
	lea	32(vp), vp

	rcr	%r11			C rotate, save acy
	rcr	%r10
	rcr	%r9
	rcr	%r8

	rcr	%rbx
	mov	%rbx, (rp)
	mov	%r8, 8(rp)
	mov	%r9, 16(rp)
	mov	%r10, 24(rp)
	mov	%r11, %rbx

	lea	32(rp), rp
	dec	n
	jne	L(top)

L(end):	mov	%rbx, (rp)
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
