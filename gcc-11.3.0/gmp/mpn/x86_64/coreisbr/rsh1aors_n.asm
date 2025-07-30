dnl  X86-64 mpn_rsh1add_n, mpn_rsh1sub_n optimised for Intel Sandy Bridge.

dnl  Copyright 2003, 2005, 2009-2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 ?
C AMD K10	 4.25
C Intel P4	 21.5
C Intel core2	 3.2
C Intel NHM	 3.87
C Intel SBR	 2.05
C Intel atom	 ?
C VIA nano	 44.9

C INPUT PARAMETERS
define(`rp', `%rdi')
define(`up', `%rsi')
define(`vp', `%rdx')
define(`n',  `%rcx')

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
	push	%rbp

	neg	%r8			C set C flag from parameter
	mov	(up), %rbp
	ADCSBB	(vp), %rbp

	jmp	L(ent)
EPILOGUE()

	ALIGN(16)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp

	mov	(up), %rbp
	ADDSUB	(vp), %rbp
L(ent):
	sbb	R32(%rbx), R32(%rbx)	C save cy
	mov	R32(%rbp), R32(%rax)
	and	$1, R32(%rax)		C return value

	mov	R32(n), R32(%r11)
	and	$3, R32(%r11)

	cmp	$1, R32(%r11)
	je	L(do)			C jump if n = 1 5 9 ...

L(n1):	cmp	$2, R32(%r11)
	jne	L(n2)			C jump unless n = 2 6 10 ...
	add	R32(%rbx), R32(%rbx)	C restore cy
	mov	8(up), %r10
	ADCSBB	8(vp), %r10
	lea	8(up), up
	lea	8(vp), vp
	lea	8(rp), rp
	sbb	R32(%rbx), R32(%rbx)	C save cy

	shrd	$1, %r10, %rbp
	mov	%rbp, -8(rp)
	jmp	L(cj1)

L(n2):	cmp	$3, R32(%r11)
	jne	L(n3)			C jump unless n = 3 7 11 ...
	add	R32(%rbx), R32(%rbx)	C restore cy
	mov	8(up), %r9
	mov	16(up), %r10
	ADCSBB	8(vp), %r9
	ADCSBB	16(vp), %r10
	lea	16(up), up
	lea	16(vp), vp
	lea	16(rp), rp
	sbb	R32(%rbx), R32(%rbx)	C save cy

	shrd	$1, %r9, %rbp
	mov	%rbp, -16(rp)
	jmp	L(cj2)

L(n3):	dec	n			C come here for n = 4 8 12 ...
	add	R32(%rbx), R32(%rbx)	C restore cy
	mov	8(up), %r8
	mov	16(up), %r9
	ADCSBB	8(vp), %r8
	ADCSBB	16(vp), %r9
	mov	24(up), %r10
	ADCSBB	24(vp), %r10
	lea	24(up), up
	lea	24(vp), vp
	lea	24(rp), rp
	sbb	R32(%rbx), R32(%rbx)	C save cy

	shrd	$1, %r8, %rbp
	mov	%rbp, -24(rp)
	shrd	$1, %r9, %r8
	mov	%r8, -16(rp)
L(cj2):	shrd	$1, %r10, %r9
	mov	%r9, -8(rp)
L(cj1):	mov	%r10, %rbp

L(do):
	shr	$2, n			C				4
	je	L(end)			C				2
	ALIGN(16)
L(top):	add	R32(%rbx), R32(%rbx)		C restore cy

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

	sbb	R32(%rbx), R32(%rbx)	C save cy

	shrd	$1, %r8, %rbp
	mov	%rbp, (rp)
	shrd	$1, %r9, %r8
	mov	%r8, 8(rp)
	shrd	$1, %r10, %r9
	mov	%r9, 16(rp)
	shrd	$1, %r11, %r10
	mov	%r10, 24(rp)

	dec	n
	mov	%r11, %rbp
	lea	32(rp), rp
	jne	L(top)

L(end):	shrd	$1, %rbx, %rbp
	mov	%rbp, (rp)
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
