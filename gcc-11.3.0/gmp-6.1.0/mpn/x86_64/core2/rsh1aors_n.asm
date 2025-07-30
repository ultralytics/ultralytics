dnl  X86-64 mpn_rsh1add_n, mpn_rsh1sub_n optimised for Intel Conroe/Penryn.

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
C AMD K8,K9	 ?
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 3.05
C Intel NHM	 3.3
C Intel SBR	 2.5
C Intel atom	 ?
C VIA nano	 ?

C TODO
C  * Loopmix to approach 2.5 c/l on NHM.

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
	mov	(up), %r8
	ADCSBB	(vp), %r8
	jmp	L(ent)
EPILOGUE()

	ALIGN(16)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp

	mov	(up), %r8
	ADDSUB	(vp), %r8
L(ent):	sbb	R32(%rbx), R32(%rbx)	C save cy
	mov	%r8, %rax
	and	$1, R32(%rax)		C return value

	lea	(up,n,8), up
	lea	(vp,n,8), vp
	lea	(rp,n,8), rp
	mov	R32(n), R32(%rbp)
	neg	n
	and	$3, R32(%rbp)
	jz	L(b0)
	cmp	$2, R32(%rbp)
	jae	L(n1)

L(b1):	mov	%r8, %rbp
	inc	n
	js	L(top)
	jmp	L(end)

L(n1):	jnz	L(b3)
	add	R32(%rbx), R32(%rbx)	C restore cy
	mov	8(up,n,8), %r11
	ADCSBB	8(vp,n,8), %r11
	sbb	R32(%rbx), R32(%rbx)	C save cy
	mov	%r8, %r10
	add	$-2, n
	jmp	L(2)

L(b3):	add	R32(%rbx), R32(%rbx)	C restore cy
	mov	8(up,n,8), %r10
	mov	16(up,n,8), %r11
	ADCSBB	8(vp,n,8), %r10
	ADCSBB	16(vp,n,8), %r11
	sbb	R32(%rbx), R32(%rbx)	C save cy
	mov	%r8, %r9
	dec	n
	jmp	L(3)

L(b0):	add	R32(%rbx), R32(%rbx)	C restore cy
	mov	8(up,n,8), %r9
	mov	16(up,n,8), %r10
	mov	24(up,n,8), %r11
	ADCSBB	8(vp,n,8), %r9
	ADCSBB	16(vp,n,8), %r10
	ADCSBB	24(vp,n,8), %r11
	sbb	R32(%rbx), R32(%rbx)	C save cy
	jmp	L(4)

	ALIGN(16)

L(top):	add	R32(%rbx), R32(%rbx)	C restore cy
	mov	(up,n,8), %r8
	mov	8(up,n,8), %r9
	mov	16(up,n,8), %r10
	mov	24(up,n,8), %r11
	ADCSBB	(vp,n,8), %r8
	ADCSBB	8(vp,n,8), %r9
	ADCSBB	16(vp,n,8), %r10
	ADCSBB	24(vp,n,8), %r11
	sbb	R32(%rbx), R32(%rbx)	C save cy
	shrd	$1, %r8, %rbp
	mov	%rbp, -8(rp,n,8)
L(4):	shrd	$1, %r9, %r8
	mov	%r8, (rp,n,8)
L(3):	shrd	$1, %r10, %r9
	mov	%r9, 8(rp,n,8)
L(2):	shrd	$1, %r11, %r10
	mov	%r10, 16(rp,n,8)
L(1):	add	$4, n
	mov	%r11, %rbp
	js	L(top)

L(end):	shrd	$1, %rbx, %rbp
	mov	%rbp, -8(rp)
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
