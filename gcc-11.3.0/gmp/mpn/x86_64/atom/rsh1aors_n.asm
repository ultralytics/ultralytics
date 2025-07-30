dnl  x86-64 mpn_rsh1add_n/mpn_rsh1sub_n.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2011, 2012 Free Software Foundation, Inc.

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

C TODO
C  * Schedule loop less.  It is now almost surely overscheduled, resulting in
C    large feed-in and wind-down code.

C	     cycles/limb
C AMD K8,K9	 ?
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 ?
C Intel NMH	 ?
C Intel SBR	 ?
C Intel atom	 5.25
C VIA nano	 ?

C INPUT PARAMETERS
define(`rp',`%rdi')
define(`up',`%rsi')
define(`vp',`%rdx')
define(`n',`%rcx')

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

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_rsh1add_n mpn_rsh1sub_n)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	(up), %r15
	ADDSUB	(vp), %r15
	sbb	R32(%rbx), R32(%rbx)
	xor	R32(%rax), R32(%rax)
	shr	%r15
	adc	R32(%rax), R32(%rax)	C return value

	mov	R32(n), R32(%rbp)
	and	$3, R32(%rbp)
	jz	L(b0)
	cmp	$2, R32(%rbp)
	jae	L(b23)

L(b1):	dec	n
	jnz	L(gt1)
	shl	$63, %rbx
	add	%rbx, %r15
	mov	%r15, (rp)
	jmp	L(cj1)
L(gt1):	lea	24(up), up
	lea	24(vp), vp
	mov	-16(up), %r9
	add	R32(%rbx), R32(%rbx)
	mov	-8(up), %r10
	lea	24(rp), rp
	mov	(up), %r11
	ADCSBB	-16(vp), %r9
	ADCSBB	-8(vp), %r10
	mov	%r15, %r12
	ADCSBB	(vp), %r11
	mov	%r9, %r13
	sbb	R32(%rbx), R32(%rbx)
	mov	%r11, %r15
	mov	%r10, %r14
	shl	$63, %r11
	shl	$63, %r10
	shl	$63, %r9
	or	%r9, %r12
	shr	%r13
	mov	8(up), %r8
	shr	%r14
	or	%r10, %r13
	shr	%r15
	or	%r11, %r14
	sub	$4, n
	jz	L(cj5)
L(gt5):	mov	16(up), %r9
	add	R32(%rbx), R32(%rbx)
	mov	24(up), %r10
	ADCSBB	8(vp), %r8
	mov	%r15, %rbp
	mov	32(up), %r11
	jmp	L(lo1)

L(b23):	jnz	L(b3)
	mov	8(up), %r8
	sub	$2, n
	jnz	L(gt2)
	add	R32(%rbx), R32(%rbx)
	ADCSBB	8(vp), %r8
	mov	%r8, %r12
	jmp	L(cj2)
L(gt2):	mov	16(up), %r9
	add	R32(%rbx), R32(%rbx)
	mov	24(up), %r10
	ADCSBB	8(vp), %r8
	mov	%r15, %rbp
	mov	32(up), %r11
	ADCSBB	16(vp), %r9
	lea	32(up), up
	ADCSBB	24(vp), %r10
	mov	%r9, %r13
	ADCSBB	32(vp), %r11
	mov	%r8, %r12
	jmp	L(lo2)

L(b3):	lea	40(up), up
	lea	8(vp), vp
	mov	%r15, %r14
	add	R32(%rbx), R32(%rbx)
	mov	-32(up), %r11
	ADCSBB	0(vp), %r11
	lea	8(rp), rp
	sbb	R32(%rbx), R32(%rbx)
	mov	%r11, %r15
	shl	$63, %r11
	mov	-24(up), %r8
	shr	%r15
	or	%r11, %r14
	sub	$3, n
	jnz	L(gt3)
	add	R32(%rbx), R32(%rbx)
	ADCSBB	8(vp), %r8
	jmp	L(cj3)
L(gt3):	mov	-16(up), %r9
	add	R32(%rbx), R32(%rbx)
	mov	-8(up), %r10
	ADCSBB	8(vp), %r8
	mov	%r15, %rbp
	mov	(up), %r11
	ADCSBB	16(vp), %r9
	ADCSBB	24(vp), %r10
	mov	%r8, %r12
	jmp	L(lo3)

L(b0):	lea	48(up), up
	lea	16(vp), vp
	add	R32(%rbx), R32(%rbx)
	mov	-40(up), %r10
	lea	16(rp), rp
	mov	-32(up), %r11
	ADCSBB	-8(vp), %r10
	mov	%r15, %r13
	ADCSBB	(vp), %r11
	sbb	R32(%rbx), R32(%rbx)
	mov	%r11, %r15
	mov	%r10, %r14
	shl	$63, %r11
	shl	$63, %r10
	mov	-24(up), %r8
	shr	%r14
	or	%r10, %r13
	shr	%r15
	or	%r11, %r14
	sub	$4, n
	jnz	L(gt4)
	add	R32(%rbx), R32(%rbx)
	ADCSBB	8(vp), %r8
	jmp	L(cj4)
L(gt4):	mov	-16(up), %r9
	add	R32(%rbx), R32(%rbx)
	mov	-8(up), %r10
	ADCSBB	8(vp), %r8
	mov	%r15, %rbp
	mov	(up), %r11
	ADCSBB	16(vp), %r9
	jmp	L(lo0)

	ALIGN(8)
L(top):	mov	16(up), %r9
	shr	%r14
	or	%r10, %r13
	shr	%r15
	or	%r11, %r14
	add	R32(%rbx), R32(%rbx)
	mov	24(up), %r10
	mov	%rbp, (rp)
	ADCSBB	8(vp), %r8
	mov	%r15, %rbp
	lea	32(rp), rp
	mov	32(up), %r11
L(lo1):	ADCSBB	16(vp), %r9
	lea	32(up), up
	mov	%r12, -24(rp)
L(lo0):	ADCSBB	24(vp), %r10
	mov	%r8, %r12
	mov	%r13, -16(rp)
L(lo3):	ADCSBB	32(vp), %r11
	mov	%r9, %r13
	mov	%r14, -8(rp)
L(lo2):	sbb	R32(%rbx), R32(%rbx)
	shl	$63, %r8
	mov	%r11, %r15
	shr	%r12
	mov	%r10, %r14
	shl	$63, %r9
	lea	32(vp), vp
	shl	$63, %r10
	or	%r8, %rbp
	shl	$63, %r11
	or	%r9, %r12
	shr	%r13
	mov	8(up), %r8
	sub	$4, n
	jg	L(top)

L(end):	shr	%r14
	or	%r10, %r13
	shr	%r15
	or	%r11, %r14
	mov	%rbp, (rp)
	lea	32(rp), rp
L(cj5):	add	R32(%rbx), R32(%rbx)
	ADCSBB	8(vp), %r8
	mov	%r12, -24(rp)
L(cj4):	mov	%r13, -16(rp)
L(cj3):	mov	%r8, %r12
	mov	%r14, -8(rp)
L(cj2):	sbb	R32(%rbx), R32(%rbx)
	shl	$63, %r8
	shr	%r12
	or	%r8, %r15
	shl	$63, %rbx
	add	%rbx, %r12
	mov	%r15, (rp)
	mov	%r12, 8(rp)
L(cj1):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
