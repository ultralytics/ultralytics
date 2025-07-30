dnl  AMD64 mpn_lshsub_n.  R = 2^k(U - V).

dnl  Copyright 2006, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 3.15	(mpn_sub_n + mpn_lshift costs about 4 c/l)
C AMD K10	 3.15	(mpn_sub_n + mpn_lshift costs about 4 c/l)
C Intel P4	16.5
C Intel core2	 4.35
C Intel corei	 ?
C Intel atom	 ?
C VIA nano	 ?

C This was written quickly and not optimized at all, but it runs very well on
C K8.  But perhaps one could get under 3 c/l.  Ideas:
C   1) Use indexing to save the 3 LEA
C   2) Write reasonable feed-in code
C   3) Be more clever about register usage
C   4) Unroll more, handling CL negation, carry save/restore cost much now
C   5) Reschedule

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')
define(`cnt',	`%r8')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_lshsub_n)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')

	push	%r12
	push	%r13
	push	%r14
	push	%r15
	push	%rbx

	mov	n, %rax
	xor	R32(%rbx), R32(%rbx)	C clear carry save register
	mov	R32(%r8), R32(%rcx)	C shift count
	xor	R32(%r15), R32(%r15)	C limb carry

	mov	R32(%rax), R32(%r11)
	and	$3, R32(%r11)
	je	L(4)
	sub	$1, R32(%r11)

L(oopette):
	add	R32(%rbx), R32(%rbx)	C restore carry flag
	mov	0(up), %r8
	lea	8(up), up
	sbb	0(vp), %r8
	mov	%r8, %r12
	sbb	R32(%rbx), R32(%rbx)	C save carry flag
	shl	R8(%rcx), %r8
	or	%r15, %r8
	mov	%r12, %r15
	lea	8(vp), vp
	neg	R8(%rcx)
	shr	R8(%rcx), %r15
	neg	R8(%rcx)
	mov	%r8, 0(rp)
	lea	8(rp), rp
	sub	$1, R32(%r11)
	jnc	L(oopette)

L(4):
	sub	$4, %rax
	jc	L(end)

	ALIGN(16)
L(oop):
	add	R32(%rbx), R32(%rbx)	C restore carry flag

	mov	0(up), %r8
	mov	8(up), %r9
	mov	16(up), %r10
	mov	24(up), %r11

	lea	32(up), up

	sbb	0(vp), %r8
	mov	%r8, %r12
	sbb	8(vp), %r9
	mov	%r9, %r13
	sbb	16(vp), %r10
	mov	%r10, %r14
	sbb	24(vp), %r11

	sbb	R32(%rbx), R32(%rbx)	C save carry flag

	shl	R8(%rcx), %r8
	shl	R8(%rcx), %r9
	shl	R8(%rcx), %r10
	or	%r15, %r8
	mov	%r11, %r15
	shl	R8(%rcx), %r11

	lea	32(vp), vp

	neg	R8(%rcx)

	shr	R8(%rcx), %r12
	shr	R8(%rcx), %r13
	shr	R8(%rcx), %r14
	shr	R8(%rcx), %r15		C used next loop

	or	%r12, %r9
	or	%r13, %r10
	or	%r14, %r11

	neg	R8(%rcx)

	mov	%r8, 0(rp)
	mov	%r9, 8(rp)
	mov	%r10, 16(rp)
	mov	%r11, 24(rp)

	lea	32(rp), rp

	sub	$4, %rax
	jnc	L(oop)
L(end):
	neg	R32(%rbx)
	shl	R8(%rcx), %rbx
	adc	%r15, %rbx
	mov	%rbx, %rax
	pop	%rbx
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12

	FUNC_EXIT()
	ret
EPILOGUE()
