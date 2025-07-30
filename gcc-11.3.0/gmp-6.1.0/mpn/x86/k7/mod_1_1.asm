dnl  x86-32 mpn_mod_1_1p, requiring cmov.

dnl  Contributed to the GNU project by Niels Möller and Torbjorn Granlund.

dnl  Copyright 2010, 2011 Free Software Foundation, Inc.

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

C			    cycles/limb
C P5				 ?
C P6 model 0-8,10-12		 ?
C P6 model 9  (Banias)		 ?
C P6 model 13 (Dothan)		 ?
C P4 model 0  (Willamette)	 ?
C P4 model 1  (?)		 ?
C P4 model 2  (Northwood)	 ?
C P4 model 3  (Prescott)	 ?
C P4 model 4  (Nocona)		 ?
C AMD K6			 ?
C AMD K7			 7
C AMD K8			 ?

define(`B2mb', `%ebx')
define(`r0', `%esi')
define(`r2', `%ebp')
define(`t0', `%edi')
define(`ap', `%ecx')  C Also shift count

C Stack frame
C	pre	36(%esp)
C	b	32(%esp)
C	n	28(%esp)
C	ap	24(%esp)
C	return	20(%esp)
C	%ebp	16(%esp)
C	%edi	12(%esp)
C	%esi	8(%esp)
C	%ebx	4(%esp)
C	B2mod	(%esp)

define(`B2modb', `(%esp)')
define(`n', `28(%esp)')
define(`b', `32(%esp)')
define(`pre', `36(%esp)')

C mp_limb_t
C mpn_mod_1_1p (mp_srcptr ap, mp_size_t n, mp_limb_t b, mp_limb_t pre[4])
C
C The pre array contains bi, cnt, B1modb, B2modb
C Note: This implementation needs B1modb only when cnt > 0

ASM_START()
	TEXT
	ALIGN(8)
PROLOGUE(mpn_mod_1_1p)
	push	%ebp
	push	%edi
	push	%esi
	push	%ebx
	mov	32(%esp), %ebp		C pre[]

	mov	12(%ebp), %eax		C B2modb
	push	%eax			C Put it on stack

	mov	n, %edx
	mov	24(%esp), ap

	lea	(ap, %edx, 4), ap
	mov	-4(ap), %eax
	cmp	$3, %edx
	jnc	L(first)
	mov	-8(ap), r0
	jmp	L(reduce_two)

L(first):
	C First iteration, no r2
	mull	B2modb
	mov	-12(ap), r0
	add	%eax, r0
	mov	-8(ap), %eax
	adc	%edx, %eax
	sbb	r2, r2
	subl	$3, n
	lea	-16(ap), ap
	jz	L(reduce_three)

	mov	B2modb, B2mb
	sub	b, B2mb
	lea	(B2mb, r0), t0
	jmp	L(mid)

	ALIGN(16)
L(top): C Loopmixed to 7 c/l on k7
	add	%eax, r0
	lea	(B2mb, r0), t0
	mov	r2, %eax
	adc	%edx, %eax
	sbb	r2, r2
L(mid):	mull	B2modb
	and	B2modb, r2
	add	r0, r2
	decl	n
	mov	(ap), r0
	cmovc(	t0, r2)
	lea	-4(ap), ap
	jnz	L(top)

	add	%eax, r0
	mov	r2, %eax
	adc	%edx, %eax
	sbb	r2, r2

L(reduce_three):
	C Eliminate r2
	and	b, r2
	sub	r2, %eax

L(reduce_two):
	mov	pre, %ebp
	movb	4(%ebp), %cl
	test	%cl, %cl
	jz	L(normalized)

	C Unnormalized, use B1modb to reduce to size < B b
	mull	8(%ebp)
	xor	t0, t0
	add	%eax, r0
	adc	%edx, t0
	mov	t0, %eax

	C Left-shift to normalize
	shld	%cl, r0, %eax C Always use shld?

	shl	%cl, r0
	jmp	L(udiv)

L(normalized):
	mov	%eax, t0
	sub	b, t0
	cmovnc(	t0, %eax)

L(udiv):
	lea	1(%eax), t0
	mull	(%ebp)
	mov	b, %ebx		C Needed in register for lea
	add	r0, %eax
	adc	t0, %edx
	imul	%ebx, %edx
	sub	%edx, r0
	cmp	r0, %eax
	lea	(%ebx, r0), %eax
	cmovnc(	r0, %eax)
	cmp	%ebx, %eax
	jnc	L(fix)
L(ok):	shr	%cl, %eax

	add	$4, %esp
	pop	%ebx
	pop	%esi
	pop	%edi
	pop	%ebp

	ret
L(fix):	sub	%ebx, %eax
	jmp	L(ok)
EPILOGUE()

PROLOGUE(mpn_mod_1_1p_cps)
	push	%ebp
	mov	12(%esp), %ebp
	push	%esi
	bsr	%ebp, %ecx
	push	%ebx
	xor	$31, %ecx
	mov	16(%esp), %esi
	sal	%cl, %ebp
	mov	%ebp, %edx
	not	%edx
	mov	$-1, %eax
	div	%ebp			C On K7, invert_limb would be a few cycles faster.
	mov	%eax, (%esi)		C store bi
	mov	%ecx, 4(%esi)		C store cnt
	neg	%ebp
	mov	$1, %edx
	shld	%cl, %eax, %edx
	imul	%ebp, %edx
	shr	%cl, %edx
	imul	%ebp, %eax
	mov	%edx, 8(%esi)		C store B1modb
	mov	%eax, 12(%esi)		C store B2modb
	pop	%ebx
	pop	%esi
	pop	%ebp
	ret
EPILOGUE()
