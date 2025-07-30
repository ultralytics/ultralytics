dnl  x86-32 mpn_mod_1s_4p, requiring cmov.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2009, 2010 Free Software Foundation, Inc.

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
C P6 model 13 (Dothan)		 6
C P4 model 0  (Willamette)	 ?
C P4 model 1  (?)		 ?
C P4 model 2  (Northwood)	15.5
C P4 model 3  (Prescott)	 ?
C P4 model 4  (Nocona)		 ?
C AMD K6			 ?
C AMD K7			 4.75
C AMD K8			 ?

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_1s_4p)
	push	%ebp
	push	%edi
	push	%esi
	push	%ebx
	sub	$28, %esp
	mov	60(%esp), %edi		C cps[]
	mov	8(%edi), %eax
	mov	12(%edi), %edx
	mov	16(%edi), %ecx
	mov	20(%edi), %esi
	mov	24(%edi), %edi
	mov	%eax, 4(%esp)
	mov	%edx, 8(%esp)
	mov	%ecx, 12(%esp)
	mov	%esi, 16(%esp)
	mov	%edi, 20(%esp)
	mov	52(%esp), %eax		C n
	xor	%edi, %edi
	mov	48(%esp), %esi		C up
	lea	-12(%esi,%eax,4), %esi
	and	$3, %eax
	je	L(b0)
	cmp	$2, %eax
	jc	L(b1)
	je	L(b2)

L(b3):	mov	4(%esi), %eax
	mull	4(%esp)
	mov	(%esi), %ebp
	add	%eax, %ebp
	adc	%edx, %edi
	mov	8(%esi), %eax
	mull	8(%esp)
	lea	-12(%esi), %esi
	jmp	L(m0)

L(b0):	mov	(%esi), %eax
	mull	4(%esp)
	mov	-4(%esi), %ebp
	add	%eax, %ebp
	adc	%edx, %edi
	mov	4(%esi), %eax
	mull	8(%esp)
	add	%eax, %ebp
	adc	%edx, %edi
	mov	8(%esi), %eax
	mull	12(%esp)
	lea	-16(%esi), %esi
	jmp	L(m0)

L(b1):	mov	8(%esi), %ebp
	lea	-4(%esi), %esi
	jmp	L(m1)

L(b2):	mov	8(%esi), %edi
	mov	4(%esi), %ebp
	lea	-8(%esi), %esi
	jmp	L(m1)

	ALIGN(16)
L(top):	mov	(%esi), %eax
	mull	4(%esp)
	mov	-4(%esi), %ebx
	xor	%ecx, %ecx
	add	%eax, %ebx
	adc	%edx, %ecx
	mov	4(%esi), %eax
	mull	8(%esp)
	add	%eax, %ebx
	adc	%edx, %ecx
	mov	8(%esi), %eax
	mull	12(%esp)
	add	%eax, %ebx
	adc	%edx, %ecx
	lea	-16(%esi), %esi
	mov	16(%esp), %eax
	mul	%ebp
	add	%eax, %ebx
	adc	%edx, %ecx
	mov	20(%esp), %eax
	mul	%edi
	mov	%ebx, %ebp
	mov	%ecx, %edi
L(m0):	add	%eax, %ebp
	adc	%edx, %edi
L(m1):	subl	$4, 52(%esp)
	ja	L(top)

L(end):	mov	4(%esp), %eax
	mul	%edi
	mov	60(%esp), %edi
	add	%eax, %ebp
	adc	$0, %edx
	mov	4(%edi), %ecx
	mov	%edx, %esi
	mov	%ebp, %eax
	sal	%cl, %esi
	mov	%ecx, %ebx
	neg	%ecx
	shr	%cl, %eax
	or	%esi, %eax
	lea	1(%eax), %esi
	mull	(%edi)
	mov	%ebx, %ecx
	mov	%eax, %ebx
	mov	%ebp, %eax
	mov	56(%esp), %ebp
	sal	%cl, %eax
	add	%eax, %ebx
	adc	%esi, %edx
	imul	%ebp, %edx
	sub	%edx, %eax
	lea	(%eax,%ebp), %edx
	cmp	%eax, %ebx
	cmovc(	%edx, %eax)
	mov	%eax, %edx
	sub	%ebp, %eax
	cmovc(	%edx, %eax)
	add	$28, %esp
	pop	%ebx
	pop	%esi
	pop	%edi
	pop	%ebp
	shr	%cl, %eax
	ret
EPILOGUE()

	ALIGN(16)
PROLOGUE(mpn_mod_1s_4p_cps)
C CAUTION: This is the same code as in pentium4/sse2/mod_1_4.asm
	push	%ebp
	push	%edi
	push	%esi
	push	%ebx
	mov	20(%esp), %ebp		C FIXME: avoid bp for 0-idx
	mov	24(%esp), %ebx
	bsr	%ebx, %ecx
	xor	$31, %ecx
	sal	%cl, %ebx		C b << cnt
	mov	%ebx, %edx
	not	%edx
	mov	$-1, %eax
	div	%ebx
	xor	%edi, %edi
	sub	%ebx, %edi
	mov	$1, %esi
	mov	%eax, (%ebp)		C store bi
	mov	%ecx, 4(%ebp)		C store cnt
	shld	%cl, %eax, %esi
	imul	%edi, %esi
	mov	%eax, %edi
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 8(%ebp)		C store B1modb

	not	%edx
	imul	%ebx, %edx
	lea	(%edx,%ebx), %esi
	cmp	%edx, %eax
	cmovnc(	%edx, %esi)
	mov	%edi, %eax
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 12(%ebp)		C store B2modb

	not	%edx
	imul	%ebx, %edx
	lea	(%edx,%ebx), %esi
	cmp	%edx, %eax
	cmovnc(	%edx, %esi)
	mov	%edi, %eax
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 16(%ebp)		C store B3modb

	not	%edx
	imul	%ebx, %edx
	lea	(%edx,%ebx), %esi
	cmp	%edx, %eax
	cmovnc(	%edx, %esi)
	mov	%edi, %eax
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 20(%ebp)		C store B4modb

	not	%edx
	imul	%ebx, %edx
	add	%edx, %ebx
	cmp	%edx, %eax
	cmovnc(	%edx, %ebx)

	shr	%cl, %ebx
	mov	%ebx, 24(%ebp)		C store B5modb

	pop	%ebx
	pop	%esi
	pop	%edi
	pop	%ebp
	ret
EPILOGUE()
