dnl  AMD K7 mpn_mul_1.

dnl  Copyright 1999-2002, 2005, 2008 Free Software Foundation, Inc.

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
C P5
C P6 model 0-8,10-12)
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C AMD K6
C AMD K7			 3.25
C AMD K8

C TODO
C  * Improve feed-in and wind-down code.  We beat the old code for all n != 1,
C    but we might be able to do even better.
C  * The feed-in code for mul_1c is crude.

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_1c)
	add	$-16, %esp
	mov	%ebp, (%esp)
	mov	%ebx, 4(%esp)
	mov	%esi, 8(%esp)
	mov	%edi, 12(%esp)

	mov	20(%esp), %edi
	mov	24(%esp), %esi
	mov	28(%esp), %ebp
	mov	32(%esp), %ecx
	mov	%ebp, %ebx
	shr	$2, %ebp
	mov	%ebp, 28(%esp)
	mov	(%esi), %eax
	and	$3, %ebx
	jz	L(c0)
	cmp	$2, %ebx
	mov	36(%esp), %ebx
	jz	L(c2)
	jg	L(c3)

L(c1):	lea	-4(%edi), %edi
	mul	%ecx
	test	%ebp, %ebp
	jnz	1f
	add	%ebx, %eax
	mov	%eax, 4(%edi)
	mov	%edx, %eax
	adc	%ebp, %eax
	jmp	L(rt)
1:	add	%eax, %ebx
	mov	$0, %ebp
	adc	%edx, %ebp
	mov	4(%esi), %eax
	jmp	L(1)

L(c2):	lea	4(%esi), %esi
	mul	%ecx
	test	%ebp, %ebp
	mov	%ebx, %ebp
	jnz	2f
	add	%eax, %ebp
	mov	$0, %ebx
	adc	%edx, %ebx
	mov	(%esi), %eax
	jmp	L(cj2)
2:	add	%eax, %ebp
	mov	$0, %ebx
	adc	%edx, %ebx
	mov	(%esi), %eax
	jmp	L(2)

L(c3):	lea	8(%esi), %esi
	lea	-12(%edi), %edi
	mul	%ecx
	add	%eax, %ebx
	mov	$0, %ebp
	adc	%edx, %ebp
	mov	-4(%esi), %eax
	incl	28(%esp)
	jmp	L(3)

L(c0):	mov	36(%esp), %ebx
	lea	-4(%esi), %esi
	lea	-8(%edi), %edi
	mul	%ecx
	mov	%ebx, %ebp
	add	%eax, %ebp
	mov	$0, %ebx
	adc	%edx, %ebx
	mov	8(%esi), %eax
	jmp	L(0)

EPILOGUE()
	ALIGN(16)
PROLOGUE(mpn_mul_1)
	add	$-16, %esp
	mov	%ebp, (%esp)
	mov	%ebx, 4(%esp)
	mov	%esi, 8(%esp)
	mov	%edi, 12(%esp)

	mov	20(%esp), %edi
	mov	24(%esp), %esi
	mov	28(%esp), %ebp
	mov	32(%esp), %ecx
	mov	%ebp, %ebx
	shr	$2, %ebp
	mov	%ebp, 28(%esp)
	mov	(%esi), %eax
	and	$3, %ebx
	jz	L(b0)
	cmp	$2, %ebx
	jz	L(b2)
	jg	L(b3)

L(b1):	lea	-4(%edi), %edi
	mul	%ecx
	test	%ebp, %ebp
	jnz	L(gt1)
	mov	%eax, 4(%edi)
	mov	%edx, %eax
	jmp	L(rt)
L(gt1):	mov	%eax, %ebx
	mov	%edx, %ebp
	mov	4(%esi), %eax
	jmp	L(1)

L(b2):	lea	4(%esi), %esi
	mul	%ecx
	test	%ebp, %ebp
	mov	%eax, %ebp
	mov	%edx, %ebx
	mov	(%esi), %eax
	jnz	L(2)
	jmp	L(cj2)

L(b3):	lea	8(%esi), %esi
	lea	-12(%edi), %edi
	mul	%ecx
	mov	%eax, %ebx
	mov	%edx, %ebp
	mov	-4(%esi), %eax
	incl	28(%esp)
	jmp	L(3)

L(b0):	lea	-4(%esi), %esi
	lea	-8(%edi), %edi
	mul	%ecx
	mov	%eax, %ebp
	mov	%edx, %ebx
	mov	8(%esi), %eax
	jmp	L(0)

	ALIGN(16)
L(top):	mov	$0, %ebx
	adc	%edx, %ebx
L(2):	mul	%ecx
	add	%eax, %ebx
	mov	%ebp, 0(%edi)
	mov	4(%esi), %eax
	mov	$0, %ebp
	adc	%edx, %ebp
L(1):	mul	%ecx
	add	%eax, %ebp
	mov	8(%esi), %eax
	mov	%ebx, 4(%edi)
	mov	$0, %ebx
	adc	%edx, %ebx
L(0):	mov	%ebp, 8(%edi)
	mul	%ecx
	add	%eax, %ebx
	mov	12(%esi), %eax
	lea	16(%esi), %esi
	mov	$0, %ebp
	adc	%edx, %ebp
L(3):	mov	%ebx, 12(%edi)
	mul	%ecx
	lea	16(%edi), %edi
	add	%eax, %ebp
	decl	28(%esp)
	mov	0(%esi), %eax
	jnz	L(top)

L(end):	mov	$0, %ebx
	adc	%edx, %ebx
L(cj2):	mul	%ecx
	add	%eax, %ebx
	mov	%ebp, (%edi)
L(cj1):	mov	%ebx, 4(%edi)
	adc	$0, %edx
	mov	%edx, %eax

L(rt):	mov	(%esp), %ebp
	mov	4(%esp), %ebx
	mov	8(%esp), %esi
	mov	12(%esp), %edi
	add	$16, %esp
	ret
EPILOGUE()
ASM_END()
