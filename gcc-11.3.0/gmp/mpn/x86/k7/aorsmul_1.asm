dnl  AMD K7 mpn_addmul_1/mpn_submul_1 -- add or subtract mpn multiple.

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
C P6 model 0-8,10-12
C P6 model 9  (Banias)		 6.5
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C AMD K6
C AMD K7			 3.75
C AMD K8

C TODO
C  * Improve feed-in and wind-down code.  We beat the old code for all n != 1,
C    but lose by 2x for n == 1.

ifdef(`OPERATION_addmul_1',`
      define(`ADDSUB',        `add')
      define(`func',  `mpn_addmul_1')
')
ifdef(`OPERATION_submul_1',`
      define(`ADDSUB',        `sub')
      define(`func',  `mpn_submul_1')
')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	add	$-16, %esp
	mov	%ebp, (%esp)
	mov	%ebx, 4(%esp)
	mov	%esi, 8(%esp)
	mov	%edi, 12(%esp)

	mov	20(%esp), %edi
	mov	24(%esp), %esi
	mov	28(%esp), %eax
	mov	32(%esp), %ecx
	mov	%eax, %ebx
	shr	$2, %eax
	mov	%eax, 28(%esp)
	mov	(%esi), %eax
	and	$3, %ebx
	jz	L(b0)
	cmp	$2, %ebx
	jz	L(b2)
	jg	L(b3)

L(b1):	lea	-4(%esi), %esi
	lea	-4(%edi), %edi
	mul	%ecx
	mov	%eax, %ebx
	mov	%edx, %ebp
	cmpl	$0, 28(%esp)
	jz	L(cj1)
	mov	8(%esi), %eax
	jmp	L(1)

L(b2):	mul	%ecx
	mov	%eax, %ebp
	mov	4(%esi), %eax
	mov	%edx, %ebx
	cmpl	$0, 28(%esp)
	jne	L(2)
	jmp	L(cj2)

L(b3):	lea	-12(%esi), %esi
	lea	-12(%edi), %edi
	mul	%ecx
	mov	%eax, %ebx
	mov	%edx, %ebp
	mov	16(%esi), %eax
	incl	28(%esp)
	jmp	L(3)

L(b0):	lea	-8(%esi), %esi
	lea	-8(%edi), %edi
	mul	%ecx
	mov	%eax, %ebp
	mov	12(%esi), %eax
	mov	%edx, %ebx
	jmp	L(0)

	ALIGN(16)
L(top):	lea	16(%edi), %edi
L(2):	mul	%ecx
	ADDSUB	%ebp, 0(%edi)
	mov	$0, %ebp
	adc	%eax, %ebx
	mov	8(%esi), %eax
	adc	%edx, %ebp
L(1):	mul	%ecx
	ADDSUB	%ebx, 4(%edi)
	mov	$0, %ebx
	adc	%eax, %ebp
	mov	12(%esi), %eax
	adc	%edx, %ebx
L(0):	mul	%ecx
	ADDSUB	%ebp, 8(%edi)
	mov	$0, %ebp
	adc	%eax, %ebx
	adc	%edx, %ebp
	mov	16(%esi), %eax
L(3):	mul	%ecx
	ADDSUB	%ebx, 12(%edi)
	adc	%eax, %ebp
	mov	20(%esi), %eax
	lea	16(%esi), %esi
	mov	$0, %ebx
	adc	%edx, %ebx
	decl	28(%esp)
	jnz	L(top)

L(end):	lea	16(%edi), %edi
L(cj2):	mul	%ecx
	ADDSUB	%ebp, (%edi)
	adc	%eax, %ebx
	adc	$0, %edx
L(cj1):	ADDSUB	%ebx, 4(%edi)
	adc	$0, %edx
	mov	%edx, %eax
	mov	(%esp), %ebp
	mov	4(%esp), %ebx
	mov	8(%esp), %esi
	mov	12(%esp), %edi
	add	$16, %esp
	ret
EPILOGUE()
ASM_END()
