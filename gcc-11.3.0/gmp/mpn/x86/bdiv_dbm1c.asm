dnl  x86 mpn_bdiv_dbm1.

dnl  Copyright 2008, 2011 Free Software Foundation, Inc.

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
C P6 model 13 (Dothan)		 5.1
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)	13.67
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom
C AMD K6
C AMD K7			 3.5
C AMD K8
C AMD K10


C TODO
C  * Optimize for more x86 processors

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_bdiv_dbm1c)
	mov	16(%esp), %ecx		C d
	push	%esi
	mov	12(%esp), %esi		C ap
	push	%edi
	mov	12(%esp), %edi		C qp
	push	%ebp
	mov	24(%esp), %ebp		C n
	push	%ebx

	mov	(%esi), %eax
	mul	%ecx
	mov	36(%esp), %ebx
	sub	%eax, %ebx
	mov	%ebx, (%edi)
	sbb	%edx, %ebx

	mov	%ebp, %eax
	and	$3, %eax
	jz	L(b0)
	cmp	$2, %eax
	jc	L(b1)
	jz	L(b2)

L(b3):	lea	-8(%esi), %esi
	lea	8(%edi), %edi
	add	$-3, %ebp
	jmp	L(3)

L(b0):	mov	4(%esi), %eax
	lea	-4(%esi), %esi
	lea	12(%edi), %edi
	add	$-4, %ebp
	jmp	L(0)

L(b2):	mov	4(%esi), %eax
	lea	4(%esi), %esi
	lea	4(%edi), %edi
	add	$-2, %ebp
	jmp	L(2)

	ALIGN(8)
L(top):	mov	4(%esi), %eax
	mul	%ecx
	lea	16(%edi), %edi
	sub	%eax, %ebx
	mov	8(%esi), %eax
	mov	%ebx, -12(%edi)
	sbb	%edx, %ebx
L(0):	mul	%ecx
	sub	%eax, %ebx
	mov	%ebx, -8(%edi)
	sbb	%edx, %ebx
L(3):	mov	12(%esi), %eax
	mul	%ecx
	sub	%eax, %ebx
	mov	%ebx, -4(%edi)
	mov	16(%esi), %eax
	lea	16(%esi), %esi
	sbb	%edx, %ebx
L(2):	mul	%ecx
	sub	%eax, %ebx
	mov	%ebx, 0(%edi)
	sbb	%edx, %ebx
L(b1):	add	$-4, %ebp
	jns	L(top)

	mov	%ebx, %eax
	pop	%ebx
	pop	%ebp
	pop	%edi
	pop	%esi
	ret
EPILOGUE()
