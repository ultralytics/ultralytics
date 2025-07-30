dnl x86-32 mpn_addmul_1 and mpn_submul_1 optimised for Intel Atom.

dnl  Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

dnl  Copyright 2011 Free Software Foundation, Inc.

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
C			    cycles/limb
C P5				 -
C P6 model 0-8,10-12		 -
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 8
C AMD K6
C AMD K7			 -
C AMD K8
C AMD K10

define(`rp', `%edi')
define(`up', `%esi')
define(`n',  `%ecx')

ifdef(`OPERATION_addmul_1',`
	define(ADDSUB,  add)
	define(func_1,  mpn_addmul_1)
	define(func_1c, mpn_addmul_1c)')
ifdef(`OPERATION_submul_1',`
	define(ADDSUB,  sub)
	define(func_1,  mpn_submul_1)
	define(func_1c, mpn_submul_1c)')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_addmul_1c mpn_submul_1 mpn_submul_1c)

	TEXT
	ALIGN(16)
PROLOGUE(func_1)
	xor	%edx, %edx
L(ent):	push	%edi
	push	%esi
	push	%ebx
	mov	16(%esp), rp
	mov	20(%esp), up
	mov	24(%esp), n
	movd	28(%esp), %mm7
	test	$1, n
	jz	L(fi0or2)
	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	shr	$2, n
	jnc	L(fi1)

L(fi3):	lea	-8(up), up
	lea	-8(rp), rp
	movd	12(up), %mm1
	movd	%mm0, %ebx
	pmuludq	%mm7, %mm1
	add	$1, n			C increment and clear carry
	jmp	L(lo3)

L(fi1):	movd	%mm0, %ebx
	jz	L(wd1)
	movd	4(up), %mm1
	pmuludq	%mm7, %mm1
	jmp	L(lo1)

L(fi0or2):
	movd	(up), %mm1
	pmuludq	%mm7, %mm1
	shr	$2, n
	movd	4(up), %mm0
	jc	L(fi2)
	lea	-4(up), up
	lea	-4(rp), rp
	movd	%mm1, %eax
	pmuludq	%mm7, %mm0
	jmp	L(lo0)

L(fi2):	lea	4(up), up
	add	$1, n			C increment and clear carry
	movd	%mm1, %eax
	lea	-12(rp), rp
	jmp	L(lo2)

C	ALIGN(16)			C alignment seems irrelevant
L(top):	movd	4(up), %mm1
	adc	$0, %edx
	ADDSUB	%eax, 12(rp)
	movd	%mm0, %ebx
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
L(lo1):	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	ADDSUB	%ebx, (rp)
L(lo0):	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	movd	%mm0, %ebx
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	ADDSUB	%eax, 4(rp)
L(lo3):	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	ADDSUB	%ebx, 8(rp)
L(lo2):	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	dec	n
	jnz	L(top)

L(end):	adc	n, %edx			C n is zero here
	ADDSUB	%eax, 12(rp)
	movd	%mm0, %ebx
	lea	16(rp), rp
L(wd1):	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %eax
	adc	n, %eax
	ADDSUB	%ebx, (rp)
	emms
	adc	n, %eax
	pop	%ebx
	pop	%esi
	pop	%edi
	ret
EPILOGUE()
PROLOGUE(func_1c)
	mov	20(%esp), %edx		C carry
	jmp	L(ent)
EPILOGUE()
