dnl  X86 mpn_cnd_add_n, mpn_cnd_sub_n

dnl  Copyright 2013 Free Software Foundation, Inc.

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
C P6 model 9   (Banias)		 ?
C P6 model 13  (Dothan)		 5.4
C P4 model 0-1 (Willamette)	 ?
C P4 model 2   (Northwood)	14.5
C P4 model 3-4 (Prescott)	21
C Intel atom			11
C AMD K6			 ?
C AMD K7			 3.4
C AMD K8			 ?


define(`rp',  `%edi')
define(`up',  `%esi')
define(`vp',  `%ebp')
define(`n',   `%ecx')
define(`cnd', `20(%esp)')
define(`cy',  `%edx')

ifdef(`OPERATION_cnd_add_n', `
	define(ADDSUB,	      add)
	define(ADCSBB,	      adc)
	define(func,	      mpn_cnd_add_n)')
ifdef(`OPERATION_cnd_sub_n', `
	define(ADDSUB,	      sub)
	define(ADCSBB,	      sbb)
	define(func,	      mpn_cnd_sub_n)')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	add	$-16, %esp
	mov	%ebp, (%esp)
	mov	%ebx, 4(%esp)
	mov	%esi, 8(%esp)
	mov	%edi, 12(%esp)

	C make cnd into a full mask
	mov	cnd, %eax
	neg	%eax
	sbb	%eax, %eax
	mov	%eax, cnd

	C load parameters into registers
	mov	24(%esp), rp
	mov	28(%esp), up
	mov	32(%esp), vp
	mov	36(%esp), n

	mov	(vp), %eax
	mov	(up), %ebx

	C put operand pointers just beyond their last limb
	lea	(vp,n,4), vp
	lea	(up,n,4), up
	lea	-4(rp,n,4), rp
	neg	n

	and	cnd, %eax
	ADDSUB	%eax, %ebx
	sbb	cy, cy
	inc	n
	je	L(end)

	ALIGN(16)
L(top):	mov	(vp,n,4), %eax
	and	cnd, %eax
	mov	%ebx, (rp,n,4)
	mov	(up,n,4), %ebx
	add	cy, cy
	ADCSBB	%eax, %ebx
	sbb	cy, cy
	inc	n
	jne	L(top)

L(end):	mov	%ebx, (rp)
	xor	%eax, %eax
	sub	cy, %eax

	mov	(%esp), %ebp
	mov	4(%esp), %ebx
	mov	8(%esp), %esi
	mov	12(%esp), %edi
	add	$16, %esp
	ret
EPILOGUE()
ASM_END()
