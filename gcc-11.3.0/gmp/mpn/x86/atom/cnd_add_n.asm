dnl  X86 mpn_cnd_add_n optimised for Intel Atom.

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
C P6 model 13  (Dothan)		 ?
C P4 model 0-1 (Willamette)	 ?
C P4 model 2   (Northwood)	 ?
C P4 model 3-4 (Prescott)	 ?
C Intel atom			 4.67
C AMD K6			 ?
C AMD K7			 ?
C AMD K8			 ?


define(`rp',  `%edi')
define(`up',  `%esi')
define(`vp',  `%ebp')
define(`n',   `%ecx')
define(`cnd', `20(%esp)')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_cnd_add_n)
	push	%edi
	push	%esi
	push	%ebx
	push	%ebp

	mov	cnd, %eax		C make cnd into a mask (1)
	mov	24(%esp), rp
	neg	%eax			C make cnd into a mask (1)
	mov	28(%esp), up
	sbb	%eax, %eax		C make cnd into a mask (1)
	mov	32(%esp), vp
	mov	%eax, cnd		C make cnd into a mask (1)
	mov	36(%esp), n

	xor	%edx, %edx

	shr	$1, n
	jnc	L(top)

	mov	0(vp), %eax
	and	cnd, %eax
	lea	4(vp), vp
	add	0(up), %eax
	lea	4(rp), rp
	lea	4(up), up
	sbb	%edx, %edx
	mov	%eax, -4(rp)
	inc	n
	dec	n
	je	L(end)

L(top):	sbb	%edx, %edx
	mov	0(vp), %eax
	and	cnd, %eax
	lea	8(vp), vp
	lea	8(rp), rp
	mov	-4(vp), %ebx
	and	cnd, %ebx
	add	%edx, %edx
	adc	0(up), %eax
	lea	8(up), up
	mov	%eax, -8(rp)
	adc	-4(up), %ebx
	dec	n
	mov	%ebx, -4(rp)
	jne	L(top)

L(end):	mov	$0, %eax
	adc	%eax, %eax

	pop	%ebp
	pop	%ebx
	pop	%esi
	pop	%edi
	ret
EPILOGUE()
ASM_END()
