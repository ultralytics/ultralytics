dnl  Intel P6 mpn_add_n/mpn_sub_n -- mpn add or subtract.

dnl  Copyright 2006 Free Software Foundation, Inc.

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

C TODO:
C  * Avoid indexed addressing, it makes us stall on the two-ported register
C    file.

C			    cycles/limb
C P6 model 0-8,10-12		3.17
C P6 model 9   (Banias)		2.15
C P6 model 13  (Dothan)		2.25


define(`rp',	`%edi')
define(`up',	`%esi')
define(`vp',	`%ebx')
define(`n',	`%ecx')

ifdef(`OPERATION_add_n', `
	define(ADCSBB,	      adc)
	define(func,	      mpn_add_n)
	define(func_nc,	      mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
	define(ADCSBB,	      sbb)
	define(func,	      mpn_sub_n)
	define(func_nc,	      mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ASM_START()

	TEXT
	ALIGN(16)

PROLOGUE(func)
	xor	%edx, %edx
L(start):
	push	%edi
	push	%esi
	push	%ebx

	mov	16(%esp), rp
	mov	20(%esp), up
	mov	24(%esp), vp
	mov	28(%esp), n

	lea	(up,n,4), up
	lea	(vp,n,4), vp
	lea	(rp,n,4), rp

	neg	n
	mov	n, %eax
	and	$-8, n
	and	$7, %eax
	shl	$2, %eax			C 4x
ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	lea	L(ent) (%eax,%eax,2), %eax	C 12x
')

	shr	%edx				C set cy flag
	jmp	*%eax

ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	lea	(%eax,%eax,2), %eax
	add	$L(ent)-L(here), %eax
	add	(%esp), %eax
	ret_internal
')

L(end):
	sbb	%eax, %eax
	neg	%eax
	pop	%ebx
	pop	%esi
	pop	%edi
	ret

	ALIGN(16)
L(top):
	jecxz	L(end)
L(ent):
Zdisp(	mov,	0,(up,n,4), %eax)
Zdisp(	ADCSBB,	0,(vp,n,4), %eax)
Zdisp(	mov,	%eax, 0,(rp,n,4))

	mov	4(up,n,4), %edx
	ADCSBB	4(vp,n,4), %edx
	mov	%edx, 4(rp,n,4)

	mov	8(up,n,4), %eax
	ADCSBB	8(vp,n,4), %eax
	mov	%eax, 8(rp,n,4)

	mov	12(up,n,4), %edx
	ADCSBB	12(vp,n,4), %edx
	mov	%edx, 12(rp,n,4)

	mov	16(up,n,4), %eax
	ADCSBB	16(vp,n,4), %eax
	mov	%eax, 16(rp,n,4)

	mov	20(up,n,4), %edx
	ADCSBB	20(vp,n,4), %edx
	mov	%edx, 20(rp,n,4)

	mov	24(up,n,4), %eax
	ADCSBB	24(vp,n,4), %eax
	mov	%eax, 24(rp,n,4)

	mov	28(up,n,4), %edx
	ADCSBB	28(vp,n,4), %edx
	mov	%edx, 28(rp,n,4)

	lea	8(n), n
	jmp	L(top)

EPILOGUE()

PROLOGUE(func_nc)
	movl	20(%esp), %edx
	jmp	L(start)
EPILOGUE()
