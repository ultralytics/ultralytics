dnl  Intel Atom mpn_lshift -- mpn left shift.

dnl  Copyright 2011 Free Software Foundation, Inc.

dnl  Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

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

C mp_limb_t mpn_lshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C			unsigned cnt);

C				  cycles/limb
C				cnt!=1	cnt==1
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 5	 2.5
C AMD K6
C AMD K7
C AMD K8
C AMD K10

defframe(PARAM_CNT, 16)
defframe(PARAM_SIZE,12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)

dnl  re-use parameter space
define(SAVE_UP,`PARAM_CNT')
define(VAR_COUNT,`PARAM_SIZE')
define(SAVE_EBX,`PARAM_SRC')
define(SAVE_EBP,`PARAM_DST')

define(`rp',  `%edi')
define(`up',  `%esi')
define(`cnt',  `%ecx')

ASM_START()
	TEXT
	ALIGN(8)
deflit(`FRAME',0)
PROLOGUE(mpn_lshift)
	mov	PARAM_CNT, cnt
	mov	PARAM_SIZE, %edx
	mov	up, SAVE_UP
	mov	PARAM_SRC, up
	push	rp			FRAME_pushl()
	mov	PARAM_DST, rp

C We can use faster code for shift-by-1 under certain conditions.
	cmp	$1,cnt
	jne	L(normal)
	cmpl	rp, up
	jnc	L(special)		C jump if s_ptr + 1 >= res_ptr
	leal	(up,%edx,4),%eax
	cmpl	%eax,rp
	jnc	L(special)		C jump if res_ptr >= s_ptr + size

L(normal):
	lea	-4(up,%edx,4), up
	mov	%ebx, SAVE_EBX
	lea	-4(rp,%edx,4), rp

	shr	%edx
	mov	(up), %eax
	mov	%edx, VAR_COUNT
	jnc	L(evn)

	mov	%eax, %ebx
	shl	%cl, %ebx
	neg	cnt
	shr	%cl, %eax
	test	%edx, %edx
	jnz	L(gt1)
	mov	%ebx, (rp)
	jmp	L(quit)

L(gt1):	mov	%ebp, SAVE_EBP
	push	%eax
	mov	-4(up), %eax
	mov	%eax, %ebp
	shr	%cl, %eax
	jmp	L(lo1)

L(evn):	mov	%ebp, SAVE_EBP
	neg	cnt
	mov	%eax, %ebp
	mov	-4(up), %edx
	shr	%cl, %eax
	mov	%edx, %ebx
	shr	%cl, %edx
	neg	cnt
	decl	VAR_COUNT
	lea	4(rp), rp
	lea	-4(up), up
	jz	L(end)
	push	%eax			FRAME_pushl()

	ALIGN(8)
L(top):	shl	%cl, %ebp
	or	%ebp, %edx
	shl	%cl, %ebx
	neg	cnt
	mov	-4(up), %eax
	mov	%eax, %ebp
	mov	%edx, -4(rp)
	shr	%cl, %eax
	lea	-8(rp), rp
L(lo1):	mov	-8(up), %edx
	or	%ebx, %eax
	mov	%edx, %ebx
	shr	%cl, %edx
	lea	-8(up), up
	neg	cnt
	mov	%eax, (rp)
	decl	VAR_COUNT
	jg	L(top)

	pop	%eax			FRAME_popl()
L(end):
	shl	%cl, %ebp
	shl	%cl, %ebx
	or	%ebp, %edx
	mov	SAVE_EBP, %ebp
	mov	%edx, -4(rp)
	mov	%ebx, -8(rp)

L(quit):
	mov	SAVE_UP, up
	mov	SAVE_EBX, %ebx
	pop	rp			FRAME_popl()
	ret

L(special):
deflit(`FRAME',4)
	lea	3(%edx), %eax		C size + 3
	dec	%edx			C size - 1
	mov	(up), %ecx
	shr	$2, %eax		C (size + 3) / 4
	and	$3, %edx		C (size - 1) % 4
	jz	L(goloop)		C jmp if  size == 1 (mod 4)
	shr	%edx
	jnc	L(odd)			C jum if  size == 3 (mod 4)

	add	%ecx, %ecx
	lea	4(up), up
	mov	%ecx, (rp)
	mov	(up), %ecx
	lea	4(rp), rp

	dec	%edx
	jnz	L(goloop)		C jump if  size == 0 (mod 4)
L(odd):	lea	-8(up), up
	lea	-8(rp), rp
	jmp	L(sentry)		C reached if size == 2 or 3 (mod 4)

L(sloop):
	adc	%ecx, %ecx
	mov	4(up), %edx
	mov	%ecx, (rp)
	adc	%edx, %edx
	mov	8(up), %ecx
	mov	%edx, 4(rp)
L(sentry):
	adc	%ecx, %ecx
	mov	12(up), %edx
	mov	%ecx, 8(rp)
	adc	%edx, %edx
	lea	16(up), up
	mov	%edx, 12(rp)
	lea	16(rp), rp
	mov	(up), %ecx
L(goloop):
	decl	%eax
	jnz	L(sloop)

L(squit):
	adc	%ecx, %ecx
	mov	%ecx, (rp)
	adc	%eax, %eax

	mov	SAVE_UP, up
	pop	rp			FRAME_popl()
	ret
EPILOGUE()
ASM_END()
