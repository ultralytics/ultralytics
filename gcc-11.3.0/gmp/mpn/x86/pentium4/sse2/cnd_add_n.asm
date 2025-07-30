dnl  Intel Pentium-4 mpn_cnd_add_n -- mpn addition.

dnl  Copyright 2001, 2002, 2013 Free Software Foundation, Inc.

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
C P6 model 0-8,10-12		 -
C P6 model 9   (Banias)		 ?
C P6 model 13  (Dothan)		 4.67
C P4 model 0-1 (Willamette)	 ?
C P4 model 2   (Northwood)	 5
C P4 model 3-4 (Prescott)	 5.25

defframe(PARAM_SIZE, 20)
defframe(PARAM_SRC2, 16)
defframe(PARAM_SRC1, 12)
defframe(PARAM_DST,  8)
defframe(PARAM_CND,  4)

dnl  re-use parameter space
define(SAVE_EBX,`PARAM_SRC1')

define(`cnd', `%mm3')

	TEXT
	ALIGN(8)

	ALIGN(8)
PROLOGUE(mpn_cnd_add_n)
deflit(`FRAME',0)
	pxor	%mm0, %mm0

	mov	PARAM_CND, %eax
	neg	%eax
	sbb	%eax, %eax
	movd	%eax, cnd

	mov	PARAM_SRC1, %eax
	mov	%ebx, SAVE_EBX
	mov	PARAM_SRC2, %ebx
	mov	PARAM_DST, %edx
	mov	PARAM_SIZE, %ecx

	lea	(%eax,%ecx,4), %eax	C src1 end
	lea	(%ebx,%ecx,4), %ebx	C src2 end
	lea	(%edx,%ecx,4), %edx	C dst end
	neg	%ecx			C -size

L(top):	movd	(%ebx,%ecx,4), %mm2
	movd	(%eax,%ecx,4), %mm1
	pand	cnd, %mm2
	paddq	%mm2, %mm1

	paddq	%mm1, %mm0
	movd	%mm0, (%edx,%ecx,4)

	psrlq	$32, %mm0

	add	$1, %ecx
	jnz	L(top)

	movd	%mm0, %eax
	mov	SAVE_EBX, %ebx
	emms
	ret

EPILOGUE()
