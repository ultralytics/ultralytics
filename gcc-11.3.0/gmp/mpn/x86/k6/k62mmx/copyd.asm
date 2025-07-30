dnl  AMD K6-2 mpn_copyd -- copy limb vector, decrementing.

dnl  Copyright 2001, 2002 Free Software Foundation, Inc.

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


C K6-2: 1.0 cycles/limb


C void mpn_copyd (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C The loop here is no faster than a rep movsl at 1.0 c/l, but it avoids a 30
C cycle startup time, which amounts for instance to a 2x speedup at 15
C limbs.
C
C If dst is 4mod8 the loop would be 1.17 c/l, but that's avoided by
C processing one limb separately to make it aligned.  This and a final odd
C limb are handled in a branch-free fashion, ending up re-copying if the
C special case isn't needed.
C
C Alternatives:
C
C There used to be a big unrolled version of this, running at 0.56 c/l if
C the destination was aligned, but that seemed rather excessive for the
C relative importance of copyd.
C
C If the destination alignment is ignored and just left to run at 1.17 c/l
C some code size and a fixed few cycles can be saved.  Considering how few
C uses copyd finds perhaps that should be favoured.  The current code has
C the attraction of being no slower than a basic rep movsl though.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

dnl  re-using parameter space
define(SAVE_EBX,`PARAM_SIZE')

	TEXT
	ALIGN(16)

PROLOGUE(mpn_copyd)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	%ebx, SAVE_EBX

	movl	PARAM_SRC, %eax
	movl	PARAM_DST, %edx

	subl	$1, %ecx		C better code alignment than decl
	jb	L(zero)

	jz	L(one_more)
	leal	4(%edx,%ecx,4), %ebx

Zdisp(	movd,	0,(%eax,%ecx,4), %mm0)	C high limb
Zdisp(	movd,	%mm0, 0,(%edx,%ecx,4))	C Zdisp for good code alignment

	cmpl	$1, %ecx
	je	L(one_more)

	shrl	$2, %ebx
	andl	$1, %ebx		C 1 if dst[size-2] unaligned

	subl	%ebx, %ecx
	nop				C code alignment

L(top):
	C eax	src
	C ebx
	C ecx	counter
	C edx	dst

	movq	-4(%eax,%ecx,4), %mm0
	subl	$2, %ecx

	movq	%mm0, 4(%edx,%ecx,4)
	ja	L(top)


L(one_more):
	movd	(%eax), %mm0
	movd	%mm0, (%edx)

	movl	SAVE_EBX, %ebx
	emms_or_femms
L(zero):
	ret

EPILOGUE()
