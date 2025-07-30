dnl  AMD Athlon mpn_com -- mpn bitwise one's complement.

dnl  Copyright 2002 Free Software Foundation, Inc.

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


C K7: 1.0 cycles/limb


C void mpn_com (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C The loop form below is necessary for the claimed speed.  It needs to be
C aligned to a 16 byte boundary and only 16 bytes long.  Maybe that's so it
C fits in a BTB entry.  The adjustments to %eax and %edx avoid offsets on
C the movq's and achieve the necessary size.
C
C If both src and dst are 4mod8, the loop runs at 1.5 c/l.  So long as one
C of the two is 0mod8, it runs at 1.0 c/l.  On that basis dst is checked
C (offset by the size, as per the loop addressing) and one high limb
C processed separately to get alignment.
C
C The padding for the nails case is unattractive, but shouldn't cost any
C cycles.  Explicit .byte's guarantee the desired instructions, at a point
C where we're probably stalled waiting for loads anyway.
C
C Enhancements:
C
C The combination load/pxor/store might be able to be unrolled to approach
C 0.5 c/l if desired.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

	TEXT
	ALIGN(16)

PROLOGUE(mpn_com)
deflit(`FRAME',0)

	movl	PARAM_DST, %edx
	movl	PARAM_SIZE, %ecx
	pcmpeqd	%mm7, %mm7

	leal	(%edx,%ecx,4), %eax
	andl	$4, %eax
ifelse(GMP_NAIL_BITS,0,,
`	psrld	$GMP_NAIL_BITS, %mm7')		C GMP_NUMB_MASK

	movl	PARAM_SRC, %eax
	movd	-4(%eax,%ecx,4), %mm0		C src high limb

ifelse(GMP_NAIL_BITS,0,,
`	C padding for alignment below
	.byte	0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00	C lea 0(%esi),%esi
	.byte	0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00	C lea 0(%edi),%edi
')

	jz	L(aligned)

	pxor	%mm7, %mm0
	movd	%mm0, -4(%edx,%ecx,4)		C dst high limb
	decl	%ecx
	jz	L(done)
L(aligned):

	addl	$4, %eax
	addl	$4, %edx
	decl	%ecx
	jz	L(one)

	C offset 0x30 for no nails, or 0x40 for nails
	ALIGN(16)
L(top):
	C eax	src
	C ebx
	C ecx	counter
	C edx	dst

	subl	$2, %ecx
	movq	(%eax,%ecx,4), %mm0
	pxor	%mm7, %mm0
	movq	%mm0, (%edx,%ecx,4)
	jg	L(top)

	jnz	L(done)				C if size even

L(one):
	movd	-4(%eax), %mm0			C src low limb
	pxor	%mm7, %mm0
	movd	%mm0, -4(%edx)			C dst low limb

L(done):
	emms

	ret

EPILOGUE()
