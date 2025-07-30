dnl  SPARC v8 mpn_addmul_1 -- Multiply a limb vector with a limb and
dnl  add the result to a second limb vector.

dnl  Copyright 1992-1995, 2000 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
C res_ptr	o0
C s1_ptr	o1
C size		o2
C s2_limb	o3

ASM_START()
PROLOGUE(mpn_addmul_1)
	orcc	%g0,%g0,%g2
	ld	[%o1+0],%o4	C 1

	sll	%o2,4,%g1
	and	%g1,(4-1)<<4,%g1
ifdef(`PIC',
`	mov	%o7,%g4		C Save return address register
0:	call	1f
	add	%o7,L(1)-0b,%g3
1:	mov	%g4,%o7		C Restore return address register
',
`	sethi	%hi(L(1)),%g3
	or	%g3,%lo(L(1)),%g3
')
	jmp	%g3+%g1
	nop
L(1):
L(L00):	add	%o0,-4,%o0
	b	L(loop00)	C 4, 8, 12, ...
	add	%o1,-4,%o1
	nop
L(L01):	b	L(loop01)	C 1, 5, 9, ...
	nop
	nop
	nop
L(L10):	add	%o0,-12,%o0	C 2, 6, 10, ...
	b	L(loop10)
	add	%o1,4,%o1
	nop
L(L11):	add	%o0,-8,%o0	C 3, 7, 11, ...
	b	L(loop11)
	add	%o1,-8,%o1
	nop

L(loop):
	addcc	%g3,%g2,%g3	C 1
	ld	[%o1+4],%o4	C 2
	rd	%y,%g2		C 1
	addx	%g0,%g2,%g2
	ld	[%o0+0],%g1	C 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+0]	C 1
L(loop00):
	umul	%o4,%o3,%g3	C 2
	ld	[%o0+4],%g1	C 2
	addxcc	%g3,%g2,%g3	C 2
	ld	[%o1+8],%o4	C 3
	rd	%y,%g2		C 2
	addx	%g0,%g2,%g2
	nop
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+4]	C 2
L(loop11):
	umul	%o4,%o3,%g3	C 3
	addxcc	%g3,%g2,%g3	C 3
	ld	[%o1+12],%o4	C 4
	rd	%y,%g2		C 3
	add	%o1,16,%o1
	addx	%g0,%g2,%g2
	ld	[%o0+8],%g1	C 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+8]	C 3
L(loop10):
	umul	%o4,%o3,%g3	C 4
	addxcc	%g3,%g2,%g3	C 4
	ld	[%o1+0],%o4	C 1
	rd	%y,%g2		C 4
	addx	%g0,%g2,%g2
	ld	[%o0+12],%g1	C 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+12]	C 4
	add	%o0,16,%o0
	addx	%g0,%g2,%g2
L(loop01):
	addcc	%o2,-4,%o2
	bg	L(loop)
	umul	%o4,%o3,%g3	C 1

	addcc	%g3,%g2,%g3	C 4
	rd	%y,%g2		C 4
	addx	%g0,%g2,%g2
	ld	[%o0+0],%g1	C 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+0]	C 4
	addx	%g0,%g2,%o0

	retl
	 nop
EPILOGUE(mpn_addmul_1)
