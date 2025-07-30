dnl  ARM64 mpn_copyd.

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

C	     cycles/limb
C Cortex-A53	 ?
C Cortex-A57	 ?

changecom(@&*$)

define(`rp', `x0')
define(`up', `x1')
define(`n',  `x2')

ASM_START()
PROLOGUE(mpn_copyd)
	add	rp, rp, n, lsl #3
	add	up, up, n, lsl #3

	cmp	n, #3
	b.le	L(bc)

C Copy until rp is 128-bit aligned
	tbz	rp, #3, L(al2)
	sub	up, up, #8
	ld1	{v22.1d}, [up]
	sub	n, n, #1
	sub	rp, rp, #8
	st1	{v22.1d}, [rp]

L(al2):	sub	up, up, #16
	ld1	{v26.2d}, [up]
	subs	n, n, #6
	sub	rp, rp, #16			C offset rp for loop
	b.lt	L(end)

	sub	up, up, #16			C offset up for loop
	mov	x12, #-16

	ALIGN(16)
L(top):	ld1	{v22.2d}, [up], x12
	st1	{v26.2d}, [rp], x12
	ld1	{v26.2d}, [up], x12
	st1	{v22.2d}, [rp], x12
	subs	n, n, #4
	b.ge	L(top)

	add	up, up, #16			C undo up offset

L(end):	st1	{v26.2d}, [rp]

C Copy last 0-3 limbs.  Note that rp is aligned after loop, but not when we
C arrive here via L(bc)
L(bc):	tbz	n, #1, L(tl1)
	sub	up, up, #16
	ld1	{v22.2d}, [up]
	sub	rp, rp, #16
	st1	{v22.2d}, [rp]
L(tl1):	tbz	n, #0, L(tl2)
	sub	up, up, #8
	ld1	{v22.1d}, [up]
	sub	rp, rp, #8
	st1	{v22.1d}, [rp]
L(tl2):	ret
EPILOGUE()
