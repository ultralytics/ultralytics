dnl  HP-PA 2.0 32-bit mpn_add_n -- Add two limb vectors of the same length > 0
dnl  and store sum in a third limb vector.

dnl  Copyright 1997, 1998, 2000-2002 Free Software Foundation, Inc.

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
C res_ptr	gr26
C s1_ptr	gr25
C s2_ptr	gr24
C size		gr23

C This runs at 2 cycles/limb on PA8000.

ASM_START()
PROLOGUE(mpn_add_n)
	sub		%r0,%r23,%r22
	zdep		%r22,30,3,%r28		C r28 = 2 * (-n & 7)
	zdep		%r22,29,3,%r22		C r22 = 4 * (-n & 7)
	sub		%r25,%r22,%r25		C offset s1_ptr
	sub		%r24,%r22,%r24		C offset s2_ptr
	sub		%r26,%r22,%r26		C offset res_ptr
	blr		%r28,%r0		C branch into loop
	add		%r0,%r0,%r0		C reset carry

LDEF(loop)
	ldw		0(%r25),%r20
	ldw		0(%r24),%r31
	addc		%r20,%r31,%r20
	stw		%r20,0(%r26)

LDEF(7)
	ldw		4(%r25),%r21
	ldw		4(%r24),%r19
	addc		%r21,%r19,%r21
	stw		%r21,4(%r26)

LDEF(6)
	ldw		8(%r25),%r20
	ldw		8(%r24),%r31
	addc		%r20,%r31,%r20
	stw		%r20,8(%r26)

LDEF(5)
	ldw		12(%r25),%r21
	ldw		12(%r24),%r19
	addc		%r21,%r19,%r21
	stw		%r21,12(%r26)

LDEF(4)
	ldw		16(%r25),%r20
	ldw		16(%r24),%r31
	addc		%r20,%r31,%r20
	stw		%r20,16(%r26)

LDEF(3)
	ldw		20(%r25),%r21
	ldw		20(%r24),%r19
	addc		%r21,%r19,%r21
	stw		%r21,20(%r26)

LDEF(2)
	ldw		24(%r25),%r20
	ldw		24(%r24),%r31
	addc		%r20,%r31,%r20
	stw		%r20,24(%r26)

LDEF(1)
	ldw		28(%r25),%r21
	ldo		32(%r25),%r25
	ldw		28(%r24),%r19
	addc		%r21,%r19,%r21
	stw		%r21,28(%r26)
	ldo		32(%r24),%r24
	addib,>		-8,%r23,L(loop)
	ldo		32(%r26),%r26

	bv		(%r2)
	addc		%r0,%r0,%r28
EPILOGUE()
