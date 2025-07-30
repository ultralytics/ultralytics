dnl  mc68020 mpn_rshift -- mpn right shift.

dnl  Copyright 1996, 1999-2003 Free Software Foundation, Inc.

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


C           cycles/limb
C        shift==1  shift>1
C 68040:    9         12


C mp_limb_t mpn_rshift (mp_ptr res_ptr, mp_srcptr s_ptr, mp_size_t s_size,
C                       unsigned cnt);
C
C The "cnt" parameter is either 16 bits or 32 bits depending on
C SIZEOF_UNSIGNED (see ABI notes in mpn/m68k/README).  The value is of
C course only 1 to 31.  When loaded as 16 bits there's garbage in the upper
C half, hence the use of cmpw.  The shift instructions take the their count
C modulo 64, so the upper part doesn't matter to them either.
C

C INPUT PARAMETERS
C res_ptr	(sp + 4)
C s_ptr		(sp + 8)
C s_size	(sp + 12)
C cnt		(sp + 16)

define(res_ptr, `a1')
define(s_ptr,   `a0')
define(s_size,  `d6')
define(cnt,     `d4')

ifdef(`SIZEOF_UNSIGNED',,
`m4_error(`SIZEOF_UNSIGNED not defined, should be in config.m4
')')

PROLOGUE(mpn_rshift)
C Save used registers on the stack.
	moveml	d2-d6/a2, M(-,sp)

C Copy the arguments to registers.
	movel	M(sp,28), res_ptr
	movel	M(sp,32), s_ptr
	movel	M(sp,36), s_size
ifelse(SIZEOF_UNSIGNED,2,
`	movew	M(sp,40), cnt',
`	movel	M(sp,40), cnt')

	moveql	#1, d5
	cmpw	d5, cnt
	bne	L(Lnormal)
	cmpl	res_ptr, s_ptr
	bls	L(Lspecial)		C jump if res_ptr >= s_ptr

ifelse(scale_available_p,1,`
	lea	M(res_ptr,s_size,l,4), a2
',`
	movel	s_size, d0
	asll	#2, d0
	lea	M(res_ptr,d0,l), a2
')
	cmpl	s_ptr, a2
	bls	L(Lspecial)		C jump if s_ptr >= res_ptr + s_size

L(Lnormal):
	moveql	#32, d5
	subl	cnt, d5
	movel	M(s_ptr,+), d2
	movel	d2, d0
	lsll	d5, d0		C compute carry limb

	lsrl	cnt, d2
	movel	d2, d1
	subql	#1, s_size
	beq	L(Lend)
	lsrl	#1, s_size
	bcs	L(L1)
	subql	#1, s_size

L(Loop):
	movel	M(s_ptr,+), d2
	movel	d2, d3
	lsll	d5, d3
	orl	d3, d1
	movel	d1, M(res_ptr,+)
	lsrl	cnt, d2
L(L1):
	movel	M(s_ptr,+), d1
	movel	d1, d3
	lsll	d5, d3
	orl	d3, d2
	movel	d2, M(res_ptr,+)
	lsrl	cnt, d1

	dbf	s_size, L(Loop)
	subl	#0x10000, s_size
	bcc	L(Loop)

L(Lend):
	movel	d1, M(res_ptr)	C store most significant limb

C Restore used registers from stack frame.
	moveml	M(sp,+), d2-d6/a2
	rts

C We loop from most significant end of the arrays, which is only permissable
C if the source and destination don't overlap, since the function is
C documented to work for overlapping source and destination.

L(Lspecial):
ifelse(scale_available_p,1,`
	lea	M(s_ptr,s_size,l,4), s_ptr
	lea	M(res_ptr,s_size,l,4), res_ptr
',`
	movel	s_size, d0
	asll	#2, d0
	addl	d0, s_ptr
	addl	d0, res_ptr
')

	clrl	d0			C initialize carry
	eorw	#1, s_size
	lsrl	#1, s_size
	bcc	L(LL1)
	subql	#1, s_size

L(LLoop):
	movel	M(-,s_ptr), d2
	roxrl	#1, d2
	movel	d2, M(-,res_ptr)
L(LL1):
	movel	M(-,s_ptr), d2
	roxrl	#1, d2
	movel	d2, M(-,res_ptr)

	dbf	s_size, L(LLoop)
	roxrl	#1, d0		C save cy in msb
	subl	#0x10000, s_size
	bcs	L(LLend)
	addl	d0, d0		C restore cy
	bra	L(LLoop)

L(LLend):
C Restore used registers from stack frame.
	moveml	M(sp,+), d2-d6/a2
	rts

EPILOGUE(mpn_rshift)
