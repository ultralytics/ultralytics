dnl  IA-64 mpn_and_n, mpn_andn_n, mpn_nand_n, mpn_ior_n, mpn_iorn_n,
dnl  mpn_nior_n, mpn_xor_n, mpn_xnor_n -- mpn bitwise logical operations.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2003-2005 Free Software Foundation, Inc.

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
C Itanium:      2
C Itanium 2:    1

C TODO
C  * Use rp,rpx scheme of aors_n.asm to allow parallel stores (useful in
C    wind-down code).

C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`vp', `r34')
define(`n', `r35')

ifdef(`OPERATION_and_n',
`	define(`func',`mpn_and_n')
	define(`logop',		`and	$1 = $2, $3')
	define(`notormov',	`mov	$1 = $2')')
ifdef(`OPERATION_andn_n',
`	define(`func',`mpn_andn_n')
	define(`logop',		`andcm	$1 = $2, $3')
	define(`notormov',	`mov	$1 = $2')')
ifdef(`OPERATION_nand_n',
`	define(`func',`mpn_nand_n')
	define(`logop',		`and	$1 = $2, $3')
	define(`notormov',	`sub	$1 = -1, $2')')
ifdef(`OPERATION_ior_n',
`	define(`func',`mpn_ior_n')
	define(`logop',		`or	$1 = $2, $3')
	define(`notormov',	`mov	$1 = $2')')
ifdef(`OPERATION_iorn_n',
`	define(`func',`mpn_iorn_n')
	define(`logop',		`andcm	$1 = $3, $2')
	define(`notormov',	`sub	$1 = -1, $2')')
ifdef(`OPERATION_nior_n',
`	define(`func',`mpn_nior_n')
	define(`logop',		`or	$1 = $2, $3')
	define(`notormov',	`sub	$1 = -1, $2')')
ifdef(`OPERATION_xor_n',
`	define(`func',`mpn_xor_n')
	define(`logop',		`xor	$1 = $2, $3')
	define(`notormov',	`mov	$1 = $2')')
ifdef(`OPERATION_xnor_n',
`	define(`func',`mpn_xnor_n')
	define(`logop',		`xor	$1 = $2, $3')
	define(`notormov',	`sub	$1 = -1, $2')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

ASM_START()
PROLOGUE(func)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',
`	addp4	rp = 0, rp			C			M I
	addp4	up = 0, up			C			M I
	addp4	vp = 0, vp			C			M I
	nop.m		0
	nop.m		0
	zxt4	n = n				C			I
	;;
')
{.mmi
	ld8		r10 = [up], 8		C			M
	ld8		r11 = [vp], 8		C			M
	mov.i		r2 = ar.lc		C			I0
}
{.mmi
	and		r14 = 3, n		C			M I
	cmp.lt		p15, p14 = 4, n		C			M I
	shr.u		n = n, 2		C			I0
	;;
}
{.mmi
	cmp.eq		p6, p0 = 1, r14		C			M I
	cmp.eq		p7, p0 = 2, r14		C			M I
	cmp.eq		p8, p0 = 3, r14		C			M I
}
{.bbb
   (p6)	br.dptk		.Lb01			C			B
   (p7)	br.dptk		.Lb10			C			B
   (p8)	br.dptk		.Lb11			C			B
}

.Lb00:	ld8		r17 = [up], 8		C			M
	ld8		r21 = [vp], 8		C			M
	add		n = -2, n		C			M I
	;;
	ld8		r18 = [up], 8		C			M
	ld8		r22 = [vp], 8		C			M
	;;
	ld8		r19 = [up], 8		C			M
	ld8		r23 = [vp], 8		C			M
  (p15)	br.cond.dpnt	.grt4			C			B

	logop(		r14, r10, r11)		C			M I
	;;
	logop(		r15, r17, r21)		C			M I
	notormov(	r8, r14)		C			M I
	br		.Lcj4			C			B

.grt4:	logop(		r14, r10, r11)		C			M I
	ld8		r16 = [up], 8		C			M
	ld8		r20 = [vp], 8		C			M
	;;
	logop(		r15, r17, r21)		C			M I
	ld8		r17 = [up], 8		C			M
	mov.i		ar.lc = n		C			I0
	notormov(	r8, r14)		C			M I
	ld8		r21 = [vp], 8		C			M
	br		.LL00			C			B

.Lb01:	add		n = -1, n		C			M I
	logop(		r15, r10, r11)		C			M I
  (p15)	br.cond.dpnt	.grt1			C			B
	;;

	notormov(	r9, r15)		C			M I
	br		.Lcj1			C			B

.grt1:	ld8		r16 = [up], 8		C			M
	ld8		r20 = [vp], 8		C			M
	;;
	ld8		r17 = [up], 8		C			M
	ld8		r21 = [vp], 8		C			M
	mov.i		ar.lc = n		C			I0
	;;
	ld8		r18 = [up], 8		C			M
	ld8		r22 = [vp], 8		C			M
	;;
	ld8		r19 = [up], 8		C			M
	ld8		r23 = [vp], 8		C			M
	br.cloop.dptk	.grt5			C			B
	;;

	logop(		r14, r16, r20)		C			M I
	notormov(	r9, r15)		C			M I
	br		.Lcj5			C			B

.grt5:	logop(		r14, r16, r20)		C			M I
	ld8		r16 = [up], 8		C			M
	notormov(	r9, r15)		C			M I
	ld8		r20 = [vp], 8		C			M
	br		.LL01			C			B

.Lb10:	ld8		r19 = [up], 8		C			M
	ld8		r23 = [vp], 8		C			M
  (p15)	br.cond.dpnt	.grt2			C			B

	logop(		r14, r10, r11)		C			M I
	;;
	logop(		r15, r19, r23)		C			M I
	notormov(	r8, r14)		C			M I
	br		.Lcj2			C			B

.grt2:	ld8		r16 = [up], 8		C			M
	ld8		r20 = [vp], 8		C			M
	add		n = -1, n		C			M I
	;;
	ld8		r17 = [up], 8		C			M
	ld8		r21 = [vp], 8		C			M
	logop(		r14, r10, r11)		C			M I
	;;
	ld8		r18 = [up], 8		C			M
	ld8		r22 = [vp], 8		C			M
	mov.i		ar.lc = n		C			I0
	;;
	logop(		r15, r19, r23)		C			M I
	ld8		r19 = [up], 8		C			M
	notormov(	r8, r14)		C			M I
	ld8		r23 = [vp], 8		C			M
	br.cloop.dptk	.Loop			C			B
	br		.Lcj6			C			B

.Lb11:	ld8		r18 = [up], 8		C			M
	ld8		r22 = [vp], 8		C			M
	add		n = -1, n		C			M I
	;;
	ld8		r19 = [up], 8		C			M
	ld8		r23 = [vp], 8		C			M
	logop(		r15, r10, r11)		C			M I
  (p15)	br.cond.dpnt	.grt3			C			B
	;;

	logop(		r14, r18, r22)		C			M I
	notormov(	r9, r15)		C			M I
	br		.Lcj3			C			B

.grt3:	ld8		r16 = [up], 8		C			M
	ld8		r20 = [vp], 8		C			M
	;;
	ld8		r17 = [up], 8		C			M
	ld8		r21 = [vp], 8		C			M
	mov.i		ar.lc = n		C			I0
	;;
	logop(		r14, r18, r22)		C			M I
	ld8		r18 = [up], 8		C			M
	notormov(	r9, r15)		C			M I
	ld8		r22 = [vp], 8		C			M
	br		.LL11			C			B

C *** MAIN LOOP START ***
	ALIGN(32)
.Loop:	st8		[rp] = r8, 8		C			M
	logop(		r14, r16, r20)		C			M I
	notormov(	r9, r15)		C			M I
	ld8		r16 = [up], 8		C			M
	ld8		r20 = [vp], 8		C			M
	nop.b		0
	;;
.LL01:	st8		[rp] = r9, 8		C			M
	logop(		r15, r17, r21)		C			M I
	notormov(	r8, r14)		C			M I
	ld8		r17 = [up], 8		C			M
	ld8		r21 = [vp], 8		C			M
	nop.b		0
	;;
.LL00:	st8		[rp] = r8, 8		C			M
	logop(		r14, r18, r22)		C			M I
	notormov(	r9, r15)		C			M I
	ld8		r18 = [up], 8		C			M
	ld8		r22 = [vp], 8		C			M
	nop.b		0
	;;
.LL11:	st8		[rp] = r9, 8		C			M
	logop(		r15, r19, r23)		C			M I
	notormov(	r8, r14)		C			M I
	ld8		r19 = [up], 8		C			M
	ld8		r23 = [vp], 8		C			M
	br.cloop.dptk	.Loop	;;		C			B
C *** MAIN LOOP END ***

.Lcj6:	st8		[rp] = r8, 8		C			M
	logop(		r14, r16, r20)		C			M I
	notormov(	r9, r15)		C			M I
	;;
.Lcj5:	st8		[rp] = r9, 8		C			M
	logop(		r15, r17, r21)		C			M I
	notormov(	r8, r14)		C			M I
	;;
.Lcj4:	st8		[rp] = r8, 8		C			M
	logop(		r14, r18, r22)		C			M I
	notormov(	r9, r15)		C			M I
	;;
.Lcj3:	st8		[rp] = r9, 8		C			M
	logop(		r15, r19, r23)		C			M I
	notormov(	r8, r14)		C			M I
	;;
.Lcj2:	st8		[rp] = r8, 8		C			M
	notormov(	r9, r15)		C			M I
	;;
.Lcj1:	st8		[rp] = r9, 8		C			M
	mov.i		ar.lc = r2		C			I0
	br.ret.sptk.many b0			C			B
EPILOGUE()
ASM_END()
