dnl  IA-64 mpn_lshift/mpn_rshift.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2000-2005 Free Software Foundation, Inc.

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

C This code is scheduled deeply since the plain shift instructions shr and shl
C have a latency of 4 (on Itanium) or 3 (on Itanium 2).  Poor scheduling of
C these instructions cause a 10 cycle replay trap on Itanium.

C The ld8 scheduling should probably be decreased to make the function smaller.
C Good lfetch  will make sure we never stall anyway.

C We should actually issue the first ld8 at cycle 0, and the first BSH/FSH pair
C at cycle 2.  Judicious use of predicates could allow us to issue more ld8's
C in the prologue.


C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`n',  `r34')
define(`cnt',`r35')

define(`tnc',`r9')

ifdef(`OPERATION_lshift',`
	define(`FSH',`shl')
	define(`BSH',`shr.u')
	define(`UPD',`-8')
	define(`POFF',`-512')
	define(`PUPD',`-32')
	define(`func',`mpn_lshift')
')
ifdef(`OPERATION_rshift',`
	define(`FSH',`shr.u')
	define(`BSH',`shl')
	define(`UPD',`8')
	define(`POFF',`512')
	define(`PUPD',`32')
	define(`func',`mpn_rshift')
')

MULFUNC_PROLOGUE(mpn_lshift mpn_rshift)

ASM_START()
PROLOGUE(func)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',
`	addp4	rp = 0, rp		C			M I
	addp4	up = 0, up		C		M I
	sxt4	n = n			C		M I
	nop.m		0
	nop.m		0
	zxt4	cnt = cnt		C		I
	;;
')

 {.mmi;	cmp.lt	p14, p15 = 4, n		C		M I
	and	r14 = 3, n		C		M I
	mov.i	r2 = ar.lc		C		I0
}{.mmi;	add	r15 = -1, n		C		M I
	sub	tnc = 64, cnt		C		M I
	add	r16 = -5, n
	;;
}{.mmi;	cmp.eq	p6, p0 = 1, r14		C		M I
	cmp.eq	p7, p0 = 2, r14		C		M I
	shr.u	n = r16, 2		C		I0
}{.mmi;	cmp.eq	p8, p0 = 3, r14		C		M I
ifdef(`OPERATION_lshift',
`	shladd	up = r15, 3, up		C		M I
	shladd	rp = r15, 3, rp')	C		M I
	;;
}{.mmi;	add	r11 = POFF, up		C		M I
	ld8	r10 = [up], UPD		C		M01
	mov.i	ar.lc = n		C		I0
}{.bbb;
   (p6)	br.dptk	.Lb01
   (p7)	br.dptk	.Lb10
   (p8)	br.dptk	.Lb11
	;; }

.Lb00:	ld8	r19 = [up], UPD
	;;
	ld8	r16 = [up], UPD
	;;
	ld8	r17 = [up], UPD
	BSH	r8 = r10, tnc		C function return value
	;;
	FSH	r24 = r10, cnt
	BSH	r25 = r19, tnc
  (p14)	br.cond.dptk	.grt4
	;;
	FSH	r26 = r19, cnt
	BSH	r27 = r16, tnc
	;;
	FSH	r20 = r16, cnt
	BSH	r21 = r17, tnc
	;;
	or	r14 = r25, r24
	FSH	r22 = r17, cnt
	BSH	r23 = r10, tnc
	br	.Lr4

.grt4:	ld8	r18 = [up], UPD
	FSH	r26 = r19, cnt
	BSH	r27 = r16, tnc
	;;
	ld8	r19 = [up], UPD
	FSH	r20 = r16, cnt
	BSH	r21 = r17, tnc
	;;
	ld8	r16 = [up], UPD
	FSH	r22 = r17, cnt
	BSH	r23 = r18, tnc
	;;
	or	r14 = r25, r24
	ld8	r17 = [up], UPD
	br.cloop.dpnt	.Ltop
	br	.Lbot

.Lb01:
  (p15)	BSH	r8 = r10, tnc		C function return value	I
  (p15)	FSH	r22 = r10, cnt		C		I
  (p15)	br.cond.dptk	.Lr1		C return	B

.grt1:	ld8	r18 = [up], UPD
	;;
	ld8	r19 = [up], UPD
	BSH	r8 = r10, tnc		C function return value
	;;
	ld8	r16 = [up], UPD
	FSH	r22 = r10, cnt
	BSH	r23 = r18, tnc
	;;
	ld8	r17 = [up], UPD
	FSH	r24 = r18, cnt
	BSH	r25 = r19, tnc
	br.cloop.dpnt	.grt5
	;;
	or	r15 = r23, r22
	FSH	r26 = r19, cnt
	BSH	r27 = r16, tnc
	;;
	FSH	r20 = r16, cnt
	BSH	r21 = r17, tnc
	br	.Lr5

.grt5:	ld8	r18 = [up], UPD
	FSH	r26 = r19, cnt
	BSH	r27 = r16, tnc
	;;
	ld8	r19 = [up], UPD
	FSH	r20 = r16, cnt
	BSH	r21 = r17, tnc
	;;
	or	r15 = r23, r22
	ld8	r16 = [up], UPD
	br	.LL01


.Lb10:	ld8	r17 = [up], UPD
  (p14)	br.cond.dptk	.grt2

	BSH	r8 = r10, tnc		C function return value
	;;
	FSH	r20 = r10, cnt
	BSH	r21 = r17, tnc
	;;
	or	r14 = r21, r20
	FSH	r22 = r17, cnt
	br	.Lr2			C return

.grt2:	ld8	r18 = [up], UPD
	BSH	r8 = r10, tnc		C function return value
	;;
	ld8	r19 = [up], UPD
	FSH	r20 = r10, cnt
	BSH	r21 = r17, tnc
	;;
	ld8	r16 = [up], UPD
	FSH	r22 = r17, cnt
	BSH	r23 = r18, tnc
	;;
 {.mmi;	ld8	r17 = [up], UPD
	or	r14 = r21, r20
	FSH	r24 = r18, cnt
}{.mib;	nop	0
	BSH	r25 = r19, tnc
	br.cloop.dpnt	.grt6
	;; }

	FSH	r26 = r19, cnt
	BSH	r27 = r16, tnc
	br	.Lr6

.grt6:	ld8	r18 = [up], UPD
	FSH	r26 = r19, cnt
	BSH	r27 = r16, tnc
	;;
	ld8	r19 = [up], UPD
	br	.LL10


.Lb11:	ld8	r16 = [up], UPD
	;;
	ld8	r17 = [up], UPD
	BSH	r8 = r10, tnc		C function return value
  (p14)	br.cond.dptk	.grt3
	;;

	FSH	r26 = r10, cnt
	BSH	r27 = r16, tnc
	;;
	FSH	r20 = r16, cnt
	BSH	r21 = r17, tnc
	;;
	or	r15 = r27, r26
	FSH	r22 = r17, cnt
	br	.Lr3			C return

.grt3:	ld8	r18 = [up], UPD
	FSH	r26 = r10, cnt
	BSH	r27 = r16, tnc
	;;
	ld8	r19 = [up], UPD
	FSH	r20 = r16, cnt
	BSH	r21 = r17, tnc
	;;
	ld8	r16 = [up], UPD
	FSH	r22 = r17, cnt
	BSH	r23 = r18, tnc
	;;
	ld8	r17 = [up], UPD
	br.cloop.dpnt	.grt7

	or	r15 = r27, r26
	FSH	r24 = r18, cnt
	BSH	r25 = r19, tnc
	br	.Lr7

.grt7:	or	r15 = r27, r26
	FSH	r24 = r18, cnt
	BSH	r25 = r19, tnc
	ld8	r18 = [up], UPD
	br	.LL11

C *** MAIN LOOP START ***
	ALIGN(32)
.Ltop:
 {.mmi;	st8	[rp] = r14, UPD		C M2
	or	r15 = r27, r26		C M3
	FSH	r24 = r18, cnt		C I0
}{.mmi;	ld8	r18 = [up], UPD		C M1
	lfetch	[r11], PUPD
	BSH	r25 = r19, tnc		C I1
	;; }
.LL11:
 {.mmi;	st8	[rp] = r15, UPD
	or	r14 = r21, r20
	FSH	r26 = r19, cnt
}{.mmi;	ld8	r19 = [up], UPD
	nop.m	0
	BSH	r27 = r16, tnc
	;; }
.LL10:
 {.mmi;	st8	[rp] = r14, UPD
	or	r15 = r23, r22
	FSH	r20 = r16, cnt
}{.mmi;	ld8	r16 = [up], UPD
	nop.m	0
	BSH	r21 = r17, tnc
	;; }
.LL01:
 {.mmi;	st8	[rp] = r15, UPD
	or	r14 = r25, r24
	FSH	r22 = r17, cnt
}{.mib;	ld8	r17 = [up], UPD
	BSH	r23 = r18, tnc
	br.cloop.dptk	.Ltop
	;; }
C *** MAIN LOOP END ***

.Lbot:
 {.mmi;	st8	[rp] = r14, UPD
	or	r15 = r27, r26
	FSH	r24 = r18, cnt
}{.mib;	nop	0
	BSH	r25 = r19, tnc
	nop	0
	;; }
.Lr7:
 {.mmi;	st8	[rp] = r15, UPD
	or	r14 = r21, r20
	FSH	r26 = r19, cnt
}{.mib;	nop	0
	BSH	r27 = r16, tnc
	nop	0
	;; }
.Lr6:
 {.mmi;	st8	[rp] = r14, UPD
	or	r15 = r23, r22
	FSH	r20 = r16, cnt
}{.mib;	nop	0
	BSH	r21 = r17, tnc
	nop	0
	;; }
.Lr5:	st8	[rp] = r15, UPD
	or	r14 = r25, r24
	FSH	r22 = r17, cnt
	;;
.Lr4:	st8	[rp] = r14, UPD
	or	r15 = r27, r26
	;;
.Lr3:	st8	[rp] = r15, UPD
	or	r14 = r21, r20
	;;
.Lr2:	st8	[rp] = r14, UPD
	;;
.Lr1:	st8	[rp] = r22, UPD		C		M23
	mov	ar.lc = r2		C		I0
	br.ret.sptk.many b0		C		B
EPILOGUE(func)
ASM_END()
