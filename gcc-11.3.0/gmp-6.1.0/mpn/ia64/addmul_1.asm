dnl  IA-64 mpn_addmul_1 -- Multiply a limb vector with a limb and add the
dnl  result to a second limb vector.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2000-2005, 2007 Free Software Foundation, Inc.

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

C         cycles/limb
C Itanium:    3.0
C Itanium 2:  2.0

C TODO
C  * Further optimize feed-in and wind-down code, both for speed and code size.
C  * Handle low limb input and results specially, using a common stf8 in the
C    epilogue.
C  * Use 1 c/l carry propagation scheme in wind-down code.
C  * Use extra pointer registers for `up' and rp to speed up feed-in loads.
C  * Work out final differences with mul_1.asm.  That function is 300 bytes
C    smaller than this due to better loop scheduling and thus simpler feed-in
C    code.

C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`n', `r34')
define(`vl', `r35')

ASM_START()
PROLOGUE(mpn_addmul_1)
	.prologue
	.save	ar.lc, r2
	.body

ifdef(`HAVE_ABI_32',
`	addp4		rp = 0, rp		C M I
	addp4		up = 0, up		C M I
	zxt4		n = n			C I
	;;
')
{.mmi
	adds		r15 = -1, n		C M I
	mov		r20 = rp		C M I
	mov.i		r2 = ar.lc		C I0
}
{.mmi
	ldf8		f7 = [up], 8		C M
	ldf8		f8 = [rp], 8		C M
	and		r14 = 3, n		C M I
	;;
}
{.mmi
	setf.sig	f6 = vl			C M2 M3
	cmp.eq		p10, p0 = 0, r14	C M I
	shr.u		r31 = r15, 2		C I0
}
{.mmi
	cmp.eq		p11, p0 = 2, r14	C M I
	cmp.eq		p12, p0 = 3, r14	C M I
	nop.i		0			C I
	;;
}
{.mii
	cmp.ne		p6, p7 = r0, r0		C M I
	mov.i		ar.lc = r31		C I0
	cmp.ne		p8, p9 = r0, r0		C M I
}
{.bbb
  (p10)	br.dptk		.Lb00			C B
  (p11)	br.dptk		.Lb10			C B
  (p12)	br.dptk		.Lb11			C B
	;;
}

.Lb01:	br.cloop.dptk	.grt1			C B

	xma.l		f39 = f7, f6, f8	C F
	xma.hu		f43 = f7, f6, f8	C F
	;;
	getf.sig	r8 = f43		C M2
	stf8		[r20] = f39		C M2 M3
	mov.i		ar.lc = r2		C I0
	br.ret.sptk.many b0			C B

.grt1:
	ldf8		f32 = [up], 8
	ldf8		f44 = [rp], 8
	;;
	ldf8		f33 = [up], 8
	ldf8		f45 = [rp], 8
	;;
	ldf8		f34 = [up], 8
	xma.l		f39 = f7, f6, f8
	ldf8		f46 = [rp], 8
	xma.hu		f43 = f7, f6, f8
	;;
	ldf8		f35 = [up], 8
	ldf8		f47 = [rp], 8
	br.cloop.dptk	.grt5

	xma.l		f36 = f32, f6, f44
	xma.hu		f40 = f32, f6, f44
	;;
	stf8		[r20] = f39, 8
	xma.l		f37 = f33, f6, f45
	xma.hu		f41 = f33, f6, f45
	;;
	getf.sig	r31 = f43
	getf.sig	r24 = f36
	xma.l		f38 = f34, f6, f46
	xma.hu		f42 = f34, f6, f46
	;;
	getf.sig	r28 = f40
	getf.sig	r25 = f37
	xma.l		f39 = f35, f6, f47
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r29 = f41
	getf.sig	r26 = f38
	br		.Lcj5

.grt5:
	mov		r30 = 0
	xma.l		f36 = f32, f6, f44
	xma.hu		f40 = f32, f6, f44
	;;
	ldf8		f32 = [up], 8
	xma.l		f37 = f33, f6, f45
	ldf8		f44 = [rp], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f33 = [up], 8
	getf.sig	r27 = f39
	;;
	getf.sig	r31 = f43
	xma.l		f38 = f34, f6, f46
	ldf8		f45 = [rp], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f34 = [up], 8
	getf.sig	r24 = f36
	;;
	getf.sig	r28 = f40
	xma.l		f39 = f35, f6, f47
	ldf8		f46 = [rp], 8
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f35 = [up], 8
	getf.sig	r25 = f37
	br.cloop.dptk	.Loop
	br		.Le0


.Lb10:	ldf8		f35 = [up], 8
	ldf8		f47 = [rp], 8
	br.cloop.dptk	.grt2

	xma.l		f38 = f7, f6, f8
	xma.hu		f42 = f7, f6, f8
	;;
	xma.l		f39 = f35, f6, f47
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r30 = f42
	stf8		[r20] = f38, 8
	getf.sig	r27 = f39
	getf.sig	r8 = f43
	br		.Lcj2

.grt2:
	ldf8		f32 = [up], 8
	ldf8		f44 = [rp], 8
	;;
	ldf8		f33 = [up], 8
	xma.l		f38 = f7, f6, f8
	ldf8		f45 = [rp], 8
	xma.hu		f42 = f7, f6, f8
	;;
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f47
	ldf8		f46 = [rp], 8
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f35 = [up], 8
	ldf8		f47 = [rp], 8
	br.cloop.dptk	.grt6

	stf8		[r20] = f38, 8
	xma.l		f36 = f32, f6, f44
	xma.hu		f40 = f32, f6, f44
	;;
	getf.sig	r30 = f42
	getf.sig	r27 = f39
	xma.l		f37 = f33, f6, f45
	xma.hu		f41 = f33, f6, f45
	;;
	getf.sig	r31 = f43
	getf.sig	r24 = f36
	xma.l		f38 = f34, f6, f46
	xma.hu		f42 = f34, f6, f46
	;;
	getf.sig	r28 = f40
	getf.sig	r25 = f37
	xma.l		f39 = f35, f6, f47
	xma.hu		f43 = f35, f6, f47
	br		.Lcj6

.grt6:
	mov		r29 = 0
	xma.l		f36 = f32, f6, f44
	xma.hu		f40 = f32, f6, f44
	;;
	ldf8		f32 = [up], 8
	getf.sig	r26 = f38
	;;
	getf.sig	r30 = f42
	xma.l		f37 = f33, f6, f45
	ldf8		f44 = [rp], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f33 = [up], 8
	getf.sig	r27 = f39
	;;
	getf.sig	r31 = f43
	xma.l		f38 = f34, f6, f46
	ldf8		f45 = [rp], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f34 = [up], 8
	getf.sig	r24 = f36
	br		.LL10


.Lb11:	ldf8		f34 = [up], 8
	ldf8		f46 = [rp], 8
	;;
	ldf8		f35 = [up], 8
	ldf8		f47 = [rp], 8
	br.cloop.dptk	.grt3
	;;

	xma.l		f37 = f7, f6, f8
	xma.hu		f41 = f7, f6, f8
	xma.l		f38 = f34, f6, f46
	xma.hu		f42 = f34, f6, f46
	xma.l		f39 = f35, f6, f47
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r29 = f41
	stf8		[r20] = f37, 8
	getf.sig	r26 = f38
	getf.sig	r30 = f42
	getf.sig	r27 = f39
	getf.sig	r8 = f43
	br		.Lcj3

.grt3:
	ldf8		f32 = [up], 8
	xma.l		f37 = f7, f6, f8
	ldf8		f44 = [rp], 8
	xma.hu		f41 = f7, f6, f8
	;;
	ldf8		f33 = [up], 8
	xma.l		f38 = f34, f6, f46
	ldf8		f45 = [rp], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f47
	ldf8		f46 = [rp], 8
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f35 = [up], 8
	getf.sig	r25 = f37		C FIXME
	ldf8		f47 = [rp], 8
	br.cloop.dptk	.grt7

	getf.sig	r29 = f41
	stf8		[r20] = f37, 8		C FIXME
	xma.l		f36 = f32, f6, f44
	getf.sig	r26 = f38
	xma.hu		f40 = f32, f6, f44
	;;
	getf.sig	r30 = f42
	xma.l		f37 = f33, f6, f45
	getf.sig	r27 = f39
	xma.hu		f41 = f33, f6, f45
	;;
	getf.sig	r31 = f43
	xma.l		f38 = f34, f6, f46
	getf.sig	r24 = f36
	xma.hu		f42 = f34, f6, f46
	br		.Lcj7

.grt7:
	getf.sig	r29 = f41
	xma.l		f36 = f32, f6, f44
	mov		r28 = 0
	xma.hu		f40 = f32, f6, f44
	;;
	ldf8		f32 = [up], 8
	getf.sig	r26 = f38
	;;
	getf.sig	r30 = f42
	xma.l		f37 = f33, f6, f45
	ldf8		f44 = [rp], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f33 = [up], 8
	getf.sig	r27 = f39
	br		.LL11


.Lb00:	ldf8		f33 = [up], 8
	ldf8		f45 = [rp], 8
	;;
	ldf8		f34 = [up], 8
	ldf8		f46 = [rp], 8
	;;
	ldf8		f35 = [up], 8
	xma.l		f36 = f7, f6, f8
	ldf8		f47 = [rp], 8
	xma.hu		f40 = f7, f6, f8
	br.cloop.dptk	.grt4

	xma.l		f37 = f33, f6, f45
	xma.hu		f41 = f33, f6, f45
	xma.l		f38 = f34, f6, f46
	xma.hu		f42 = f34, f6, f46
	;;
	getf.sig	r28 = f40
	stf8		[r20] = f36, 8
	xma.l		f39 = f35, f6, f47
	getf.sig	r25 = f37
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r29 = f41
	getf.sig	r26 = f38
	getf.sig	r30 = f42
	getf.sig	r27 = f39
	br		.Lcj4

.grt4:
	ldf8		f32 = [up], 8
	xma.l		f37 = f33, f6, f45
	ldf8		f44 = [rp], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f33 = [up], 8
	xma.l		f38 = f34, f6, f46
	ldf8		f45 = [rp], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f34 = [up], 8
	getf.sig	r24 = f36		C FIXME
	xma.l		f39 = f35, f6, f47
	ldf8		f46 = [rp], 8
	getf.sig	r28 = f40
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f35 = [up], 8
	getf.sig	r25 = f37
	ldf8		f47 = [rp], 8
	br.cloop.dptk	.grt8

	getf.sig	r29 = f41
	stf8		[r20] = f36, 8		C FIXME
	xma.l		f36 = f32, f6, f44
	getf.sig	r26 = f38
	getf.sig	r30 = f42
	xma.hu		f40 = f32, f6, f44
	;;
	xma.l		f37 = f33, f6, f45
	getf.sig	r27 = f39
	xma.hu		f41 = f33, f6, f45
	br		.Lcj8

.grt8:
	getf.sig	r29 = f41
	xma.l		f36 = f32, f6, f44
	mov		r31 = 0
	xma.hu		f40 = f32, f6, f44
	;;
	ldf8		f32 = [up], 8
	getf.sig	r26 = f38
	br		.LL00


C *** MAIN LOOP START ***
	ALIGN(32)				C insn	fed	cycle #
.Loop:
	.pred.rel "mutex", p6, p7		C num	by	i1 i2
	getf.sig	r29 = f41		C 00	16	0   0
	xma.l		f36 = f32, f6, f44	C 01	06,15	0   0
   (p6)	add		r14 = r30, r27, 1	C 02		0   0
	ldf8		f47 = [rp], 8		C 03		0   0
	xma.hu		f40 = f32, f6, f44	C 04	06,15	0   0
   (p7)	add		r14 = r30, r27		C 05		0   0
	;;
	.pred.rel "mutex", p6, p7
	ldf8		f32 = [up], 8		C 06		1   1
   (p6)	cmp.leu		p8, p9 = r14, r27	C 07		1   1
   (p7)	cmp.ltu		p8, p9 = r14, r27	C 08		1   1
	getf.sig	r26 = f38		C 09	25	2   1
	st8		[r20] = r14, 8		C 10		2   1
	nop.b		0			C 11		2   1
	;;
.LL00:
	.pred.rel "mutex", p8, p9
	getf.sig	r30 = f42		C 12	28	3   2
	xma.l		f37 = f33, f6, f45	C 13	18,27	3   2
   (p8)	add		r16 = r31, r24, 1	C 14		3   2
	ldf8		f44 = [rp], 8		C 15		3   2
	xma.hu		f41 = f33, f6, f45	C 16	18,27	3   2
   (p9)	add		r16 = r31, r24		C 17		3   2
	;;
	.pred.rel "mutex", p8, p9
	ldf8		f33 = [up], 8		C 18		4   3
   (p8)	cmp.leu		p6, p7 = r16, r24	C 19		4   3
   (p9)	cmp.ltu		p6, p7 = r16, r24	C 20		4   3
	getf.sig	r27 = f39		C 21	37	5   3
	st8		[r20] = r16, 8		C 22		5   3
	nop.b		0			C 23		5   3
	;;
.LL11:
	.pred.rel "mutex", p6, p7
	getf.sig	r31 = f43		C 24	40	6   4
	xma.l		f38 = f34, f6, f46	C 25	30,39	6   4
   (p6)	add		r14 = r28, r25, 1	C 26		6   4
	ldf8		f45 = [rp], 8		C 27		6   4
	xma.hu		f42 = f34, f6, f46	C 28	30,39	6   4
   (p7)	add		r14 = r28, r25		C 29		6   4
	;;
	.pred.rel "mutex", p6, p7
	ldf8		f34 = [up], 8		C 30		7   5
   (p6)	cmp.leu		p8, p9 = r14, r25	C 31		7   5
   (p7)	cmp.ltu		p8, p9 = r14, r25	C 32		7   5
	getf.sig	r24 = f36		C 33	01	8   5
	st8		[r20] = r14, 8		C 34		8   5
	nop.b		0			C 35		8   5
	;;
.LL10:
	.pred.rel "mutex", p8, p9
	getf.sig	r28 = f40		C 36	04	9   6
	xma.l		f39 = f35, f6, f47	C 37	42,03	9   6
   (p8)	add		r16 = r29, r26, 1	C 38		9   6
	ldf8		f46 = [rp], 8		C 39		9   6
	xma.hu		f43 = f35, f6, f47	C 40	42,03	9   6
   (p9)	add		r16 = r29, r26		C 41		9   6
	;;
	.pred.rel "mutex", p8, p9
	ldf8		f35 = [up], 8		C 42	       10   7
   (p8)	cmp.leu		p6, p7 = r16, r26	C 43	       10   7
   (p9)	cmp.ltu		p6, p7 = r16, r26	C 44	       10   7
	getf.sig	r25 = f37		C 45	13     11   7
	st8		[r20] = r16, 8		C 46	       11   7
	br.cloop.dptk	.Loop			C 47	       11   7
C *** MAIN LOOP END ***
	;;
.Le0:
	.pred.rel "mutex", p6, p7
	getf.sig	r29 = f41		C
	xma.l		f36 = f32, f6, f44	C
   (p6)	add		r14 = r30, r27, 1	C
	ldf8		f47 = [rp], 8		C
	xma.hu		f40 = f32, f6, f44	C
   (p7)	add		r14 = r30, r27		C
	;;
	.pred.rel "mutex", p6, p7
   (p6)	cmp.leu		p8, p9 = r14, r27	C
   (p7)	cmp.ltu		p8, p9 = r14, r27	C
	getf.sig	r26 = f38		C
	st8		[r20] = r14, 8		C
	;;
	.pred.rel "mutex", p8, p9
	getf.sig	r30 = f42		C
	xma.l		f37 = f33, f6, f45	C
   (p8)	add		r16 = r31, r24, 1	C
	xma.hu		f41 = f33, f6, f45	C
   (p9)	add		r16 = r31, r24		C
	;;
	.pred.rel "mutex", p8, p9
   (p8)	cmp.leu		p6, p7 = r16, r24	C
   (p9)	cmp.ltu		p6, p7 = r16, r24	C
	getf.sig	r27 = f39		C
	st8		[r20] = r16, 8		C
	;;
.Lcj8:
	.pred.rel "mutex", p6, p7
	getf.sig	r31 = f43		C
	xma.l		f38 = f34, f6, f46	C
   (p6)	add		r14 = r28, r25, 1	C
	xma.hu		f42 = f34, f6, f46	C
   (p7)	add		r14 = r28, r25		C
	;;
	.pred.rel "mutex", p6, p7
   (p6)	cmp.leu		p8, p9 = r14, r25	C
   (p7)	cmp.ltu		p8, p9 = r14, r25	C
	getf.sig	r24 = f36		C
	st8		[r20] = r14, 8		C
	;;
.Lcj7:
	.pred.rel "mutex", p8, p9
	getf.sig	r28 = f40		C
	xma.l		f39 = f35, f6, f47	C
   (p8)	add		r16 = r29, r26, 1	C
	xma.hu		f43 = f35, f6, f47	C
   (p9)	add		r16 = r29, r26		C
	;;
	.pred.rel "mutex", p8, p9
   (p8)	cmp.leu		p6, p7 = r16, r26	C
   (p9)	cmp.ltu		p6, p7 = r16, r26	C
	getf.sig	r25 = f37		C
	st8		[r20] = r16, 8		C
	;;
.Lcj6:
	.pred.rel "mutex", p6, p7
	getf.sig	r29 = f41		C
   (p6)	add		r14 = r30, r27, 1	C
   (p7)	add		r14 = r30, r27		C
	;;
	.pred.rel "mutex", p6, p7
   (p6)	cmp.leu		p8, p9 = r14, r27	C
   (p7)	cmp.ltu		p8, p9 = r14, r27	C
	getf.sig	r26 = f38		C
	st8		[r20] = r14, 8		C
	;;
.Lcj5:
	.pred.rel "mutex", p8, p9
	getf.sig	r30 = f42		C
   (p8)	add		r16 = r31, r24, 1	C
   (p9)	add		r16 = r31, r24		C
	;;
	.pred.rel "mutex", p8, p9
   (p8)	cmp.leu		p6, p7 = r16, r24	C
   (p9)	cmp.ltu		p6, p7 = r16, r24	C
	getf.sig	r27 = f39		C
	st8		[r20] = r16, 8		C
	;;
.Lcj4:
	.pred.rel "mutex", p6, p7
	getf.sig	r8 = f43		C
   (p6)	add		r14 = r28, r25, 1	C
   (p7)	add		r14 = r28, r25		C
	;;
	.pred.rel "mutex", p6, p7
	st8		[r20] = r14, 8		C
   (p6)	cmp.leu		p8, p9 = r14, r25	C
   (p7)	cmp.ltu		p8, p9 = r14, r25	C
	;;
.Lcj3:
	.pred.rel "mutex", p8, p9
   (p8)	add		r16 = r29, r26, 1	C
   (p9)	add		r16 = r29, r26		C
	;;
	.pred.rel "mutex", p8, p9
	st8		[r20] = r16, 8		C
   (p8)	cmp.leu		p6, p7 = r16, r26	C
   (p9)	cmp.ltu		p6, p7 = r16, r26	C
	;;
.Lcj2:
	.pred.rel "mutex", p6, p7
   (p6)	add		r14 = r30, r27, 1	C
   (p7)	add		r14 = r30, r27		C
	;;
	.pred.rel "mutex", p6, p7
	st8		[r20] = r14		C
   (p6)	cmp.leu		p8, p9 = r14, r27	C
   (p7)	cmp.ltu		p8, p9 = r14, r27	C
	;;
   (p8)	add		r8 = 1, r8		C M I
	mov.i		ar.lc = r2		C I0
	br.ret.sptk.many b0			C B
EPILOGUE()
ASM_END()
