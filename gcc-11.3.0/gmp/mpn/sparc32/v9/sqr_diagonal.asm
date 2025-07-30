dnl  SPARC v9 32-bit mpn_sqr_diagonal.

dnl  Copyright 2001, 2003 Free Software Foundation, Inc.

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
C rp	i0
C up	i1
C n	i2

C This code uses a very deep software pipeline, due to the need for moving data
C forth and back between the integer registers and floating-point registers.
C
C A VIS variant of this code would make the pipeline less deep, since the
C masking now done in the integer unit could take place in the floating-point
C unit using the FAND instruction.  It would be possible to save several cycles
C too.
C
C On UltraSPARC 1 and 2, this code runs at 11 cycles/limb from the Dcache and
C not much slower from the Ecache.  It would perhaps be possible to shave off
C one cycle, but not easily.  We cannot do better than 10 cycles/limb with the
C used instructions, since we have 10 memory operations per limb.  But a VIS
C variant could run three cycles faster than the corresponding non-VIS code.

C This is non-pipelined code showing the algorithm:
C
C .Loop:
C	lduw	[up+0],%g4		C 00000000hhhhllll
C	sllx	%g4,16,%g3		C 0000hhhhllll0000
C	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
C	andn	%g2,%g5,%g2		C 0000hhhh0000llll
C	stx	%g2,[%fp+80]
C	ldd	[%fp+80],%f0
C	fitod	%f0,%f4			C hi16
C	fitod	%f1,%f6			C lo16
C	ld	[up+0],%f9
C	fxtod	%f8,%f2
C	fmuld	%f2,%f4,%f4
C	fmuld	%f2,%f6,%f6
C	fdtox	%f4,%f4
C	fdtox	%f6,%f6
C	std	%f4,[%fp-24]
C	std	%f6,[%fp-16]
C	ldx	[%fp-24],%g2
C	ldx	[%fp-16],%g1
C	sllx	%g2,16,%g2
C	add	%g2,%g1,%g1
C	stw	%g1,[rp+0]
C	srlx	%g1,32,%l0
C	stw	%l0,[rp+4]
C	add	up,4,up
C	subcc	n,1,n
C	bne,pt	%icc,.Loop
C	add	rp,8,rp

define(`fanop',`fitod %f12,%f10')	dnl  A quasi nop running in the FA pipe

ASM_START()

	TEXT
	ALIGN(4)
.Lnoll:
	.word	0

PROLOGUE(mpn_sqr_diagonal)
	save	%sp,-256,%sp

ifdef(`PIC',
`.Lpc:	rd	%pc,%o7
	ld	[%o7+.Lnoll-.Lpc],%f8',
`	sethi	%hi(.Lnoll),%g1
	ld	[%g1+%lo(.Lnoll)],%f8')

	sethi	%hi(0xffff0000),%g5
	add	%i1,-8,%i1

	lduw	[%i1+8],%g4
	add	%i1,4,%i1		C s1_ptr++
	sllx	%g4,16,%g3		C 0000hhhhllll0000
	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
	subcc	%i2,1,%i2
	bne,pt	%icc,.L_grt_1
	andn	%g2,%g5,%g2		C 0000hhhh0000llll

	add	%i1,4,%i1		C s1_ptr++
	stx	%g2,[%fp+80]
	ld	[%i1],%f9
	ldd	[%fp+80],%f0
	fxtod	%f8,%f2
	fitod	%f0,%f4
	fitod	%f1,%f6
	fmuld	%f2,%f4,%f4
	fmuld	%f2,%f6,%f6
	fdtox	%f4,%f4
	fdtox	%f6,%f6
	std	%f4,[%fp-24]
	std	%f6,[%fp-16]

	add	%fp, 80, %l3
	add	%fp, -24, %l4
	add	%fp, 72, %l5
	b	.L1
	add	%fp, -40, %l6

.L_grt_1:
	stx	%g2,[%fp+80]
	lduw	[%i1+8],%g4
	add	%i1,4,%i1		C s1_ptr++
	sllx	%g4,16,%g3		C 0000hhhhllll0000
	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
	subcc	%i2,1,%i2
	bne,pt	%icc,.L_grt_2
	andn	%g2,%g5,%g2		C 0000hhhh0000llll

	stx	%g2,[%fp+72]
	ld	[%i1],%f9
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+80],%f0
	fxtod	%f8,%f2
	fitod	%f0,%f4
	fitod	%f1,%f6
	fmuld	%f2,%f4,%f4
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
	ldd	[%fp+72],%f0
	fdtox	%f4,%f4
	fdtox	%f6,%f6
	std	%f4,[%fp-24]
	fxtod	%f8,%f2
	std	%f6,[%fp-16]
	fitod	%f0,%f4
	fitod	%f1,%f6
	fmuld	%f2,%f4,%f4
	fmuld	%f2,%f6,%f6
	fdtox	%f4,%f4

	add	%fp, 72, %l3
	add	%fp, -40, %l4
	add	%fp, 80, %l5
	b	.L2
	add	%fp, -24, %l6

.L_grt_2:
	stx	%g2,[%fp+72]
	lduw	[%i1+8],%g4
	ld	[%i1],%f9
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+80],%f0
	sllx	%g4,16,%g3		C 0000hhhhllll0000
	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
	subcc	%i2,1,%i2
	fxtod	%f8,%f2
	bne,pt	%icc,.L_grt_3
	andn	%g2,%g5,%g2		C 0000hhhh0000llll

	stx	%g2,[%fp+80]
	fitod	%f0,%f4
	fitod	%f1,%f6
	fmuld	%f2,%f4,%f4
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+72],%f0
	fdtox	%f4,%f4
	fdtox	%f6,%f6
	std	%f4,[%fp-24]
	fxtod	%f8,%f2
	std	%f6,[%fp-16]
	fitod	%f0,%f4
	fitod	%f1,%f6
	fmuld	%f2,%f4,%f4
	ld	[%i1],%f9
	add	%fp, 80, %l3
	fmuld	%f2,%f6,%f6
	add	%fp, -24, %l4
	ldd	[%fp+80],%f0
	add	%fp, 72, %l5
	fdtox	%f4,%f4
	b	.L3
	add	%fp, -40, %l6

.L_grt_3:
	stx	%g2,[%fp+80]
	fitod	%f0,%f4
	lduw	[%i1+8],%g4
	fitod	%f1,%f6
	fmuld	%f2,%f4,%f4
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+72],%f0
	fdtox	%f4,%f4
	sllx	%g4,16,%g3		C 0000hhhhllll0000
	fdtox	%f6,%f6
	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
	subcc	%i2,1,%i2
	std	%f4,[%fp-24]
	fxtod	%f8,%f2
	std	%f6,[%fp-16]
	bne,pt	%icc,.L_grt_4
	andn	%g2,%g5,%g2		C 0000hhhh0000llll

	stx	%g2,[%fp+72]
	fitod	%f0,%f4
	fitod	%f1,%f6
	add	%fp, 72, %l3
	fmuld	%f2,%f4,%f4
	add	%fp, -40, %l4
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+80],%f0
	add	%fp, 80, %l5
	fdtox	%f4,%f4
	b	.L4
	add	%fp, -24, %l6

.L_grt_4:
	stx	%g2,[%fp+72]
	fitod	%f0,%f4
	lduw	[%i1+8],%g4
	fitod	%f1,%f6
	fmuld	%f2,%f4,%f4
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+80],%f0
	fdtox	%f4,%f4
	sllx	%g4,16,%g3		C 0000hhhhllll0000
	fdtox	%f6,%f6
	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
	subcc	%i2,1,%i2
	std	%f4,[%fp-40]
	fxtod	%f8,%f2
	std	%f6,[%fp-32]
	be,pn	%icc,.L5
	andn	%g2,%g5,%g2		C 0000hhhh0000llll

	b,a	.Loop

	.align	16
C --- LOOP BEGIN
.Loop:	nop
	nop
	stx	%g2,[%fp+80]
	fitod	%f0,%f4
C ---
	nop
	nop
	lduw	[%i1+8],%g4
	fitod	%f1,%f6
C ---
	nop
	nop
	ldx	[%fp-24],%g2		C p16
	fanop
C ---
	nop
	nop
	ldx	[%fp-16],%g1		C p0
	fmuld	%f2,%f4,%f4
C ---
	sllx	%g2,16,%g2		C align p16
	add	%i0,8,%i0		C res_ptr++
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
C ---
	add	%g2,%g1,%g1		C add p16 to p0 (ADD1)
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+72],%f0
	fanop
C ---
	srlx	%g1,32,%l0
	nop
	stw	%g1,[%i0-8]
	fdtox	%f4,%f4
C ---
	sllx	%g4,16,%g3		C 0000hhhhllll0000
	nop
	stw	%l0,[%i0-4]
	fdtox	%f6,%f6
C ---
	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
	subcc	%i2,1,%i2
	std	%f4,[%fp-24]
	fxtod	%f8,%f2
C ---
	std	%f6,[%fp-16]
	andn	%g2,%g5,%g2		C 0000hhhh0000llll
	be,pn	%icc,.Lend
	fanop
C ---  LOOP MIDDLE
	nop
	nop
	stx	%g2,[%fp+72]
	fitod	%f0,%f4
C ---
	nop
	nop
	lduw	[%i1+8],%g4
	fitod	%f1,%f6
C ---
	nop
	nop
	ldx	[%fp-40],%g2		C p16
	fanop
C ---
	nop
	nop
	ldx	[%fp-32],%g1		C p0
	fmuld	%f2,%f4,%f4
C ---
	sllx	%g2,16,%g2		C align p16
	add	%i0,8,%i0		C res_ptr++
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
C ---
	add	%g2,%g1,%g1		C add p16 to p0 (ADD1)
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%fp+80],%f0
	fanop
C ---
	srlx	%g1,32,%l0
	nop
	stw	%g1,[%i0-8]
	fdtox	%f4,%f4
C ---
	sllx	%g4,16,%g3		C 0000hhhhllll0000
	nop
	stw	%l0,[%i0-4]
	fdtox	%f6,%f6
C ---
	or	%g3,%g4,%g2		C 0000hhhhXXXXllll
	subcc	%i2,1,%i2
	std	%f4,[%fp-40]
	fxtod	%f8,%f2
C ---
	std	%f6,[%fp-32]
	andn	%g2,%g5,%g2		C 0000hhhh0000llll
	bne,pt	%icc,.Loop
	fanop
C --- LOOP END

.L5:	add	%fp, 80, %l3
	add	%fp, -24, %l4
	add	%fp, 72, %l5
	b	.Ltail
	add	%fp, -40, %l6

.Lend:	add	%fp, 72, %l3
	add	%fp, -40, %l4
	add	%fp, 80, %l5
	add	%fp, -24, %l6
.Ltail:	stx	%g2,[%l3]
	fitod	%f0,%f4
	fitod	%f1,%f6
	ldx	[%l4],%g2		C p16
	ldx	[%l4+8],%g1		C p0
	fmuld	%f2,%f4,%f4
	sllx	%g2,16,%g2		C align p16
	add	%i0,8,%i0		C res_ptr++
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
	add	%g2,%g1,%g1		C add p16 to p0 (ADD1)
	add	%i1,4,%i1		C s1_ptr++
	ldd	[%l5],%f0
	srlx	%g1,32,%l0
	stw	%g1,[%i0-8]
	fdtox	%f4,%f4
	stw	%l0,[%i0-4]
.L4:	fdtox	%f6,%f6
	std	%f4,[%l4]
	fxtod	%f8,%f2
	std	%f6,[%l4+8]

	fitod	%f0,%f4
	fitod	%f1,%f6
	ldx	[%l6],%g2		C p16
	ldx	[%l6+8],%g1		C p0
	fmuld	%f2,%f4,%f4
	sllx	%g2,16,%g2		C align p16
	add	%i0,8,%i0		C res_ptr++
	ld	[%i1],%f9
	fmuld	%f2,%f6,%f6
	add	%g2,%g1,%g1		C add p16 to p0 (ADD1)
	ldd	[%l3],%f0
	srlx	%g1,32,%l0
	stw	%g1,[%i0-8]
	fdtox	%f4,%f4
	stw	%l0,[%i0-4]
.L3:	fdtox	%f6,%f6
	std	%f4,[%l6]
	fxtod	%f8,%f2
	std	%f6,[%l6+8]

	fitod	%f0,%f4
	fitod	%f1,%f6
	ldx	[%l4],%g2		C p16
	ldx	[%l4+8],%g1		C p0
	fmuld	%f2,%f4,%f4
	sllx	%g2,16,%g2		C align p16
	add	%i0,8,%i0		C res_ptr++
	fmuld	%f2,%f6,%f6
	add	%g2,%g1,%g1		C add p16 to p0 (ADD1)
	srlx	%g1,32,%l0
	stw	%g1,[%i0-8]
	fdtox	%f4,%f4
	stw	%l0,[%i0-4]
.L2:	fdtox	%f6,%f6
	std	%f4,[%l4]
	std	%f6,[%l4+8]

	ldx	[%l6],%g2		C p16
	ldx	[%l6+8],%g1		C p0
	sllx	%g2,16,%g2		C align p16
	add	%i0,8,%i0		C res_ptr++
	add	%g2,%g1,%g1		C add p16 to p0 (ADD1)
	srlx	%g1,32,%l0
	stw	%g1,[%i0-8]
	stw	%l0,[%i0-4]

.L1:	ldx	[%l4],%g2		C p16
	ldx	[%l4+8],%g1		C p0
	sllx	%g2,16,%g2		C align p16
	add	%i0,8,%i0		C res_ptr++
	add	%g2,%g1,%g1		C add p16 to p0 (ADD1)
	srlx	%g1,32,%l0
	stw	%g1,[%i0-8]
	stw	%l0,[%i0-4]

	ret
	restore	%g0,%g0,%o0

EPILOGUE(mpn_sqr_diagonal)
