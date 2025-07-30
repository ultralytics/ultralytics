dnl  SPARC v9 64-bit mpn_addmul_2 -- Multiply an n limb number with 2-limb
dnl  number and add the result to a n limb vector.

dnl  Copyright 2002, 2003 Free Software Foundation, Inc.

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

C                  cycles/limb
C UltraSPARC 1&2:      9
C UltraSPARC 3:       10

C Algorithm: We use 16 floating-point multiplies per limb product, with the
C 2-limb v operand split into eight 16-bit pieces, and the n-limb u operand
C split into 32-bit pieces.  We sum four 48-bit partial products using
C floating-point add, then convert the resulting four 50-bit quantities and
C transfer them to the integer unit.

C Possible optimizations:
C   1. Align the stack area where we transfer the four 50-bit product-sums
C      to a 32-byte boundary.  That would minimize the cache collision.
C      (UltraSPARC-1/2 use a direct-mapped cache.)  (Perhaps even better would
C      be to align the area to map to the area immediately before up?)
C   2. Perform two of the fp->int conversions with integer instructions.  We
C      can get almost ten free IEU slots, if we clean up bookkeeping and the
C      silly carry-limb code.
C   3. For an mpn_addmul_1 based on this, we need to fix the silly carry-limb
C      code.

C OSP (Overlapping software pipeline) version of mpn_mul_basecase:
C Operand swap will require 8 LDDA and 8 FXTOD, which will mean 8 cycles.
C FI	= 20
C L	=  9 x un * vn
C WDFI	= 10 x vn / 2
C WD	= 4

C Instruction classification (as per UltraSPARC functional units).
C Assuming silly carry code is fixed.  Includes bookkeeping.
C
C               mpn_addmul_X     mpn_mul_X
C                1       2       1       2
C               ==========      ==========
C      FM        8      16       8      16
C      FA       10      18      10      18
C     MEM       12      12      10      10
C  ISHIFT        6       6       6       6
C IADDLOG       11      11      10      10
C  BRANCH        1       1       1       1
C
C TOTAL IEU     17      17      16      16
C TOTAL         48      64      45      61
C
C IEU cycles     8.5     8.5     8       8
C MEM cycles    12      12      10      10
C ISSUE cycles  12      16      11.25   15.25
C FPU cycles    10      18      10      18
C cycles/loop   12      18      12      18
C cycles/limb   12       9      12       9


C INPUT PARAMETERS
C rp[n + 1]	i0
C up[n]		i1
C n		i2
C vp[2]		i3


ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)

C Combine registers:
C u00_hi= u32_hi
C u00_lo= u32_lo
C a000  = out000
C a016  = out016
C Free: f52 f54


define(`p000', `%f8')  define(`p016',`%f10')
define(`p032',`%f12')  define(`p048',`%f14')
define(`p064',`%f16')  define(`p080',`%f18')
define(`p096a',`%f20') define(`p112a',`%f22')
define(`p096b',`%f56') define(`p112b',`%f58')

define(`out000',`%f0') define(`out016',`%f6')

define(`v000',`%f24')  define(`v016',`%f26')
define(`v032',`%f28')  define(`v048',`%f30')
define(`v064',`%f44')  define(`v080',`%f46')
define(`v096',`%f48')  define(`v112',`%f50')

define(`u00',`%f32')   define(`u32', `%f34')

define(`a000',`%f36')  define(`a016',`%f38')
define(`a032',`%f40')  define(`a048',`%f42')
define(`a064',`%f60')  define(`a080',`%f62')

define(`u00_hi',`%f2') define(`u32_hi',`%f4')
define(`u00_lo',`%f3') define(`u32_lo',`%f5')

define(`cy',`%g1')
define(`rlimb',`%g3')
define(`i00',`%l0')    define(`i16',`%l1')
define(`r00',`%l2')    define(`r32',`%l3')
define(`xffffffff',`%l7')
define(`xffff',`%o0')


PROLOGUE(mpn_addmul_2)

C Initialization.  (1) Split v operand into eight 16-bit chunks and store them
C as IEEE double in fp registers.  (2) Clear upper 32 bits of fp register pairs
C f2 and f4.  (3) Store masks in registers aliased to `xffff' and `xffffffff'.
C This code could be better scheduled.

	save	%sp, -256, %sp

ifdef(`HAVE_VIS',
`	mov	-1, %g4
	wr	%g0, 0xD2, %asi
	srlx	%g4, 32, xffffffff	C store mask in register `xffffffff'
	ldda	[%i3+6] %asi, v000
	ldda	[%i3+4] %asi, v016
	ldda	[%i3+2] %asi, v032
	ldda	[%i3+0] %asi, v048
	fxtod	v000, v000
	ldda	[%i3+14] %asi, v064
	fxtod	v016, v016
	ldda	[%i3+12] %asi, v080
	fxtod	v032, v032
	ldda	[%i3+10] %asi, v096
	fxtod	v048, v048
	ldda	[%i3+8] %asi, v112
	fxtod	v064, v064
	fxtod	v080, v080
	fxtod	v096, v096
	fxtod	v112, v112
	fzero	u00_hi
	fzero	u32_hi
',
`	mov	-1, %g4
	ldx	[%i3+0], %l0		C vp[0]
	srlx	%g4, 48, xffff		C store mask in register `xffff'
	ldx	[%i3+8], %l1		C vp[1]

	and	%l0, xffff, %g2
	stx	%g2, [%sp+2223+0]
	srlx	%l0, 16, %g3
	and	%g3, xffff, %g3
	stx	%g3, [%sp+2223+8]
	srlx	%l0, 32, %g2
	and	%g2, xffff, %g2
	stx	%g2, [%sp+2223+16]
	srlx	%l0, 48, %g3
	stx	%g3, [%sp+2223+24]
	and	%l1, xffff, %g2
	stx	%g2, [%sp+2223+32]
	srlx	%l1, 16, %g3
	and	%g3, xffff, %g3
	stx	%g3, [%sp+2223+40]
	srlx	%l1, 32, %g2
	and	%g2, xffff, %g2
	stx	%g2, [%sp+2223+48]
	srlx	%l1, 48, %g3
	stx	%g3, [%sp+2223+56]

	srlx	%g4, 32, xffffffff	C store mask in register `xffffffff'

	ldd	[%sp+2223+0], v000
	ldd	[%sp+2223+8], v016
	ldd	[%sp+2223+16], v032
	ldd	[%sp+2223+24], v048
	fxtod	v000, v000
	ldd	[%sp+2223+32], v064
	fxtod	v016, v016
	ldd	[%sp+2223+40], v080
	fxtod	v032, v032
	ldd	[%sp+2223+48], v096
	fxtod	v048, v048
	ldd	[%sp+2223+56], v112
	fxtod	v064, v064
	ld	[%sp+2223+0], u00_hi	C zero u00_hi
	fxtod	v080, v080
	ld	[%sp+2223+0], u32_hi	C zero u32_hi
	fxtod	v096, v096
	fxtod	v112, v112
')
C Initialization done.
	mov	0, %g2
	mov	0, rlimb
	mov	0, %g4
	add	%i0, -8, %i0		C BOOKKEEPING

C Start software pipeline.

	ld	[%i1+4], u00_lo		C read low 32 bits of up[i]
	fxtod	u00_hi, u00
C mid
	ld	[%i1+0], u32_lo		C read high 32 bits of up[i]
	fmuld	u00, v000, a000
	fmuld	u00, v016, a016
	fmuld	u00, v032, a032
	fmuld	u00, v048, a048
	add	%i2, -1, %i2		C BOOKKEEPING
	fmuld	u00, v064, p064
	add	%i1, 8, %i1		C BOOKKEEPING
	fxtod	u32_hi, u32
	fmuld	u00, v080, p080
	fmuld	u00, v096, p096a
	brnz,pt	%i2, .L_2_or_more
	 fmuld	u00, v112, p112a

.L1:	fdtox	a000, out000
	fmuld	u32, v000, p000
	fdtox	a016, out016
	fmuld	u32, v016, p016
	fmovd	p064, a064
	fmuld	u32, v032, p032
	fmovd	p080, a080
	fmuld	u32, v048, p048
	std	out000, [%sp+2223+16]
	faddd	p000, a032, a000
	fmuld	u32, v064, p064
	std	out016, [%sp+2223+24]
	fxtod	u00_hi, u00
	faddd	p016, a048, a016
	fmuld	u32, v080, p080
	faddd	p032, a064, a032
	fmuld	u32, v096, p096b
	faddd	p048, a080, a048
	fmuld	u32, v112, p112b
C mid
	fdtox	a000, out000
	fdtox	a016, out016
	faddd	p064, p096a, a064
	faddd	p080, p112a, a080
	std	out000, [%sp+2223+0]
	b	.L_wd2
	 std	out016, [%sp+2223+8]

.L_2_or_more:
	ld	[%i1+4], u00_lo		C read low 32 bits of up[i]
	fdtox	a000, out000
	fmuld	u32, v000, p000
	fdtox	a016, out016
	fmuld	u32, v016, p016
	fmovd	p064, a064
	fmuld	u32, v032, p032
	fmovd	p080, a080
	fmuld	u32, v048, p048
	std	out000, [%sp+2223+16]
	faddd	p000, a032, a000
	fmuld	u32, v064, p064
	std	out016, [%sp+2223+24]
	fxtod	u00_hi, u00
	faddd	p016, a048, a016
	fmuld	u32, v080, p080
	faddd	p032, a064, a032
	fmuld	u32, v096, p096b
	faddd	p048, a080, a048
	fmuld	u32, v112, p112b
C mid
	ld	[%i1+0], u32_lo		C read high 32 bits of up[i]
	fdtox	a000, out000
	fmuld	u00, v000, p000
	fdtox	a016, out016
	fmuld	u00, v016, p016
	faddd	p064, p096a, a064
	fmuld	u00, v032, p032
	faddd	p080, p112a, a080
	fmuld	u00, v048, p048
	add	%i2, -1, %i2		C BOOKKEEPING
	std	out000, [%sp+2223+0]
	faddd	p000, a032, a000
	fmuld	u00, v064, p064
	add	%i1, 8, %i1		C BOOKKEEPING
	std	out016, [%sp+2223+8]
	fxtod	u32_hi, u32
	faddd	p016, a048, a016
	fmuld	u00, v080, p080
	faddd	p032, a064, a032
	fmuld	u00, v096, p096a
	faddd	p048, a080, a048
	brnz,pt	%i2, .L_3_or_more
	 fmuld	u00, v112, p112a

	b	.Lend
	 nop

C  64      32       0
C   .       .       .
C   .       |__rXXX_|	32
C   .      |___cy___|	34
C   .  |_______i00__|	50
C  |_______i16__|   .	50


C BEGIN MAIN LOOP
	.align	16
.L_3_or_more:
.Loop:	ld	[%i1+4], u00_lo		C read low 32 bits of up[i]
	and	%g2, xffffffff, %g2
	fdtox	a000, out000
	fmuld	u32, v000, p000
C
	lduw	[%i0+4+8], r00		C read low 32 bits of rp[i]
	add	%g2, rlimb, %l5
	fdtox	a016, out016
	fmuld	u32, v016, p016
C
	srlx	%l5, 32, cy
	ldx	[%sp+2223+16], i00
	faddd	p064, p096b, a064
	fmuld	u32, v032, p032
C
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+24], i16
	faddd	p080, p112b, a080
	fmuld	u32, v048, p048
C
	nop
	std	out000, [%sp+2223+16]
	faddd	p000, a032, a000
	fmuld	u32, v064, p064
C
	add	i00, r00, rlimb
	add	%i0, 8, %i0		C BOOKKEEPING
	std	out016, [%sp+2223+24]
	fxtod	u00_hi, u00
C
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	faddd	p016, a048, a016
	fmuld	u32, v080, p080
C
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	faddd	p032, a064, a032
	fmuld	u32, v096, p096b
C
	stw	%l5, [%i0+4]
	nop
	faddd	p048, a080, a048
	fmuld	u32, v112, p112b
C midloop
	ld	[%i1+0], u32_lo		C read high 32 bits of up[i]
	and	%g2, xffffffff, %g2
	fdtox	a000, out000
	fmuld	u00, v000, p000
C
	lduw	[%i0+0], r32		C read high 32 bits of rp[i]
	add	%g2, rlimb, %l5
	fdtox	a016, out016
	fmuld	u00, v016, p016
C
	srlx	%l5, 32, cy
	ldx	[%sp+2223+0], i00
	faddd	p064, p096a, a064
	fmuld	u00, v032, p032
C
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+8], i16
	faddd	p080, p112a, a080
	fmuld	u00, v048, p048
C
	add	%i2, -1, %i2		C BOOKKEEPING
	std	out000, [%sp+2223+0]
	faddd	p000, a032, a000
	fmuld	u00, v064, p064
C
	add	i00, r32, rlimb
	add	%i1, 8, %i1		C BOOKKEEPING
	std	out016, [%sp+2223+8]
	fxtod	u32_hi, u32
C
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	faddd	p016, a048, a016
	fmuld	u00, v080, p080
C
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	faddd	p032, a064, a032
	fmuld	u00, v096, p096a
C
	stw	%l5, [%i0+0]
	faddd	p048, a080, a048
	brnz,pt	%i2, .Loop
	 fmuld	u00, v112, p112a
C END MAIN LOOP

C WIND-DOWN PHASE 1
.Lend:	and	%g2, xffffffff, %g2
	fdtox	a000, out000
	fmuld	u32, v000, p000
	lduw	[%i0+4+8], r00		C read low 32 bits of rp[i]
	add	%g2, rlimb, %l5
	fdtox	a016, out016
	fmuld	u32, v016, p016
	srlx	%l5, 32, cy
	ldx	[%sp+2223+16], i00
	faddd	p064, p096b, a064
	fmuld	u32, v032, p032
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+24], i16
	faddd	p080, p112b, a080
	fmuld	u32, v048, p048
	std	out000, [%sp+2223+16]
	faddd	p000, a032, a000
	fmuld	u32, v064, p064
	add	i00, r00, rlimb
	add	%i0, 8, %i0		C BOOKKEEPING
	std	out016, [%sp+2223+24]
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	faddd	p016, a048, a016
	fmuld	u32, v080, p080
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	faddd	p032, a064, a032
	fmuld	u32, v096, p096b
	stw	%l5, [%i0+4]
	faddd	p048, a080, a048
	fmuld	u32, v112, p112b
C mid
	and	%g2, xffffffff, %g2
	fdtox	a000, out000
	lduw	[%i0+0], r32		C read high 32 bits of rp[i]
	add	%g2, rlimb, %l5
	fdtox	a016, out016
	srlx	%l5, 32, cy
	ldx	[%sp+2223+0], i00
	faddd	p064, p096a, a064
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+8], i16
	faddd	p080, p112a, a080
	std	out000, [%sp+2223+0]
	add	i00, r32, rlimb
	std	out016, [%sp+2223+8]
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	stw	%l5, [%i0+0]

C WIND-DOWN PHASE 2
.L_wd2:	and	%g2, xffffffff, %g2
	fdtox	a032, out000
	lduw	[%i0+4+8], r00		C read low 32 bits of rp[i]
	add	%g2, rlimb, %l5
	fdtox	a048, out016
	srlx	%l5, 32, cy
	ldx	[%sp+2223+16], i00
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+24], i16
	std	out000, [%sp+2223+16]
	add	i00, r00, rlimb
	add	%i0, 8, %i0		C BOOKKEEPING
	std	out016, [%sp+2223+24]
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	stw	%l5, [%i0+4]
C mid
	and	%g2, xffffffff, %g2
	fdtox	a064, out000
	lduw	[%i0+0], r32		C read high 32 bits of rp[i]
	add	%g2, rlimb, %l5
	fdtox	a080, out016
	srlx	%l5, 32, cy
	ldx	[%sp+2223+0], i00
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+8], i16
	std	out000, [%sp+2223+0]
	add	i00, r32, rlimb
	std	out016, [%sp+2223+8]
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	stw	%l5, [%i0+0]

C WIND-DOWN PHASE 3
.L_wd3:	and	%g2, xffffffff, %g2
	fdtox	p096b, out000
	add	%g2, rlimb, %l5
	fdtox	p112b, out016
	srlx	%l5, 32, cy
	ldx	[%sp+2223+16], rlimb
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+24], i16
	std	out000, [%sp+2223+16]
	add	%i0, 8, %i0		C BOOKKEEPING
	std	out016, [%sp+2223+24]
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	stw	%l5, [%i0+4]
C mid
	and	%g2, xffffffff, %g2
	add	%g2, rlimb, %l5
	srlx	%l5, 32, cy
	ldx	[%sp+2223+0], rlimb
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+8], i16
	sllx	i16, 16, %g2
	add	cy, rlimb, rlimb
	srlx	i16, 16, %g4
	add	%g2, rlimb, %l5
	stw	%l5, [%i0+0]

	and	%g2, xffffffff, %g2
	add	%g2, rlimb, %l5
	srlx	%l5, 32, cy
	ldx	[%sp+2223+16], i00
	add	%g4, cy, cy		C new cy
	ldx	[%sp+2223+24], i16

	sllx	i16, 16, %g2
	add	i00, cy, cy
	return	%i7+8
	add	%g2, cy, %o0
EPILOGUE(mpn_addmul_2)
