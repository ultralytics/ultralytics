dnl  SPARC v9 64-bit mpn_addmul_1 -- Multiply a limb vector with a limb and add
dnl  the result to a second limb vector.

dnl  Copyright 1998, 2000-2004 Free Software Foundation, Inc.

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

C		   cycles/limb
C UltraSPARC 1&2:     14
C UltraSPARC 3:	      17.5

C Algorithm: We use eight floating-point multiplies per limb product, with the
C invariant v operand split into four 16-bit pieces, and the up operand split
C into 32-bit pieces.  We sum pairs of 48-bit partial products using
C floating-point add, then convert the four 49-bit product-sums and transfer
C them to the integer unit.

C Possible optimizations:
C   0. Rewrite to use algorithm of mpn_addmul_2.
C   1. Align the stack area where we transfer the four 49-bit product-sums
C      to a 32-byte boundary.  That would minimize the cache collision.
C      (UltraSPARC-1/2 use a direct-mapped cache.)  (Perhaps even better would
C      be to align the area to map to the area immediately before up?)
C   2. Sum the 4 49-bit quantities using 32-bit operations, as in the
C      develop mpn_addmul_2.  This would save many integer instructions.
C   3. Unrolling.  Questionable if it is worth the code expansion, given that
C      it could only save 1 cycle/limb.
C   4. Specialize for particular v values.  If its upper 32 bits are zero, we
C      could save many operations, in the FPU (fmuld), but more so in the IEU
C      since we'll be summing 48-bit quantities, which might be simpler.
C   5. Ideally, we should schedule the f2/f3 and f4/f5 RAW further apart, and
C      the i00,i16,i32,i48 RAW less apart.  The latter apart-scheduling should
C      not be greater than needed for L2 cache latency, and also not so great
C      that i16 needs to be copied.
C   6. Avoid performing mem+fa+fm in the same cycle, at least not when we want
C      to get high IEU bandwidth.  (12 of the 14 cycles will be free for 2 IEU
C      ops.)

C Instruction classification (as per UltraSPARC-1/2 functional units):
C    8 FM
C   10 FA
C   12 MEM
C   10 ISHIFT + 14 IADDLOG
C    1 BRANCH
C   55 insns totally (plus one mov insn that should be optimized out)

C The loop executes 56 instructions in 14 cycles on UltraSPARC-1/2, i.e we
C sustain the peak execution rate of 4 instructions/cycle.

C INPUT PARAMETERS
C rp	i0
C up	i1
C n	i2
C v	i3

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)

define(`p00', `%f8') define(`p16',`%f10') define(`p32',`%f12') define(`p48',`%f14')
define(`r32',`%f16') define(`r48',`%f18') define(`r64',`%f20') define(`r80',`%f22')
define(`v00',`%f24') define(`v16',`%f26') define(`v32',`%f28') define(`v48',`%f30')
define(`u00',`%f32') define(`u32', `%f34')
define(`a00',`%f36') define(`a16',`%f38') define(`a32',`%f40') define(`a48',`%f42')
define(`cy',`%g1')
define(`rlimb',`%g3')
define(`i00',`%l0') define(`i16',`%l1') define(`i32',`%l2') define(`i48',`%l3')
define(`xffffffff',`%l7')
define(`xffff',`%o0')

PROLOGUE(mpn_addmul_1)

C Initialization.  (1) Split v operand into four 16-bit chunks and store them
C as IEEE double in fp registers.  (2) Clear upper 32 bits of fp register pairs
C f2 and f4.  (3) Store masks in registers aliased to `xffff' and `xffffffff'.

	save	%sp, -256, %sp
	mov	-1, %g4
	srlx	%g4, 48, xffff		C store mask in register `xffff'
	and	%i3, xffff, %g2
	stx	%g2, [%sp+2223+0]
	srlx	%i3, 16, %g3
	and	%g3, xffff, %g3
	stx	%g3, [%sp+2223+8]
	srlx	%i3, 32, %g2
	and	%g2, xffff, %g2
	stx	%g2, [%sp+2223+16]
	srlx	%i3, 48, %g3
	stx	%g3, [%sp+2223+24]
	srlx	%g4, 32, xffffffff	C store mask in register `xffffffff'

	sllx	%i2, 3, %i2
	mov	0, cy			C clear cy
	add	%i0, %i2, %i0
	add	%i1, %i2, %i1
	neg	%i2
	add	%i1, 4, %i5
	add	%i0, -32, %i4
	add	%i0, -16, %i0

	ldd	[%sp+2223+0], v00
	ldd	[%sp+2223+8], v16
	ldd	[%sp+2223+16], v32
	ldd	[%sp+2223+24], v48
	ld	[%sp+2223+0],%f2	C zero f2
	ld	[%sp+2223+0],%f4	C zero f4
	ld	[%i5+%i2], %f3		C read low 32 bits of up[i]
	ld	[%i1+%i2], %f5		C read high 32 bits of up[i]
	fxtod	v00, v00
	fxtod	v16, v16
	fxtod	v32, v32
	fxtod	v48, v48

C Start real work.  (We sneakingly read f3 and f5 above...)
C The software pipeline is very deep, requiring 4 feed-in stages.

	fxtod	%f2, u00
	fxtod	%f4, u32
	fmuld	u00, v00, a00
	fmuld	u00, v16, a16
	fmuld	u00, v32, p32
	fmuld	u32, v00, r32
	fmuld	u00, v48, p48
	addcc	%i2, 8, %i2
	bnz,pt	%xcc, .L_two_or_more
	fmuld	u32, v16, r48

.L_one:
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
	fdtox	a00, a00
	faddd	p48, r48, a48
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
	fdtox	a32, a32
	fdtox	a48, a48
	std	a00, [%sp+2223+0]
	std	a16, [%sp+2223+8]
	std	a32, [%sp+2223+16]
	std	a48, [%sp+2223+24]
	add	%i2, 8, %i2

	fdtox	r64, a00
	ldx	[%i0+%i2], rlimb	C read rp[i]
	fdtox	r80, a16
	ldx	[%sp+2223+0], i00
	ldx	[%sp+2223+8], i16
	ldx	[%sp+2223+16], i32
	ldx	[%sp+2223+24], i48
	std	a00, [%sp+2223+0]
	std	a16, [%sp+2223+8]
	add	%i2, 8, %i2

	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	add	i00, %g5, %g5		C i00+ now in g5
	ldx	[%sp+2223+0], i00
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	sllx	i48, 32, %l6		C (i48 << 32)
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	std	a00, [%sp+2223+0]
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	std	a16, [%sp+2223+8]
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	add	%l6, %o2, %o2		C mi64- in %o2
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	add	cy, %g5, %o4		C x = prev(i00) + cy
	b	.L_out_1
	add	%i2, 8, %i2

.L_two_or_more:
	ld	[%i5+%i2], %f3		C read low 32 bits of up[i]
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
	ld	[%i1+%i2], %f5		C read high 32 bits of up[i]
	fdtox	a00, a00
	faddd	p48, r48, a48
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
	fdtox	a32, a32
	fxtod	%f2, u00
	fxtod	%f4, u32
	fdtox	a48, a48
	std	a00, [%sp+2223+0]
	fmuld	u00, v00, p00
	std	a16, [%sp+2223+8]
	fmuld	u00, v16, p16
	std	a32, [%sp+2223+16]
	fmuld	u00, v32, p32
	std	a48, [%sp+2223+24]
	faddd	p00, r64, a00
	fmuld	u32, v00, r32
	faddd	p16, r80, a16
	fmuld	u00, v48, p48
	addcc	%i2, 8, %i2
	bnz,pt	%xcc, .L_three_or_more
	fmuld	u32, v16, r48

.L_two:
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
	fdtox	a00, a00
	ldx	[%i0+%i2], rlimb	C read rp[i]
	faddd	p48, r48, a48
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
	ldx	[%sp+2223+0], i00
	fdtox	a32, a32
	ldx	[%sp+2223+8], i16
	ldx	[%sp+2223+16], i32
	ldx	[%sp+2223+24], i48
	fdtox	a48, a48
	std	a00, [%sp+2223+0]
	std	a16, [%sp+2223+8]
	std	a32, [%sp+2223+16]
	std	a48, [%sp+2223+24]
	add	%i2, 8, %i2

	fdtox	r64, a00
	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	ldx	[%i0+%i2], rlimb	C read rp[i]
	add	i00, %g5, %g5		C i00+ now in g5
	fdtox	r80, a16
	ldx	[%sp+2223+0], i00
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	ldx	[%sp+2223+16], i32
	sllx	i48, 32, %l6		C (i48 << 32)
	ldx	[%sp+2223+24], i48
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	std	a00, [%sp+2223+0]
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	std	a16, [%sp+2223+8]
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	add	%l6, %o2, %o2		C mi64- in %o2
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	add	cy, %g5, %o4		C x = prev(i00) + cy
	b	.L_out_2
	add	%i2, 8, %i2

.L_three_or_more:
	ld	[%i5+%i2], %f3		C read low 32 bits of up[i]
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
	ld	[%i1+%i2], %f5		C read high 32 bits of up[i]
	fdtox	a00, a00
	ldx	[%i0+%i2], rlimb	C read rp[i]
	faddd	p48, r48, a48
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
	ldx	[%sp+2223+0], i00
	fdtox	a32, a32
	ldx	[%sp+2223+8], i16
	fxtod	%f2, u00
	ldx	[%sp+2223+16], i32
	fxtod	%f4, u32
	ldx	[%sp+2223+24], i48
	fdtox	a48, a48
	std	a00, [%sp+2223+0]
	fmuld	u00, v00, p00
	std	a16, [%sp+2223+8]
	fmuld	u00, v16, p16
	std	a32, [%sp+2223+16]
	fmuld	u00, v32, p32
	std	a48, [%sp+2223+24]
	faddd	p00, r64, a00
	fmuld	u32, v00, r32
	faddd	p16, r80, a16
	fmuld	u00, v48, p48
	addcc	%i2, 8, %i2
	bnz,pt	%xcc, .L_four_or_more
	fmuld	u32, v16, r48

.L_three:
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
	fdtox	a00, a00
	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	ldx	[%i0+%i2], rlimb	C read rp[i]
	faddd	p48, r48, a48
	add	i00, %g5, %g5		C i00+ now in g5
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
	ldx	[%sp+2223+0], i00
	fdtox	a32, a32
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	ldx	[%sp+2223+16], i32
	sllx	i48, 32, %l6		C (i48 << 32)
	ldx	[%sp+2223+24], i48
	fdtox	a48, a48
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	std	a00, [%sp+2223+0]
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	std	a16, [%sp+2223+8]
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	std	a32, [%sp+2223+16]
	add	%l6, %o2, %o2		C mi64- in %o2
	std	a48, [%sp+2223+24]
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	add	cy, %g5, %o4		C x = prev(i00) + cy
	b	.L_out_3
	add	%i2, 8, %i2

.L_four_or_more:
	ld	[%i5+%i2], %f3		C read low 32 bits of up[i]
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
	ld	[%i1+%i2], %f5		C read high 32 bits of up[i]
	fdtox	a00, a00
	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	ldx	[%i0+%i2], rlimb	C read rp[i]
	faddd	p48, r48, a48
	add	i00, %g5, %g5		C i00+ now in g5
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
	ldx	[%sp+2223+0], i00
	fdtox	a32, a32
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	fxtod	%f2, u00
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	ldx	[%sp+2223+16], i32
	fxtod	%f4, u32
	sllx	i48, 32, %l6		C (i48 << 32)
	ldx	[%sp+2223+24], i48
	fdtox	a48, a48
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	std	a00, [%sp+2223+0]
	fmuld	u00, v00, p00
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	std	a16, [%sp+2223+8]
	fmuld	u00, v16, p16
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	std	a32, [%sp+2223+16]
	fmuld	u00, v32, p32
	add	%l6, %o2, %o2		C mi64- in %o2
	std	a48, [%sp+2223+24]
	faddd	p00, r64, a00
	fmuld	u32, v00, r32
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	faddd	p16, r80, a16
	fmuld	u00, v48, p48
	add	cy, %g5, %o4		C x = prev(i00) + cy
	addcc	%i2, 8, %i2
	bnz,pt	%xcc, .Loop
	fmuld	u32, v16, r48

.L_four:
	b,a	.L_out_4

C BEGIN MAIN LOOP
	.align	16
.Loop:
C 00
	srlx	%o4, 16, %o5		C (x >> 16)
	ld	[%i5+%i2], %f3		C read low 32 bits of up[i]
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
C 01
	add	%o5, %o2, %o2		C mi64 in %o2   2nd ASSIGNMENT
	and	%o4, xffff, %o5		C (x & 0xffff)
	ld	[%i1+%i2], %f5		C read high 32 bits of up[i]
	fdtox	a00, a00
C 02
	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	ldx	[%i0+%i2], rlimb	C read rp[i]
	faddd	p48, r48, a48
C 03
	srlx	%o2, 48, %o7		C (mi64 >> 48)
	add	i00, %g5, %g5		C i00+ now in g5
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
C 04
	sllx	%o2, 16, %i3		C (mi64 << 16)
	add	%o7, %o1, cy		C new cy
	ldx	[%sp+2223+0], i00
	fdtox	a32, a32
C 05
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	fxtod	%f2, u00
C 06
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	ldx	[%sp+2223+16], i32
	fxtod	%f4, u32
C 07
	sllx	i48, 32, %l6		C (i48 << 32)
	or	%i3, %o5, %o5
	ldx	[%sp+2223+24], i48
	fdtox	a48, a48
C 08
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	std	a00, [%sp+2223+0]
	fmuld	u00, v00, p00
C 09
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	std	a16, [%sp+2223+8]
	fmuld	u00, v16, p16
C 10
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	std	a32, [%sp+2223+16]
	fmuld	u00, v32, p32
C 11
	add	%l6, %o2, %o2		C mi64- in %o2
	std	a48, [%sp+2223+24]
	faddd	p00, r64, a00
	fmuld	u32, v00, r32
C 12
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	stx	%o5, [%i4+%i2]
	faddd	p16, r80, a16
	fmuld	u00, v48, p48
C 13
	add	cy, %g5, %o4		C x = prev(i00) + cy
	addcc	%i2, 8, %i2
	bnz,pt	%xcc, .Loop
	fmuld	u32, v16, r48
C END MAIN LOOP

.L_out_4:
	srlx	%o4, 16, %o5		C (x >> 16)
	fmuld	u32, v32, r64	C FIXME not urgent
	faddd	p32, r32, a32
	add	%o5, %o2, %o2		C mi64 in %o2   2nd ASSIGNMENT
	and	%o4, xffff, %o5		C (x & 0xffff)
	fdtox	a00, a00
	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	ldx	[%i0+%i2], rlimb	C read rp[i]
	faddd	p48, r48, a48
	srlx	%o2, 48, %o7		C (mi64 >> 48)
	add	i00, %g5, %g5		C i00+ now in g5
	fmuld	u32, v48, r80	C FIXME not urgent
	fdtox	a16, a16
	sllx	%o2, 16, %i3		C (mi64 << 16)
	add	%o7, %o1, cy		C new cy
	ldx	[%sp+2223+0], i00
	fdtox	a32, a32
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	ldx	[%sp+2223+16], i32
	sllx	i48, 32, %l6		C (i48 << 32)
	or	%i3, %o5, %o5
	ldx	[%sp+2223+24], i48
	fdtox	a48, a48
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	std	a00, [%sp+2223+0]
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	std	a16, [%sp+2223+8]
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	std	a32, [%sp+2223+16]
	add	%l6, %o2, %o2		C mi64- in %o2
	std	a48, [%sp+2223+24]
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	stx	%o5, [%i4+%i2]
	add	cy, %g5, %o4		C x = prev(i00) + cy
	add	%i2, 8, %i2
.L_out_3:
	srlx	%o4, 16, %o5		C (x >> 16)
	add	%o5, %o2, %o2		C mi64 in %o2   2nd ASSIGNMENT
	and	%o4, xffff, %o5		C (x & 0xffff)
	fdtox	r64, a00
	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	ldx	[%i0+%i2], rlimb	C read rp[i]
	srlx	%o2, 48, %o7		C (mi64 >> 48)
	add	i00, %g5, %g5		C i00+ now in g5
	fdtox	r80, a16
	sllx	%o2, 16, %i3		C (mi64 << 16)
	add	%o7, %o1, cy		C new cy
	ldx	[%sp+2223+0], i00
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	ldx	[%sp+2223+16], i32
	sllx	i48, 32, %l6		C (i48 << 32)
	or	%i3, %o5, %o5
	ldx	[%sp+2223+24], i48
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	std	a00, [%sp+2223+0]
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	std	a16, [%sp+2223+8]
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	add	%l6, %o2, %o2		C mi64- in %o2
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	stx	%o5, [%i4+%i2]
	add	cy, %g5, %o4		C x = prev(i00) + cy
	add	%i2, 8, %i2
.L_out_2:
	srlx	%o4, 16, %o5		C (x >> 16)
	add	%o5, %o2, %o2		C mi64 in %o2   2nd ASSIGNMENT
	and	%o4, xffff, %o5		C (x & 0xffff)
	srlx	rlimb, 32, %g4		C HI(rlimb)
	and	rlimb, xffffffff, %g5	C LO(rlimb)
	srlx	%o2, 48, %o7		C (mi64 >> 48)
	add	i00, %g5, %g5		C i00+ now in g5
	sllx	%o2, 16, %i3		C (mi64 << 16)
	add	%o7, %o1, cy		C new cy
	ldx	[%sp+2223+0], i00
	srlx	i16, 48, %l4		C (i16 >> 48)
	mov	i16, %g2
	ldx	[%sp+2223+8], i16
	srlx	i48, 16, %l5		C (i48 >> 16)
	add	i32, %g4, %g4		C i32+ now in g4
	sllx	i48, 32, %l6		C (i48 << 32)
	or	%i3, %o5, %o5
	srlx	%g4, 32, %o3		C (i32 >> 32)
	add	%l5, %l4, %o1		C hi64- in %o1
	sllx	%g4, 16, %o2		C (i32 << 16)
	add	%o3, %o1, %o1		C hi64 in %o1   1st ASSIGNMENT
	sllx	%o1, 48, %o3		C (hi64 << 48)
	add	%g2, %o2, %o2		C mi64- in %o2
	add	%l6, %o2, %o2		C mi64- in %o2
	sub	%o2, %o3, %o2		C mi64 in %o2   1st ASSIGNMENT
	stx	%o5, [%i4+%i2]
	add	cy, %g5, %o4		C x = prev(i00) + cy
	add	%i2, 8, %i2
.L_out_1:
	srlx	%o4, 16, %o5		C (x >> 16)
	add	%o5, %o2, %o2		C mi64 in %o2   2nd ASSIGNMENT
	and	%o4, xffff, %o5		C (x & 0xffff)
	srlx	%o2, 48, %o7		C (mi64 >> 48)
	sllx	%o2, 16, %i3		C (mi64 << 16)
	add	%o7, %o1, cy		C new cy
	or	%i3, %o5, %o5
	stx	%o5, [%i4+%i2]

	sllx	i00, 0, %g2
	add	%g2, cy, cy
	sllx	i16, 16, %g3
	add	%g3, cy, cy

	return	%i7+8
	mov	cy, %o0
EPILOGUE(mpn_addmul_1)
