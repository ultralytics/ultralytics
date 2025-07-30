; mc88110 __gmpn_addmul_1 -- Multiply a limb vector with a single limb and
; store the product in a second limb vector.

; Copyright 1996, 2000 Free Software Foundation, Inc.

;  This file is part of the GNU MP Library.
;
;  The GNU MP Library is free software; you can redistribute it and/or modify
;  it under the terms of either:
;
;    * the GNU Lesser General Public License as published by the Free
;      Software Foundation; either version 3 of the License, or (at your
;      option) any later version.
;
;  or
;
;    * the GNU General Public License as published by the Free Software
;      Foundation; either version 2 of the License, or (at your option) any
;      later version.
;
;  or both in parallel, as here.
;
;  The GNU MP Library is distributed in the hope that it will be useful, but
;  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
;  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
;  for more details.
;
;  You should have received copies of the GNU General Public License and the
;  GNU Lesser General Public License along with the GNU MP Library.  If not,
;  see https://www.gnu.org/licenses/.


; INPUT PARAMETERS
; res_ptr	r2
; s1_ptr	r3
; size		r4
; s2_limb	r5

	text
	align	16
	global	___gmpn_addmul_1
___gmpn_addmul_1:
	lda	 r3,r3[r4]
	lda	 r8,r2[r4]		; RES_PTR in r8 since r2 is retval
	subu	 r4,r0,r4
	addu.co	 r2,r0,r0		; r2 = cy = 0

	ld	 r6,r3[r4]
	addu	 r4,r4,1
	subu	 r8,r8,4
	bcnd.n	 eq0,r4,Lend
	 mulu.d	 r10,r6,r5

Loop:	ld	 r7,r8[r4]
	ld	 r6,r3[r4]
	addu.cio r9,r11,r2
	addu.ci	 r2,r10,r0
	addu.co	 r9,r9,r7
	st	 r9,r8[r4]
	addu	 r4,r4,1
	mulu.d	 r10,r6,r5
	bcnd	 ne0,r4,Loop

Lend:	ld	 r7,r8,0
	addu.cio r9,r11,r2
	addu.ci	 r2,r10,r0
	addu.co	 r9,r9,r7
	st	 r9,r8,0
	jmp.n	 r1
	 addu.ci r2,r2,r0
