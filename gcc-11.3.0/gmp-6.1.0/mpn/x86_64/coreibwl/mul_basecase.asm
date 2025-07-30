dnl  AMD64 mpn_mul_basecase optimised for Intel Broadwell.

dnl  Copyright 2015 Free Software Foundation, Inc.

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

C cycles/limb	mul_1		addmul_1
C AMD K8,K9	n/a		n/a
C AMD K10	n/a		n/a
C AMD bull	n/a		n/a
C AMD pile	n/a		n/a
C AMD steam	n/a		n/a
C AMD excavator	 ?		 ?
C AMD bobcat	n/a		n/a
C AMD jaguar	n/a		n/a
C Intel P4	n/a		n/a
C Intel core2	n/a		n/a
C Intel NHM	n/a		n/a
C Intel SBR	n/a		n/a
C Intel IBR	n/a		n/a
C Intel HWL	 1.68		n/a
C Intel BWL	 1.69	      1.8-1.9
C Intel atom	n/a		n/a
C Intel SLM	n/a		n/a
C VIA nano	n/a		n/a

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * Do overlapped software pipelining.
C  * When changing this, make sure the code which falls into the inner loops
C    does not execute too many no-ops (for both PIC and non-PIC).

define(`rp',      `%rdi')
define(`up',      `%rsi')
define(`un_param',`%rdx')
define(`vp_param',`%rcx')
define(`vn',      `%r8')

define(`n',       `%rcx')
define(`n_save',  `%rbp')
define(`vp',      `%r14')
define(`unneg',   `%rbx')
define(`v0',      `%rdx')
define(`jaddr',   `%rax')

define(`w0',	`%r12')
define(`w1',	`%r9')
define(`w2',	`%r10')
define(`w3',	`%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_basecase)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')

	cmp	$2, un_param
	ja	L(gen)
	mov	(vp_param), %rdx
	mulx(	(up), %rax, %r9)	C 0 1
	je	L(s2x)

L(s11):	mov	%rax, (rp)
	mov	%r9, 8(rp)
	ret

L(s2x):	cmp	$2, vn
	mulx(	8,(up), %r8, %r10)	C 1 2
	je	L(s22)

L(s21):	add	%r8, %r9
	adc	$0, %r10
	mov	%rax, (rp)
	mov	%r9, 8(rp)
	mov	%r10, 16(rp)
	ret

L(s22):	add	%r8, %r9		C 1
	adc	$0, %r10		C 2
	mov	8(vp_param), %rdx
	mov	%rax, (rp)
	mulx(	(up), %r8, %r11)	C 1 2
	mulx(	8,(up), %rax, %rdx)	C 2 3
	add	%r11, %rax		C 2
	adc	$0, %rdx		C 3
	add	%r8, %r9		C 1
	adc	%rax, %r10		C 2
	adc	$0, %rdx		C 3
	mov	%r9, 8(rp)
	mov	%r10, 16(rp)
	mov	%rdx, 24(rp)
	ret

	ALIGN(16)
L(gen):
	push	%rbx
	push	%rbp
	push	%r12
	push	%r14

	mov	vp_param, vp
	lea	1(un_param), unneg
	mov	un_param, n_save
	mov	R32(un_param), R32(%rax)
	and	$-8, unneg
	shr	$3, n_save		C loop count
	neg	unneg
	and	$7, R32(%rax)		C clear CF for adc as side-effect
					C note that rax lives very long
	mov	n_save, n
	mov	(vp), v0
	lea	8(vp), vp

	lea	L(mtab)(%rip), %r10
ifdef(`PIC',
`	movslq	(%r10,%rax,4), %r11
	lea	(%r11, %r10), %r10
	jmp	*%r10
',`
	jmp	*(%r10,%rax,8)
')

L(mf0):	mulx(	(up), w2, w3)
	lea	56(up), up
	lea	-8(rp), rp
	jmp	L(mb0)

L(mf3):	mulx(	(up), w0, w1)
	lea	16(up), up
	lea	16(rp), rp
	inc	n
	jmp	L(mb3)

L(mf4):	mulx(	(up), w2, w3)
	lea	24(up), up
	lea	24(rp), rp
	inc	n
	jmp	L(mb4)

L(mf5):	mulx(	(up), w0, w1)
	lea	32(up), up
	lea	32(rp), rp
	inc	n
	jmp	L(mb5)

L(mf6):	mulx(	(up), w2, w3)
	lea	40(up), up
	lea	40(rp), rp
	inc	n
	jmp	L(mb6)

L(mf7):	mulx(	(up), w0, w1)
	lea	48(up), up
	lea	48(rp), rp
	inc	n
	jmp	L(mb7)

L(mf1):	mulx(	(up), w0, w1)
	jmp	L(mb1)

L(mf2):	mulx(	(up), w2, w3)
	lea	8(up), up
	lea	8(rp), rp
	mulx(	(up), w0, w1)

	ALIGN(16)
L(m1top):
	mov	w2, -8(rp)
	adc	w3, w0
L(mb1):	mulx(	8,(up), w2, w3)
	adc	w1, w2
	lea	64(up), up
	mov	w0, (rp)
L(mb0):	mov	w2, 8(rp)
	mulx(	-48,(up), w0, w1)
	lea	64(rp), rp
	adc	w3, w0
L(mb7):	mulx(	-40,(up), w2, w3)
	mov	w0, -48(rp)
	adc	w1, w2
L(mb6):	mov	w2, -40(rp)
	mulx(	-32,(up), w0, w1)
	adc	w3, w0
L(mb5):	mulx(	-24,(up), w2, w3)
	mov	w0, -32(rp)
	adc	w1, w2
L(mb4):	mulx(	-16,(up), w0, w1)
	mov	w2, -24(rp)
	adc	w3, w0
L(mb3):	mulx(	-8,(up), w2, w3)
	adc	w1, w2
	mov	w0, -16(rp)
	dec	n
	mulx(	(up), w0, w1)
	jnz	L(m1top)

L(m1end):
	mov	w2, -8(rp)
	adc	w3, w0
	mov	w0, (rp)
	adc	%rcx, w1		C relies on rcx = 0
	mov	w1, 8(rp)

	dec	vn
	jz	L(done)

	lea	L(atab)(%rip), %r10
ifdef(`PIC',
`	movslq	(%r10,%rax,4), %rax
	lea	(%rax, %r10), jaddr
',`
	mov	(%r10,%rax,8), jaddr
')

L(outer):
	lea	(up,unneg,8), up
	mov	n_save, n
	mov	(vp), v0
	lea	8(vp), vp
	jmp	*jaddr

L(f0):	mulx(	8,(up), w2, w3)
	lea	8(rp,unneg,8), rp
	lea	-1(n), n
	jmp	L(b0)

L(f3):	mulx(	-16,(up), w0, w1)
	lea	-56(rp,unneg,8), rp
	jmp	L(b3)

L(f4):	mulx(	-24,(up), w2, w3)
	lea	-56(rp,unneg,8), rp
	jmp	L(b4)

L(f5):	mulx(	-32,(up), w0, w1)
	lea	-56(rp,unneg,8), rp
	jmp	L(b5)

L(f6):	mulx(	-40,(up), w2, w3)
	lea	-56(rp,unneg,8), rp
	jmp	L(b6)

L(f7):	mulx(	16,(up), w0, w1)
	lea	8(rp,unneg,8), rp
	jmp	L(b7)

L(f1):	mulx(	(up), w0, w1)
	lea	8(rp,unneg,8), rp
	jmp	L(b1)

L(am1end):
	adox(	(rp), w0)
	adox(	%rcx, w1)		C relies on rcx = 0
	mov	w0, (rp)
	adc	%rcx, w1		C relies on rcx = 0
	mov	w1, 8(rp)

	dec	vn			C clear CF and OF as side-effect
	jnz	L(outer)
L(done):
	pop	%r14
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret

L(f2):
	mulx(	-8,(up), w2, w3)
	lea	8(rp,unneg,8), rp
	mulx(	(up), w0, w1)

	ALIGN(16)
L(am1top):
	adox(	-8,(rp), w2)
	adcx(	w3, w0)
	mov	w2, -8(rp)
	jrcxz	L(am1end)
L(b1):	mulx(	8,(up), w2, w3)
	adox(	(rp), w0)
	lea	-1(n), n
	mov	w0, (rp)
	adcx(	w1, w2)
L(b0):	mulx(	16,(up), w0, w1)
	adcx(	w3, w0)
	adox(	8,(rp), w2)
	mov	w2, 8(rp)
L(b7):	mulx(	24,(up), w2, w3)
	lea	64(up), up
	adcx(	w1, w2)
	adox(	16,(rp), w0)
	mov	w0, 16(rp)
L(b6):	mulx(	-32,(up), w0, w1)
	adox(	24,(rp), w2)
	adcx(	w3, w0)
	mov	w2, 24(rp)
L(b5):	mulx(	-24,(up), w2, w3)
	adcx(	w1, w2)
	adox(	32,(rp), w0)
	mov	w0, 32(rp)
L(b4):	mulx(	-16,(up), w0, w1)
	adox(	40,(rp), w2)
	adcx(	w3, w0)
	mov	w2, 40(rp)
L(b3):	adox(	48,(rp), w0)
	mulx(	-8,(up), w2, w3)
	mov	w0, 48(rp)
	lea	64(rp), rp
	adcx(	w1, w2)
	mulx(	(up), w0, w1)
	jmp	L(am1top)

	JUMPTABSECT
	ALIGN(8)
L(mtab):JMPENT(	L(mf0), L(mtab))
	JMPENT(	L(mf1), L(mtab))
	JMPENT(	L(mf2), L(mtab))
	JMPENT(	L(mf3), L(mtab))
	JMPENT(	L(mf4), L(mtab))
	JMPENT(	L(mf5), L(mtab))
	JMPENT(	L(mf6), L(mtab))
	JMPENT(	L(mf7), L(mtab))
L(atab):JMPENT(	L(f0), L(atab))
	JMPENT(	L(f1), L(atab))
	JMPENT(	L(f2), L(atab))
	JMPENT(	L(f3), L(atab))
	JMPENT(	L(f4), L(atab))
	JMPENT(	L(f5), L(atab))
	JMPENT(	L(f6), L(atab))
	JMPENT(	L(f7), L(atab))
	TEXT
EPILOGUE()
