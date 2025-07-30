dnl  x86 mpn_sec_tabselect.

dnl  Copyright 2011 Free Software Foundation, Inc.

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


C			    cycles/limb
C P5				 ?
C P6 model 0-8,10-12		 ?
C P6 model 9  (Banias)		 ?
C P6 model 13 (Dothan)		 ?
C P4 model 0  (Willamette)	 ?
C P4 model 1  (?)		 ?
C P4 model 2  (Northwood)	 4.5
C P4 model 3  (Prescott)	 ?
C P4 model 4  (Nocona)		 ?
C Intel Atom			 ?
C AMD K6			 ?
C AMD K7			 3.4
C AMD K8			 ?
C AMD K10			 ?

C NOTES
C  * This has not been tuned for any specific processor.  Its speed should not
C    be too bad, though.
C  * Using SSE2 could result in many-fold speedup.

C mpn_sec_tabselect (mp_limb_t *rp, mp_limb_t *tp, mp_size_t n, mp_size_t nents, mp_size_t which)
define(`rp',     `%edi')
define(`tp',     `%esi')
define(`n',      `%ebx')
define(`nents',  `%ecx')
define(`which',  `36(%esp)')

define(`i',      `%ebp')
define(`maskp',  `20(%esp)')
define(`maskn',  `32(%esp)')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_sec_tabselect)
	push	%edi
	push	%esi
	push	%ebx
	push	%ebp
	mov	20(%esp), rp
	mov	24(%esp), tp
	mov	28(%esp), n
	mov	32(%esp), nents

	lea	(rp,n,4), rp
	lea	(tp,n,4), tp
	sub	nents, which
L(outer):
	mov	which, %eax
	add	nents, %eax
	neg	%eax			C set CF iff 'which' != k
	sbb	%eax, %eax
	mov	%eax, maskn
	not	%eax
	mov	%eax, maskp

	mov	n, i
	neg	i

	ALIGN(16)
L(top):	mov	(tp,i,4), %eax
	and	maskp, %eax
	mov	(rp,i,4), %edx
	and	maskn, %edx
	or	%edx, %eax
	mov	%eax, (rp,i,4)
	inc	i
	js	L(top)

L(end):	mov	n, %eax
	lea	(tp,%eax,4), tp
	dec	nents
	jne	L(outer)

L(outer_end):
	pop	%ebp
	pop	%ebx
	pop	%esi
	pop	%edi
	ret
EPILOGUE()
