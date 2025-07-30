## Process this file with automake to generate Makefile.in

# Copyright 2001, 2002 Free Software Foundation, Inc.
#
#  This file is part of the GNU MP Library.
#
#  The GNU MP Library is free software; you can redistribute it and/or modify
#  it under the terms of either:
#
#    * the GNU Lesser General Public License as published by the Free
#      Software Foundation; either version 3 of the License, or (at your
#      option) any later version.
#
#  or
#
#    * the GNU General Public License as published by the Free Software
#      Foundation; either version 2 of the License, or (at your option) any
#      later version.
#
#  or both in parallel, as here.
#
#  The GNU MP Library is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received copies of the GNU General Public License and the
#  GNU Lesser General Public License along with the GNU MP Library.  If not,
#  see https://www.gnu.org/licenses/.


AM_CPPFLAGS = -D__GMP_WITHIN_GMP -I$(top_srcdir)

noinst_LTLIBRARIES = libprintf.la

libprintf_la_SOURCES =							 \
  asprintf.c asprntffuns.c doprnt.c doprntf.c doprnti.c			 \
  fprintf.c obprintf.c obvprintf.c obprntffuns.c			 \
  printf.c printffuns.c snprintf.c snprntffuns.c sprintf.c sprintffuns.c \
  vasprintf.c vfprintf.c vprintf.c vsnprintf.c vsprintf.c		 \
  repl-vsnprintf.c
