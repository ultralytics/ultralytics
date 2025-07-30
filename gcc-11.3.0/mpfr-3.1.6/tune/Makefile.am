# Copyright 2010-2017 Free Software Foundation, Inc.
# This Makefile.am is free software; the Free Software Foundation
# gives unlimited permission to copy and/or distribute it,
# with or without modifications, as long as this notice is preserved.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.


EXTRA_PROGRAMS = tuneup speed bidimensional_sample

tuneup_SOURCES = tuneup.c
tuneup_LDADD = -lspeed $(top_builddir)/src/libmpfr.la $(TUNE_LIBS)
tuneup_LDFLAGS = -static

speed_SOURCES = speed.c
speed_LDADD = -lspeed $(top_builddir)/src/libmpfr.la $(TUNE_LIBS)
speed_LDFLAGS = -static

bidimensional_sample_SOURCES = bidimensional_sample.c
bidimensional_sample_LDADD = -lspeed $(top_builddir)/src/libmpfr.la $(TUNE_LIBS)
bidimensional_sample_LDFLAGS = -static

AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_builddir)/src

tune:
	$(MAKE) $(AM_MAKEFLAGS) tuneup$(EXEEXT)
	./tuneup$(EXEEXT) -v
	mv mparam.h $(top_builddir)/src/
	cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) clean
	cd $(top_builddir)/src && $(MAKE) $(AM_MAKEFLAGS) libmpfr.la

$(top_builddir)/src/libmpfr.la:
	cd $(top_builddir)/src && $(MAKE) $(AM_MAKEFLAGS) libmpfr.la

CLEANFILES = $(EXTRA_PROGRAMS) mparam.h
