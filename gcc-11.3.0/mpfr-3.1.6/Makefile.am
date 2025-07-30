# Copyright 2000-2017 Free Software Foundation, Inc.
# This Makefile.am is free software; the Free Software Foundation
# gives unlimited permission to copy and/or distribute it,
# with or without modifications, as long as this notice is preserved.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.


AUTOMAKE_OPTIONS = gnu

# ACLOCAL_AMFLAGS will be fully deprecated in Automake 2.0;
# AC_CONFIG_MACRO_DIRS (introduced in Automake 1.13) is now used instead,
# but we still set ACLOCAL_AMFLAGS to avoid a warning message from
# libtoolize and in case some developer needs to switch back to an
# old Automake version.
ACLOCAL_AMFLAGS = -I m4

SUBDIRS = doc src tests tune

nobase_dist_doc_DATA = AUTHORS BUGS COPYING COPYING.LESSER NEWS TODO \
  examples/ReadMe examples/divworst.c examples/rndo-add.c examples/sample.c \
  examples/version.c

EXTRA_DIST = PATCHES VERSION \
  tools/ck-copyright-notice tools/ck-news tools/ck-version-info \
  tools/get_patches.sh \
  $(DATAFILES)

# Various checks for "make dist".
# * Check consistency concerning -version-info. Moreover if the VERSION
#   file doesn't end with "-dev", check that the -version-info value is
#   up-to-date. Note: this is a heuristic, to detect some mistakes.
# * Check that copyright notices exist and appear to be correct.
# * Check the NEWS file.
dist-hook:
	cd $(srcdir) && tools/ck-version-info
	cd $(srcdir) && tools/ck-copyright-notice
	cd $(srcdir) && tools/ck-news
