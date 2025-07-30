/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_ARG_H
#define ISL_ARG_H

#include <stddef.h>
#include <stdlib.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_arg_choice {
	const char	*name;
	unsigned	 value;
};

struct isl_arg_flags {
	const char	*name;
	unsigned	 mask;
	unsigned	 value;
};

enum isl_arg_type {
	isl_arg_end,
	isl_arg_alias,
	isl_arg_arg,
	isl_arg_bool,
	isl_arg_child,
	isl_arg_choice,
	isl_arg_flags,
	isl_arg_footer,
	isl_arg_int,
	isl_arg_user,
	isl_arg_long,
	isl_arg_ulong,
	isl_arg_str,
	isl_arg_str_list,
	isl_arg_version
};

struct isl_args;

struct isl_arg {
	enum isl_arg_type	 type;
	char			 short_name;
	const char		*long_name;
	const char		*argument_name;
	size_t			 offset;
	const char		*help_msg;
#define ISL_ARG_SINGLE_DASH	(1 << 0)
#define ISL_ARG_BOOL_ARG	(1 << 1)
#define ISL_ARG_HIDDEN		(1 << 2)
	unsigned		 flags;
	union {
	struct {
		struct isl_arg_choice	*choice;
		unsigned	 	 default_value;
		unsigned	 	 default_selected;
		int (*set)(void *opt, unsigned val);
	} choice;
	struct {
		struct isl_arg_flags	*flags;
		unsigned	 	 default_value;
	} flags;
	struct {
		unsigned		 default_value;
		int (*set)(void *opt, unsigned val);
	} b;
	struct {
		int			default_value;
	} i;
	struct {
		long		 	default_value;
		long		 	default_selected;
		int (*set)(void *opt, long val);
	} l;
	struct {
		unsigned long		default_value;
	} ul;
	struct {
		const char		*default_value;
	} str;
	struct {
		size_t			 offset_n;
	} str_list;
	struct {
		struct isl_args		*child;
	} child;
	struct {
		void (*print_version)(void);
	} version;
	struct {
		int (*init)(void*);
		void (*clear)(void*);
	} user;
	} u;
};

struct isl_args {
	size_t			 options_size;
	struct isl_arg		*args;
};

#define ISL_ARGS_START(s,name)						\
	struct isl_arg name ## LIST[];					\
	struct isl_args name = { sizeof(s), name ## LIST };		\
	struct isl_arg name ## LIST[] = {
#define ISL_ARGS_END							\
	{ isl_arg_end } };

#define ISL_ARG_ALIAS(l)	{					\
	.type = isl_arg_alias,						\
	.long_name = l,							\
},
#define ISL_ARG_ARG(st,f,a,d)	{					\
	.type = isl_arg_arg,						\
	.argument_name = a,						\
	.offset = offsetof(st, f),					\
	.u = { .str = { .default_value = d } }				\
},
#define ISL_ARG_FOOTER(h)	{					\
	.type = isl_arg_footer,						\
	.help_msg = h,							\
},
#define ISL_ARG_CHOICE(st,f,s,l,c,d,h)	{				\
	.type = isl_arg_choice,						\
	.short_name = s,						\
	.long_name = l,							\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .choice = { .choice = c, .default_value = d,		\
			    .default_selected = d, .set = NULL } }	\
},
#define ISL_ARG_OPT_CHOICE(st,f,s,l,c,d,ds,h)	{			\
	.type = isl_arg_choice,						\
	.short_name = s,						\
	.long_name = l,							\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .choice = { .choice = c, .default_value = d,		\
			    .default_selected = ds, .set = NULL } }	\
},
#define ISL_ARG_PHANTOM_USER_CHOICE_F(s,l,c,setter,d,h,fl)	{	\
	.type = isl_arg_choice,						\
	.short_name = s,						\
	.long_name = l,							\
	.offset = -1,							\
	.help_msg = h,							\
	.flags = fl,							\
	.u = { .choice = { .choice = c, .default_value = d,		\
			    .default_selected = d, .set = setter } }	\
},
#define ISL_ARG_USER_OPT_CHOICE(st,f,s,l,c,setter,d,ds,h)	{	\
	.type = isl_arg_choice,						\
	.short_name = s,						\
	.long_name = l,							\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .choice = { .choice = c, .default_value = d,		\
			    .default_selected = ds, .set = setter } }	\
},
#define _ISL_ARG_BOOL_F(o,s,l,setter,d,h,fl)	{			\
	.type = isl_arg_bool,						\
	.short_name = s,						\
	.long_name = l,							\
	.offset = o,							\
	.help_msg = h,							\
	.flags = fl,							\
	.u = { .b = { .default_value = d, .set = setter } }		\
},
#define ISL_ARG_BOOL_F(st,f,s,l,d,h,fl)					\
	_ISL_ARG_BOOL_F(offsetof(st, f),s,l,NULL,d,h,fl)
#define ISL_ARG_BOOL(st,f,s,l,d,h)					\
	ISL_ARG_BOOL_F(st,f,s,l,d,h,0)
#define ISL_ARG_PHANTOM_BOOL_F(s,l,setter,h,fl)				\
	_ISL_ARG_BOOL_F(-1,s,l,setter,0,h,fl)
#define ISL_ARG_PHANTOM_BOOL(s,l,setter,h)				\
	ISL_ARG_PHANTOM_BOOL_F(s,l,setter,h,0)
#define ISL_ARG_INT_F(st,f,s,l,a,d,h,fl)	{			\
	.type = isl_arg_int,						\
	.short_name = s,						\
	.long_name = l,							\
	.argument_name = a,						\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.flags = fl,							\
	.u = { .ul = { .default_value = d } }				\
},
#define ISL_ARG_INT(st,f,s,l,a,d,h)					\
	ISL_ARG_INT_F(st,f,s,l,a,d,h,0)
#define ISL_ARG_LONG(st,f,s,lo,d,h)	{				\
	.type = isl_arg_long,						\
	.short_name = s,						\
	.long_name = lo,						\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .l = { .default_value = d, .default_selected = d,	\
		      .set = NULL } }					\
},
#define ISL_ARG_USER_LONG(st,f,s,lo,setter,d,h)	{			\
	.type = isl_arg_long,						\
	.short_name = s,						\
	.long_name = lo,						\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .l = { .default_value = d, .default_selected = d,	\
		      .set = setter } }					\
},
#define ISL_ARG_OPT_LONG(st,f,s,lo,d,ds,h)	{			\
	.type = isl_arg_long,						\
	.short_name = s,						\
	.long_name = lo,						\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .l = { .default_value = d, .default_selected = ds,	\
		      .set = NULL } }					\
},
#define ISL_ARG_ULONG(st,f,s,l,d,h)	{				\
	.type = isl_arg_ulong,						\
	.short_name = s,						\
	.long_name = l,							\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .ul = { .default_value = d } }				\
},
#define ISL_ARG_STR_F(st,f,s,l,a,d,h,fl)	{			\
	.type = isl_arg_str,						\
	.short_name = s,						\
	.long_name = l,							\
	.argument_name = a,						\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.flags = fl,							\
	.u = { .str = { .default_value = d } }				\
},
#define ISL_ARG_STR(st,f,s,l,a,d,h)					\
	ISL_ARG_STR_F(st,f,s,l,a,d,h,0)
#define ISL_ARG_STR_LIST(st,f_n,f_l,s,l,a,h)	{			\
	.type = isl_arg_str_list,					\
	.short_name = s,						\
	.long_name = l,							\
	.argument_name = a,						\
	.offset = offsetof(st, f_l),					\
	.help_msg = h,							\
	.u = { .str_list = { .offset_n = offsetof(st, f_n) } }		\
},
#define _ISL_ARG_CHILD(o,l,c,h,fl)	{				\
	.type = isl_arg_child,						\
	.long_name = l,							\
	.offset = o,							\
	.help_msg = h,							\
	.flags = fl,							\
	.u = { .child = { .child = c } }				\
},
#define ISL_ARG_CHILD(st,f,l,c,h)					\
	_ISL_ARG_CHILD(offsetof(st, f),l,c,h,0)
#define ISL_ARG_GROUP_F(l,c,h,fl)					\
	_ISL_ARG_CHILD(-1,l,c,h,fl)
#define ISL_ARG_GROUP(l,c,h)						\
	ISL_ARG_GROUP_F(l,c,h,0)
#define ISL_ARG_FLAGS(st,f,s,l,c,d,h)	{				\
	.type = isl_arg_flags,						\
	.short_name = s,						\
	.long_name = l,							\
	.offset = offsetof(st, f),					\
	.help_msg = h,							\
	.u = { .flags = { .flags = c, .default_value = d } }		\
},
#define ISL_ARG_USER(st,f,i,c) {					\
	.type = isl_arg_user,						\
	.offset = offsetof(st, f),					\
	.u = { .user = { .init = i, .clear = c} }			\
},
#define ISL_ARG_VERSION(print) {					\
	.type = isl_arg_version,					\
	.u = { .version = { .print_version = print } }			\
},

#define ISL_ARG_ALL		(1 << 0)
#define ISL_ARG_SKIP_HELP	(1 << 1)

void isl_args_set_defaults(struct isl_args *args, void *opt);
void isl_args_free(struct isl_args *args, void *opt);
int isl_args_parse(struct isl_args *args, int argc, char **argv, void *opt,
	unsigned flags);

#define ISL_ARG_DECL(prefix,st,args)					\
extern struct isl_args args;						\
st *prefix ## _new_with_defaults(void);					\
void prefix ## _free(st *opt);						\
int prefix ## _parse(st *opt, int argc, char **argv, unsigned flags);

#define ISL_ARG_DEF(prefix,st,args)					\
st *prefix ## _new_with_defaults()					\
{									\
	st *opt = (st *)calloc(1, sizeof(st));				\
	if (opt)							\
		isl_args_set_defaults(&(args), opt);			\
	return opt;							\
}									\
									\
void prefix ## _free(st *opt)						\
{									\
	isl_args_free(&(args), opt);					\
}									\
									\
int prefix ## _parse(st *opt, int argc, char **argv, unsigned flags)	\
{									\
	return isl_args_parse(&(args), argc, argv, opt, flags);		\
}

#if defined(__cplusplus)
}
#endif

#endif
