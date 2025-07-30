/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_LIST_H
#define ISL_LIST_H

#include <isl/ctx.h>
#include <isl/printer_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define ISL_DECLARE_LIST_TYPE(EL)					\
struct isl_##EL;							\
struct isl_##EL##_list;							\
typedef struct isl_##EL##_list isl_##EL##_list;
#define ISL_DECLARE_LIST_FN(EL)						\
isl_ctx *isl_##EL##_list_get_ctx(__isl_keep isl_##EL##_list *list);	\
__isl_give isl_##EL##_list *isl_##EL##_list_from_##EL(			\
	__isl_take struct isl_##EL *el);				\
__isl_give isl_##EL##_list *isl_##EL##_list_alloc(isl_ctx *ctx, int n);	\
__isl_give isl_##EL##_list *isl_##EL##_list_copy(			\
	__isl_keep isl_##EL##_list *list);				\
__isl_null isl_##EL##_list *isl_##EL##_list_free(			\
	__isl_take isl_##EL##_list *list);				\
__isl_give isl_##EL##_list *isl_##EL##_list_add(			\
	__isl_take isl_##EL##_list *list,				\
	__isl_take struct isl_##EL *el);				\
__isl_give isl_##EL##_list *isl_##EL##_list_insert(			\
	__isl_take isl_##EL##_list *list, unsigned pos,			\
	__isl_take struct isl_##EL *el);				\
__isl_give isl_##EL##_list *isl_##EL##_list_drop(			\
	__isl_take isl_##EL##_list *list, unsigned first, unsigned n);	\
__isl_give isl_##EL##_list *isl_##EL##_list_concat(			\
	__isl_take isl_##EL##_list *list1,				\
	__isl_take isl_##EL##_list *list2);				\
int isl_##EL##_list_n_##EL(__isl_keep isl_##EL##_list *list);		\
__isl_give struct isl_##EL *isl_##EL##_list_get_##EL(			\
	__isl_keep isl_##EL##_list *list, int index);			\
__isl_give struct isl_##EL##_list *isl_##EL##_list_set_##EL(		\
	__isl_take struct isl_##EL##_list *list, int index,		\
	__isl_take struct isl_##EL *el);				\
isl_stat isl_##EL##_list_foreach(__isl_keep isl_##EL##_list *list,	\
	isl_stat (*fn)(__isl_take struct isl_##EL *el, void *user),	\
	void *user);							\
__isl_give isl_##EL##_list *isl_##EL##_list_sort(			\
	__isl_take isl_##EL##_list *list,				\
	int (*cmp)(__isl_keep struct isl_##EL *a,			\
		__isl_keep struct isl_##EL *b,				\
		void *user), void *user);				\
isl_stat isl_##EL##_list_foreach_scc(__isl_keep isl_##EL##_list *list,	\
	isl_bool (*follows)(__isl_keep struct isl_##EL *a,		\
			__isl_keep struct isl_##EL *b, void *user),	\
	void *follows_user,						\
	isl_stat (*fn)(__isl_take isl_##EL##_list *scc, void *user),	\
	void *fn_user);							\
__isl_give isl_printer *isl_printer_print_##EL##_list(			\
	__isl_take isl_printer *p, __isl_keep isl_##EL##_list *list);	\
void isl_##EL##_list_dump(__isl_keep isl_##EL##_list *list);

#define ISL_DECLARE_LIST(EL)						\
	ISL_DECLARE_LIST_TYPE(EL)					\
	ISL_DECLARE_LIST_FN(EL)

#if defined(__cplusplus)
}
#endif

#endif
