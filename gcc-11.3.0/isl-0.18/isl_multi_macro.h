#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef EL
#define EL CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)
#define xMULTI(BASE) isl_multi_ ## BASE
#define MULTI(BASE) xMULTI(BASE)
#undef DOM
#define DOM CAT(isl_,DOMBASE)
