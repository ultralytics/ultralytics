#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)
#define xLIST(EL) EL ## _list
#define LIST(EL) xLIST(EL)

struct LIST(EL) {
	int ref;
	isl_ctx *ctx;

	int n;

	size_t size;
	struct EL *p[1];
};

__isl_give LIST(EL) *FN(LIST(EL),dup)(__isl_keep LIST(EL) *list);
