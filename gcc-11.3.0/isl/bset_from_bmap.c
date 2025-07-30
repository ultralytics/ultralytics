#include <isl/map_type.h>

/* Return the basic set that was treated as the basic map "bmap".
 */
static __isl_give isl_basic_set *bset_from_bmap(__isl_take isl_basic_map *bmap)
{
	return (isl_basic_set *) bmap;
}
