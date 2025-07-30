# Example from the manual
domain: "{ A[i,j] : 0 <= i,j and i + j <= 100 }"
child:
  schedule: "[{ A[i,j] -> [floor(i/10)] }, \
	{ A[i,j] -> [floor(j/10)] }, \
	{ A[i,j] -> [i] }, { A[i,j] -> [j] }]"
  options: "{ isolate[[] -> [a,b,c,d]] : 0 <= 10a,10b and \
	10a+9+10b+9 <= 100; [isolate[] -> unroll[3]] }"
