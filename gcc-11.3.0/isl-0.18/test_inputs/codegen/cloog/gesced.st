domain: "[M, N] -> { S3[i0, i1] : i0 >= 1 and i0 <= N and i1 >= 1 and i1 <= M; S1[i0] : i0 >= 1 and i0 <= N; S2[i0, i1] : i0 >= 1 and i0 <= N and i1 >= 1 and i1 <= M }"
child:
  context: "[M, N] -> { [] : N <= M and M >= 2 and N >= 2 }"
  child:
    schedule: "[M, N] -> [{ S2[i0, i1] -> [(N + i1)]; S3[i0, i1] -> [(2N + i1)]; S1[i0] -> [(i0)] }]"
    options: "[M, N] -> { separate[i0] }"
