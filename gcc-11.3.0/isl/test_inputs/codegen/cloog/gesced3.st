domain: "[M, N] -> { S1[i0] : i0 >= 1 and i0 <= N; S2[i0] : i0 >= 1 and i0 <= N }"
child:
  context: "[M, N] -> { [] : N >= M and M >= 2 }"
  child:
    schedule: "[M, N] -> [{ S2[i0] -> [(2M + i0)]; S1[i0] -> [(M + i0)] }]"
    options: "[M, N] -> { separate[i0] }"
