domain: "[M, N] -> { S1[i0, i1] : i1 >= 0 and i1 <= i0 and i0 <= M; S2[i0, i1] : i1 >= 0 and i1 <= i0 and i0 <= M and i1 <= N }"
child:
  context: "[M, N] -> { [] : N <= M and N >= 1 }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S1[i0, i1] }"
      - filter: "[M, N] -> { S2[i0, i1] }"
