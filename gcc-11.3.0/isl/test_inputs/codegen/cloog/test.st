domain: "[M, N] -> { S1[i0, i1] : i0 >= 1 and i0 <= N and i1 >= 1 and i1 <= M; S2[i0, i0] : i0 >= 3 and i0 <= N }"
child:
  context: "[M, N] -> { [] : N >= M and M >= 4 }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S1[i0, i1] }"
      - filter: "[M, N] -> { S2[i0, i1] }"
