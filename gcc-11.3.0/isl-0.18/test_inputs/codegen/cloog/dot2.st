domain: "[M, N] -> { S1[i0] : i0 >= 1 and i0 <= M; S2[i0, i1] : i0 >= 1 and i0 <= N and i1 >= 1 and i1 <= M }"
child:
  context: "[M, N] -> { [] : M >= 1 and N >= 1 }"
  child:
    schedule: "[M, N] -> [{ S2[i0, i1] -> [(i0)]; S1[i0] -> [(i0)] }, { S2[i0, i1] -> [(i1)]; S1[i0] -> [(0)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S1[i0] }"
      - filter: "[M, N] -> { S2[i0, i1] }"
