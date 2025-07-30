domain: "[M, N] -> { S1[i0] : (i0 >= 1 and i0 <= N and i0 <= 2M) or (i0 >= M and i0 >= 1 and i0 <= N); S2[i0] : i0 >= 1 + N and i0 <= 2N }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S2[i0] -> [(i0)]; S1[i0] -> [(i0)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S1[i0] }"
      - filter: "[M, N] -> { S2[i0] }"
