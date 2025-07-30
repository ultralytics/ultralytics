domain: "[N] -> { S1[i0, i1] : i0 >= 0 and i0 <= -1 + N and i1 >= 0 and i1 <= -1 + N; S2[i0, i1] : i0 >= 0 and i0 <= -1 + N and i1 >= 0 and i1 <= -1 + N }"
child:
  context: "[N] -> { [] }"
  child:
    schedule: "[N] -> [{ S1[i0, i1] -> [(i0 - i1)]; S2[i0, i1] -> [(1 + i0 - i1)] }]"
    options: "[N] -> { separate[i0] }"
