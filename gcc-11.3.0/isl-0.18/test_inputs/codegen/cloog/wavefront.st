domain: "[n, m] -> { S1[i0, i1] : i0 >= 1 and i0 <= n and i1 >= 1 and i1 <= m }"
child:
  context: "[n, m] -> { [] }"
  child:
    schedule: "[n, m] -> [{ S1[i0, i1] -> [(i0 + i1)] }, { S1[i0, i1] -> [(i0)] }]"
    options: "[n, m] -> { separate[i0] }"
