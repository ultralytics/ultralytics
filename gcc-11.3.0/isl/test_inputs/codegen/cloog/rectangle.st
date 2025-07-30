domain: "[n] -> { S1[i0, i1] : i0 >= 0 and i0 <= n and i1 >= 0 and i1 <= n }"
child:
  context: "[n] -> { [] : n >= 0 }"
  child:
    schedule: "[n] -> [{ S1[i0, i1] -> [(i0 + i1)] }]"
    options: "[n] -> { separate[i0] }"
