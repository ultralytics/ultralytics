domain: "[n] -> { S1[i0, i1] : i0 >= 0 and i0 <= n and i1 >= 0 and i1 <= n }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S1[i0, i1] -> [(2i0 + 3i1)] }, { S1[i0, i1] -> [(2i0 + 2i1)] }]"
    options: "[n] -> { separate[i0] }"
