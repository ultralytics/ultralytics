domain: "[n] -> { S1[j, 0, 0] : j >= 0 and j <= n; S2[0, l, 0] : l >= 0 and l <= n }"
child:
  context: "[n] -> { [] }"
  child:
    sequence:
    - filter: "[n] -> { S1[i0, i1, i2] }"
      child:
        schedule: "[n] -> [{ S1[i0, i1, i2] -> [(i0)] }]"
        options: "[n] -> { separate[i0] }"
    - filter: "[n] -> { S2[i0, i1, i2] }"
      child:
        schedule: "[n] -> [{ S2[i0, i1, i2] -> [(i1)] }]"
        options: "[n] -> { separate[i0] }"
