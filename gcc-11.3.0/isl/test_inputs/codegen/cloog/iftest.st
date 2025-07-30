domain: "[m, n] -> { S1[i0] : (i0 >= m and i0 >= 1 and i0 <= n) or (i0 >= 1 and i0 <= n and i0 <= 2m) }"
child:
  context: "[m, n] -> { [] }"
  child:
    schedule: "[m, n] -> [{ S1[i0] -> [(i0)] }]"
    options: "[m, n] -> { separate[i0] }"
